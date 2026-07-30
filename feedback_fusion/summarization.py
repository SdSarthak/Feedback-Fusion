"""Abstractive summarization of large feedback sets.

`transformers` and its torch backend are heavy and optional, so the import
is deferred and a frequency-based extractive fallback covers installs that
do not have them.
"""

import heapq
import re

from .config import settings
from .text import clean_many, get_stopwords

_summarizer = None
_summarizer_failed = False

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z']+")


class SummaryUnavailable(RuntimeError):
    """Raised when no summarization backend can be constructed."""


def _build_corpus(texts, max_chars=None):
    """Join and truncate texts to a model-friendly size."""
    limit = settings.summary_max_chars if max_chars is None else max_chars
    corpus = " ".join(clean_many(texts))
    if limit and len(corpus) > limit:
        # Cut on a word boundary so the model never sees a half word.
        corpus = corpus[:limit].rsplit(" ", 1)[0]
    return corpus.strip()


def get_summarizer():
    """Lazily construct the HuggingFace summarization pipeline.

    Returns None (and remembers the failure) when transformers/torch are
    unavailable, so callers can fall back without paying the import cost
    on every request.
    """
    global _summarizer, _summarizer_failed
    if _summarizer is not None or _summarizer_failed:
        return _summarizer
    try:
        from transformers import pipeline

        _summarizer = pipeline("summarization", model=settings.summarizer_model, device=-1)
    except Exception:
        _summarizer_failed = True
        _summarizer = None
    return _summarizer


def extractive_summary(texts, max_sentences=5):
    """Frequency-scored extractive summary; pure Python, no model needed.

    Scores each sentence by the summed frequency of its non-stopword terms,
    then returns the best sentences in their original order.
    """
    corpus = _build_corpus(texts)
    if not corpus:
        return ""

    sentences = [s.strip() for s in _SENTENCE_RE.split(corpus) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    stop = get_stopwords()
    frequencies = {}
    for word in _WORD_RE.findall(corpus.lower()):
        if word in stop or len(word) < 3:
            continue
        frequencies[word] = frequencies.get(word, 0) + 1

    if not frequencies:
        return " ".join(sentences[:max_sentences])

    peak = max(frequencies.values())
    scored = []
    for index, sentence in enumerate(sentences):
        words = _WORD_RE.findall(sentence.lower())
        if not words:
            continue
        # Normalise by length so long sentences do not automatically win.
        score = sum(frequencies.get(w, 0) / peak for w in words) / len(words)
        scored.append((score, index, sentence))

    best = heapq.nlargest(max_sentences, scored)
    return " ".join(sentence for _, _, sentence in sorted(best, key=lambda item: item[1]))


def summarize(texts, max_length=250, min_length=60, allow_fallback=True):
    """Summarize feedback, preferring the neural model.

    Falls back to `extractive_summary` when transformers is missing or the
    model errors out. Raises SummaryUnavailable only when the fallback is
    explicitly disabled.
    """
    corpus = _build_corpus(texts)
    if not corpus:
        return ""

    summarizer = get_summarizer()
    if summarizer is not None:
        try:
            # min_length must stay below the input size or the model loops.
            word_count = len(corpus.split())
            bounded_min = max(10, min(min_length, word_count // 2))
            bounded_max = max(bounded_min + 10, min(max_length, word_count))
            output = summarizer(
                corpus,
                max_length=bounded_max,
                min_length=bounded_min,
                do_sample=False,
                truncation=True,
            )
            text = (output[0].get("summary_text") or "").strip()
            if text:
                return text
        except Exception:
            if not allow_fallback:
                raise

    if not allow_fallback:
        raise SummaryUnavailable(
            "No summarization backend available. Install the optional extras "
            "with: pip install -r requirements-ml.txt"
        )
    return extractive_summary(texts)
