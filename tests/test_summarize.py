"""Tests for the summarization layer and its extractive fallback."""

import pytest

from feedback_fusion import summarization as summarize_module
from feedback_fusion.summarization import (
    SummaryUnavailable,
    _build_corpus,
    extractive_summary,
    summarize,
)

PARAGRAPH = [
    "The battery life on this laptop is outstanding and lasts a full working day.",
    "Build quality feels premium and the keyboard is comfortable to type on.",
    "The screen is bright but the colours look washed out in direct sunlight.",
    "Customer support was slow to respond when I raised a warranty question.",
    "Overall it is good value for money given the specification on offer.",
    "Shipping arrived two days earlier than the estimate which was a nice surprise.",
]


class TestBuildCorpus:
    def test_joins_and_cleans(self):
        corpus = _build_corpus(["one <br> two", "three"])
        assert "<br>" not in corpus
        assert "one" in corpus and "three" in corpus

    def test_truncates_on_word_boundary(self):
        corpus = _build_corpus(["word " * 500], max_chars=50)
        assert len(corpus) <= 50
        assert not corpus.endswith("wor")

    def test_empty_input(self):
        assert _build_corpus([]) == ""
        assert _build_corpus(["", None]) == ""


class TestExtractiveSummary:
    def test_empty_input_returns_empty_string(self):
        assert extractive_summary([]) == ""

    def test_short_input_returned_whole(self):
        summary = extractive_summary(["Only one sentence here."], max_sentences=5)
        assert "Only one sentence here." in summary

    def test_limits_sentence_count(self):
        summary = extractive_summary(PARAGRAPH, max_sentences=2)
        assert summary
        assert len(summary) < len(" ".join(PARAGRAPH))

    def test_preserves_original_order(self):
        summary = extractive_summary(PARAGRAPH, max_sentences=3)
        chosen = [s for s in PARAGRAPH if s.rstrip(".") in summary or s in summary]
        positions = [PARAGRAPH.index(s) for s in chosen]
        assert positions == sorted(positions)


class TestSummarize:
    def test_empty_input_returns_empty_string(self):
        assert summarize([]) == ""

    def test_falls_back_when_model_unavailable(self, monkeypatch):
        """With no transformers backend we must still get usable text."""
        monkeypatch.setattr(summarize_module, "get_summarizer", lambda: None)
        summary = summarize(PARAGRAPH)
        assert summary
        assert isinstance(summary, str)

    def test_falls_back_when_model_raises(self, monkeypatch):
        def exploding_summarizer(*args, **kwargs):
            raise RuntimeError("model blew up")

        monkeypatch.setattr(summarize_module, "get_summarizer", lambda: exploding_summarizer)
        summary = summarize(PARAGRAPH)
        assert summary

    def test_uses_model_output_when_available(self, monkeypatch):
        def fake_summarizer(text, **kwargs):
            return [{"summary_text": "A concise model summary."}]

        monkeypatch.setattr(summarize_module, "get_summarizer", lambda: fake_summarizer)
        assert summarize(PARAGRAPH) == "A concise model summary."

    def test_blank_model_output_falls_back(self, monkeypatch):
        def blank_summarizer(text, **kwargs):
            return [{"summary_text": "   "}]

        monkeypatch.setattr(summarize_module, "get_summarizer", lambda: blank_summarizer)
        summary = summarize(PARAGRAPH)
        assert summary.strip()

    def test_malformed_model_output_falls_back(self, monkeypatch):
        for output in ([], [{}], [{"summary_text": None}], "not a list"):
            monkeypatch.setattr(
                summarize_module, "get_summarizer", lambda o=output: (lambda *a, **k: o)
            )
            assert summarize(PARAGRAPH).strip()

    def test_fallback_does_not_reclean_the_corpus(self, monkeypatch):
        """The corpus is cleaned once in summarize and reused by the fallback."""
        calls = []
        original = summarize_module.clean_many

        def counting_clean_many(texts):
            calls.append(1)
            return original(texts)

        monkeypatch.setattr(summarize_module, "clean_many", counting_clean_many)
        monkeypatch.setattr(summarize_module, "get_summarizer", lambda: None)
        summarize(PARAGRAPH)
        assert len(calls) == 1

    def test_model_receives_bounded_lengths(self, monkeypatch):
        seen = {}

        def recording_summarizer(text, **kwargs):
            seen.update(kwargs)
            return [{"summary_text": "ok"}]

        monkeypatch.setattr(
            summarize_module, "get_summarizer", lambda: recording_summarizer
        )
        summarize(["short feedback here"], max_length=250, min_length=60)
        assert seen["min_length"] >= 10
        assert seen["max_length"] > seen["min_length"]

    def test_no_backend_and_no_fallback_raises(self, monkeypatch):
        monkeypatch.setattr(summarize_module, "get_summarizer", lambda: None)
        with pytest.raises(SummaryUnavailable):
            summarize(PARAGRAPH, allow_fallback=False)

    def test_summarizer_is_built_once_under_concurrency(self, monkeypatch):
        """Each pipeline is ~1 GB; two threads must not build two of them."""
        import threading
        import time

        builds = []

        def slow_pipeline(*args, **kwargs):
            builds.append(1)
            time.sleep(0.15)
            return lambda text, **kw: [{"summary_text": "ok"}]

        monkeypatch.setattr(summarize_module, "_summarizer", None)
        monkeypatch.setattr(summarize_module, "_summarizer_failed", False)
        monkeypatch.setitem(
            __import__("sys").modules,
            "transformers",
            type("m", (), {"pipeline": staticmethod(slow_pipeline)}),
        )

        threads = [
            threading.Thread(target=summarize_module.get_summarizer) for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summarize_module._summarizer = None
        summarize_module._summarizer_failed = False
        assert len(builds) == 1


class TestExtractiveGuards:
    def test_non_positive_sentence_count_returns_empty(self):
        assert extractive_summary(PARAGRAPH, max_sentences=0) == ""
        assert extractive_summary(PARAGRAPH, max_sentences=-2) == ""

    def test_only_stopwords_still_returns_text(self):
        texts = ["the and of. a is the. it was so."]
        assert extractive_summary(texts, max_sentences=1)
