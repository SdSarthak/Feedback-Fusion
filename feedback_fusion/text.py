"""Text cleaning, word clouds and keyword extraction."""

import re

from nltk.corpus import stopwords

from .sentiment import ensure_nltk_data

# YouTube comments arrive as HTML fragments (<br>, &quot;, ...), so strip
# markup and entities before any of it reaches a word cloud.
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")

_stopwords_cache = None


def get_stopwords():
    """Return the English stopword set, downloading the corpus if needed."""
    global _stopwords_cache
    if _stopwords_cache is None:
        ensure_nltk_data()
        _stopwords_cache = set(stopwords.words("english"))
    return _stopwords_cache


def clean_text(text):
    """Strip HTML, URLs and excess whitespace from a single string."""
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", str(text))
    cleaned = _ENTITY_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def clean_many(texts):
    """Clean an iterable of strings, dropping any that become empty."""
    return [c for c in (clean_text(t) for t in texts or []) if c]


def generate_word_cloud(texts, width=800, height=400, background_color="white"):
    """Build a WordCloud from a collection of texts.

    Returns None when there is no usable vocabulary, because WordCloud
    raises ValueError on empty input and that used to crash the page.
    """
    from wordcloud import WordCloud

    corpus = " ".join(clean_many(texts))
    if not corpus.strip():
        return None

    try:
        return WordCloud(
            stopwords=get_stopwords(),
            background_color=background_color,
            width=width,
            height=height,
        ).generate(corpus)
    except ValueError:
        # Raised when every token is a stopword.
        return None


def top_keywords(texts, top_n=10):
    """Rank terms by summed TF-IDF weight across the corpus.

    Returns a list of (keyword, score) tuples, highest first.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = clean_many(texts)
    if not corpus:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max(top_n, 1))
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary: every document was stopwords or punctuation.
        return []

    names = vectorizer.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    ranked = sorted(zip(names, scores), key=lambda pair: pair[1], reverse=True)
    return [(name, round(float(score), 4)) for name, score in ranked[:top_n]]


def keywords_dataframe(texts, top_n=10):
    """Return top keywords as a DataFrame with Keyword / TF-IDF Score."""
    import pandas as pd

    return pd.DataFrame(top_keywords(texts, top_n), columns=["Keyword", "TF-IDF Score"])
