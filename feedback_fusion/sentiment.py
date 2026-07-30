"""VADER-based sentiment analysis.

This module holds the single implementation of the sentiment logic that the
YouTube, Amazon and company-feedback pages all share.
"""

from dataclasses import dataclass, field

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from .config import settings

POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"

SENTIMENT_LABELS = (POSITIVE, NEGATIVE, NEUTRAL)

_analyzer = None


def ensure_nltk_data():
    """Download the NLTK corpora we depend on, only if they are missing.

    The original code called `nltk.download()` on every page render, which
    hit the network on each Streamlit rerun. Checking first makes repeated
    runs instant and lets the app work offline once warmed up.
    """
    resources = (
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("corpora/stopwords", "stopwords"),
    )
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def get_analyzer():
    """Return a lazily-built, process-wide SentimentIntensityAnalyzer."""
    global _analyzer
    if _analyzer is None:
        ensure_nltk_data()
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def classify(compound, positive_threshold=None, negative_threshold=None):
    """Map a VADER compound score onto a sentiment label."""
    high = settings.positive_threshold if positive_threshold is None else positive_threshold
    low = settings.negative_threshold if negative_threshold is None else negative_threshold
    if compound > high:
        return POSITIVE
    if compound < low:
        return NEGATIVE
    return NEUTRAL


@dataclass
class SentimentResult:
    """Sentiment outcome for a single piece of feedback."""

    text: str
    sentiment: str
    scores: dict = field(default_factory=dict)

    @property
    def compound(self):
        return self.scores.get("compound", 0.0)

    def as_record(self, text_field="text"):
        """Flatten to a plain dict suitable for pandas or MongoDB."""
        return {
            text_field: self.text,
            "sentiment": self.sentiment,
            "compound": self.compound,
            "scores": dict(self.scores),
        }


def analyze_text(text):
    """Analyze one string and return a SentimentResult."""
    cleaned = (text or "").strip()
    if not cleaned:
        neutral_scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return SentimentResult(text=cleaned, sentiment=NEUTRAL, scores=neutral_scores)
    scores = get_analyzer().polarity_scores(cleaned)
    return SentimentResult(text=cleaned, sentiment=classify(scores["compound"]), scores=scores)


def analyze_many(texts):
    """Analyze an iterable of strings, skipping blanks and non-strings."""
    results = []
    for item in texts or []:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        results.append(analyze_text(text))
    return results


def count_sentiments(results):
    """Return a label -> count dict covering all three labels."""
    counts = {label: 0 for label in SENTIMENT_LABELS}
    for result in results:
        label = result.sentiment if isinstance(result, SentimentResult) else result.get("sentiment")
        if label in counts:
            counts[label] += 1
    return counts


def summarize_scores(results):
    """Aggregate statistics over a batch of results.

    Returns total, per-label counts, per-label percentages and the mean
    compound score. Safe to call on an empty list.
    """
    counts = count_sentiments(results)
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "counts": counts,
            "percentages": {label: 0.0 for label in SENTIMENT_LABELS},
            "average_compound": 0.0,
            "overall": NEUTRAL,
        }

    compounds = [
        r.compound if isinstance(r, SentimentResult) else r.get("compound", 0.0) for r in results
    ]
    average = sum(compounds) / total
    return {
        "total": total,
        "counts": counts,
        "percentages": {
            label: round(counts[label] * 100.0 / total, 2) for label in SENTIMENT_LABELS
        },
        "average_compound": round(average, 4),
        "overall": classify(average),
    }


def to_dataframe(results, text_field="text"):
    """Build a pandas DataFrame from results, with stable columns."""
    import pandas as pd

    if not results:
        return pd.DataFrame(columns=[text_field, "sentiment", "compound"])
    rows = [r.as_record(text_field) for r in results]
    return pd.DataFrame(rows)[[text_field, "sentiment", "compound"]]
