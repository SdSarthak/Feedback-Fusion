"""Loading company/employee feedback from uploaded CSV or JSON files."""

import pandas as pd

from ..sentiment import analyze_many, summarize_scores
from ..text import top_keywords

# Accept whatever the user happened to call their columns.
FEEDBACK_COLUMNS = ("feedback", "review", "comment", "text", "response", "message")
CATEGORY_COLUMNS = ("category", "type", "source", "group", "role")


class FeedbackFileError(ValueError):
    """Raised when an uploaded file cannot be interpreted as feedback."""


def _resolve_column(df, candidates):
    """Find the first matching column, comparing case-insensitively."""
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def load_feedback_frame(file, filename=None):
    """Read an uploaded CSV or JSON file into a DataFrame.

    `file` may be a path or any file-like object (such as the object
    Streamlit's uploader hands back).
    """
    name = (filename or getattr(file, "name", "") or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith(".json"):
            df = pd.read_json(file)
        else:
            raise FeedbackFileError(
                "Unsupported file type. Upload a .csv or .json file."
            )
    except FeedbackFileError:
        raise
    except ValueError as exc:
        raise FeedbackFileError(f"Could not parse {name or 'file'}: {exc}") from exc

    if df.empty:
        raise FeedbackFileError("The uploaded file contains no rows.")
    return df


def load_feedback(file, filename=None):
    """Load a file and normalise it to `feedback` / `category` columns.

    Returns (dataframe, feedback_column, category_column). The category
    column is None when the file does not carry one.
    """
    df = load_feedback_frame(file, filename)

    feedback_col = _resolve_column(df, FEEDBACK_COLUMNS)
    if feedback_col is None:
        raise FeedbackFileError(
            "No feedback column found. Expected one of: "
            + ", ".join(FEEDBACK_COLUMNS)
            + f". Found: {', '.join(str(c) for c in df.columns)}"
        )

    category_col = _resolve_column(df, CATEGORY_COLUMNS)
    return df, feedback_col, category_col


def _column_texts(df, column):
    """Extract usable feedback strings from a column.

    Drops nulls *before* casting to str, otherwise pandas turns None/NaN
    into the literal strings "None"/"nan" and they get analysed as feedback.
    """
    series = df[column].dropna().astype(str)
    return [text for text in (value.strip() for value in series) if text]


def analyze_frame(df, feedback_col, category_col=None):
    """Analyze a loaded feedback frame, optionally split by category.

    Returns overall stats plus a `by_category` mapping when a category
    column is present.
    """
    texts = _column_texts(df, feedback_col)
    results = analyze_many(texts)

    payload = {
        "texts": texts,
        "results": results,
        "stats": summarize_scores(results),
        "keywords": top_keywords(texts),
        "by_category": {},
    }

    if category_col and category_col in df.columns:
        for category in df[category_col].dropna().unique():
            subset = _column_texts(df[df[category_col] == category], feedback_col)
            if not subset:
                continue
            payload["by_category"][str(category)] = summarize_scores(analyze_many(subset))

    return payload
