"""Loading company/employee feedback from uploaded CSV or JSON files."""

import pandas as pd

from ..sentiment import analyze_text, summarize_scores
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


def _rewind(file):
    """Seek a file-like object back to the start before reading it.

    Streamlit hands the *same* UploadedFile object back on every rerun, and a
    rerun is triggered by any widget interaction on the results page (the CSV
    download button, for one). Without this the second read sees an exhausted
    buffer and pandas fails with "No columns to parse from file".
    """
    if hasattr(file, "seek") and getattr(file, "seekable", lambda: False)():
        try:
            file.seek(0)
        except (OSError, ValueError):
            # Closed or non-rewindable stream; let the reader report it.
            pass


def load_feedback_frame(file, filename=None):
    """Read an uploaded CSV or JSON file into a DataFrame.

    `file` may be a path or any file-like object (such as the object
    Streamlit's uploader hands back).
    """
    name = (filename or getattr(file, "name", "") or "").lower()

    if not name.endswith((".csv", ".json")):
        raise FeedbackFileError("Unsupported file type. Upload a .csv or .json file.")

    _rewind(file)
    try:
        df = pd.read_csv(file) if name.endswith(".csv") else pd.read_json(file)
    except FileNotFoundError:
        # Callers distinguish "no such file" from "unreadable file".
        raise
    except ValueError as exc:
        # Covers pandas ParserError, EmptyDataError and UnicodeDecodeError.
        raise FeedbackFileError(f"Could not parse {name or 'file'}: {exc}") from exc
    except OSError as exc:
        raise FeedbackFileError(f"Could not read {name or 'file'}: {exc}") from exc

    if df.empty:
        raise FeedbackFileError("The uploaded file contains no rows.")
    if df.columns.empty:
        raise FeedbackFileError("The uploaded file has no columns.")
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


def _extract_texts(df, column):
    """Extract usable feedback strings plus their row positions.

    Drops nulls *before* casting to str, otherwise pandas turns None/NaN
    into the literal strings "None"/"nan" and they get analysed as feedback.
    Positions are integer offsets, not index labels, so a frame with a
    duplicated or non-unique index still lines up with the category column.
    """
    series = df[column]
    values = series.tolist()
    missing = series.isna().tolist()

    texts, positions = [], []
    for position, value in enumerate(values):
        if missing[position]:
            continue
        text = str(value).strip()
        if text:
            texts.append(text)
            positions.append(position)
    return texts, positions


def _is_missing(value):
    """NaN/None check that tolerates non-scalar cells."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Lists/arrays in a cell make pd.isna return an array.
        return False


def analyze_frame(df, feedback_col, category_col=None):
    """Analyze a loaded feedback frame, optionally split by category.

    Returns overall stats plus a `by_category` mapping when a category
    column is present.

    Each row is scored exactly once. The previous version re-ran the whole
    sentiment pipeline for every category and rebuilt a boolean mask over the
    full frame each time, which is O(rows x categories) work on top of a
    second complete pass over the data.
    """
    texts, positions = _extract_texts(df, feedback_col)
    results = [analyze_text(text) for text in texts]

    payload = {
        "texts": texts,
        "results": results,
        "stats": summarize_scores(results),
        "keywords": top_keywords(texts),
        "by_category": {},
    }

    if category_col and category_col in df.columns:
        labels = df[category_col].tolist()
        grouped = {}
        for position, result in zip(positions, results):
            label = labels[position]
            if _is_missing(label):
                continue
            grouped.setdefault(str(label), []).append(result)
        payload["by_category"] = {
            category: summarize_scores(group) for category, group in grouped.items()
        }

    return payload
