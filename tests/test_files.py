"""Tests for loading and analysing uploaded feedback files."""

import io
import json

import pandas as pd
import pytest

from feedback_fusion.sources.files import (
    FeedbackFileError,
    analyze_frame,
    load_feedback,
    load_feedback_frame,
)

ROWS = [
    {"feedback": "Great place to work, very supportive.", "category": "Employee"},
    {"feedback": "Customer service was slow, not happy.", "category": "Customer"},
    {"feedback": "I love the products and the team.", "category": "Employee"},
    {"feedback": "The meeting is on Tuesday.", "category": "Customer"},
]


def csv_upload(rows=None, name="feedback.csv"):
    frame = pd.DataFrame(rows if rows is not None else ROWS)
    buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    buffer.name = name
    return buffer


def json_upload(rows=None, name="feedback.json"):
    buffer = io.BytesIO(json.dumps(rows if rows is not None else ROWS).encode("utf-8"))
    buffer.name = name
    return buffer


class TestLoadFeedbackFrame:
    def test_loads_csv(self):
        assert len(load_feedback_frame(csv_upload())) == 4

    def test_loads_json(self):
        assert len(load_feedback_frame(json_upload())) == 4

    def test_rejects_unsupported_extension(self):
        buffer = io.BytesIO(b"nope")
        buffer.name = "notes.txt"
        with pytest.raises(FeedbackFileError):
            load_feedback_frame(buffer)

    def test_rejects_empty_file(self):
        with pytest.raises(FeedbackFileError):
            load_feedback_frame(json_upload([]))


class TestColumnResolution:
    def test_finds_standard_columns(self):
        _, feedback_col, category_col = load_feedback(csv_upload())
        assert feedback_col == "feedback"
        assert category_col == "category"

    def test_accepts_alternative_names(self):
        rows = [{"review": "Excellent!", "type": "Customer"}]
        _, feedback_col, category_col = load_feedback(csv_upload(rows))
        assert feedback_col == "review"
        assert category_col == "type"

    def test_is_case_insensitive(self):
        rows = [{"Feedback": "Excellent!", "Category": "Customer"}]
        _, feedback_col, category_col = load_feedback(csv_upload(rows))
        assert feedback_col == "Feedback"
        assert category_col == "Category"

    def test_category_optional(self):
        rows = [{"feedback": "Excellent!"}]
        _, feedback_col, category_col = load_feedback(csv_upload(rows))
        assert feedback_col == "feedback"
        assert category_col is None

    def test_missing_feedback_column_raises(self):
        rows = [{"title": "x", "score": 1}]
        with pytest.raises(FeedbackFileError, match="No feedback column"):
            load_feedback(csv_upload(rows))


class TestAnalyzeFrame:
    def test_overall_stats(self):
        frame, feedback_col, category_col = load_feedback(csv_upload())
        payload = analyze_frame(frame, feedback_col, category_col)
        assert payload["stats"]["total"] == 4
        assert payload["keywords"]

    def test_breaks_down_by_category(self):
        frame, feedback_col, category_col = load_feedback(csv_upload())
        payload = analyze_frame(frame, feedback_col, category_col)
        assert set(payload["by_category"]) == {"Employee", "Customer"}
        assert payload["by_category"]["Employee"]["counts"]["Positive"] == 2

    def test_no_category_gives_empty_breakdown(self):
        frame, feedback_col, category_col = load_feedback(csv_upload([{"feedback": "Nice!"}]))
        payload = analyze_frame(frame, feedback_col, category_col)
        assert payload["by_category"] == {}

    def test_skips_nan_rows(self):
        frame = pd.DataFrame({"feedback": ["Great!", None, "Awful!"]})
        payload = analyze_frame(frame, "feedback")
        assert payload["stats"]["total"] == 2
