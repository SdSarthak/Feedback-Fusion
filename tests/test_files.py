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

    def test_same_upload_can_be_read_twice(self):
        """Streamlit reuses one UploadedFile across reruns, so the reader must
        rewind it. Without that the second read hit an exhausted buffer."""
        buffer = csv_upload()
        assert len(load_feedback_frame(buffer)) == 4
        assert len(load_feedback_frame(buffer)) == 4

    def test_rewinds_a_partially_consumed_buffer(self):
        buffer = csv_upload()
        buffer.read(5)
        assert len(load_feedback_frame(buffer)) == 4

    def test_json_upload_can_be_read_twice(self):
        buffer = json_upload()
        assert len(load_feedback_frame(buffer)) == 4
        assert len(load_feedback_frame(buffer)) == 4

    def test_unsupported_extension_is_rejected_before_reading(self):
        """An unreadable stream with a bad name must still give the type error."""
        buffer = io.BytesIO(b"nope")
        buffer.name = "notes.txt"
        buffer.close()
        with pytest.raises(FeedbackFileError, match="Unsupported file type"):
            load_feedback_frame(buffer)

    def test_malformed_csv_raises_feedback_error(self):
        buffer = io.BytesIO(b"")
        buffer.name = "broken.csv"
        with pytest.raises(FeedbackFileError):
            load_feedback_frame(buffer)

    def test_unicode_content_survives(self):
        rows = [{"feedback": "Sehr gut — überzeugend! 太好了 🎉", "category": "Kunde"}]
        frame = load_feedback_frame(csv_upload(rows))
        assert "太好了" in frame["feedback"].iloc[0]


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

    def test_category_totals_add_up_to_the_overall_total(self):
        frame, feedback_col, category_col = load_feedback(csv_upload())
        payload = analyze_frame(frame, feedback_col, category_col)
        per_category = sum(s["total"] for s in payload["by_category"].values())
        assert per_category == payload["stats"]["total"]

    def test_rows_with_a_missing_category_are_excluded_from_the_breakdown(self):
        frame = pd.DataFrame(
            {
                "feedback": ["Great!", "Awful!", "Fine."],
                "category": ["Employee", None, "Employee"],
            }
        )
        payload = analyze_frame(frame, "feedback", "category")
        assert payload["stats"]["total"] == 3
        assert set(payload["by_category"]) == {"Employee"}
        assert payload["by_category"]["Employee"]["total"] == 2

    def test_blank_feedback_rows_do_not_shift_category_alignment(self):
        """Rows dropped for empty text must not misalign the category labels."""
        frame = pd.DataFrame(
            {
                "feedback": ["   ", "I love this team.", None, "Terrible experience."],
                "category": ["Employee", "Employee", "Customer", "Customer"],
            }
        )
        payload = analyze_frame(frame, "feedback", "category")
        assert payload["stats"]["total"] == 2
        assert payload["by_category"]["Employee"]["counts"]["Positive"] == 1
        assert payload["by_category"]["Customer"]["counts"]["Negative"] == 1

    def test_handles_a_duplicated_index(self):
        frame = pd.DataFrame(
            {"feedback": ["Great!", "Awful!"], "category": ["Employee", "Customer"]},
            index=[0, 0],
        )
        payload = analyze_frame(frame, "feedback", "category")
        assert payload["stats"]["total"] == 2
        assert set(payload["by_category"]) == {"Employee", "Customer"}

    def test_all_blank_feedback_yields_empty_payload(self):
        frame = pd.DataFrame({"feedback": ["", "   ", None]})
        payload = analyze_frame(frame, "feedback")
        assert payload["texts"] == []
        assert payload["results"] == []
        assert payload["stats"]["total"] == 0
        assert payload["keywords"] == []

    def test_missing_category_column_is_ignored(self):
        frame = pd.DataFrame({"feedback": ["Great!"]})
        payload = analyze_frame(frame, "feedback", "not_a_column")
        assert payload["by_category"] == {}

    def test_numeric_categories_are_stringified(self):
        frame = pd.DataFrame({"feedback": ["Great!", "Awful!"], "category": [1, 2]})
        payload = analyze_frame(frame, "feedback", "category")
        assert set(payload["by_category"]) == {"1", "2"}
