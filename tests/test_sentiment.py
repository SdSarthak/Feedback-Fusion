"""Tests for the VADER sentiment layer."""

import pytest

from feedback_fusion.sentiment import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    analyze_many,
    analyze_text,
    classify,
    count_sentiments,
    summarize_scores,
    to_dataframe,
)
from feedback_fusion.text import clean_text


class TestEscapedNegation:
    """Regression: entity deletion split contractions and flipped the label."""

    @pytest.mark.parametrize(
        "raw",
        ["I don&#39;t like it at all", "This isn&#39;t good", "I wouldn&#39;t recommend it"],
    )
    def test_escaped_contractions_keep_their_negation(self, raw):
        assert analyze_text(clean_text(raw)).sentiment == NEGATIVE


class TestClassify:
    def test_positive_above_threshold(self):
        assert classify(0.8) == POSITIVE

    def test_negative_below_threshold(self):
        assert classify(-0.8) == NEGATIVE

    def test_neutral_inside_band(self):
        assert classify(0.0) == NEUTRAL
        assert classify(0.05) == NEUTRAL
        assert classify(-0.05) == NEUTRAL

    def test_custom_thresholds(self):
        assert classify(0.3, positive_threshold=0.5) == NEUTRAL
        assert classify(0.6, positive_threshold=0.5) == POSITIVE


class TestAnalyzeText:
    def test_clearly_positive(self):
        result = analyze_text("This is absolutely wonderful, I love it!")
        assert result.sentiment == POSITIVE
        assert result.compound > 0.05

    def test_clearly_negative(self):
        result = analyze_text("This is awful, terrible and a complete waste of money.")
        assert result.sentiment == NEGATIVE
        assert result.compound < -0.05

    def test_empty_string_is_neutral(self):
        result = analyze_text("")
        assert result.sentiment == NEUTRAL
        assert result.compound == 0.0

    def test_none_is_neutral(self):
        assert analyze_text(None).sentiment == NEUTRAL

    def test_whitespace_is_neutral(self):
        assert analyze_text("     ").sentiment == NEUTRAL

    def test_as_record_uses_custom_field(self):
        record = analyze_text("Great!").as_record("comment")
        assert "comment" in record
        assert record["sentiment"] == POSITIVE
        assert "compound" in record


class TestAnalyzeMany:
    def test_skips_blanks_and_none(self):
        results = analyze_many(["Good stuff", "", None, "   ", "Bad stuff"])
        assert len(results) == 2

    def test_handles_empty_input(self):
        assert analyze_many([]) == []
        assert analyze_many(None) == []

    def test_coerces_non_strings(self):
        results = analyze_many([123, 45.6])
        assert len(results) == 2


class TestAggregation:
    def test_count_sentiments_covers_all_labels(self):
        counts = count_sentiments(analyze_many(["Wonderful!"]))
        assert set(counts) == {POSITIVE, NEGATIVE, NEUTRAL}
        assert counts[POSITIVE] == 1

    def test_summarize_scores_on_empty_list(self):
        stats = summarize_scores([])
        assert stats["total"] == 0
        assert stats["average_compound"] == 0.0
        assert stats["overall"] == NEUTRAL
        assert stats["percentages"][POSITIVE] == 0.0

    def test_summarize_scores_percentages_sum_to_100(self):
        stats = summarize_scores(analyze_many(["Great!", "Awful!", "A table."]))
        assert stats["total"] == 3
        assert sum(stats["percentages"].values()) == pytest.approx(100.0, abs=0.1)

    def test_overall_reflects_majority(self):
        stats = summarize_scores(analyze_many(["Fantastic!", "Wonderful!", "I love it!"]))
        assert stats["overall"] == POSITIVE


class TestDataFrame:
    def test_empty_results_give_empty_frame_with_columns(self):
        frame = to_dataframe([], text_field="comment")
        assert list(frame.columns) == ["comment", "sentiment", "compound"]
        assert frame.empty

    def test_columns_and_rows(self):
        frame = to_dataframe(analyze_many(["Great!", "Awful!"]), text_field="review")
        assert list(frame.columns) == ["review", "sentiment", "compound"]
        assert len(frame) == 2
