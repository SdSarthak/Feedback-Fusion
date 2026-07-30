"""Tests for text cleaning, word clouds and keyword extraction."""

from feedback_fusion.text import (
    clean_many,
    clean_text,
    generate_word_cloud,
    keywords_dataframe,
    top_keywords,
)


class TestCleanText:
    def test_strips_html_tags(self):
        assert "<br>" not in clean_text("line one<br>line two")

    def test_strips_html_entities(self):
        cleaned = clean_text("she said &quot;hello&quot; &amp; left")
        assert "&quot;" not in cleaned
        assert "&amp;" not in cleaned

    def test_strips_urls(self):
        cleaned = clean_text("check https://example.com/page now")
        assert "example.com" not in cleaned
        assert "check" in cleaned

    def test_collapses_whitespace(self):
        assert clean_text("too    many\n\nspaces") == "too many spaces"

    def test_handles_empty_and_none(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_many_drops_empties(self):
        assert clean_many(["good", "", None, "<br>", "fine"]) == ["good", "fine"]


class TestWordCloud:
    def test_returns_none_for_empty_input(self):
        assert generate_word_cloud([]) is None
        assert generate_word_cloud(["", None]) is None

    def test_returns_none_when_only_stopwords(self):
        # Every token is a stopword, so there is no vocabulary to draw.
        assert generate_word_cloud(["the and of the", "is a the"]) is None

    def test_builds_cloud_from_real_text(self):
        cloud = generate_word_cloud(["battery life is excellent", "excellent battery"])
        assert cloud is not None
        assert cloud.to_array().shape[2] == 3


class TestKeywords:
    def test_empty_corpus_returns_empty_list(self):
        assert top_keywords([]) == []
        assert top_keywords(["", None]) == []

    def test_only_stopwords_returns_empty_list(self):
        assert top_keywords(["the and of", "a is the"]) == []

    def test_ranks_by_score_descending(self):
        keywords = top_keywords(
            ["battery battery battery life", "battery drains fast", "screen is nice"],
            top_n=5,
        )
        assert keywords
        scores = [score for _, score in keywords]
        assert scores == sorted(scores, reverse=True)
        assert "battery" in [name for name, _ in keywords]

    def test_respects_top_n(self):
        corpus = ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"]
        assert len(top_keywords(corpus, top_n=3)) <= 3

    def test_dataframe_shape(self):
        frame = keywords_dataframe(["great camera quality", "camera is great"], top_n=4)
        assert list(frame.columns) == ["Keyword", "TF-IDF Score"]
        assert len(frame) <= 4
