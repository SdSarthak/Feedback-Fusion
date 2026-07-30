"""Tests for YouTube URL parsing and the analysis pipeline."""

import pytest

from feedback_fusion.config import ConfigError
from feedback_fusion.sources.youtube import YouTubeAnalyzer, extract_video_id

VIDEO_ID = "dQw4w9WgXcQ"


class TestExtractVideoId:
    @pytest.mark.parametrize(
        "url",
        [
            f"https://www.youtube.com/watch?v={VIDEO_ID}",
            f"http://youtube.com/watch?v={VIDEO_ID}",
            f"https://youtu.be/{VIDEO_ID}",
            f"https://www.youtube.com/shorts/{VIDEO_ID}",
            f"https://www.youtube.com/embed/{VIDEO_ID}",
            f"https://www.youtube.com/live/{VIDEO_ID}",
            f"www.youtube.com/watch?v={VIDEO_ID}",
            VIDEO_ID,
        ],
    )
    def test_recognised_forms(self, url):
        assert extract_video_id(url) == VIDEO_ID

    def test_extra_query_params(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s&list=PL123"
        assert extract_video_id(url) == VIDEO_ID

    def test_whitespace_is_tolerated(self):
        assert extract_video_id(f"  https://youtu.be/{VIDEO_ID}  ") == VIDEO_ID

    @pytest.mark.parametrize("value", [None, "", "   ", "https://example.com/watch?v=abc"])
    def test_rejects_non_youtube_input(self, value):
        assert extract_video_id(value) is None

    def test_youtube_url_without_id_returns_none(self):
        assert extract_video_id("https://www.youtube.com/") is None


class TestYouTubeAnalyzer:
    def test_missing_api_key_raises_config_error(self):
        analyzer = YouTubeAnalyzer(api_key="")
        with pytest.raises(ConfigError):
            analyzer.extract_comments(VIDEO_ID)

    def test_analyze_with_supplied_comments_skips_api(self):
        """Passing comments in must not touch the network or need a key."""
        analyzer = YouTubeAnalyzer(api_key="")
        payload = analyzer.analyze(
            VIDEO_ID,
            comments=["Love this video!", "This was boring and bad.", "Uploaded today."],
            persist=False,
        )
        assert payload["video_id"] == VIDEO_ID
        assert payload["stats"]["total"] == 3
        assert payload["stats"]["counts"]["Positive"] == 1
        assert payload["stats"]["counts"]["Negative"] == 1
        assert payload["storage"] is None

    def test_analyze_handles_empty_comment_list(self):
        analyzer = YouTubeAnalyzer(api_key="")
        payload = analyzer.analyze(VIDEO_ID, comments=[], persist=False)
        assert payload["stats"]["total"] == 0
        assert payload["keywords"] == []

    def test_max_comments_defaults_from_settings(self):
        assert YouTubeAnalyzer(api_key="x").max_comments > 0
