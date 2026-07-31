"""Tests for YouTube URL parsing and the analysis pipeline."""

import pytest

from feedback_fusion.config import ConfigError
from feedback_fusion.sources.youtube import YouTubeAnalyzer, YouTubeError, extract_video_id

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

    @pytest.mark.parametrize(
        "value",
        [
            "hello world",          # 11 characters, but not a video id
            "!!!!!!!!!!!",          # 11 punctuation characters
            "https://www.youtube.com/watch?v=short",
            "https://youtu.be/x",
            "https://www.youtube.com/shorts/waytoolongtobeanid",
            "https://www.youtube.com/watch?v=has spaces",
        ],
    )
    def test_rejects_malformed_video_ids(self, value):
        """Junk must not be forwarded to the Data API as if it were an id."""
        assert extract_video_id(value) is None

    def test_accepts_ids_with_dashes_and_underscores(self):
        assert extract_video_id("https://youtu.be/a-b_c1D2E3f") == "a-b_c1D2E3f"

    def test_malformed_url_does_not_raise(self):
        assert extract_video_id("https://[oops/watch?v=x") is None


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

    @pytest.mark.parametrize("bad", [0, -5, "not-a-number", None])
    def test_unusable_comment_cap_falls_back(self, bad):
        """A cap of 0 or -5 made the fetch loop exit before its first request,
        reporting an empty comment section for a busy video."""
        assert YouTubeAnalyzer(api_key="x", max_comments=bad).max_comments > 0


def _thread_item(text):
    return {"snippet": {"topLevelComment": {"snippet": {"textDisplay": text}}}}


class _FakeRequest:
    def __init__(self, page):
        self._page = page

    def execute(self):
        if isinstance(self._page, Exception):
            raise self._page
        return self._page


class _PagedComments:
    """Stand-in for the commentThreads endpoint, backed by a fixed list."""

    def __init__(self, comments, fail_after=None, page=None):
        self.comments = list(comments)
        self.fail_after = fail_after
        self.forced_page = page
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeRequest(self._build(kwargs))

    def _build(self, kwargs):
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            return RuntimeError("quotaExceeded")
        if self.forced_page is not None:
            return self.forced_page
        start = int(kwargs.get("pageToken") or 0)
        chunk = self.comments[start:start + int(kwargs["maxResults"])]
        response = {"items": [_thread_item(c) for c in chunk]}
        if start + len(chunk) < len(self.comments):
            response["nextPageToken"] = str(start + len(chunk))
        return response


class _FakeYouTube:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def commentThreads(self):
        return self.endpoint


def build(comments, max_comments=None, **kwargs):
    endpoint = _PagedComments(comments, **kwargs)
    analyzer = YouTubeAnalyzer(api_key="fake", max_comments=max_comments)
    analyzer._client = lambda: _FakeYouTube(endpoint)
    return analyzer, endpoint


class TestCommentPagination:
    def test_pages_until_the_cap_is_reached(self):
        analyzer, endpoint = build([f"comment number {i}" for i in range(250)], 150)
        comments, partial = analyzer.fetch_comments(VIDEO_ID)
        assert len(comments) == 150
        assert partial is None
        assert len(endpoint.calls) > 1

    def test_never_requests_an_out_of_range_page_size(self):
        """maxResults outside 1..100 is rejected by the Data API."""
        analyzer, endpoint = build([f"comment number {i}" for i in range(250)], 150)
        analyzer.fetch_comments(VIDEO_ID)
        assert all(1 <= call["maxResults"] <= 100 for call in endpoint.calls)

    def test_stops_when_the_video_runs_out_of_comments(self):
        analyzer, _ = build([f"comment number {i}" for i in range(30)], 500)
        comments, partial = analyzer.fetch_comments(VIDEO_ID)
        assert len(comments) == 30
        assert partial is None

    def test_mid_fetch_failure_keeps_results_and_reports_why(self):
        analyzer, _ = build([f"comment number {i}" for i in range(250)], 250, fail_after=1)
        comments, partial = analyzer.fetch_comments(VIDEO_ID)
        assert len(comments) == 100
        assert partial is not None
        assert "quota" in partial.lower()

    def test_first_page_failure_raises(self):
        analyzer, _ = build(["one comment"], 10, fail_after=0)
        with pytest.raises(YouTubeError):
            analyzer.fetch_comments(VIDEO_ID)

    def test_repeating_page_token_does_not_loop_forever(self):
        page = {"items": [_thread_item("stuck here")], "nextPageToken": "same"}
        analyzer, endpoint = build([], 500, page=page)
        comments, _ = analyzer.fetch_comments(VIDEO_ID)
        # Two calls: the repeat is only detectable once the token comes back.
        assert len(endpoint.calls) == 2
        assert comments == ["stuck here", "stuck here"]

    def test_empty_pages_do_not_loop_forever(self):
        page = {"items": [], "nextPageToken": "next"}
        analyzer, endpoint = build([], 500, page=page)
        comments, _ = analyzer.fetch_comments(VIDEO_ID)
        assert comments == []
        assert len(endpoint.calls) <= YouTubeAnalyzer.MAX_EMPTY_PAGES

    def test_malformed_items_are_skipped_not_fatal(self):
        page = {"items": [{"snippet": {}}, None, _thread_item("this one is fine")]}
        analyzer, _ = build([], 50, page=page)
        comments, partial = analyzer.fetch_comments(VIDEO_ID)
        assert comments == ["this one is fine"]
        assert partial is None

    def test_analyze_surfaces_a_partial_fetch(self):
        analyzer, _ = build([f"comment number {i}" for i in range(250)], 250, fail_after=1)
        payload = analyzer.analyze(VIDEO_ID, persist=False)
        assert payload["partial_reason"]
        assert payload["stats"]["total"] == 100

    def test_analyze_with_supplied_comments_has_no_partial_reason(self):
        analyzer = YouTubeAnalyzer(api_key="")
        payload = analyzer.analyze(VIDEO_ID, comments=["Nice work!"], persist=False)
        assert payload["partial_reason"] is None

    def test_blank_video_id_raises(self):
        analyzer = YouTubeAnalyzer(api_key="fake")
        with pytest.raises(YouTubeError):
            analyzer.fetch_comments("")
