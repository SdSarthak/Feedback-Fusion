"""YouTube comment collection and analysis."""

import re
from urllib.parse import parse_qs, urlparse

from ..config import ConfigError, settings
from ..sentiment import analyze_many, summarize_scores
from ..storage import store_results
from ..text import clean_many, generate_word_cloud, top_keywords

# youtu.be/<id>, /shorts/<id>, /embed/<id> and /live/<id> all carry the id
# in the path rather than the ?v= query parameter.
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")


# YouTube video ids are exactly 11 characters of base64url. Validating this
# stops arbitrary junk ("hello world" is also 11 characters) from being
# forwarded to the Data API, where it returns an opaque error instead of a
# clear "that is not a YouTube link".
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeError(RuntimeError):
    """Raised when the YouTube Data API cannot be queried."""


def _valid(video_id):
    """Return the id only if it is a well-formed YouTube video id."""
    return video_id if video_id and _VIDEO_ID_RE.match(video_id) else None


def _comment_text(item):
    """Pull the comment body out of a commentThreads item.

    One malformed item used to raise KeyError out of the whole pagination
    loop, which silently truncated an otherwise healthy fetch.
    """
    try:
        snippet = item["snippet"]["topLevelComment"]["snippet"]
    except (KeyError, TypeError, IndexError):
        return ""
    if not isinstance(snippet, dict):
        return ""
    return snippet.get("textDisplay") or snippet.get("textOriginal") or ""


def extract_video_id(url):
    """Pull the 11-character video id out of any common YouTube URL form.

    Returns None if the input is not a recognisable YouTube link or does not
    contain a well-formed video id. A bare video id is passed straight
    through.
    """
    if not url:
        return None
    candidate = str(url).strip()

    # Already a bare video id.
    if "/" not in candidate and "." not in candidate:
        return _valid(candidate)

    if "//" not in candidate:
        candidate = "https://" + candidate

    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError:
        # Malformed IPv6 literals and bad ports make urlparse raise.
        return None

    if host in ("youtu.be", "youtube.be"):
        return _valid(parsed.path.lstrip("/").split("/")[0])

    if "youtube" not in host:
        return None

    values = parse_qs(parsed.query).get("v")
    if values and values[0]:
        return _valid(values[0])

    for prefix in _PATH_PREFIXES:
        if parsed.path.startswith(prefix):
            return _valid(parsed.path[len(prefix):].split("/")[0])

    return None


class YouTubeAnalyzer:
    """Fetch YouTube comments and run the full analysis pipeline."""

    # A page whose items are all filtered out still advances the cursor; give
    # up after this many in a row rather than paging forever.
    MAX_EMPTY_PAGES = 3

    def __init__(self, api_key=None, max_comments=None):
        self.api_key = api_key if api_key is not None else settings.youtube_api_key
        self.max_comments = self._bounded_limit(max_comments)

    @staticmethod
    def _bounded_limit(max_comments):
        """Coerce the comment cap to a usable positive integer.

        A zero or negative cap made the fetch loop exit before its first
        request, so the caller was told the video had no comments.
        """
        for candidate in (max_comments, settings.max_comments):
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 1

    def _client(self):
        if not self.api_key:
            raise ConfigError(
                "YOUTUBE_API_KEY is not set. Copy .env.example to .env and add a "
                "YouTube Data API v3 key."
            )
        import googleapiclient.discovery

        return googleapiclient.discovery.build(
            "youtube", "v3", developerKey=self.api_key, cache_discovery=False
        )

    def fetch_comments(self, video_id):
        """Return (comments, partial_reason) for a video.

        `partial_reason` is None on a complete fetch and a message when
        pagination stopped early — quota exhaustion partway through, for
        example. The previous version discarded that reason entirely, so a
        truncated result was indistinguishable from a complete one.
        """
        if not video_id:
            raise YouTubeError("No video id supplied.")

        youtube = self._client()
        comments = []
        page_token = None
        seen_tokens = set()
        empty_pages = 0
        partial_reason = None

        try:
            while len(comments) < self.max_comments:
                remaining = self.max_comments - len(comments)
                response = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=min(100, remaining),
                        pageToken=page_token,
                        textFormat="plainText",
                    )
                    .execute()
                )

                before = len(comments)
                for item in response.get("items") or []:
                    text = _comment_text(item)
                    if text:
                        comments.append(text)

                # Guard against a page that yields nothing and a cursor that
                # never terminates; either one used to spin indefinitely.
                empty_pages = 0 if len(comments) > before else empty_pages + 1
                if empty_pages >= self.MAX_EMPTY_PAGES:
                    break

                page_token = response.get("nextPageToken")
                if not page_token or page_token in seen_tokens:
                    break
                seen_tokens.add(page_token)
        except Exception as exc:
            # Comments disabled, bad key and quota-exceeded all land here;
            # keep whatever we already collected but surface the reason.
            if not comments:
                raise YouTubeError(f"Could not fetch comments: {exc}") from exc
            partial_reason = str(exc)

        return clean_many(comments), partial_reason

    def extract_comments(self, video_id):
        """Return up to `max_comments` top-level comment strings."""
        return self.fetch_comments(video_id)[0]

    def analyze(self, video_id, comments=None, persist=True):
        """Run sentiment, keywords and stats over a video's comments.

        Pass `comments` to analyze an existing list without calling the API.
        """
        partial_reason = None
        if comments is None:
            comments, partial_reason = self.fetch_comments(video_id)

        results = analyze_many(comments)
        storage = (
            store_results(results, "comments", video_id, text_field="comment")
            if persist
            else None
        )

        return {
            "video_id": video_id,
            "comments": comments,
            "results": results,
            "stats": summarize_scores(results),
            "keywords": top_keywords(comments),
            "storage": storage,
            "partial_reason": partial_reason,
        }

    @staticmethod
    def word_cloud(comments):
        return generate_word_cloud(comments)
