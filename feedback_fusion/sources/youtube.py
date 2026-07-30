"""YouTube comment collection and analysis."""

from urllib.parse import parse_qs, urlparse

from ..config import ConfigError, settings
from ..sentiment import analyze_many, summarize_scores
from ..storage import store_results
from ..text import clean_many, generate_word_cloud, top_keywords

# youtu.be/<id>, /shorts/<id>, /embed/<id> and /live/<id> all carry the id
# in the path rather than the ?v= query parameter.
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")


class YouTubeError(RuntimeError):
    """Raised when the YouTube Data API cannot be queried."""


def extract_video_id(url):
    """Pull the 11-character video id out of any common YouTube URL form.

    Returns None if the input is not a recognisable YouTube link. A bare
    video id is accepted and passed straight through.
    """
    if not url:
        return None
    candidate = str(url).strip()

    # Already a bare video id.
    if len(candidate) == 11 and "/" not in candidate and "." not in candidate:
        return candidate

    if "//" not in candidate:
        candidate = "https://" + candidate

    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower().removeprefix("www.")

    if host in ("youtu.be", "youtube.be"):
        video_id = parsed.path.lstrip("/").split("/")[0]
        return video_id or None

    if "youtube" not in host:
        return None

    values = parse_qs(parsed.query).get("v")
    if values and values[0]:
        return values[0]

    for prefix in _PATH_PREFIXES:
        if parsed.path.startswith(prefix):
            video_id = parsed.path[len(prefix):].split("/")[0]
            return video_id or None

    return None


class YouTubeAnalyzer:
    """Fetch YouTube comments and run the full analysis pipeline."""

    def __init__(self, api_key=None, max_comments=None):
        self.api_key = api_key if api_key is not None else settings.youtube_api_key
        self.max_comments = max_comments or settings.max_comments

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

    def extract_comments(self, video_id):
        """Return up to `max_comments` top-level comment strings.

        Pages through the API until the cap is reached. Errors are raised
        rather than swallowed so the UI can show why nothing came back.
        """
        youtube = self._client()
        comments = []
        page_token = None

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

                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(snippet.get("textDisplay", ""))

                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except Exception as exc:
            # Comments disabled, bad key and quota-exceeded all land here;
            # keep whatever we already collected but surface the reason.
            if not comments:
                raise YouTubeError(f"Could not fetch comments: {exc}") from exc

        return clean_many(comments)

    def analyze(self, video_id, comments=None, persist=True):
        """Run sentiment, keywords and stats over a video's comments.

        Pass `comments` to analyze an existing list without calling the API.
        """
        if comments is None:
            comments = self.extract_comments(video_id)

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
        }

    @staticmethod
    def word_cloud(comments):
        return generate_word_cloud(comments)
