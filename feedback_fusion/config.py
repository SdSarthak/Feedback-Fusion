"""Environment-driven configuration for Feedback-Fusion.

Every secret and deployment-specific value is read from the environment so
that nothing sensitive has to live in the source tree. See `.env.example`
for the full list of supported variables.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a .env file from the project root if one is present. Values already
# set in the real environment always win, which keeps container/CI overrides
# working without editing files.
load_dotenv(override=False)


def _get_int(name, default, minimum=None):
    """Read an int from the environment, falling back on unusable values.

    A value below `minimum` falls back to the default rather than being
    passed through: MAX_COMMENTS=0 or -5 used to reach the YouTube fetch
    loop, which then exited immediately and reported "no comments found"
    for a video that had thousands.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _get_float(name, default, minimum=None, maximum=None):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value != value:  # NaN: every comparison against it is False.
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


def _get_bool(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    youtube_api_key: str
    mongo_uri: str
    mongo_db: str
    storage_enabled: bool
    max_comments: int
    positive_threshold: float
    negative_threshold: float
    summarizer_model: str
    summary_max_chars: int
    chrome_headless: bool
    scrape_max_pages: int

    def require_youtube_api_key(self):
        """Return the YouTube key, or explain how to set it."""
        if not self.youtube_api_key:
            raise ConfigError(
                "YOUTUBE_API_KEY is not set. Copy .env.example to .env and add "
                "a YouTube Data API v3 key from "
                "https://console.cloud.google.com/apis/credentials"
            )
        return self.youtube_api_key


DEFAULT_POSITIVE_THRESHOLD = 0.05
DEFAULT_NEGATIVE_THRESHOLD = -0.05


def load_settings():
    """Build a Settings object from the current environment.

    Out-of-range values fall back to the documented defaults instead of being
    honoured, because every one of them silently corrupts results rather than
    failing loudly.
    """
    # VADER compound scores are bounded to [-1, 1]; anything outside that
    # makes a label unreachable.
    positive = _get_float(
        "POSITIVE_THRESHOLD", DEFAULT_POSITIVE_THRESHOLD, minimum=-1.0, maximum=1.0
    )
    negative = _get_float(
        "NEGATIVE_THRESHOLD", DEFAULT_NEGATIVE_THRESHOLD, minimum=-1.0, maximum=1.0
    )
    if negative > positive:
        # Inverted thresholds make "Neutral" unreachable and label ordinary
        # neutral feedback as Negative. Fall back rather than emit nonsense.
        positive = DEFAULT_POSITIVE_THRESHOLD
        negative = DEFAULT_NEGATIVE_THRESHOLD

    return Settings(
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        # An empty override must not win: MongoClient("") and client[""] both
        # raise, which would take down storage that is meant to be optional.
        mongo_uri=os.getenv("MONGO_URI", "").strip() or "mongodb://localhost:27017/",
        mongo_db=os.getenv("MONGO_DB", "").strip() or "feedback_fusion",
        storage_enabled=_get_bool("STORAGE_ENABLED", True),
        max_comments=_get_int("MAX_COMMENTS", 500, minimum=1),
        positive_threshold=positive,
        negative_threshold=negative,
        summarizer_model=os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6").strip(),
        summary_max_chars=_get_int("SUMMARY_MAX_CHARS", 3000, minimum=1),
        chrome_headless=_get_bool("CHROME_HEADLESS", True),
        scrape_max_pages=_get_int("SCRAPE_MAX_PAGES", 5, minimum=1),
    )


settings = load_settings()
