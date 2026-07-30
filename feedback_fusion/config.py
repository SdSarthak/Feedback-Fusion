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


def _get_int(name, default):
    """Read an int from the environment, falling back on malformed values."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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


def load_settings():
    """Build a Settings object from the current environment."""
    return Settings(
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017/").strip(),
        mongo_db=os.getenv("MONGO_DB", "feedback_fusion").strip(),
        storage_enabled=_get_bool("STORAGE_ENABLED", True),
        max_comments=_get_int("MAX_COMMENTS", 500),
        positive_threshold=_get_float("POSITIVE_THRESHOLD", 0.05),
        negative_threshold=_get_float("NEGATIVE_THRESHOLD", -0.05),
        summarizer_model=os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6").strip(),
        summary_max_chars=_get_int("SUMMARY_MAX_CHARS", 3000),
        chrome_headless=_get_bool("CHROME_HEADLESS", True),
        scrape_max_pages=_get_int("SCRAPE_MAX_PAGES", 5),
    )


settings = load_settings()
