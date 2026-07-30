"""Feedback-Fusion: sentiment analysis across YouTube, Amazon and uploaded feedback."""

from .config import ConfigError, Settings, load_settings, settings
from .sentiment import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SentimentResult,
    analyze_many,
    analyze_text,
    classify,
    count_sentiments,
    summarize_scores,
    to_dataframe,
)
from .sources.amazon import AmazonAnalyzer
from .sources.files import load_feedback, load_feedback_frame
from .sources.youtube import YouTubeAnalyzer, extract_video_id
from .storage import store_results
from .summarization import extractive_summary, summarize
from .text import clean_text, generate_word_cloud, keywords_dataframe, top_keywords

__version__ = "1.0.0"

__all__ = [
    "AmazonAnalyzer",
    "ConfigError",
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
    "SentimentResult",
    "Settings",
    "YouTubeAnalyzer",
    "analyze_many",
    "analyze_text",
    "classify",
    "clean_text",
    "count_sentiments",
    "extract_video_id",
    "extractive_summary",
    "generate_word_cloud",
    "keywords_dataframe",
    "load_feedback",
    "load_feedback_frame",
    "load_settings",
    "settings",
    "store_results",
    "summarize",
    "summarize_scores",
    "to_dataframe",
    "top_keywords",
]
