"""Tests for configuration parsing and best-effort storage."""

import dataclasses

import pytest

from feedback_fusion import config as config_module
from feedback_fusion.config import ConfigError, load_settings
from feedback_fusion.sentiment import analyze_many
from feedback_fusion.storage import safe_collection_name, store_results


class TestSettings:
    def test_reads_values_from_environment(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
        monkeypatch.setenv("MAX_COMMENTS", "42")
        monkeypatch.setenv("POSITIVE_THRESHOLD", "0.25")
        settings = load_settings()
        assert settings.youtube_api_key == "test-key"
        assert settings.max_comments == 42
        assert settings.positive_threshold == 0.25

    def test_malformed_numbers_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("MAX_COMMENTS", "not-a-number")
        monkeypatch.setenv("POSITIVE_THRESHOLD", "")
        settings = load_settings()
        assert settings.max_comments == 500
        assert settings.positive_threshold == 0.05

    @pytest.mark.parametrize(
        "raw,expected",
        [("true", True), ("True", True), ("1", True), ("yes", True),
         ("false", False), ("0", False), ("no", False)],
    )
    def test_boolean_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("STORAGE_ENABLED", raw)
        assert load_settings().storage_enabled is expected

    def test_require_youtube_api_key_raises_when_absent(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "")
        with pytest.raises(ConfigError, match="YOUTUBE_API_KEY"):
            load_settings().require_youtube_api_key()

    def test_no_api_key_is_baked_into_the_source(self):
        """Guard against a real key ever being hardcoded again."""
        default = load_settings()
        assert not default.youtube_api_key.startswith("AIza") or bool(
            __import__("os").getenv("YOUTUBE_API_KEY")
        )


class TestSafeCollectionName:
    def test_replaces_unsafe_characters(self):
        name = safe_collection_name("reviews", "Lenovo IdeaPad $500 / 15\"")
        assert "$" not in name
        assert " " not in name
        assert "/" not in name
        assert name.startswith("reviews_")

    def test_handles_empty_identifier(self):
        assert safe_collection_name("reviews", "") == "reviews_unknown"
        assert safe_collection_name("reviews", None) == "reviews_unknown"

    def test_truncates_long_identifiers(self):
        name = safe_collection_name("reviews", "x" * 500)
        assert len(name) <= len("reviews_") + 80


class TestStorage:
    """Settings is a frozen dataclass, so swap the whole object rather than
    mutating an attribute on it."""

    @staticmethod
    def _with_settings(monkeypatch, **overrides):
        import feedback_fusion.storage as storage_module

        replacement = dataclasses.replace(config_module.settings, **overrides)
        monkeypatch.setattr(storage_module, "settings", replacement)
        return storage_module

    def test_returns_falsy_result_when_disabled(self, monkeypatch):
        self._with_settings(monkeypatch, storage_enabled=False)
        result = store_results(analyze_many(["Great!"]), "comments", "abc")
        assert not result
        assert "disabled" in result.reason

    def test_empty_results_are_not_stored(self):
        result = store_results([], "comments", "abc")
        assert not result
        assert result.reason == "nothing to store"

    def test_storage_failure_does_not_raise(self, monkeypatch):
        """An unreachable Mongo must degrade, never crash the analysis."""
        storage_module = self._with_settings(monkeypatch, storage_enabled=True)
        monkeypatch.setattr(storage_module, "get_client", lambda: None)
        result = store_results(analyze_many(["Great!"]), "comments", "abc")
        assert not result
        assert "MongoDB" in result.reason
