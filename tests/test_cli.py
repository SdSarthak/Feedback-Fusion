"""Tests for the command-line interface."""

import json

import pytest

from feedback_fusion.cli import main

SAMPLE = "data/sample_feedback.csv"


class TestFileMode:
    def test_analyses_sample_file(self, capsys):
        assert main([SAMPLE, "--no-store"]) == 0
        output = capsys.readouterr().out
        assert "Total analysed" in output
        assert "Top keywords" in output

    def test_breaks_down_by_category(self, capsys):
        main([SAMPLE, "--no-store"])
        output = capsys.readouterr().out
        assert "Category: Employee" in output
        assert "Category: Customer" in output

    def test_json_output_is_valid(self, capsys):
        assert main([SAMPLE, "--json", "--no-store"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["stats"]["total"] == 20
        assert payload["keywords"]

    def test_missing_file_exits_nonzero(self, capsys):
        assert main(["does_not_exist.csv", "--no-store"]) == 2
        assert "no such file" in capsys.readouterr().err


class TestYouTubeMode:
    def test_invalid_url_exits_nonzero(self, capsys):
        assert main(["--youtube", "https://example.com/nope"]) == 2
        assert "not a valid YouTube URL" in capsys.readouterr().err


class TestArgParsing:
    def test_requires_a_source(self):
        with pytest.raises(SystemExit):
            main([])
