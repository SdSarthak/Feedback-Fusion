"""Tests for the Amazon analyzer's driver resolution and analysis pipeline.

None of these start a real browser; the selenium surface is faked so the
fallback logic can be exercised on machines without Chrome.
"""

import pytest

from feedback_fusion.sources.amazon import AmazonAnalyzer, ScraperUnavailable


class FakeOptions:
    def __init__(self):
        self.arguments = []

    def add_argument(self, value):
        self.arguments.append(value)


class FakeService:
    def __init__(self, path=None):
        self.path = path


class FakeDriver:
    def __init__(self, service, options):
        self.service = service
        self.options = options


class FakeWebdriver:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.attempts = []

    def Chrome(self, service=None, options=None):
        self.attempts.append(service.path)
        if service.path in self.fail_for:
            raise RuntimeError(f"driver {service.path} does not match Chrome")
        return FakeDriver(service, options)


def fake_selenium(webdriver):
    return lambda: (webdriver, FakeOptions, FakeService, object())


class TestDriverResolution:
    def test_prefers_selenium_manager(self, monkeypatch):
        """Selenium Manager matches the installed Chrome; webdriver_manager's
        cached driver goes stale and used to be tried first."""
        webdriver = FakeWebdriver()
        analyzer = AmazonAnalyzer()
        monkeypatch.setattr(analyzer, "_selenium", fake_selenium(webdriver))
        monkeypatch.setattr(
            AmazonAnalyzer, "_webdriver_manager_service", staticmethod(
                lambda Service: pytest.fail("webdriver_manager must not be reached")
            )
        )
        assert isinstance(analyzer._driver(), FakeDriver)
        assert webdriver.attempts == [None]

    def test_falls_back_to_webdriver_manager(self, monkeypatch):
        webdriver = FakeWebdriver(fail_for={None})
        analyzer = AmazonAnalyzer()
        monkeypatch.setattr(analyzer, "_selenium", fake_selenium(webdriver))
        monkeypatch.setattr(
            AmazonAnalyzer,
            "_webdriver_manager_service",
            staticmethod(lambda Service: Service("/cached/chromedriver")),
        )
        assert isinstance(analyzer._driver(), FakeDriver)
        assert webdriver.attempts == [None, "/cached/chromedriver"]

    def test_reports_every_failure_when_no_driver_starts(self, monkeypatch):
        webdriver = FakeWebdriver(fail_for={None, "/cached/chromedriver"})
        analyzer = AmazonAnalyzer()
        monkeypatch.setattr(analyzer, "_selenium", fake_selenium(webdriver))
        monkeypatch.setattr(
            AmazonAnalyzer,
            "_webdriver_manager_service",
            staticmethod(lambda Service: Service("/cached/chromedriver")),
        )
        with pytest.raises(ScraperUnavailable) as info:
            analyzer._driver()
        message = str(info.value)
        assert "selenium-manager" in message
        assert "webdriver-manager" in message

    def test_missing_webdriver_manager_is_not_fatal(self, monkeypatch):
        webdriver = FakeWebdriver()
        analyzer = AmazonAnalyzer()
        monkeypatch.setattr(analyzer, "_selenium", fake_selenium(webdriver))
        assert isinstance(analyzer._driver(), FakeDriver)

    def test_missing_selenium_raises_scraper_unavailable(self, monkeypatch):
        analyzer = AmazonAnalyzer()

        def explode():
            raise ScraperUnavailable("Selenium is not installed.")

        monkeypatch.setattr(analyzer, "_selenium", explode)
        with pytest.raises(ScraperUnavailable):
            analyzer.scrape_reviews("https://www.amazon.in/dp/B0TEST")

    def test_headless_flag_reaches_the_options(self, monkeypatch):
        webdriver = FakeWebdriver()
        analyzer = AmazonAnalyzer(headless=True)
        monkeypatch.setattr(analyzer, "_selenium", fake_selenium(webdriver))
        assert "--headless=new" in analyzer._driver().options.arguments

        visible = AmazonAnalyzer(headless=False)
        monkeypatch.setattr(visible, "_selenium", fake_selenium(FakeWebdriver()))
        assert "--headless=new" not in visible._driver().options.arguments


class TestScrapeGuards:
    @pytest.mark.parametrize("url", ["", "   ", None])
    def test_blank_url_is_rejected_before_launching_chrome(self, url):
        with pytest.raises(ValueError):
            AmazonAnalyzer().scrape_reviews(url)


class TestAnalyze:
    def test_supplied_reviews_skip_scraping(self):
        payload = AmazonAnalyzer().analyze(
            reviews=["Battery life is excellent.", "It broke after a week, awful."],
            product_name="Test Widget",
            persist=False,
        )
        assert payload["product_name"] == "Test Widget"
        assert payload["stats"]["total"] == 2
        assert payload["stats"]["counts"]["Positive"] == 1
        assert payload["stats"]["counts"]["Negative"] == 1
        assert payload["storage"] is None

    def test_empty_review_list_does_not_crash(self):
        payload = AmazonAnalyzer().analyze(reviews=[], persist=False)
        assert payload["stats"]["total"] == 0
        assert payload["keywords"] == []
        assert payload["product_name"] == "Unknown product"

    def test_blank_product_name_falls_back(self):
        payload = AmazonAnalyzer().analyze(reviews=["Nice."], product_name="", persist=False)
        assert payload["product_name"] == "Unknown product"

    def test_max_pages_defaults_to_a_positive_number(self):
        assert AmazonAnalyzer().max_pages > 0
