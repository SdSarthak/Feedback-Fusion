"""Amazon product review scraping and analysis.

Selenium and webdriver-manager are optional extras, so the imports live
inside the methods that need them.
"""

import time

from ..config import settings
from ..sentiment import analyze_many, summarize_scores
from ..storage import store_results
from ..text import clean_many, generate_word_cloud, top_keywords

REVIEW_XPATH = "//span[@data-hook='review-body']"
TITLE_ID = "productTitle"
NEXT_PAGE_XPATH = "//li[@class='a-last']/a"


class ScraperUnavailable(RuntimeError):
    """Raised when Selenium or a Chrome driver cannot be set up."""


class AmazonAnalyzer:
    """Scrape Amazon product reviews and run the analysis pipeline."""

    def __init__(self, headless=None, max_pages=None):
        self.headless = settings.chrome_headless if headless is None else headless
        self.max_pages = max_pages or settings.scrape_max_pages

    def _driver(self):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
        except ImportError as exc:
            raise ScraperUnavailable(
                "Selenium is not installed. Install the scraping extras with: "
                "pip install -r requirements-scraping.txt"
            ) from exc

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # Amazon serves a bot-check page to the default Selenium UA.
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )

        try:
            from webdriver_manager.chrome import ChromeDriverManager

            service = Service(ChromeDriverManager().install())
        except Exception:
            # Fall back to a chromedriver already on PATH.
            service = Service()

        try:
            return webdriver.Chrome(service=service, options=options)
        except Exception as exc:
            raise ScraperUnavailable(f"Could not start Chrome: {exc}") from exc

    def scrape_reviews(self, product_url):
        """Return (product_name, reviews) for an Amazon product page.

        Pages through review lists up to `max_pages`. The original version
        looped with `while True` and could spin forever on pages where the
        next button never disappears; this one is bounded and also stops
        when a page yields no new reviews.
        """
        from selenium.webdriver.common.by import By

        driver = self._driver()
        reviews = []
        product_name = "Unknown product"

        try:
            driver.get(product_url)
            time.sleep(3)

            try:
                product_name = driver.find_element(By.ID, TITLE_ID).text.strip() or product_name
            except Exception:
                pass

            for _ in range(self.max_pages):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)

                found = [
                    element.text.strip()
                    for element in driver.find_elements(By.XPATH, REVIEW_XPATH)
                    if element.text.strip()
                ]
                new = [r for r in found if r not in reviews]
                if not new:
                    break
                reviews.extend(new)

                next_links = driver.find_elements(By.XPATH, NEXT_PAGE_XPATH)
                if not next_links:
                    break
                try:
                    next_links[0].click()
                except Exception:
                    break
                time.sleep(2.5)
        finally:
            try:
                driver.quit()
            except Exception:
                pass

        return product_name, clean_many(reviews)

    def analyze(self, product_url=None, reviews=None, product_name=None, persist=True):
        """Analyze reviews, scraping them first when not supplied."""
        if reviews is None:
            product_name, reviews = self.scrape_reviews(product_url)
        product_name = product_name or "Unknown product"

        results = analyze_many(reviews)
        storage = (
            store_results(results, "reviews", product_name, text_field="review")
            if persist
            else None
        )

        return {
            "product_name": product_name,
            "reviews": reviews,
            "results": results,
            "stats": summarize_scores(results),
            "keywords": top_keywords(reviews),
            "storage": storage,
        }

    @staticmethod
    def word_cloud(reviews):
        return generate_word_cloud(reviews)
