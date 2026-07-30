# Feedback-Fusion

Feedback analysis platform that turns raw feedback — YouTube comments, Amazon
reviews, or an uploaded CSV/JSON of customer and employee responses — into
sentiment breakdowns, keyword rankings, word clouds and summaries.

Built as a V-semester minor project by CSDS 2026, Acropolis Institute of
Technology and Research.

---

## What it actually does

| Capability | Status | Notes |
|---|---|---|
| VADER sentiment classification (positive / negative / neutral + scores) | Working | Core of the platform |
| Aggregate stats (counts, percentages, mean compound, overall label) | Working | |
| TF-IDF keyword extraction | Working | |
| Word clouds | Working | |
| CSV / JSON feedback upload with per-category breakdown | Working | Flexible column naming |
| YouTube comment analysis | Working | Needs a YouTube Data API v3 key |
| Extractive summarisation | Working | Pure Python, no model download |
| Abstractive summarisation | Optional | Needs `requirements-ml.txt` |
| Amazon review scraping | Best-effort | Amazon actively blocks bots; see below |
| MongoDB persistence | Optional | App degrades gracefully if unavailable |
| Firebase email/password auth | Working | Static frontend pages |
| Command-line interface | Working | `python -m feedback_fusion` |

Things the earlier README advertised that are **not** implemented: emotion
detection, topic modelling, predictive/trend forecasting, anomaly detection,
PDF report export, multi-language analysis and social-media integrations.

---

## Quick start

```bash
git clone https://github.com/SdSarthak/Feedback-Fusion
cd Feedback-Fusion

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

cp .env.example .env         # then edit .env
streamlit run Final.py
```

The app opens at <http://localhost:8501>.

You can try it immediately without any API keys: open the **Company Review**
page and upload `data/sample_feedback.csv`.

### Optional extras

```bash
pip install -r requirements-ml.txt        # neural summarisation (transformers + torch)
pip install -r requirements-scraping.txt  # Amazon scraping (selenium + chrome)
pip install -r requirements-dev.txt       # pytest
```

Without the ML extras the app still summarises, using a frequency-based
extractive summariser instead of a transformer model.

---

## Configuration

All configuration is read from the environment, with `.env` support. Copy
`.env.example` to `.env` and fill it in. **`.env` is gitignored — never commit
real credentials.**

| Variable | Default | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | *(empty)* | YouTube Data API v3 key. Required for the YouTube page. |
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection string. |
| `MONGO_DB` | `feedback_fusion` | Database name. |
| `STORAGE_ENABLED` | `true` | Set `false` to skip persistence entirely. |
| `MAX_COMMENTS` | `500` | Cap on comments/reviews fetched per run. |
| `POSITIVE_THRESHOLD` | `0.05` | VADER compound score above which feedback is positive. |
| `NEGATIVE_THRESHOLD` | `-0.05` | Compound score below which feedback is negative. |
| `SUMMARIZER_MODEL` | `sshleifer/distilbart-cnn-12-6` | HuggingFace summarisation model. |
| `SUMMARY_MAX_CHARS` | `3000` | Input truncation limit for the summariser. |
| `CHROME_HEADLESS` | `true` | Run the scraper's Chrome headless. |
| `SCRAPE_MAX_PAGES` | `5` | Review pages to walk before stopping. |

Getting a YouTube key: [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
→ create an API key → enable **YouTube Data API v3** for the project.

MongoDB is entirely optional. If `mongod` is not running the analysis still
completes and the UI notes that results were not persisted.

---

## Usage

### Streamlit app

```bash
streamlit run Final.py
```

Three analysis pages:

- **YouTube Video** — paste any video URL (`watch?v=`, `youtu.be/`, `/shorts/`,
  `/embed/`, `/live/`) or a bare video ID.
- **Amazon Product** — paste a product URL. Requires the scraping extras.
- **Company Review** — upload a CSV or JSON file.

Each page produces headline metrics, a sentiment distribution chart, a score
histogram, a word cloud, TF-IDF keywords, a results table with CSV download,
and a summary.

### Command line

```bash
# Analyse a feedback file
python -m feedback_fusion data/sample_feedback.csv

# Include a summary, skip MongoDB
python -m feedback_fusion data/sample_feedback.csv --summary --no-store

# Machine-readable output
python -m feedback_fusion data/sample_feedback.csv --json

# Analyse a YouTube video (needs YOUTUBE_API_KEY)
python -m feedback_fusion --youtube https://youtu.be/VIDEO_ID
```

### As a library

```python
from feedback_fusion import analyze_many, summarize_scores, top_keywords

results = analyze_many(["I love this!", "Terrible experience.", "It arrived."])
for r in results:
    print(r.sentiment, round(r.compound, 3), r.text)

print(summarize_scores(results))
# {'total': 3, 'counts': {'Positive': 1, 'Negative': 1, 'Neutral': 1}, ...}

print(top_keywords(["battery life is great", "battery drains fast"]))
```

```python
from feedback_fusion import YouTubeAnalyzer, extract_video_id

analyzer = YouTubeAnalyzer()                       # key from YOUTUBE_API_KEY
video_id = extract_video_id("https://youtu.be/VIDEO_ID")
payload = analyzer.analyze(video_id)

print(payload["stats"])
print(payload["keywords"])
```

```python
from feedback_fusion.sources.files import load_feedback, analyze_frame

frame, feedback_col, category_col = load_feedback("data/sample_feedback.csv")
payload = analyze_frame(frame, feedback_col, category_col)
print(payload["by_category"])
```

---

## Feedback file format

CSV or JSON. Column names are matched case-insensitively.

- **Feedback column** (required): one of `feedback`, `review`, `comment`,
  `text`, `response`, `message`.
- **Category column** (optional): one of `category`, `type`, `source`, `group`,
  `role`. When present, results are additionally broken down per category.
- Any other columns are preserved and displayed but not required.

```csv
feedback,category,rating,date
"Great place to work, very supportive.",Employee,5,2024-12-01
"Customer service was slow, not happy.",Customer,2,2024-12-02
```

See `data/sample_feedback.csv` for a working 20-row example.

---

## Project structure

```
feedback_fusion/            Core package (all analysis logic)
  config.py                 Environment-driven settings
  sentiment.py              VADER analysis, aggregation, DataFrame export
  text.py                   Cleaning, word clouds, TF-IDF keywords
  summarization.py          Neural summariser + extractive fallback
  storage.py                Best-effort MongoDB persistence
  cli.py                    Command-line interface
  sources/
    youtube.py              YouTubeAnalyzer, URL parsing
    amazon.py               AmazonAnalyzer, Selenium scraping
    files.py                CSV/JSON loading and analysis
Final.py                    Streamlit UI (presentation layer only)
tests/                      pytest suite (100 tests)
data/sample_feedback.csv    Sample data to try the app immediately
Feedback Fusion.html        Marketing landing page
LogIn.html/.css/.js         Firebase email/password sign in
Reg.html/.css/.js           Firebase registration
firebase-config.js          Firebase web config (public by design)
notebooks/                  Original exploration notebooks
```

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

100 tests covering sentiment classification and aggregation, text cleaning,
keyword extraction, YouTube URL parsing, file loading and column resolution,
config parsing, storage degradation, summarisation fallbacks and the CLI.

Tests never hit the network: YouTube analysis is exercised by passing comments
in directly, and the summariser is monkeypatched.

---

## Frontend

The HTML pages are static — no build step, no `npm install` required. The
Firebase modular SDK loads from the gstatic CDN.

```bash
npm start        # serves the folder at http://localhost:3000
```

For Firebase auth to work against your own project:

1. Replace the values in `firebase-config.js` with your project's web config.
2. Enable **Email/Password** under Authentication → Sign-in method.
3. Add your domain under Authentication → Settings → Authorized domains.

Firebase web API keys are safe to commit — they identify the project rather
than granting access. Security comes from Authentication, Security Rules and
the authorized-domain list. See
[Firebase docs on API keys](https://firebase.google.com/docs/projects/api-keys).

---

## Known limitations

- **Amazon scraping is unreliable.** Amazon serves bot-check pages to
  automated traffic. The scraper sets a realistic user agent and bounds its
  paging, but expect it to return nothing on many runs. For dependable
  results, export reviews and use the Company Review upload page.
- **VADER is lexicon-based.** It handles informal English, emoji and
  intensifiers well, but not sarcasm, domain jargon or non-English text.
- **Summarisation truncates.** Input is capped at `SUMMARY_MAX_CHARS`, so
  summaries reflect the beginning of a large feedback set, not all of it.
- **No user-scoped data.** Firebase auth gates the frontend pages, but the
  Streamlit app has no per-user access control.

---

## Contributors

CSDS 3rd Year, Acropolis Institute of Technology and Research, Batch 2026:

- **Sarthak Doshi** — Project Lead
- **Atishay Jain**
- **Om Chouksey**
- **Shambhavi Bhadoria**
- **Uday Vashishtha**

## License

MIT
