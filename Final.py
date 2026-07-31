"""Feedback-Fusion Streamlit application.

All analysis logic lives in the `feedback_fusion` package; this module is
the presentation layer. Run with:

    streamlit run Final.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from feedback_fusion import __version__, settings
from feedback_fusion.config import ConfigError
from feedback_fusion.sentiment import NEGATIVE, NEUTRAL, POSITIVE, to_dataframe
from feedback_fusion.sources.amazon import AmazonAnalyzer, ScraperUnavailable
from feedback_fusion.sources.files import FeedbackFileError, analyze_frame, load_feedback
from feedback_fusion.sources.youtube import YouTubeAnalyzer, YouTubeError, extract_video_id
from feedback_fusion.summarization import summarize

ACCENT = "#00FFFF"

SENTIMENT_COLORS = {POSITIVE: "#2ECC71", NEGATIVE: "#E74C3C", NEUTRAL: "#95A5A6"}

st.set_page_config(page_title="Feedback-Fusion", page_icon="📊", layout="wide")


def heading(text, level=2):
    st.markdown(f"<h{level} style='color: {ACCENT};'>{text}</h{level}>", unsafe_allow_html=True)


def go_to(page):
    st.session_state.page = page
    st.rerun()


def back_button(key):
    if st.button("← Return to Main Page", key=key):
        go_to("main")


# --------------------------------------------------------------------------
# Shared rendering helpers
# --------------------------------------------------------------------------

def render_stats(stats):
    """Show headline metrics for a batch of analysed feedback."""
    counts = stats["counts"]
    percentages = stats["percentages"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total analysed", stats["total"])
    col2.metric("Positive", counts[POSITIVE], f"{percentages[POSITIVE]}%")
    col3.metric("Negative", counts[NEGATIVE], f"{percentages[NEGATIVE]}%")
    col4.metric("Neutral", counts[NEUTRAL], f"{percentages[NEUTRAL]}%")

    st.caption(
        f"Overall sentiment: **{stats['overall']}** "
        f"(mean compound score {stats['average_compound']})"
    )


def render_distribution(stats, title="Sentiment Distribution"):
    counts = stats["counts"]
    frame = pd.DataFrame({"Sentiment": list(counts), "Count": list(counts.values())})
    figure = px.bar(
        frame,
        x="Sentiment",
        y="Count",
        title=title,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    st.plotly_chart(figure, use_container_width=True)


def render_score_histogram(results):
    if not results:
        return
    scores = [r.compound for r in results]
    figure = px.histogram(
        pd.DataFrame({"Sentiment Score": scores}),
        x="Sentiment Score",
        nbins=30,
        title="Sentiment Score Distribution",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_word_cloud(texts):
    """Render a word cloud, or explain why one could not be produced."""
    from feedback_fusion.text import generate_word_cloud

    cloud = generate_word_cloud(texts)
    if cloud is None:
        st.info("Not enough distinct words to build a word cloud.")
        return
    st.image(cloud.to_array(), use_container_width=True)


def render_keywords(keywords):
    if not keywords:
        st.info("No keywords could be extracted.")
        return
    frame = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Score"])
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_storage(storage):
    if storage is None:
        return
    if storage.stored:
        st.caption(f"Saved {storage.count} records to MongoDB collection `{storage.collection}`.")
    else:
        st.caption(f"Results not persisted ({storage.reason}).")


@st.cache_data(show_spinner=False)
def cached_summary(texts):
    """Cache summaries so a Streamlit rerun does not re-run the model."""
    return summarize(texts)


def render_summary(texts):
    """Summarize once, cached. The old code ran the model twice per render."""
    heading("Summary", level=3)
    with st.spinner("Generating summary..."):
        try:
            text = cached_summary(tuple(texts))
        except Exception as exc:
            st.error(f"Could not generate a summary: {exc}")
            return
    if text:
        st.success(text)
    else:
        st.info("Not enough text to summarise.")


def render_analysis(payload, texts, text_field, summary=True):
    """Render the standard analysis blocks shared by every source page."""
    render_stats(payload["stats"])
    render_storage(payload.get("storage"))

    left, right = st.columns(2)
    with left:
        render_distribution(payload["stats"])
    with right:
        render_score_histogram(payload["results"])

    heading("Word Cloud", level=3)
    render_word_cloud(texts)

    heading("Top Keywords (TF-IDF)", level=3)
    render_keywords(payload["keywords"])

    heading("Analysed Feedback", level=3)
    frame = to_dataframe(payload["results"], text_field=text_field)
    st.dataframe(frame.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download full results as CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"feedback_fusion_{text_field}_analysis.csv",
        mime="text/csv",
    )

    if summary:
        render_summary(texts)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def main_page():
    heading("Feedback-Fusion: Empowering Better Decisions with Feedback Analysis", level=1)

    heading("Project Vision")
    st.write(
        "The vision of **Feedback-Fusion** is to change how organisations understand and "
        "use feedback from customers, employees and other stakeholders. By combining "
        "sentiment analysis with keyword extraction and summarisation, the platform turns "
        "large volumes of raw feedback into insights that support data-driven decisions."
    )

    heading("Services We Provide")
    st.write(
        """
    - **YouTube Video Feedback Analysis**: analyse video comments to understand viewer sentiment.
    - **Amazon Product Review Analysis**: gauge customer satisfaction and product performance.
    - **Company Review Analysis**: upload employee and customer feedback as CSV or JSON.
    - **Sentiment Analysis**: classify feedback as positive, negative or neutral with scores.
    - **Word Cloud & Keywords**: surface the themes that dominate a feedback set.
    - **Summarisation**: condense large feedback sets into a short readable summary.
    """
    )

    heading("Explore Our Services")
    st.write("Choose a service to begin analysing feedback:")

    col1, col2, col3 = st.columns(3)
    if col1.button("YouTube Video", use_container_width=True):
        go_to("youtube")
    if col2.button("Amazon Product", use_container_width=True):
        go_to("amazon")
    if col3.button("Company Review", use_container_width=True):
        go_to("company")

    heading("Contributors")
    st.write(
        """
    **Feedback-Fusion** is the result of the work of the following contributors from
    **CSDS 3rd Year, Acropolis Institute of Technology and Research, Batch 2026**:
    - **Sarthak Doshi** (Project Lead)
    - **Atishay Jain**
    - **Om Chouksey**
    - **Shambhavi Bhadoria**
    - **Uday Vashishtha**
    """
    )

    st.markdown(
        f"<h4 style='color: #FFFFFF;'>Feedback-Fusion v{__version__} | All rights reserved © 2024</h4>",
        unsafe_allow_html=True,
    )


def youtube_page():
    heading("YouTube Video Feedback Analyzer")
    st.write("Analyse the comment section of any public YouTube video.")

    if not settings.youtube_api_key:
        st.warning(
            "No YouTube API key configured. Copy `.env.example` to `.env` and set "
            "`YOUTUBE_API_KEY` to a YouTube Data API v3 key."
        )

    video_url = st.text_input("YouTube video URL or ID:", placeholder="https://www.youtube.com/watch?v=...")

    if video_url:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.warning("That does not look like a valid YouTube URL or video ID.")
        else:
            analyzer = YouTubeAnalyzer()
            try:
                with st.spinner(f"Fetching comments for `{video_id}`..."):
                    payload = analyzer.analyze(video_id)
            except ConfigError as exc:
                st.error(str(exc))
            except YouTubeError as exc:
                st.error(
                    f"{exc}\n\nComments may be disabled, or the API key may be invalid "
                    "or out of quota."
                )
            else:
                if payload.get("partial_reason"):
                    st.warning(
                        f"Only {len(payload['comments'])} comments could be fetched "
                        f"before the API stopped responding ({payload['partial_reason']}). "
                        "The results below cover that subset only."
                    )
                if payload["results"]:
                    render_analysis(payload, payload["comments"], "comment")
                else:
                    st.warning("No comments found for this video.")

    back_button("back_youtube")


def amazon_page():
    heading("Amazon Product Review Analyzer")
    st.write("Analyse customer reviews from an Amazon product page.")
    st.info(
        "Scraping needs Chrome plus the optional extras: "
        "`pip install -r requirements-scraping.txt`. Amazon actively blocks automated "
        "traffic, so if scraping returns nothing use the Company Review page to upload "
        "an exported review file instead."
    )

    product_url = st.text_input("Amazon product URL:", placeholder="https://www.amazon.in/dp/...")

    if product_url:
        analyzer = AmazonAnalyzer()
        try:
            with st.spinner("Fetching reviews, this can take a minute..."):
                payload = analyzer.analyze(product_url)
        except ScraperUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Scraping failed: {exc}")
        else:
            if payload["results"]:
                st.subheader(payload["product_name"])
                render_analysis(payload, payload["reviews"], "review")
            else:
                st.warning(
                    "No reviews were found. Amazon may have served a bot-check page."
                )

    back_button("back_amazon")


def company_page():
    heading("Company Review Analyzer")
    st.write("Upload employee or customer feedback and analyse it in one pass.")

    with st.expander("File format requirements", expanded=False):
        st.write(
            """
        Upload a **CSV** or **JSON** file containing feedback.

        - **Feedback column** (required): the feedback text. Name it `feedback`,
          `review`, `comment`, `text`, `response` or `message`.
        - **Category column** (optional): who the feedback came from, for example
          `Customer` or `Employee`. Name it `category`, `type`, `source`, `group`
          or `role`. When present, results are also broken down per category.
        - Any other columns (rating, date, department) are preserved and shown
          but not required.

        **Example CSV**

        ```csv
        feedback,category,rating,date
        "Great place to work, very supportive.",Employee,5,2024-12-01
        "Customer service was slow, not happy.",Customer,2,2024-12-02
        ```
        """
        )

    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            frame, feedback_col, category_col = load_feedback(uploaded_file)
        except FeedbackFileError as exc:
            st.error(str(exc))
        else:
            st.caption(
                f"Using `{feedback_col}` as the feedback column"
                + (f" and `{category_col}` as the category column." if category_col else ".")
            )
            heading("Uploaded Data", level=3)
            st.dataframe(frame.head(10), use_container_width=True)

            payload = analyze_frame(frame, feedback_col, category_col)
            if not payload["results"]:
                st.warning("No usable feedback text was found in that file.")
            else:
                render_analysis(payload, payload["texts"], "feedback")

                if payload["by_category"]:
                    heading("Sentiment by Category", level=3)
                    for category, stats in payload["by_category"].items():
                        st.markdown(f"**{category}**")
                        render_stats(stats)
                        render_distribution(stats, title=f"Sentiment Distribution — {category}")

    back_button("back_company")


PAGES = {
    "main": main_page,
    "youtube": youtube_page,
    "amazon": amazon_page,
    "company": company_page,
}


def run():
    if "page" not in st.session_state:
        st.session_state.page = "main"
    PAGES.get(st.session_state.page, main_page)()


run()
