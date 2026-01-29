import streamlit as st

# Main page layout with buttons
def main_page():
     # Title of the page
    # Title of the page with custom color
    st.markdown("<h1 style='color: #00FFFF;'>Feedback-Fusion: Empowering Better Decisions with Feedback Analysis</h1>", unsafe_allow_html=True)
    
    # Vision of the Project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Project Vision</h2>", unsafe_allow_html=True)
    st.write("""
    The vision of **Feedback-Fusion** is to revolutionize how organizations and businesses understand and utilize feedback from their customers, employees, and other stakeholders. 
    By harnessing the power of sentiment analysis and feedback processing, our goal is to help companies gain deep insights into the mindset of their audiences, enabling them to make informed, data-driven decisions for continuous improvement.
    """)
    
    # Services provided by the project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Services We Provide</h2>", unsafe_allow_html=True)
    st.write("""
    **Feedback-Fusion** offers a range of services designed to analyze feedback in various forms, whether it's customer reviews, employee surveys, or product feedback:
    - **YouTube Video Feedback Analysis**: Analyze YouTube video comments to gain insights into viewer sentiment.
    - **Amazon Product Review Analysis**: Analyze product reviews from Amazon to gauge customer satisfaction and product performance.
    - **Company Review Analysis**: Analyze feedback from both employees and customers within your company, enabling you to improve workplace satisfaction and customer experience.
    - **Sentiment Analysis**: Automatically classify feedback as positive, negative, or neutral and generate insightful reports.
    - **Word Cloud Visualization**: Generate word clouds to visualize the most common terms in the feedback, helping to highlight key themes.
    - **Summarization**: Provide summaries of large sets of feedback for quick and easy understanding.
    """)
    
    # Impact of the Project with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Impact of Feedback-Fusion</h2>", unsafe_allow_html=True)
    st.write("""
    The **Feedback-Fusion** platform is designed to have a transformative impact on businesses, organizations, and individuals:
    - **Improved Decision-Making**: By understanding the sentiments of customers, employees, and stakeholders, organizations can make more informed decisions.
    - **Better Customer Experience**: Analyzing customer feedback helps identify areas of improvement, allowing companies to enhance their offerings and services.
    - **Employee Satisfaction**: Analyzing employee feedback ensures that companies can maintain a positive work environment and improve employee engagement.
    - **Data-Driven Insights**: Feedback-Fusion empowers organizations to move from gut-feeling decisions to data-backed strategies, ultimately leading to better outcomes and growth.
    """)

    st.markdown("<h2 style='color: #00FFFF;'>Explore Our Services", unsafe_allow_html=True)
    st.write("Choose a service to begin analyzing feedback:")
    
    # Button for YouTube Video
    if st.button("YouTube Video"):
        st.session_state.page = "youtube"
        st.rerun()


    # Button for Amazon Product
    if st.button("Amazon Product"):
        st.session_state.page = "amazon"
        st.rerun()


    # Button for Company Review
    if st.button("Company Review"):
        st.session_state.page = "company"
        st.rerun()

    # Project Contributors
    # Project Contributors with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Contributors</h2>", unsafe_allow_html=True)
    st.write("""
    **Feedback-Fusion** is the result of the hard work and dedication of the following contributors from the **CSDS 3rd Year, Acropolis Institute of Technology and Research, Batch 2026**:
    - **Sarthak Doshi** (Project Lead)
    - **Atishay Jain** 
    - **Om Chouksey**
    - **Shambhavi Bhadoria**
    - **Uday Vashishtha**
    
    We thank each contributor for their valuable input, collaboration, and commitment to making this project a success.
    """)
    
    # Contact Information with custom color
    st.markdown("<h2 style='color: #00FFFF;'>Contact Us</h2>", unsafe_allow_html=True)
    st.write("""
    For more information, feedback, or collaboration inquiries, please contact us at:
    - **Email**: sarthakdoshi231@gmail.com
    - **Phone**: +91 7898770372
    """)

    # Footer Section with custom color
    st.markdown("<h3 style='color: #FFFFFF;'>Feedback-Fusion | All rights reserved © 2024</h3>", unsafe_allow_html=True)    


# YouTube Video Page
def youtube_page():
    import streamlit as st
    st.markdown("<h2 style='color: #00FFFF;'>YouTube Video Feedback Analyzer", unsafe_allow_html=True)
    st.write("This page will allow you to analyze feedback for YouTube videos.")
    
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import pandas as pd
    import matplotlib.pyplot as plt
    from pymongo import MongoClient
    import googleapiclient.discovery
    from urllib.parse import urlparse, parse_qs
    import plotly.express as px
    import plotly.graph_objects as go
    from wordcloud import WordCloud
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import word_tokenize
    from transformers import pipeline

    # Ensure necessary NLTK data is downloaded
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # MongoDB setup
    client = MongoClient("mongodb://localhost:27017/")
    db = client["feedback_fusion"]

    # Define the YouTube API key here (predefined)
    api_key = "AIzaSyC9rAzBhULMEtzaq5XeZVra-VKjk089x98 "  # Replace with your actual API key

    # Function to extract comments using YouTube Data API
    def extract_comments(video_id, api_key):
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        comments = []

        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            while "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response["nextPageToken"],
                    maxResults=100
                )
                response = request.execute()

                for item in response.get("items", []):
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(comment)

        except Exception as e:
            print(f"Error extracting comments: {e}")
        
        return comments

    # Function to process comments and perform sentiment analysis
    def process_comments(comments):
        results = []
        for comment in comments:
            sentiment_score = sia.polarity_scores(comment)
            sentiment = (
                "Positive" if sentiment_score['compound'] > 0.05 else
                "Negative" if sentiment_score['compound'] < -0.05 else
                "Neutral"
            )
            results.append({
                "comment": comment,
                "sentiment": sentiment,
                "scores": sentiment_score
            })

        return results

    # Function to generate word cloud
    def generate_word_cloud(comments):
        stop_words = set(stopwords.words('english'))
        all_comments = ' '.join(comments)
        wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(all_comments)
        return wordcloud

    # Function to store results in MongoDB (creates a new collection based on video ID)
    def store_results(results, video_id):
        collection_name = f"comments_{video_id}"  # Dynamic collection name based on video ID
        collection = db[collection_name]  # Create or access collection
        collection.insert_many(results)

    # Function to extract video ID from URL
    def extract_video_id(url):
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get("v")
        return video_id[0] if video_id else None

    def summarize_comments(comments):
        # Combine all comments into a single text
        combined_text = " ".join(comments)
        
        # Truncate text to prevent overloading the model
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]
        
        # Use a pre-trained Hugging Face summarization model on CPU
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        
        try:
            # Generate summary
            summary = summarizer(combined_text, max_length=1000, min_length=100, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error in summarization: {e}")
            return "Error in generating summary. Please check your input or GPU configuration."


    # Streamlit App Layout
    # Input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        video_id = extract_video_id(video_url)
        
        if video_id:
            comments = extract_comments(video_id, api_key)
                
            if comments:
                processed_comments = process_comments(comments)
                store_results(processed_comments, video_id)  # Store results in MongoDB
                total_comments = len(comments)
                st.metric("Total Comments", total_comments)
                # Create sentiment distribution (Bar chart)
                st.markdown("<h2 style='color: #00FFFF;'>Sentiment Distribution Graph",unsafe_allow_html=True)
                sentiment_counts = pd.DataFrame(processed_comments)["sentiment"].value_counts()
                sentiment_fig = px.bar(
                    sentiment_counts, 
                    x=sentiment_counts.index, 
                    y=sentiment_counts.values, 
                    labels={"x": "Sentiment", "y": "Number of Comments"},
                    color=sentiment_counts.index
                )
                st.plotly_chart(sentiment_fig)
                st.markdown("<h2 style='color: #00FFFF;'>Sentiment Score Distribution",unsafe_allow_html=True)
                # Sentiment score distribution (Histogram)
                sentiment_scores = pd.DataFrame(processed_comments)["scores"].apply(lambda x: x['compound'])
                sentiment_hist_fig = px.histogram(
                    sentiment_scores, 
                    nbins=30, 
                    labels={"value": "Sentiment Score"}
                )
                st.plotly_chart(sentiment_hist_fig)
                st.markdown("<h2 style='color: #00FFFF;'>Word Cloud for Reviews",unsafe_allow_html=True)
                # Word Cloud (Word frequency)
                wordcloud = generate_word_cloud(comments)
                st.image(wordcloud.to_array(), use_container_width=True)

                # Display processed comments (first 10)
                st.markdown("<h2 style='color: #00FFFF;'>Processed Comments (First 10)",unsafe_allow_html=True)
                st.write(pd.DataFrame(processed_comments)[['comment', 'sentiment']].head(10))
                    
                summary = summarize_comments(comments)
                
                st.markdown("<h2 style='color: #00FFFF;'>Comments Summary",unsafe_allow_html=True)
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_comments(comments)
                        st.success("Summary Generated!")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error in summarization: {e}")
                
            else:
                st.warning("No comments found.")
        else:
            st.warning("Invalid YouTube video URL.")
            
    if st.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()

# Amazon Product Page
def amazon_page():
    import streamlit as st
    st.markdown("<h2 style='color: #00FFFF;'>Amazon Product Feedback Analyzer",unsafe_allow_html=True)
    st.write("This page will allow you to analyze reviews for Amazon products.")
    # You can place your existing Amazon product analysis code here
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import pandas as pd
    import matplotlib.pyplot as plt
    from pymongo import MongoClient
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    from wordcloud import WordCloud
    from transformers import pipeline
    import streamlit as st
    import plotly.express as px

    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # MongoDB setup
    client = MongoClient("mongodb://localhost:27017/")
    db = client["feedback_fusion"]

    # Set up headless mode for Streamlit Cloud
    def get_amazon_reviews_selenium(product_url):
        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        # Path to chromedriver (make sure it's correct)
        service = Service(ChromeDriverManager().install())  # Update with your path
        
        # Initialize WebDriver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get('https://www.amazon.com')
        try:
            # Open the product URL
            driver.get(product_url)
            time.sleep(5)  # Wait for dynamic content to load
            
            # Extract product name
            try:
                product_name = driver.find_element(By.ID, "productTitle").text.strip()
            except Exception:
                product_name = "Product title not found"
            
            # Extract reviews
            reviews = []
            while True:
                review_elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
                reviews.extend([review.text.strip() for review in review_elements])

                # Scroll down to load more reviews
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                # Check if there's a "Next" button to load more reviews
                next_button = driver.find_elements(By.XPATH, "//li[@class='a-last']/a")
                if next_button:
                    next_button[0].click()
                    time.sleep(3)
                else:
                    break

            return product_name, reviews
        
        finally:
            # Quit the driver after scraping
            driver.quit()

    # Function to process reviews and perform sentiment analysis
    def process_reviews(reviews):
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for review in reviews:
            sentiment_score = sia.polarity_scores(review)
            sentiment = (
                "Positive" if sentiment_score['compound'] > 0.05 else
                "Negative" if sentiment_score['compound'] < -0.05 else
                "Neutral"
            )
            results.append({
                "review": review,
                "sentiment": sentiment,
                "scores": sentiment_score
            })

            sentiment_counts[sentiment] += 1

        return results, sentiment_counts

    # Function to generate word cloud
    def generate_word_cloud(reviews):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        all_reviews = ' '.join(reviews)
        wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(all_reviews)
        return wordcloud

    # Function to store results in MongoDB (creates a new collection based on product name)
    def store_results(results, product_name):
        collection_name = f"reviews_{product_name.replace(' ', '_')}"  # Dynamic collection name based on product name
        collection = db[collection_name]  # Create or access collection
        collection.insert_many(results)

    # Function to summarize reviews using Hugging Face summarization
    def summarize_reviews(reviews):
        # Combine all reviews into a single text
        combined_text = " ".join(reviews)
        
        # Truncate text to prevent overloading the model
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]
        
        # Use a pre-trained Hugging Face summarization model on CPU
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        
        try:
            # Generate summary
            summary = summarizer(combined_text, max_length=1000, min_length=100, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error in summarization: {e}")
            return "Error in generating summary. Please check your input or GPU configuration."

    # Streamlit UI
    st.title("Amazon Product Review Analyzer")
    st.subheader("Enter the Amazon product URL below to analyze reviews:")

    product_url = st.text_input("Product URL")

    if product_url:
        st.write("Fetching reviews... Please wait.")
        product_name, reviews = get_amazon_reviews_selenium(product_url)

        if reviews:
            # Process the reviews
            processed_reviews, sentiment_counts = process_reviews(reviews)

            # Store the results in MongoDB
            store_results(processed_reviews, product_name)

            # Display results
            st.subheader(f"Sentiment Analysis Results for {product_name}")
            
            
            st.metric("Total Comments", len(reviews))
            # Sentiment Distribution Visualization (Bar chart)
            st.subheader("Sentiment Distribution")
            sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
            fig = px.bar(sentiment_df, x='Sentiment', y='Count', title="Sentiment Distribution of Reviews", color='Sentiment')
            st.plotly_chart(fig)
            
            # Generate a word cloud
            st.subheader("Word Cloud for Reviews")
            wordcloud = generate_word_cloud(reviews)
            st.image(wordcloud.to_array(), use_container_width=True)

            # Show first 10 reviews in an organized table
            st.subheader("First 10 Reviews")
            reviews_df = pd.DataFrame(processed_reviews[:10])  # Display first 10 reviews
            st.write(reviews_df[['review', 'sentiment']])
            

            # Summarize the reviews
            st.subheader("Summary of Reviews")
            summary = summarize_reviews(reviews)
            st.write(summary)

        

        else:
            st.write("No reviews found for this product.")

            
            
    if st.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
            
            
# Company Review Page
def company_page():
    import streamlit as st
    st.markdown("<h2 style='color: #00FFFF;'>Company Review Analyzer",unsafe_allow_html=True)
    st.write("This page will allow you to analyze company reviews from employees and customers.")
    # You can place your existing Company review analysis code here
    import pandas as pd
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from wordcloud import WordCloud
    import streamlit as st
    import plotly.express as px
    from sklearn.feature_extraction.text import TfidfVectorizer
    import json

    # Ensure necessary NLTK data is downloaded
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Function to load CSV or JSON feedback file
    def load_feedback(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            raise ValueError("File format not supported. Please upload a CSV or JSON file.")
    
        return df

    # Function to process feedback and perform sentiment analysis
    def process_feedback(feedbacks):
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for feedback in feedbacks:
            sentiment_score = sia.polarity_scores(feedback)
            sentiment = (
                "Positive" if sentiment_score['compound'] > 0.05 else
                "Negative" if sentiment_score['compound'] < -0.05 else
                "Neutral"
            )
            results.append({
                "feedback": feedback,
                "sentiment": sentiment,
                "scores": sentiment_score
            })

            sentiment_counts[sentiment] += 1

        return results, sentiment_counts

    # Function to generate word cloud
    def generate_word_cloud(feedbacks):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        all_feedbacks = ' '.join(feedbacks)
        wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(all_feedbacks)
        return wordcloud

    # Function to generate a TF-IDF visualization for keyword extraction
    def generate_tfidf_keywords(feedbacks):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        tfidf_matrix = vectorizer.fit_transform(feedbacks)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keyword_df = pd.DataFrame(list(zip(feature_names, tfidf_scores)), columns=["Keyword", "TF-IDF Score"])
        keyword_df = keyword_df.sort_values(by="TF-IDF Score", ascending=False)
        return keyword_df

    # Streamlit UI
    

    st.subheader("Instructions for Uploading Feedback Files")
    st.write("""
        Please upload a CSV or JSON file containing feedback from both customers and employees.
        
        **File Format Requirements**:
        1. The file should have at least two columns:
            - **Feedback Column**: Contains the actual feedback text. You can name it `feedback`, `review`, or similar.
            - **Category Column**: Specifies whether the feedback is from a `Customer` or an `Employee`. You can name it `category`, `type`, or similar.
        
        2. Optional Columns:
            - **Rating**: A numeric rating (e.g., 1 to 5 or 1 to 10).
            - **Date**: The date when the feedback was given.
            - **Department**: For employee feedback, you may include a `department` column to specify which department the feedback is related to (e.g., `Sales`, `HR`).
        
        **Example CSV Format**:
        | feedback                                | category   | rating | date       |
        |-----------------------------------------|------------|--------|------------|
        | "Great place to work, very supportive." | Employee   | 5      | 2024-12-01 |
        | "Customer service was slow, not happy." | Customer   | 2      | 2024-12-02 |
        | "I love the products and the team."     | Employee   | 4      | 2024-12-03 |
        | "The product quality is excellent."     | Customer   | 5      | 2024-12-04 |
        | "Need better management."               | Employee   | 3      | 2024-12-05 |
        
        **Example JSON Format**:
        ```json
        [
        {"feedback": "Great place to work, very supportive.", "category": "Employee", "rating": 5, "date": "2024-12-01"},
        {"feedback": "Customer service was slow, not happy.", "category": "Customer", "rating": 2, "date": "2024-12-02"},
        {"feedback": "I love the products and the team.", "category": "Employee", "rating": 4, "date": "2024-12-03"},
        {"feedback": "The product quality is excellent.", "category": "Customer", "rating": 5, "date": "2024-12-04"},
        {"feedback": "Need better management.", "category": "Employee", "rating": 3, "date": "2024-12-05"}
        ]
        ```
    """)
    st.subheader("Upload your feedback data (CSV or JSON file):")

    # File uploader for CSV or JSON
    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            # Load the uploaded feedback file
            feedback_df = load_feedback(uploaded_file)
            
            # Display the first few rows of the dataset
            st.subheader("Uploaded Feedback Data")
            st.write(feedback_df.head())

            # Process the feedbacks (assuming the column containing feedback text is named 'feedback')
            feedbacks = feedback_df['feedback'].astype(str).tolist()

            # Process the feedback for sentiment analysis
            processed_feedback, sentiment_counts = process_feedback(feedbacks)

            # Display sentiment analysis results
            st.markdown("<h2 style='color: #00FFFF;'>Sentiment Analysis Results",unsafe_allow_html=True)
            st.write(f"Positive: {sentiment_counts['Positive']}")
            st.write(f"Negative: {sentiment_counts['Negative']}")
            st.write(f"Neutral: {sentiment_counts['Neutral']}")

            # Generate a word cloud
            st.markdown("<h2 style='color: #00FFFF;'>Word Cloud for Feedback",unsafe_allow_html=True)
            wordcloud = generate_word_cloud(feedbacks)
            st.image(wordcloud.to_array(), use_container_width=True)

            # Display TF-IDF keyword analysis
            st.markdown("<h2 style='color: #00FFFF;'>Top Keywords Based on TF-IDF",unsafe_allow_html=True)
            tfidf_keywords = generate_tfidf_keywords(feedbacks)
            st.write(tfidf_keywords.head(10))  # Show top 10 keywords

            # Visualize sentiment distribution
            sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
            fig = px.bar(sentiment_df, x='Sentiment', y='Count', title="Sentiment Distribution of Feedback", color='Sentiment')
            st.plotly_chart(fig)

            # Optionally: Separate feedback into Customer and Employee (assuming the file contains this information)
            if 'category' in feedback_df.columns:
                st.markdown("<h2 style='color: #00FFFF;'>Sentiment Analysis by Category",unsafe_allow_html=True)
                categories = feedback_df['category'].unique()
                for category in categories:
                    category_feedbacks = feedback_df[feedback_df['category'] == category]['feedback'].astype(str).tolist()
                    category_results, category_counts = process_feedback(category_feedbacks)
                    st.write(f"Sentiment Analysis for {category} Feedback")
                    st.write(f"Positive: {category_counts['Positive']}")
                    st.write(f"Negative: {category_counts['Negative']}")
                    st.write(f"Neutral: {category_counts['Neutral']}")
                    category_df = pd.DataFrame(list(category_counts.items()), columns=['Sentiment', 'Count'])
                    category_fig = px.bar(category_df, x='Sentiment', y='Count', title=f"Sentiment Distribution for {category}", color='Sentiment')
                    st.plotly_chart(category_fig)
                    
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Return to Main Page"):
        st.session_state["page"] = "main"
        st.rerun()
        
        
# Main execution block
if 'page' not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "youtube":
    youtube_page()
elif st.session_state.page == "amazon":
    amazon_page()
elif st.session_state.page == "company":
    company_page()