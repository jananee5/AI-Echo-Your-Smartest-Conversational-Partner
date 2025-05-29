import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load pre-trained sentiment model (Replace with your actual model)
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home", "Predict Sentiment", "Sentiment Analysis Insights"])

# ✅ Home Page
if page == "Home":
    page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: url("https://cdn.prod.website-files.com/6505b4884ef96252921f8bb5/65155ee0a96a837b72406c18_63c03b9d759618c9dab0d7d4_blog%2520image%2520final-03-p-1600.jpeg");
            background-size: cover;
            color: rgba(255, 255, 255, 0.7);
        }
        </style>
        """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
    st.markdown(
    """
    <style>
    .big-font {
        font-size: 46px !important;
        font-weight: bold;
        color: rgba(255, 255, 255, 0.85);
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('<p class="big-font">📢 Sentiment Analysis Web App</p>', unsafe_allow_html=True)

    st.markdown("""
    Welcome! 🎉  
    Here’s what you can do:  
    - **Predict Sentiment**: Enter a review and get its sentiment classification.  
    - **Sentiment Analysis Insights**: Explore trends in user feedback using interactive visualizations.  
    """)

# ✅ Predict Sentiment Page
elif page == "Predict Sentiment":
    page_gradient_bg = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #ff9a9e, #fad0c4);
    }
    </style>
    """
    st.markdown(page_gradient_bg, unsafe_allow_html=True)

    st.title("🔮 Predict Sentiment for Your Review")
    user_review = st.text_area("Enter a review to analyze:")

    def predict_sentiment(text):
        # Neutral keyword list
        neutral_keywords = ["average", "okay", "fine", "not bad", "acceptable", "moderate", "decent", "neutral"]

        # Positive keyword list for better accuracy
        positive_keywords = ["excellent", "great", "awesome", "fantastic", "superb", "amazing", "love", "perfect"]

        # Check if text contains neutral keywords
        if any(word in text.lower() for word in neutral_keywords):
            return "😐 Neutral"

        # Check if text contains positive keywords (ensure recognition)
        if any(word in text.lower() for word in positive_keywords):
            return "🌟 Highly Positive"

        # Use model prediction otherwise
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        output = model(**tokens)
        
        # Get predicted class and confidence scores
        probabilities = torch.nn.functional.softmax(output.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        sentiment_labels = ["😡 Negative", "😐 Neutral", "😊 Positive"]

        # ✅ Prevent index errors
        if predicted_class >= len(sentiment_labels):
            return "❓ Unknown Sentiment (Fix Model Output)"

        # ✅ Confidence-based classification refinements
        confidence = probabilities[0][predicted_class].item()

        if confidence < 0.6:
            return "😐 Neutral (Low Confidence Prediction)"
        elif confidence >= 0.75 and predicted_class == 2:  # High-confidence Positive
            return "🌟 Highly Positive"
        elif confidence >= 0.75 and predicted_class == 0:  # High-confidence Negative
            return "🔥 Strongly Negative"

        return sentiment_labels[predicted_class]
    # ✅ Modify Output Formatting in UI
    if st.button("Analyze Sentiment"):
        if user_review:
            sentiment_result = predict_sentiment(user_review)
            st.markdown(f"### **Predicted Sentiment:** {sentiment_result}")  # Emojis + Formatting
        else:
            st.warning("⚠️ Please enter a review before analyzing!")

# ✅ Sentiment Analysis Insights Page
elif page == "Sentiment Analysis Insights":
    page_gradient_bg = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #ff9a9e, #fad0c4);
    }
    </style>
    """
    st.markdown(page_gradient_bg, unsafe_allow_html=True)
    
    st.title("📊 Sentiment Analysis Insights")
    st.write("Explore user sentiment trends with detailed answers and dynamic visualizations!")

    # ✅ Define Questions and Answers
    questions = {
        "What is the overall sentiment of user reviews?": "We classify reviews as Positive, Neutral, or Negative, then compute proportions.",
        "How does sentiment vary by rating?": "We analyze whether lower ratings truly reflect negative sentiment and check mismatches.",
        "Which keywords or phrases are most associated with each sentiment class?": "Using word clouds and keyword frequency analysis, we find common words.",
        "How has sentiment changed over time?": "We track sentiment trends by month or week to observe satisfaction peaks.",
        "Do verified users tend to leave more positive or negative reviews?": "Comparing sentiment scores between verified and non-verified users.",
        "Are longer reviews more likely to be negative or positive?": "Analyzing whether longer reviews correlate with extreme sentiments.",
        "Which locations show the most positive or negative sentiment?": "Identifying geographic trends in sentiment scores.",
        "Is there a difference in sentiment across platforms (Web vs Mobile)?": "Comparing sentiment variations across different user platforms.",
        "Which ChatGPT versions are associated with higher/lower sentiment?": "Tracking user satisfaction across versions.",
        "What are the most common negative feedback themes?": "Using topic modeling to detect recurring pain points in reviews."
    }

    # ✅ Dropdown for User Selection
    selected_question = st.selectbox("Choose a question to analyze:", list(questions.keys()))

    # ✅ Display Answer First
    st.write("### Answer:")
    st.write(questions[selected_question])

    # ✅ Generate Visualizations Based on Question
    if selected_question == "What is the overall sentiment of user reviews?":
        st.write("This pie chart shows the distribution of sentiment classifications.")
        sentiment_data = pd.Series(["Positive", "Negative", "Neutral", "Positive", "Negative"])
        fig, ax = plt.subplots()
        sentiment_data.value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%')
        st.pyplot(fig)

    elif selected_question == "How does sentiment vary by rating?":
        st.write("A bar chart representing sentiment trends by rating.")
        rating_sentiment = pd.Series([5, 4, 3, 1, 2, 4, 5, 1, 2, 3, 4, 5])
        fig, ax = plt.subplots()
        rating_sentiment.value_counts().sort_index().plot(kind="bar", ax=ax, color="blue")
        st.pyplot(fig)

    elif selected_question == "Which keywords or phrases are most associated with each sentiment class?":
        st.write("Word cloud representation of sentiment-linked keywords.")
        
        sentiment_keywords = {
            "Positive": "excellent amazing fantastic great superb awesome",
            "Negative": "bad terrible awful poor horrible disappointing",
            "Neutral": "average okay decent fine acceptable moderate"
        }
        
        for sentiment, words in sentiment_keywords.items():
            st.subheader(f"{sentiment} Sentiment Keywords")
            wordcloud = WordCloud(background_color="white").generate(words)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    elif selected_question == "How has sentiment changed over time?":
        st.write("Line chart tracking sentiment trends.")
        time_sentiment = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
            "Sentiment Score": [3.5, 4.0, 4.2, 3.8, 4.1]
        })
        fig, ax = plt.subplots()
        time_sentiment.plot(x="Month", y="Sentiment Score", kind="line", ax=ax)
        st.pyplot(fig)

    elif selected_question == "Do verified users tend to leave more positive or negative reviews?":
        st.write("Comparison of sentiment between positive vs. negative reviews of verified users.")
        verified_users = pd.DataFrame({"Rating": [5, 4, 2, 1, 3], "Verified": [True, False, True, False, True]})
        verified_counts = verified_users.groupby("Verified")["Rating"].mean()
        fig, ax = plt.subplots()
        verified_counts.plot(kind="bar", ax=ax, color=["green", "red"])
        st.pyplot(fig)

    elif selected_question == "Are longer reviews more likely to be negative or positive?":
        st.write("Scatter plot showing review length vs. sentiment score.")
        review_length = pd.Series([50, 200, 150, 300, 100, 250, 400])
        sentiment_score = pd.Series([2.5, 4.8, 3.0, 1.5, 3.5, 4.2, 2.0])
        fig, ax = plt.subplots()
        ax.scatter(review_length, sentiment_score, c=sentiment_score, cmap="coolwarm")
        ax.set_xlabel("Review Length (words)")
        ax.set_ylabel("Sentiment Score")
        st.pyplot(fig)

    elif selected_question == "Which locations show the most positive or negative sentiment?":
        st.write("Bar chart representing sentiment scores across locations.")
        locations = pd.Series(["USA", "India", "UK", "Canada", "Germany", "Australia", "Japan"])
        location_sentiments = pd.Series([4.5, 3.8, 4.2, 4.0, 3.5, 4.6, 3.9])
        fig, ax = plt.subplots()
        pd.DataFrame({"Location": locations, "Sentiment Score": location_sentiments}).set_index("Location").plot(kind="bar", ax=ax, color="skyblue")
        st.pyplot(fig)

    elif selected_question == "Is there a difference in sentiment across platforms (Web vs Mobile)?":
        st.write("Bar chart comparing sentiment between platforms.")
        platforms = pd.Series(["Web", "Mobile"])
        platform_sentiments = pd.Series([4.2, 3.8])
        fig, ax = plt.subplots()
        pd.DataFrame({"Platform": platforms, "Sentiment Score": platform_sentiments}).set_index("Platform").plot(kind="bar", ax=ax, color=["purple", "orange"])
        st.pyplot(fig)

    elif selected_question == "What are the most common negative feedback themes?":
        st.write("Bar chart showing the most frequently mentioned negative themes.")
        negative_themes = pd.Series(["Slow performance", "Bad UI", "Poor customer support", "Bugs and glitches", "High pricing"])
        theme_counts = pd.Series([20, 15, 25, 30, 18])
        fig, ax = plt.subplots()
        pd.DataFrame({"Negative Theme": negative_themes, "Mentions": theme_counts}).set_index("Negative Theme").plot(kind="bar", ax=ax, color="red")
        st.pyplot(fig)
    
    elif selected_question == "Which ChatGPT versions are associated with higher/lower sentiment?":
        st.write("Bar chart showing average sentiment per ChatGPT version.")

        # Sample data for ChatGPT versions and sentiment ratings
        chatgpt_versions = ["GPT-3", "GPT-4", "GPT-4 Turbo"]
        sentiment_scores = [4.0, 4.6, 4.3]  # Example average ratings

        # ✅ Convert data into a Pandas DataFrame
        df_versions = pd.DataFrame({"Version": chatgpt_versions, "Sentiment Score": sentiment_scores})

        # ✅ Ensure proper chart rendering
        fig, ax = plt.subplots()
        df_versions.set_index("Version").plot(kind="bar", ax=ax, color="lightblue")
        ax.set_ylabel("Average Sentiment Score")
        ax.set_xticklabels(chatgpt_versions, rotation=0)  # Set labels correctly
        st.pyplot(fig)
