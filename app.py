import streamlit as st
import pandas as pd
import re
import pickle
import os
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def cleaner(string):
    # Remove URLs
    string = re.sub(r"http\S+|www\S+|https\S+", "", string)
    # Replace mentions with token
    string = re.sub(r"@\w+", "[USER]", string)
    # Remove hashtag symbol but keep text
    string = re.sub(r"#(\w+)", r"\1", string)
    # Remove special characters but keep emoticons
    string = re.sub(r"[^a-zA-Z0-9\s:)(:D;)\-_]", "", string)
    # Handle repeated characters (e.g., "sooooo" -> "soo")
    string = re.sub(r"(.)\1{2,}", r"\1\1", string)
    # Remove extra whitespace
    string = re.sub(r"\s+", " ", string)
    string = string.lower().strip()
    return string

def lemmatize_text(string):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in string.split()]
    return ' '.join(words)

@st.cache_data
def load_data():
    return pd.read_csv("database/tweets.csv", names=['sentiment', 'ids', 'date', 'flag', 'user', 'text'])

def create_visualizations(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sentiment_counts = df['sentiment'].value_counts()
        ax1.bar(['Negative (0)', 'Positive (4)'], sentiment_counts.values, color=['red', 'green'])
        ax1.set_title("Sentiment Distribution (Raw Data)")
        ax1.set_ylabel("Number of tweets")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Tweet Length Distribution")
        df['tweet_length'] = df['text'].apply(len)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(df['tweet_length'], bins=50, color='skyblue', edgecolor='black')
        ax2.set_title("Tweet Length Distribution (Raw Data)")
        ax2.set_xlabel("Length (characters)")
        ax2.set_ylabel("Number of Tweets")
        st.pyplot(fig2)

@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    processed_text = cleaner(text)
    lemmatized_text = lemmatize_text(processed_text)
    text_vec = vectorizer.transform([lemmatized_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    # Calculate confidence based on how close probabilities are
    prob_diff = abs(probability[0] - probability[1])
    
    # If probabilities are close (< 0.3 difference), it's neutral
    if prob_diff < 0.3:
        sentiment = "Neutral"
        # Neutral confidence: how close the probabilities are (inverted)
        confidence = 1 - prob_diff
    elif prediction == 0:
        sentiment = "Negative"
        confidence = probability[0]
    else:
        sentiment = "Positive"
        confidence = probability[1]
    
    return sentiment, confidence

# Streamlit App
st.title("🐦 Tweet Sentiment Analysis")
st.write("Enter a tweet or text to analyze its sentiment: **Positive**, **Negative**, or **Neutral**!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Sentiment Analyzer", "Data Visualization"])

if page == "Data Visualization":
    st.header("📊 Dataset Analysis")
    try:
        df = load_data()
        st.write(f"**Dataset Shape:** {df.shape[0]:,} tweets")
        create_visualizations(df)
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df[['sentiment', 'text']].head())
        
    except FileNotFoundError:
        st.error("Dataset not found! Please make sure 'database/tweets.csv' exists.")

else:
    st.header("🔍 Analyze Your Text")
    
    # Load model
    try:
        model, vectorizer = load_model()
        
        # Text input
        user_input = st.text_area("Enter your text here:", placeholder="Type your tweet or message...")
        
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                sentiment, confidence = predict_sentiment(user_input, model, vectorizer)
                
                # Display result
                col1, col2 = st.columns(2)
                
                with col1:
                    if sentiment == "Positive":
                        st.success(f"😊 **{sentiment}**")
                    elif sentiment == "Negative":
                        st.error(f"😞 **{sentiment}**")
                    else:
                        st.info(f"😐 **{sentiment}**")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                    if confidence < 0.2:
                        st.caption("⚠️ Low confidence")
                    elif confidence > 0.5:
                        st.caption("✅ High confidence")
                
                # Show processed text
                with st.expander("See processed text"):
                    processed = cleaner(user_input)
                    lemmatized = lemmatize_text(processed)
                    st.write(f"**Original:** {user_input}")
                    st.write(f"**Processed:** {lemmatized}")
            else:
                st.warning("Please enter some text to analyze!")
                
    except FileNotFoundError:
        st.error("Model files not found! Please run 'python train_model.py' first to train the model.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Make sure to run 'python train_model.py' to create the model files.")