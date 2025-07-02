import streamlit as st
import pandas as pd
import re
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained('./distilbert_sentiment')
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_sentiment')

# Cleaning and prediction functions
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s#]', '', text)
    text = text.lower()
    return text

def predict_sentiment(tweet):
    cleaned_tweet = clean_text(tweet)
    inputs = tokenizer(cleaned_tweet, return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1)
    pred = tf.argmax(probs, axis=1).numpy()[0]
    score = probs[0][pred].numpy()
    sentiment = 'Positive' if pred == 1 else 'Negative'
    return sentiment, score

# Streamlit interface
st.title("Tweet Sentiment Analysis Dashboard")
st.sidebar.header("Sections")
section = st.sidebar.selectbox("Choose Section", 
                              ["EDA Visuals", "Model Predictions", "Model Performance"])

if section == "EDA Visuals":
    st.subheader("Exploratory Data Analysis")
    st.write("Visualizations from Sentiment140 dataset analysis")
    st.image('Length_Dist.png', caption='Tweet Length Distribution')
    st.image('Length_box.png', caption='Tweet Length by Sentiment')
    st.image('pos_wordcloud.png', caption='Positive Word Cloud')
    st.image('neg_wordcloud.png', caption='Negative Word Cloud')
    st.image('temporal_line.png', caption='Sentiment by Hour')
    st.image('temporal_heatmap.png', caption='Average Sentiment by Hour')
    st.image('hashtag_bar.png', caption='Top Hashtags')

elif section == "Model Predictions":
    # Single tweet prediction
    st.subheader("Predict Sentiment for a Single Tweet")
    user_input = st.text_area("Enter a Tweet", "I love #AI!", key="single_tweet")
    if st.button("Predict", key="predict_single"):
        if user_input:
            sentiment, score = predict_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}, Confidence: {score:.2f}")
        else:
            st.warning("Please enter a tweet.")

    # Batch prediction
    st.subheader("Analyze Multiple Tweets")
    uploaded_file = st.file_uploader("Upload Tweet CSV (with 'text' column)", type="csv", key="upload")
    if uploaded_file and st.button("Analyze", key="analyze_batch"):
        api_df = pd.read_csv(uploaded_file)
        if 'text' in api_df.columns and not api_df.empty:
            predictions = []
            scores = []
            for tweet in api_df['text']:
                sentiment, score = predict_sentiment(tweet)
                predictions.append(sentiment)
                scores.append(score)
            api_df['sentiment'] = predictions
            api_df['confidence'] = scores
            st.write(f"Analyzed {len(api_df)} tweets")
            st.dataframe(api_df[['text', 'sentiment', 'confidence']].head())
            st.download_button("Download Predictions", api_df.to_csv(index=False), "predictions.csv")
            st.image('api_sentiment_pie.png', caption='Sentiment Distribution for Uploaded Tweets')
            st.image('api_wordcloud.png', caption='Positive Tweet Word Cloud')
            st.image('api_word_bar.png', caption='Top Words in Tweets')
            st.image('api_conf_hist.png', caption='Confidence Score Distribution')
        else:
            st.error("CSV must contain a 'text' column and not be empty.")

elif section == "Model Performance":
    st.subheader("Model Performance")
    st.write("Metrics from test set evaluation")
    st.write("Accuracy: 0.87, F1-Score: 0.85")
    st.image('confusion_matrix.png', caption='Confusion Matrix')
