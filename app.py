import pandas as pd
import numpy as np
import re
import string
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the datasets
true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

# Prepare the data
true['label'] = 1
fake['label'] = 0
news = pd.concat([fake, true], axis=0)
news = news.drop(['title', 'subject', 'date'], axis=1)

# Shuffle and reset index
news = news.sample(frac=1).reset_index(drop=True)


# Function to process text
def textProcessing(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Preprocess the text
news['text'] = news['text'].apply(textProcessing)
x = news['text']
y = news['label']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Vectorize the text
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)

# Train the model
LR = LogisticRegression()
LR.fit(xv_train, y_train)


# Function to get output label
def output_label(n):
    if n == 0:
        return "It's Fake News"
    elif n == 1:
        return "It's Real News"


# Streamlit app
def main():
    st.title("Fake News Detection")

    # Input for news article
    news_article = st.text_area("Enter the news article text:")

    if st.button("Detect"):
        if news_article:
            # Process and predict the input
            processed_article = textProcessing(news_article)
            new_xv_test = vectorization.transform([processed_article])
            prediction = LR.predict(new_xv_test)
            st.write("Prediction:", output_label(prediction[0]))
        else:
            st.warning("Please enter a news article to analyze.")


if __name__ == "__main__":
    main()
