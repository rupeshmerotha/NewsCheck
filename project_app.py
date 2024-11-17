import pandas as pd
import re
import string
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load datasets and cache preprocessing
@st.cache_data
def load_and_prepare_data():
    true = pd.read_csv('True.csv')
    fake = pd.read_csv('Fake.csv')
    true['label'] = 1
    fake['label'] = 0
    news = pd.concat([fake, true], axis=0)
    news = news.drop(['title', 'subject', 'date'], axis=1)
    news = news.sample(frac=1).reset_index(drop=True)
    news['text'] = news['text'].apply(textProcessing)
    x = news['text']
    y = news['label']
    return train_test_split(x, y, test_size=0.3)


# Text preprocessing function
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


# Train model and vectorizer
@st.cache_resource
def train_model():
    x_train, x_test, y_train, y_test = load_and_prepare_data()
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    model = LogisticRegression()
    model.fit(xv_train, y_train)
    return model, vectorization


# Main app
def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.title("📰 Fake News Detection")
    st.markdown(
        """
        <style>
        .stTextArea {
            margin-top: 20px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    model, vectorizer = train_model()
    st.subheader("Enter a news article to check its authenticity.")
    news_article = st.text_area("News Article:", height=150, placeholder="Type or paste the news article here...")

    if st.button("Check Authenticity"):
        if news_article.strip():
            with st.spinner("Analyzing..."):
                processed_text = textProcessing(news_article)
                vectorized_text = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized_text)[0]
            if prediction == 1:
                st.success("This news article appears to be **Real**.")
            else:
                st.error("This news article appears to be **Fake**.")
        else:
            st.warning("Please enter a valid news article.")


if __name__ == "__main__":
    main()
