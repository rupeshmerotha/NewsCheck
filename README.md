# 🔎 NewsCheck – Fake News Detection App

**NewsCheck** is a web-based fake news detection tool built using **Python**, **Streamlit**, and **scikit-learn**. It allows users to input a news article and instantly check its authenticity using a trained machine learning model.

---

## 🚀 Features

- Classifies news articles as **Real** or **Fake**
- Clean and interactive **Streamlit UI**
- Uses **Logistic Regression** with **TF-IDF Vectorization**
- Displays Accuracy, Precision, Recall, and F1 Score
- Smart preprocessing of text: lowercasing, punctuation removal, URL cleaning, etc.
- Caching enabled for faster performance

---

## 📁 Project Structure

```
.
├── project_app.py        # Main Streamlit application
├── test_app.py           # Unit test for text preprocessing
├── True.csv              # Dataset of real news (required)
├── Fake.csv              # Dataset of fake news (required)
├── model.joblib          # Trained model (auto-generated)
├── vectorizer.joblib     # TF-IDF vectorizer (auto-generated)
├── metrics.joblib        # Evaluation metrics (auto-generated)
```

---

## 🧠 How It Works

1. Loads and combines real (`True.csv`) and fake (`Fake.csv`) news datasets.
2. Cleans and processes the text using regex-based preprocessing.
3. Splits data into training and testing sets.
4. Trains a Logistic Regression model on TF-IDF features.
5. Caches model, vectorizer, and performance metrics.
6. Launches a Streamlit UI for users to input news and get predictions.

---

## ✅ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies

Create a virtual environment (recommended), then install required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pip install streamlit pandas scikit-learn seaborn matplotlib joblib
```

### 3. Add Required Datasets

Ensure the following files are present in the root directory:

- `True.csv`
- `Fake.csv`

You can download them from [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## ▶️ Run the App

```bash
streamlit run project_app.py
```

Then open the app in your browser at `http://localhost:8501`.

---

## 🧪 Run Tests

To test the preprocessing function:

```bash
python test_app.py
```

Expected output:

```
textProcessing() passed.
All tests passed.
```

---

## 📊 Example Model Metrics

| Metric     | Score  | Description                                       |
|------------|--------|---------------------------------------------------|
| Accuracy   | ~99%   | Overall prediction correctness                    |
| Precision  | ~99%   | How many predicted Fake articles were actually fake |
| Recall     | ~99%   | How many actual Fake articles were correctly detected |
| F1 Score   | ~99%   | Balance between precision and recall              |

---

## 📌 Notes

- If `model.joblib` or `vectorizer.joblib` don’t exist, they will be created automatically.
- To force retraining, delete `model.joblib`, `vectorizer.joblib`, and `metrics.joblib`.
- Make sure your CSVs contain a `text` column.

---

## 🙌 Acknowledgements

- [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Streamlit, scikit-learn, and the open-source ML community
