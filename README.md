# Fake News Detection

A simple web app to detect fake news using machine learning. Built with Streamlit, pandas, and scikit-learn.

## Features
- Upload or paste a news article and check if it's real or fake
- Uses logistic regression and TF-IDF vectorization
- Handles large datasets (`True.csv` and `Fake.csv`)

## Setup Instructions

1. **Clone or download this repository**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place `True.csv` and `Fake.csv` in the project directory** (these are required for the app to work).

5. **Run the Streamlit app:**
   ```bash
   streamlit run project_app.py
   ```

## Troubleshooting
- If you see a missing file error, make sure `True.csv` and `Fake.csv` are present.
- If you have dependency issues, try upgrading pip: `pip install --upgrade pip` and reinstall requirements.
- For any other errors, check the error message in the Streamlit UI.

## License
MIT