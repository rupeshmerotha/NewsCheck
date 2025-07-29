import pandas as pd
from project_app import textProcessing

def test_text_processing():
    sample = "Breaking: COVID-19 cases rise! See more at https://news.com."
    processed = textProcessing(sample)
    assert 'covid' in processed
    assert 'https' not in processed
    assert '!' not in processed
    print("textProcessing() passed.")

if __name__ == "__main__":
    test_text_processing()
    print("All tests passed.")
