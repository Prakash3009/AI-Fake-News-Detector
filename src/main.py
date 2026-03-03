import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text
from model import train_and_save_model, predict_single
import pandas as pd
import numpy as np

# Files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'news_dataset.csv')
MODEL_FILE = os.path.join(DATA_DIR, 'model.pkl')
VEC_FILE = os.path.join(DATA_DIR, 'vectorizer.pkl')

def create_sample_data():
    """Create a small synthetic dataset for demonstration."""
    real_news = [
        "The scientific community reports a new breakthrough in renewable energy.",
        "Local government announces new public park project in the city center.",
        "The stock market showed steady growth in the final quarter of the year.",
        "Researchers find that regular exercise significantly improves mental health.",
        "The space agency successfully launched its latest observation satellite.",
        "New study highlights the benefits of a balanced diet for long-term health.",
        "City council votes to increase funding for public transportation.",
        "Astronomers discover a new planet in a distant star system.",
        "The manufacturing sector expects an increase in productivity this year.",
        "Health officials report a decrease in seasonal flu cases."
    ] * 20
    
    fake_news = [
        "Shocking discovery: lemons can cure all known diseases overnight!",
        "Government secretly planning to replace all currency with gold coins by Monday.",
        "Aliens have landed in a remote desert and are communicating with world leaders.",
        "Drinking magic water can make you live forever, says mysterious expert.",
        "The moon is actually made of cheese, new leaked documents reveal.",
        "Celebrity caught in a scandal involving secret underground tunnels.",
        "New law requires everyone to wear purple on Fridays or face huge fines.",
        "Dinosaurs are still alive and living in a hidden valley in the mountains.",
        "Instant wealth guaranteed by following this one weird trick with salt.",
        "The Eiffel Tower was actually built as a secret antenna to talk to Mars."
    ] * 20
    
    data = []
    for text in real_news:
        data.append({'text': clean_text(text), 'label': 0})
    for text in fake_news:
        data.append({'text': clean_text(text), 'label': 1})
        
    df = pd.DataFrame(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Sample dataset created at {DATA_FILE}")

def main():
    # Ensure data exists
    if not os.path.exists(DATA_FILE):
        create_sample_data()
    
    # Training Stage
    print("--- Model Training & Evaluation ---")
    model, vectorizer, accuracy, cm, report = train_and_save_model(DATA_FILE, MODEL_FILE, VEC_FILE)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("-" * 35)

    # Prediction Stage
    print("\n--- Fake News Detector ---")
    news_input = input("Enter news article text to predict: ") if sys.stdin.isatty() else \
                 "Breaking: Lemons secretively heal every illness according to leaked reports."
    
    if not news_input:
        news_input = "Scientists find new evidence of water on Mars."
        
    print(f"\nProcessing article: \"{news_input[:100]}...\"")
    label, confidence = predict_single(model, vectorizer, news_input, clean_text)
    
    print(f"\nRESULT: {label}")
    print(f"Confidence Score: {confidence:.4f}")

if __name__ == "__main__":
    main()
