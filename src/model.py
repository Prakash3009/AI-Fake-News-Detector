import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def train_and_save_model(data_path, model_path, vectorizer_path):
    """
    Train a Logistic Regression model on the dataset.
    """
    # Dataset Loading
    df = pd.read_csv(data_path)
    
    # Preprocessing (already done in data_gen or on the fly)
    # Using 'text' as the feature and 'label' as the target (0 for Real, 1 for Fake)
    X = df['text']
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Logistic Regression Classifier
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Accuracy Calculation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    return model, vectorizer, accuracy, cm, classification_report(y_test, y_pred)

def predict_single(model, vectorizer, text, clean_func):
    """
    Predict if a single news article is Real or Fake.
    """
    cleaned_text = clean_func(text)
    tfidf_text = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(tfidf_text)[0]
    confidence = model.predict_proba(tfidf_text)[0][prediction]
    
    label = "FAKE" if prediction == 1 else "REAL"
    return label, confidence
