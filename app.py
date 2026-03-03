import streamlit as st
import pandas as pd
import joblib
import os
import sys
import plotly.express as px
import nltk
from pathlib import Path

# --- Constants & Path Handling ---
# Use Path for robust path handling in cloud environments
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# Add src to sys.path for internal imports
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# --- Safe NLTK Setup ---
@st.cache_resource
def setup_nltk():
    """Ensure NLTK resources are available without repeated downloads."""
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

setup_nltk()

from preprocess import clean_text
from model import predict_single

# --- Model Loading ---
@st.cache_resource
def load_assets():
    """Load model and vectorizer with caching to prevent re-loading on every run."""
    model_path = DATA_DIR / "model.pkl"
    vec_path = DATA_DIR / "vectorizer.pkl"
    
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"Missing model files at {DATA_DIR}. Ensure they are committed to the repository.")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🕵️",
    layout="centered"
)

# --- UI Components ---
st.sidebar.title("Model Details")
st.sidebar.info("""
**Architecture:** Logistic Regression
**Feature Engineering:** TF-IDF Vectorization
**Max Features:** 5000
**Preprocessing:** Lowercasing, Punctuation Removal, Stopword Removal
""")
st.sidebar.divider()
st.sidebar.markdown("### Model Provenance")
st.sidebar.text("Training data: 400 articles")
st.sidebar.text("Status: Operational")

st.title("🕵️ AI Fake News Detector")
st.markdown("""
This professional AI tool uses advanced **Natural Language Processing (NLP)** and **Logistic Regression** to analyze news articles and determine their credibility.
""")

st.subheader("Article Analysis")
news_input = st.text_area("Paste the news article text below for instant classification:", height=250, placeholder="Enter text here...")

if st.button("Analyze Article", type="primary"):
    if not news_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            model, vectorizer = load_assets()
            
            # Prediction
            label, confidence = predict_single(model, vectorizer, news_input, clean_text)
            
            # Visualization logic
            cleaned_text = clean_text(news_input)
            tfidf_text = vectorizer.transform([cleaned_text])
            probs = model.predict_proba(tfidf_text)[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if label == "REAL":
                    st.success(f"### Result: {label} NEWS")
                else:
                    st.error(f"### Result: {label} NEWS")
                
                st.metric("Confidence Score", f"{confidence:.4f}")
            
            with col2:
                prob_df = pd.DataFrame({
                    'Category': ['REAL', 'FAKE'],
                    'Probability': probs
                })
                fig = px.bar(prob_df, x='Category', y='Probability', 
                             color='Category', 
                             color_discrete_map={'REAL': '#2ecc71', 'FAKE': '#e74c3c'},
                             range_y=[0, 1])
                fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "AI Project Credit | Developed by Senior AI Engineering Team"
    "</div>", 
    unsafe_allow_html=True
)
