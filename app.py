import streamlit as st
import pandas as pd
import joblib
import os
import sys
import plotly.express as px

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import clean_text
from model import predict_single

# Page Configuration
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🕵️",
    layout="centered"
)

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_FILE = os.path.join(DATA_DIR, 'model.pkl')
VEC_FILE = os.path.join(DATA_DIR, 'vectorizer.pkl')

# Cache model loading
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VEC_FILE)
    return model, vectorizer

# Sidebar
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

# Title & Description
st.title("🕵️ AI Fake News Detector")
st.markdown("""
This professional AI tool uses advanced **Natural Language Processing (NLP)** and **Logistic Regression** to analyze news articles and determine their credibility.
""")

# Text Input
st.subheader("Article Analysis")
news_input = st.text_area("Paste the news article text below for instant classification:", height=250, placeholder="Enter text here...")

if st.button("Analyze Article", type="primary"):
    if not news_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            model, vectorizer = load_assets()
            
            # Get Prediction
            label, confidence = predict_single(model, vectorizer, news_input, clean_text)
            
            # Probabilities for visualization
            cleaned_text = clean_text(news_input)
            tfidf_text = vectorizer.transform([cleaned_text])
            probs = model.predict_proba(tfidf_text)[0]
            
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if label == "REAL":
                    st.success(f"### Result: {label} NEWS")
                else:
                    st.error(f"### Result: {label} NEWS")
                
                st.metric("Confidence Score", f"{confidence:.4f}")
            
            with col2:
                # Visualization
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
            
        except FileNotFoundError:
            st.error("Model files not found. Please ensure the model has been trained and saved in the 'data/' folder.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "AI Project Credit | Developed by Senior AI Engineering Team"
    "</div>", 
    unsafe_allow_html=True
)
