# app.py

import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Emotion Recognition App",
    page_icon="😊",
    layout="wide"
)

# --- App Title and Description ---
st.title("Emotion Recognition from Text")
st.markdown("""
This app demonstrates a comparative analysis of different Transformer models 
for multi-label emotion recognition. Choose a model, enter some text, and see its predictions!
""")

# --- API Endpoint ---
API_URL = "http://127.0.0.1:8000/predict"

# --- User Interface ---

# Model selection dropdown
model_options = ['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base']
selected_model = st.selectbox("1. Choose a model for inference", model_options)

# Text input area
text_to_analyze = st.text_area("2. Enter the text you want to analyze",
                               "I can't believe I finally finished my bachelor's project. It was so much work, but I'm thrilled with the result!",
                               height=150)

# Analyze button
if st.button("Analyze Emotion"):
    if text_to_analyze:
        with st.spinner(f"Asking the {selected_model} model what it thinks..."):
            # Prepare the request payload
            payload = {
                "text": text_to_analyze,
                "model_name": selected_model
            }

            # Send request to the API
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes

                results = response.json()

                # --- Display Results ---
                st.subheader("Analysis Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Model Used", value=results.get("model_used"))
                    st.metric(label="Response Time (Latency)", value=f"{results.get('latency'):.4f} seconds")

                with col2:
                    st.write("Predicted Emotions:")
                    if results.get("predictions"):
                        # Create a DataFrame for plotting
                        df = pd.DataFrame(results["predictions"])
                        df.rename(columns={'label': 'Emotion', 'score': 'Confidence'}, inplace=True)
                        st.dataframe(df)

                        # Bar chart for better visualization
                        st.bar_chart(df.set_index('Emotion')['Confidence'])
                    else:
                        st.info("No emotions detected above the confidence threshold.")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please make sure the backend is running. Error: {e}")

    else:
        st.warning("Please enter some text to analyze.")