import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Toxicity Detector", layout="wide")
st.title("Multi-Platform Toxicity Detection System")
st.write("Enter a comment to analyze toxicity across 6 categories.")

text = st.text_area("Input Comment", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text.")
        st.stop()

    try:
        response = requests.post(API_URL, json={"text": text}, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            scores = result["toxicity_scores"]

            df = pd.DataFrame({
                "Label": list(scores.keys()),
                "Score": list(scores.values())
            })

            fig = px.bar(
                df,
                x="Label",
                y="Score",
                range_y=[0, 1],
                title="Toxicity Breakdown"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Raw Output")
            st.json(result)
        else:
            st.error(f"API returned status code: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Failed to connect to API at: {API_URL}")
        st.code(str(e))

st.sidebar.title("System Info")
st.sidebar.write("GRU + Transformer Hybrid")
st.sidebar.write("6-class multi-label toxicity detection system")
st.sidebar.write(f"Target API Endpoint: `{API_URL}`")