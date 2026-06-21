import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://localhost:8000/predict"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# =====================================================
# UI CONFIG
# =====================================================

st.set_page_config(
    page_title="Toxicity Detection System",
    layout="wide"
)

st.title("Multi-Platform Toxicity & Cyberbullying Detection")

st.write(
    "Enter a comment below to analyze toxicity across multiple categories."
)

# =====================================================
# INPUT
# =====================================================

text = st.text_area(
    "Comment Input",
    height=150,
    placeholder="Type or paste a comment here..."
)

# =====================================================
# PREDICTION FLOW
# =====================================================

if st.button("Analyze Comment"):

    if not text.strip():
        st.warning("Please enter a valid comment.")
        st.stop()

    try:
        response = requests.post(
            API_URL,
            json={"text": text}
        )

        result = response.json()

        scores = result.get("toxicity_scores", {})

        # =================================================
        # METRICS DISPLAY
        # =================================================

        st.subheader("Toxicity Breakdown")

        df = pd.DataFrame({
            "Label": list(scores.keys()),
            "Score": list(scores.values())
        })

        fig = px.bar(
            df,
            x="Label",
            y="Score",
            color="Score",
            range_y=[0, 1],
            title="Multi-Label Toxicity Scores"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =================================================
        # RAW OUTPUT
        # =================================================

        st.subheader("Raw Model Output")

        st.json(result)

        # =================================================
        # SIMPLE INTERPRETATION LAYER
        # =================================================

        max_label = max(scores, key=scores.get)
        max_score = scores[max_label]

        st.subheader("Interpretation")

        if max_score > 0.7:
            st.error(f"High confidence toxicity detected: {max_label}")
        elif max_score > 0.4:
            st.warning(f"Moderate toxicity detected: {max_label}")
        else:
            st.success("Low toxicity detected")

    except Exception as e:
        st.error("Failed to connect to FastAPI backend.")
        st.code(str(e))

# =====================================================
# SIDEBAR (SYSTEM INFO)
# =====================================================

st.sidebar.title("System Info")

st.sidebar.write(
    "This dashboard connects to a FastAPI inference backend."
)

st.sidebar.write(
    "Model: GRU + Transformer Ensemble (planned)"
)

st.sidebar.write(
    "Labels: 6-class multi-label toxicity detection"
)