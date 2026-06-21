import streamlit as st
from toxicity_model import predict

st.set_page_config(
    page_title="Toxicity Detection Dashboard",
    layout="wide"
)

st.title("Multi-Platform Toxicity Detection Pipeline")

st.write(
    """
    Enter a comment below to analyze toxicity levels.
    """
)

comment = st.text_area(
    "Comment",
    height=150
)

if st.button("Analyze"):

    result = predict(comment)

    st.subheader("Prediction")

    st.write(
        f"Toxicity Score: {result['toxicity_score']:.2f}"
    )

    st.write(
        f"Detected Labels: {result['labels']}"
    )

    st.json(result)