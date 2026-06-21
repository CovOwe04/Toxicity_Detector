import streamlit as st
import requests
import plotly.bar as bar
import pandas as pd

API_URL = "http://localhost:8000/predict"

st.title("Toxicity & Cyberbullying Detection System")

text = st.text_area("Enter a comment")

if st.button("Analyze"):

    response = requests.post(
        API_URL,
        json={"text": text}
    ).json()

    scores = response["toxicity_scores"]

    st.subheader("Results")

    df = pd.DataFrame({
        "Label": list(scores.keys()),
        "Score": list(scores.values())
    })

    st.bar_chart(df.set_index("Label"))

    st.json(response)