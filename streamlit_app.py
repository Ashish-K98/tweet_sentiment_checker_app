import streamlit as st
import requests
from typing import List

API_URL = "http://localhost:8000/predict"

st.title("Customer Feedback Sentiment (Demo)")

texts = st.text_area("Paste reviews (one per line)").strip().splitlines()
if st.button("Predict") and texts:
    resp = requests.post(API_URL, json={"texts": texts}, timeout=10)
    if resp.ok:
        data = resp.json()
        for txt, p, prob in zip(texts, data["predictions"], data["probabilities"]):
            st.write(f"**{txt}** -> {p} (prob: {max(prob):.2f})")
    else:
        st.error(f"API error: {resp.status_code} {resp.text}")