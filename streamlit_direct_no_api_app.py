import streamlit as st
from model_utils import predict  # directly call your model instead of API

st.title("Customer Feedback Sentiment (Demo)")

# Input box
texts = st.text_area("Paste reviews (one per line)").strip().splitlines()

if st.button("Predict") and texts:
    preds, probs = predict(texts)  # directly call predict()
    
    for txt, p, prob in zip(texts, preds, probs):
        st.write(f"**{txt}** -> {p} (prob: {max(prob):.2f})")