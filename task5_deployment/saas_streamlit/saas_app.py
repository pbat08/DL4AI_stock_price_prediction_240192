import streamlit as st
import requests
import numpy as np

# Replace this with your live API URL from the previous cell
API_URL = "https://clever-steaks-knock.loca.lt"   # <-- update

st.title("📈 Vietnam Stock Price Prediction SaaS")

ticker = st.selectbox("Stock", ["VIC", "HPG", "VCB", "GAS", "PLX"])
open_p = st.number_input("Open", value=100.0)
high   = st.number_input("High", value=102.0)
low    = st.number_input("Low", value=99.0)
close  = st.number_input("Close", value=101.0)
volume = st.number_input("Volume", value=1_000_000)

if st.button("Predict"):
    payload = {
        "ticker": ticker,
        "features": {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume}
    }
    resp = requests.post(API_URL + '/predict', json=payload).json()
    if "prediction" in resp:
        st.success(f"Predicted price: **{resp['prediction'][0][0]:.2f}**")
    else:
        st.error(resp.get("error", "Unknown error"))
