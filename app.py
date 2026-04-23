
import streamlit as st
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load('ecommerce_model_pipeline.pkl')

st.set_page_config(page_title="Smart Shopper Predictor", page_icon="🛒", layout="wide")
st.title("E-Commerce Purchase Intent Predictor")

col1, col2 = st.columns(2)
with col1:
    product  = st.slider("Product pages visited", 0, 200, 25)
    duration = st.slider("Total time on site (seconds)", 0, 6000, 800)
    bounce   = st.slider("Bounce rate", 0.0, 1.0, 0.02, 0.01)
    exit_r   = st.slider("Exit rate", 0.0, 1.0, 0.04, 0.01)

with col2:
    pg_val   = st.slider("Page value score", 0.0, 400.0, 10.0)
    month    = st.selectbox("Month", list(range(1, 13)), index=10)
    visitor  = st.selectbox("Visitor type (0=Other, 1=Returning, 2=New)", [0, 1, 2])
    weekend  = st.checkbox("Weekend visit?")

if st.button("Predict Purchase Intent", type="primary"):
    admin = 3
    info  = 2
    total_pages   = admin + info + product
    total_dur     = duration
    bounce_diff   = bounce - exit_r
    is_engaged    = int(total_dur > 400)
    val_per_page  = pg_val / (product + 1)
    prod_ratio    = product / (total_pages + 1)
    high_val      = int(pg_val > 50)
    dur_per_page  = total_dur / (total_pages + 1)
    is_returning  = int(visitor == 1)
    info_ratio    = info / (total_pages + 1)
    weekend_sp    = int(weekend) * 0.0

    row = [[admin, 0, info, 0, product, 0,
            bounce, exit_r, pg_val, 0.0, month,
            1, 1, 1, 0, visitor, int(weekend), 0,
            total_pages, total_dur, bounce_diff,
            is_engaged, val_per_page, prod_ratio,
            high_val, dur_per_page, is_returning,
            info_ratio, weekend_sp]]

    prob = pipeline.predict_proba(row)[0][1]
    pred = pipeline.predict(row)[0]

    st.metric("Purchase Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    if pred == 1:
        st.success("High purchase intent — show this customer a discount!")
    else:
        st.warning("Low purchase intent — customer likely to leave without buying.")
