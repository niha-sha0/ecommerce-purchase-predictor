app_code = '''
import streamlit as st
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load('ecommerce_model_pipeline.pkl')

st.set_page_config(page_title="Smart Shopper Predictor", layout="wide")
st.title("E-Commerce Purchase Intent Predictor")
st.caption("Predict whether a visitor will make a purchase — powered by XGBoost")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Browsing behaviour")
    admin    = st.slider("Admin pages visited",       0, 30,   3)
    admin_d  = st.slider("Admin duration (secs)",     0, 3000, 100)
    info     = st.slider("Info pages visited",        0, 30,   2)
    info_d   = st.slider("Info duration (secs)",      0, 3000, 50)
    product  = st.slider("Product pages visited",     0, 200,  25)
    product_d= st.slider("Product duration (secs)",   0, 6000, 600)

with col2:
    st.subheader("Session quality")
    bounce   = st.slider("Bounce rate",               0.0, 1.0, 0.02, 0.01)
    exit_r   = st.slider("Exit rate",                 0.0, 1.0, 0.04, 0.01)
    pg_val   = st.slider("Page value score",          0.0, 400.0, 10.0)
    special  = st.slider("Special day proximity",     0.0, 1.0, 0.0, 0.1)
    month    = st.selectbox("Month (1=Jan ... 12=Dec)", list(range(1, 13)), index=10)
    os_type  = st.selectbox("OS type",                list(range(1, 9)))
    browser  = st.selectbox("Browser type",           list(range(1, 9)))
    region   = st.selectbox("Region",                 list(range(1, 10)))
    traffic  = st.selectbox("Traffic type",           list(range(1, 21)))
    visitor  = st.selectbox("Visitor (0=Other, 1=Returning, 2=New)", [0, 1, 2])
    weekend  = st.checkbox("Weekend visit?")

if st.button("Predict Purchase Intent", type="primary"):

    total_pages    = admin + info + product
    total_dur      = admin_d + info_d + product_d
    bounce_diff    = bounce - exit_r
    is_engaged     = int(total_dur > 400)
    val_per_page   = pg_val / (product + 1)
    prod_ratio     = product / (total_pages + 1)
    high_val       = int(pg_val > 50)
    dur_per_page   = total_dur / (total_pages + 1)
    is_returning   = int(visitor == 1)
    info_ratio     = info / (total_pages + 1)
    weekend_sp     = int(weekend) * special

    input_data = pd.DataFrame([[
        admin, admin_d, info, info_d, product, product_d,
        bounce, exit_r, pg_val, special, month,
        os_type, browser, region, traffic, visitor, int(weekend), 0,
        total_pages, total_dur, bounce_diff,
        is_engaged, val_per_page, prod_ratio,
        high_val, dur_per_page, is_returning,
        info_ratio, weekend_sp
    ]], columns=[
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "Month", "OperatingSystems", "Browser", "Region",
        "TrafficType", "VisitorType", "Weekend", "SpecialDay_2",
        "total_pages", "total_duration", "bounce_exit_diff",
        "is_engaged", "value_per_page", "product_page_ratio",
        "high_value_session", "duration_per_page",
        "is_returning", "info_ratio", "weekend_special"
    ])

    # Keep only columns the model was trained on
    model_features = pipeline.feature_names_in_
    input_data = input_data[model_features]

    prob = pipeline.predict_proba(input_data)[0][1]
    pred = pipeline.predict(input_data)[0]

    st.metric("Purchase Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    if pred == 1:
        st.success("High purchase intent — show this customer a discount now!")
    else:
        st.warning("Low purchase intent — customer likely browsing only.")
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("app.py saved!")
