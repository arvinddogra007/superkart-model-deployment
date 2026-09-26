import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")
st.title("SuperKart Sales Forecast")
st.caption("Forecast product-level store sales using the deployed regression pipeline.")

st.subheader("Online prediction")
with st.form("prediction_form"):
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.66)
    sugar_content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input(
        "Allocated display-area ratio", min_value=0.0, max_value=1.0, value=0.027, format="%.3f"
    )
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
    city_type = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    store_type = st.selectbox(
        "Store type",
        ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"],
        index=3,
    )
    product_family = st.selectbox("Product family", ["FD", "DR", "NC"])
    store_age = st.number_input("Store age (years)", min_value=0, value=16)
    product_category = st.selectbox(
        "Product category", ["Perishables", "Non Perishables"], index=1
    )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_family,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_category,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        response.raise_for_status()
        st.success(f"Predicted sales: {response.json()['Predicted Sales']:,.2f}")
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

st.subheader("Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV with all model features", type=["csv"])
if uploaded_file is not None and st.button("Predict batch", type="primary"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
            timeout=60,
        )
        response.raise_for_status()
        result = pd.DataFrame.from_dict(
            response.json(), orient="index", columns=["Predicted Sales"]
        )
        st.success(f"Generated {len(result)} predictions.")
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions",
            result.to_csv(index_label="Row").encode("utf-8"),
            "superkart_predictions.csv",
            "text/csv",
        )
    except requests.RequestException as exc:
        st.error(f"Batch request failed: {exc}")
