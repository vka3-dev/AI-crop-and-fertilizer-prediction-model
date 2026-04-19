# app.py
import streamlit as st
import numpy as np
import joblib


# 1. Load saved models & encoders
crop_model = joblib.load("crop_model.plk")
crop_le    = joblib.load("crop_encode.plk")

fert_model = joblib.load("fertilizer.plk")
fert_le    = joblib.load("fertilizer_encode.plk")
crop_to_fert_crop_le = joblib.load("ferti_crop_encode.plk")
soil_le    = joblib.load("soil_encode.plk")


# ---------- PAGE CONFIG & TITLE ----------
st.set_page_config(page_title="Smart Crop & Fertilizer Advisor", layout="centered")

st.title("Smart Crop & Fertilizer Advisor 🌾💧")
st.write(
    "Enter your field and soil details below to get a recommended crop "
    "and a matching fertilizer suggestion."
)


# ---------- INPUT FORM (SINGLE BLOCK) ----------
with st.form("input_form"):
    st.subheader("Field & Soil Details")

    col1, col2 = st.columns(2)

    with col1:
        N  = st.number_input("Nitrogen (N)", value=50.0)
        P  = st.number_input("Phosphorus (P)", value=50.0)
        K  = st.number_input("Potassium (K)", value=50.0)
        ph = st.number_input("Soil pH", value=6.5)

    with col2:
        temperature = st.number_input("Temperature (°C)", value=25.0)
        humidity    = st.number_input("Humidity (%)", value=80.0)
        rainfall    = st.number_input("Rainfall (mm)", value=200.0)
        moisture    = st.number_input("Soil Moisture", value=30.0)

    st.markdown("### Additional Soil Properties")
    carbon = st.number_input("Organic Carbon", value=1.0)

    soil_options = list(soil_le.classes_)
    soil_str = st.selectbox("Soil Type", soil_options)
    soil_en  = soil_le.transform([soil_str])[0]

    submitted = st.form_submit_button("Recommend Crop & Fertilizer")


# ---------- PREDICTION PIPELINE ----------
if submitted:
    # 1) Crop prediction
    crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_pred_enc = crop_model.predict(crop_features)[0]
    crop_name     = crop_le.inverse_transform([crop_pred_enc])[0]

    # 2) Prepare inputs for fertilizer model
    crop_en_for_fert = crop_to_fert_crop_le.transform([crop_name])[0]

    fert_features = np.array([[
        temperature,        # Temperature
        moisture,           # Moisture
        rainfall,           # Rainfall
        ph,                 # PH
        N,                  # Nitrogen
        P,                  # Phosphorous
        K,                  # Potassium
        carbon,             # Carbon
        soil_en,            # Soil_en (encoded)
        crop_en_for_fert    # Crop_en (encoded)
    ]])

    fert_pred_enc = fert_model.predict(fert_features)[0]
    fert_name     = fert_le.inverse_transform([fert_pred_enc])[0]

    # ---------- RESULTS SECTION ----------
    st.markdown("---")
    st.subheader("Recommendation")

    col_crop, col_fert = st.columns(2)

    with col_crop:
        st.markdown("**Recommended Crop**")
        st.success(crop_name)

    with col_fert:
        st.markdown("**Recommended Fertilizer**")
        st.success(fert_name)

    st.markdown(
        f"*Soil type:* **{soil_str}** &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"*Rainfall:* **{rainfall} mm** &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"*Temperature:* **{temperature} °C**"
    )
