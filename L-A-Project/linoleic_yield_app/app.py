import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained components
model = joblib.load("linoleic_yield_app\linoleic_model.pkl")
scaler = joblib.load("linoleic_yield_app\scaler.pkl")
label_encoders = joblib.load("linoleic_yield_app\label_encoders.pkl")

st.set_page_config(page_title="Linoleic Acid Yield Predictor", layout="centered")

st.title("🌿 Linoleic Acid Yield Prediction App")
st.markdown("Use this tool to predict linoleic acid yield based on process parameters.")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Reaction Parameters")

    temperature = st.slider("Temperature (°C)", 25.0, 65.0, 45.0)
    ph = st.slider("pH", 5.5, 9.0, 7.2)
    substrate_conc = st.slider("Substrate Concentration (g/L)", 80.0, 250.0, 150.0)
    reaction_time = st.slider("Reaction Time (hours)", 2.0, 16.0, 8.0)
    enzyme_conc = st.slider("Enzyme Concentration (U/mL)", 0.5, 5.0, 2.5)
    agitation_speed = st.slider("Agitation Speed (rpm)", 100.0, 350.0, 200.0)
    pressure = st.slider("Pressure (bar)", 1.0, 4.0, 2.0)

    catalyst_type = st.selectbox("Catalyst Type", ["Lipase_A", "Lipase_B", "Lipase_C", "Chemical_Cat"])
    feedstock_type = st.selectbox("Feedstock Type", ["Sunflower_Oil", "Soybean_Oil", "Corn_Oil", "Safflower_Oil"])

    submitted = st.form_submit_button("Predict Yield")

# Prediction
if submitted:
    # Encode categorical features
    catalyst_encoded = label_encoders['catalyst_type'].transform([catalyst_type])[0]
    feedstock_encoded = label_encoders['feedstock_type'].transform([feedstock_type])[0]

    # Create input array
    input_features = np.array([[temperature, ph, substrate_conc, reaction_time, 
                                 enzyme_conc, agitation_speed, pressure, 
                                 catalyst_encoded, feedstock_encoded]])

    # Scale if needed
    if hasattr(model, 'predict') and 'Regressor' not in str(type(model)):
        input_scaled = scaler.transform(input_features)
    else:
        input_scaled = input_features  # Tree-based models don't need scaling

    # Predict
    yield_pred = model.predict(input_scaled)[0]
    yield_pred = np.clip(yield_pred, 0, 100)

    st.success(f"✅ **Predicted Linoleic Acid Yield:** {yield_pred:.2f}%")
    st.markdown("📊 The yield is based on optimal biochemical conditions and ML model inference.")
