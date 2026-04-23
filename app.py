import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
MODEL_PATH = "model1.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model1.pkl not found. Please upload it.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model, scaler, le, columns = pickle.load(f)

# -----------------------------
# TITLE
# -----------------------------
st.title("🌱 Smart Agriculture Prediction App")
st.write("Enter input values to predict output")

# -----------------------------
# USER INPUT
# -----------------------------
user_input = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    try:
        input_data = pd.DataFrame([user_input], columns=columns)

        # Ensure correct column order
        input_data = input_data[columns]

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        # Decode safely
        try:
            result = le.inverse_transform(prediction)
            output = result[0]
        except:
            output = prediction[0]

        st.success(f"🌾 Prediction: {output}")

        # Show confidence if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)
            st.info(f"Confidence: {np.max(probs):.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -----------------------------
# DATASET (LOCAL FILE)
# -----------------------------
st.subheader("Dataset Preview")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")  # <-- Put your dataset here

if st.button("Show Dataset"):
    try:
        df = load_data()
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
