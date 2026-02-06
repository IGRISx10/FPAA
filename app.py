import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load model and feature schema
# -----------------------------
model = joblib.load("ensemble_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Football Injury Category Prediction",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("⚽ Football Injury Category Prediction")
st.subheader("Machine Learning–based Decision Support Prototype")

st.markdown("""
This application demonstrates how an **ensemble machine learning model**
predicts **injury categories** based on player context.
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Player Information")

age = st.sidebar.slider("Age", min_value=16, max_value=40, value=25)

position = st.sidebar.selectbox(
    "Playing Position",
    [
        "Centre Back",
        "Defensive Midfielder",
        "Central Midfielder",
        "Winger",
        "Forward",
        "Goalkeeper"
    ]
)

season = st.sidebar.selectbox(
    "Season",
    ["Summer", "Autumn", "Winter", "Spring"]
)

# -----------------------------
# Create input dataframe
# -----------------------------
input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

# Set AGE
if "AGE" in input_data.columns:
    input_data["AGE"] = age

# Set POSITION one-hot
position_col = f"POSITION_{position}"
if position_col in input_data.columns:
    input_data[position_col] = 1

# Set SEASON one-hot
season_col = f"SEASON_{season}"
if season_col in input_data.columns:
    input_data[season_col] = 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Injury Category"):

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    # Top-2 predictions
    top_indices = np.argsort(probs)[::-1][:2]
    top1, top2 = classes[top_indices[0]], classes[top_indices[1]]

    st.success("### 🏥 Prediction Results")

    st.markdown(f"""
    **Most Likely Injury Category:**  
    🥇 **{top1}**

    **Second Most Likely Injury Category:**  
    🥈 **{top2}**
    """)

    # -----------------------------
    # Confidence display
    # -----------------------------
    confidence_df = pd.DataFrame({
        "Injury Category": classes,
        "Confidence": probs
    }).sort_values(by="Confidence", ascending=False)

    st.markdown("### 📊 Model Confidence Scores")
    st.bar_chart(confidence_df.set_index("Injury Category"))

    # -----------------------------
    # Explanation
    # -----------------------------
    st.markdown("### 🧠 Model Explanation")
    st.info(
        "Predictions are influenced primarily by **player position** and **age**, "
        "with seasonal context playing a secondary role. "
        "Soft tissue injuries are more strongly associated with wide and midfield positions."
    )

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.warning("""
⚠️ **Disclaimer**

This application is an **academic machine learning prototype** developed for analytical
and educational purposes only.

It does **not** provide medical advice, diagnosis, or injury prevention guidance and
must not be used for clinical decision-making.
""")
