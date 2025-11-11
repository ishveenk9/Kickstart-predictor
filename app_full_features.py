import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load model ---
with open("model_full.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]
features = data["features"]  # all features including one-hot
categorical_options = data["categorical_options"]

# --- Custom CSS for a clean, cohesive look ---
st.markdown(
    """
    <style>
    /* Background and font */
    body, .stApp {
        background-color: #f8f9fa;  /* light gray */
        color: #212529;              /* dark gray for text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main title */
    .stTitle {
        color: #0d6efd;  /* Bootstrap primary blue */
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Section headers */
    h3 {
        color: #0d6efd;
        border-bottom: 2px solid #0d6efd;
        padding-bottom: 5px;
        margin-top: 25px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #0d6efd;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        background-color: #0b5ed7;  /* darker blue on hover */
        color: white;
    }

    /* Input fields */
    .stNumberInput, .stSelectbox, .stTextInput {
        background-color: #ffffff;
        padding: 8px;
        border-radius: 6px;
        border: 1px solid #ced4da;  /* subtle gray border */
        margin-bottom: 10px;
    }

    /* Prediction output */
    .stMarkdown div {
        background-color: #e7f1ff;  /* soft blue highlight */
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        font-weight: bold;
        font-size: 18px;
        border: 1px solid #0d6efd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Random Forest Predictor")

st.markdown("### Select categorical values:")

# --- Collect categorical inputs ---
user_data = {}
for cat_col, options in categorical_options.items():
    choice = st.selectbox(f"{cat_col}", options)
    user_data[cat_col] = choice

st.markdown("### Enter numeric features:")

# --- Identify numeric features ---
numeric_features = [f for f in features if all(not f.startswith(cat + "_") for cat in categorical_options.keys())]
numeric_input = {}
for num_feat in numeric_features:
    val = st.number_input(f"{num_feat}", value=0.0)
    numeric_input[num_feat] = val

# --- Build input DataFrame matching model features ---
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0  # initialize all zeros

# Set categorical selections
for cat_col, choice in user_data.items():
    col_name = f"{cat_col}_{choice}"
    if col_name in input_df.columns:
        input_df.at[0, col_name] = 1

# Set numeric inputs
for num_feat, val in numeric_input.items():
    input_df.at[0, num_feat] = val

# --- Scale features ---
user_input_scaled = scaler.transform(input_df)

# --- Predict ---
if st.button("Predict"):
    pred = model.predict(user_input_scaled)
    pred_label = le.inverse_transform(pred)
    st.markdown(f"### Predicted class: {pred_label[0]}")
