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
features = data["features"]
categorical_options = data["categorical_options"]

# --- Custom CSS ---
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #6699CC;
        color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stTitle h1, .stTitle h1 a {
        color: #000000 !important;
        text-decoration: none !important;
        pointer-events: none !important;
    }

    div[data-testid="stVerticalBlock"] > div:first-child:empty {
        display: none;
    }

    .stNumberInput, .stSelectbox, .stTextInput {
        color: #000000;
    }

    div.stButton > button:first-child {
        background-color: #1a73e8;
        color: #ffffff;
        font-size: 16px;
        padding: 12px 25px;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1558b0;
    }

    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ced4da;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        color: #000000;
        margin-top: 20px;
    }

    label, div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("Random Forest Predictor")

# --- Sidebar link to feature explanations ---
st.sidebar.markdown("### Navigation")
st.sidebar.button("Feature Explanations", on_click=lambda: st.experimental_set_query_params(page="explain_features"))

# --- Collect categorical inputs ---
user_data = {}
for cat_col, options in categorical_options.items():
    user_data[cat_col] = st.selectbox(f"{cat_col}", options)

# --- Collect numeric inputs ---
numeric_features = [f for f in features if all(not f.startswith(cat + "_") for cat in categorical_options.keys())]
numeric_input = {}
for num_feat in numeric_features:
    numeric_input[num_feat] = st.number_input(f"{num_feat}", value=0.0)

# --- Build input DataFrame matching model features ---
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0

for cat_col, choice in user_data.items():
    col_name = f"{cat_col}_{choice}"
    if col_name in input_df.columns:
        input_df.at[0, col_name] = 1

for num_feat, val in numeric_input.items():
    input_df.at[0, num_feat] = val

# --- Scale features ---
user_input_scaled = scaler.transform(input_df)

# --- Predict ---
if st.button("Predict"):
    pred = model.predict(user_input_scaled)
    pred_label = le.inverse_transform(pred)
    st.markdown(f'<div class="prediction-box">Predicted class: {pred_label[0]}</div>', unsafe_allow_html=True)
