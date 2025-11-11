import streamlit as st

# --- Custom CSS (same as main page) ---
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #6699CC;
        color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stTitle h1 {
        color: #000000 !important;
        text-decoration: none !important;
        pointer-events: none !important;
    }

    label, div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Feature Explanations")
st.write("Here you can explain all the features of your model.")

st.markdown("""
Here you can explain all the features used in your Random Forest model.  
For example:

- **Feature A**: Description of Feature A
- **Feature B**: Description of Feature B
- **Feature C**: Description of Feature C

Add all features and their explanations here to help users understand the model.
""")
