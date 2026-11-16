
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
model_path = Path(__file__).parent / "house_price_model.joblib"
model = joblib.load(model_path)

# Page settings
st.set_page_config(
    page_title="Sukkur House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Sukkur House Price Prediction")
st.write("Enter the house details to estimate its price.")

st.divider()

# House information
st.subheader("📍 Property Details")

location = st.selectbox(
    "Location",
    [
        "Bunder Road",
        "Old Sukkur",
        "Shikarpur Road",
        "Barrage Colony",
        "Military Road",
        "Rohri",
        "Airport Road"
    ]
)

area_marla = st.number_input(
    "Area (Marla)", min_value=3.0, max_value=50.0, value=10.0
)

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input(
        "Bedrooms", min_value=1, max_value=8, value=3
    )

    bathrooms = st.number_input(
        "Bathrooms", min_value=1, max_value=8, value=2
    )

with col2:
    floors = st.number_input(
        "Floors", min_value=1, max_value=3, value=2
    )

    house_age = st.number_input(
        "House Age (Years)", min_value=0.0, max_value=45.0, value=10.0
    )

condition = st.selectbox(
    "House Condition",
    ["Poor", "Average", "Good", "Excellent"]
)

st.subheader("🏡 Additional Features")

col1, col2 = st.columns(2)

with col1:
    garage = st.selectbox("Garage", ["Yes", "No"])
    corner_plot = st.selectbox("Corner Plot", ["Yes", "No"])

with col2:
    near_main_road = st.selectbox(
        "Near Main Road", ["Yes", "No"]
    )
    listed_year = st.selectbox(
        "Listed Year", [2026, 2025, 2024]
    )

st.divider()

# Predict house price
if st.button("Predict House Price", type="primary", use_container_width=True):

    # Create input with the same feature names as model training
    input_data = pd.DataFrame([{
        "location": location,
        "area_marla": area_marla,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "house_age": house_age,
        "condition": condition,
        "garage": garage,
        "corner_plot": corner_plot,
        "near_main_road": near_main_road,
        "listed_year": listed_year
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Estimated House Price",
        value=f"Rs. {prediction:,.2f} Lakh"
    )

    st.caption(
        "This is an educational prediction based on a course practice dataset, "
        "not a verified market valuation."
    )
