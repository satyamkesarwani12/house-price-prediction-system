import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)
st.image("banner.png", use_container_width=True)

# Load Model
model = joblib.load("model.pkl")

#side bar
st.sidebar.title("🏠 House Price Prediction")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.info("""
📌 Algorithm:
Random Forest Regressor

📈 Accuracy (R²):
80.75%

👨‍💻 Developer:
Satyam Kesarwani
""")

# Title
#st.title("🏠 House Price Prediction System")
st.write("Predict California House Prices using Machine Learning")

st.markdown("---")
col1, col2 = st.columns(2)
# Input Section
with col1:
    longitude = st.number_input(
        "Longitude",
        value=-122.23,
        help="Location of the house"
    )

with col2:
    latitude = st.number_input(
        "Latitude",
         value=37.88,
        help="Location of the house"
    )
with col1:
    housing_median_age = st.number_input(
        "Housing Median Age",
         value=41
    )
with col2:
    total_rooms = st.number_input(
        "Total Rooms",
        value=880
    )
with col1:
    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=129
    )
with col2:
    population = st.number_input(
        "Population",
        value=322
    )
with col1:
    households = st.number_input(
        "Households",
        value=126
    )
with col2:
    median_income = st.number_input(
        "Median Income",
        value=8.3252
    )

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
)

mapping = {
    "NEAR BAY": 3,
    "<1H OCEAN": 0,
    "INLAND": 1,
    "NEAR OCEAN": 4,
    "ISLAND": 2
}

st.markdown("---")

if st.button("🔍 Predict Price"):

    data = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        mapping[ocean_proximity]
    ]])

    prediction = model.predict(data)
    
    st.markdown("---")
    st.subheader("📈 Feature Importance")

    feature_names = [
        "Longitude",
        "Latitude",
        "Housing Age",
        "Total Rooms",
        "Total Bedrooms",
        "Population",
        "Households",
        "Median Income",
        "Ocean Proximity"
    ]

    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(feature_names, importance)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")

    st.pyplot(fig)

    
    # Prediction History
    st.markdown("---")
    st.subheader("📊 Prediction Details")

    result = {
    "Feature": [
        "Longitude",
        "Latitude",
        "Housing Median Age",
        "Total Rooms",
        "Total Bedrooms",
        "Population",
        "Households",
        "Median Income",
        "Ocean Proximity",
        "Predicted Price"
        ],
        "Value": [
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity,
            f"${prediction[0]:,.2f}"
        ]
    }

    st.table(result)
    st.balloons()

    st.success(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")

    st.info(f"""
        ### 📋 Prediction Summary

        - 💰 Predicted Price: **${prediction[0]:,.2f}**
        - 🤖 Model Used: **Random Forest Regressor**
        - 📊 Accuracy (R²): **80.75%**
        """)
    st.write("### Prediction Summary")

    st.table({
        "Feature": [
            "Longitude",
            "Latitude",
            "Median Income",
            "Ocean Proximity"
        ],
        "Value": [
            longitude,
            latitude,
            median_income,
            ocean_proximity
        ]
    })

st.markdown("---")

st.markdown("---")
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", "20,640")
    st.metric("Total Features", "9")

with col2:
    st.metric("Best Model", "Random Forest")
    st.metric("R² Score", "80.75%")
st.markdown("---")

st.subheader("📊 Model Performance")

comparison = {
    "Model": ["Linear Regression", "Random Forest"],
    "R² Score": [0.6137, 0.8075]
    
}

st.table(comparison)
import pandas as pd

comparison_df = pd.DataFrame(comparison)

st.bar_chart(comparison_df.set_index("Model"))
st.caption("Developed using Python, Scikit-learn and Streamlit")