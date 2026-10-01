import streamlit as st
import pandas as pd
import joblib

# Set page layout and title
st.set_page_config(page_title="Rwanda House Price Predictor", layout="centered")

st.title("🏡 House Price Prediction App (Rwanda)")
st.write("Estimate house market prices in million RWF based on structural and location features.")

# Load saved model pipeline using Streamlit caching
@st.cache_resource
def load_model():
    return joblib.load('house_price_model.sav')

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'house_price_model.sav' not found. Please train and save the model first.")

# Input fields in sidebar
st.sidebar.header("House Features")

area = st.sidebar.number_input("Area (m²)", min_value=20, max_value=1000, value=120)
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=8, value=2)
age = st.sidebar.number_input("House Age (Years)", min_value=0, max_value=100, value=5)
distance = st.sidebar.number_input("Distance to City Centre (km)", min_value=0.0, max_value=50.0, value=5.0)
parking = st.sidebar.slider("Parking Spaces", min_value=0, max_value=5, value=1)
neighborhood = st.sidebar.selectbox("Neighborhood", ['Gasabo', 'Huye', 'Kicukiro', 'Kigali City', 'Musanze', 'Nyarugenge'])

# Construct input dataframe matching exact column names
input_data = pd.DataFrame([{
    'Area_m2': area,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'House_Age_Years': age,
    'Distance_to_City_km': distance,
    'Parking_Spaces': parking,
    'Neighborhood': neighborhood
}])

# Make prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"### Estimated Price: {prediction:,.2f} Million RWF")
