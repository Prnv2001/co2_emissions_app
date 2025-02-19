import streamlit as st
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

# Load trained XGBoost model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Mean and standard deviation of CO₂ emissions (from training data)
mean_y = 200.5  # Replace with actual mean
std_y = 45.3    # Replace with actual std

# Streamlit UI
st.title("🚗 CO2 Emissions Prediction App")
st.write("Enter the car features below to predict CO2 emissions (g/km):")

# User input fields
make = st.text_input("Car Make (e.g., Toyota, Ford, BMW)", value="Unknown")
model_name = st.text_input("Car Model", value="Unknown")
vehicle_class = st.text_input("Vehicle Class (e.g., Sedan, SUV)", value="Unknown")
transmission = st.selectbox("Transmission Type", ["Automatic", "Manual", "Automated Manual", "CVT"])
fuel_type = st.selectbox("Fuel Type", ["Regular Gasoline", "Premium Gasoline", "Diesel", "Ethanol", "Natural Gas"])
engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, value=4, step=1)
fuel_consumption_city = st.number_input("City Fuel Consumption (L/100km)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
fuel_consumption_hwy = st.number_input("Highway Fuel Consumption (L/100km)", min_value=0.0, max_value=30.0, value=6.0, step=0.1)
fuel_consumption_comb_l = st.number_input("Combined Fuel Consumption (L/100km)", min_value=0.0, max_value=30.0, value=7.0, step=0.1)
fuel_consumption_comb_mpg = st.number_input("Combined Fuel Consumption (mpg)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

# Convert categorical inputs into numerical format
transmission_dict = {"Automatic": 0, "Manual": 1, "Automated Manual": 2, "CVT": 3}
fuel_type_dict = {"Regular Gasoline": 0, "Premium Gasoline": 1, "Diesel": 2, "Ethanol": 3, "Natural Gas": 4}

# Apply encoding
transmission_encoded = transmission_dict[transmission]
fuel_type_encoded = fuel_type_dict[fuel_type]

# Encode text categories using hash values (consistent but unique for each input)
make_encoded = hash(make) % 1000
model_encoded = hash(model_name) % 1000
vehicle_class_encoded = hash(vehicle_class) % 1000

# Prepare input as DataFrame with all expected features
input_data = pd.DataFrame([[make_encoded, model_encoded, vehicle_class_encoded, engine_size, cylinders, transmission_encoded, fuel_type_encoded,
                            fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb_l, fuel_consumption_comb_mpg]],
                          columns=['make', 'model', 'vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel_type',
                                   'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)', 'fuel_consumption_comb(mpg)'])

# Predict button
if st.button("Predict CO2 Emissions"):
    # Make prediction (standardized output)
    prediction_scaled = model.predict(input_data)[0]  # Get single prediction

    # De-standardize the prediction
    prediction_actual = (prediction_scaled * std_y) + mean_y

    # Display result
    st.success(f"🔹 Predicted CO2 Emissions: {prediction_actual:.2f} g/km")
