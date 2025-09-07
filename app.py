import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("models/LinerRegressionModel.pkl", "rb"))

# Load car names
car = pd.read_csv("data/cleaned_car.csv")
car_names = sorted(car['name'].unique())
companies = sorted(car['company'].unique())
fuel_types = sorted(car['fuel_type'].unique())

st.title("🚗 Car Price Prediction App")

# Dropdowns
name = st.selectbox("Select Car", car_names)
company = st.selectbox("Select Company", companies)
fuel = st.selectbox("Select Fuel Type", fuel_types)
year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("KMs Driven", min_value=0, step=500)

if st.button("Predict Price"):
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {int(price):,}")
