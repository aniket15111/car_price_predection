import pandas as pd
import joblib

def predict_price(model_path: str, car_features: list):
    pipe = joblib.load(model_path)
    df = pd.DataFrame([car_features], columns=["name", "company", "year", "kms_driven", "fuel_type"])
    return pipe.predict(df)[0]

if __name__ == "__main__":
    price = predict_price("models/LinearRegressionModel.pkl", 
                          ['Mahindra Jeep CL550', 'Mahindra', 2006, 100, 'Diesel'])
    print(f"Predicted Price: ₹{int(price)}")
