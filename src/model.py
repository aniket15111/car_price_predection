import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def train_and_save_model(df: pd.DataFrame, model_path: str = "models/LinearRegressionModel.pkl"):
    X = df.drop(columns="Price")
    y = df["Price"]

    ohe = OneHotEncoder()
    column_trans = make_column_transformer(
        (OneHotEncoder(categories='auto'), ['name', 'company', 'fuel_type']),
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = make_pipeline(column_trans, LinearRegression())
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    score = r2_score(y_test, y_pred)

    joblib.dump(pipe, model_path)
    print(f"Model saved at {model_path} with R2 score: {score:.2f}")

    return pipe, score
