# app/main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API (Logistic Regression)")

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/churn_model.pkl")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    X = preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": bool(prob >= 0.5),
    }