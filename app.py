# main.py

import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------- Load dataset + model once ----------
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))


# ---------- Input schema ----------
class CustomerInput(BaseModel):
    SeniorCitizen: str
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
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
    tenure: int


# ---------- Home route ----------
@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}


# ---------- Prediction route ----------
@app.post("/predict")
def predict(data: CustomerInput):

    # Convert input → DataFrame
    new_df = pd.DataFrame([data.dict()])

    # Merge with original dataset (same as Flask)
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # tenure grouping
    labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(
        df_2.tenure.astype(int),
        range(1, 80, 12),
        right=False,
        labels=labels
    )

    df_2.drop(columns=["tenure"], inplace=True)

    # One hot encoding
    new_df_dummies = pd.get_dummies(
        df_2[
            [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'tenure_group'
            ]
        ]
    )

    # Prediction
    single = model.predict(new_df_dummies.tail(1))[0]
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1][0]

    # Output (same logic as Flask)
    if single == 1:
        result = "This customer is likely to be churned"
    else:
        result = "This customer is likely to continue"

    return {
        "prediction": result,
        "confidence": float(probability * 100)
    }