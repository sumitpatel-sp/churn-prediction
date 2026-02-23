# main.py

import os
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------- Safe paths ----------
BASE_DIR = os.path.dirname(__file__)

df_1 = pd.read_csv(os.path.join(BASE_DIR, "first_telc.csv"))
model = pickle.load(open(os.path.join(BASE_DIR, "model.sav"), "rb"))


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


@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}


@app.post("/predict")
def predict(data: CustomerInput):

    try:
        # ---------- Convert types (VERY IMPORTANT) ----------
        d = data.dict()

        d["SeniorCitizen"] = int(d["SeniorCitizen"])
        d["MonthlyCharges"] = float(d["MonthlyCharges"])
        d["TotalCharges"] = float(d["TotalCharges"])
        d["tenure"] = int(d["tenure"])

        new_df = pd.DataFrame([d])

        # ---------- Same logic as Flask ----------
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]

        df_2["tenure_group"] = pd.cut(
            df_2.tenure.astype(int),
            range(1, 80, 12),
            right=False,
            labels=labels
        )

        df_2.drop(columns=["tenure"], inplace=True)

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

        # ---------- Align columns with model ----------
        if hasattr(model, "feature_names_in_"):
            new_df_dummies = new_df_dummies.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        single = model.predict(new_df_dummies.tail(1))[0]
        probability = model.predict_proba(new_df_dummies.tail(1))[:, 1][0]

        result = (
            "This customer is likely to be churned"
            if single == 1
            else "This customer is likely to continue"
        )

        return {
            "prediction": result,
            "confidence": float(probability * 100)
        }

    except Exception as e:
        # ⭐ IMPORTANT — show real error instead of 500
        return {"error": str(e)}