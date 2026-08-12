from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load model + preprocessor
rf_model = joblib.load('notebook/rf_model.joblib')
preprocessor = joblib.load('notebook/preprocessor.joblib')



app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent transactions using a pre-trained Random Forest"
)

class TransactionData(BaseModel):
    source_currency: str = Field(example="usd")
    dest_currency: str = Field(example="cad")
    channel: str = Field(example="web")
    amount_src: float = Field(example=10.0)
    exchange_rate_src_to_dest: float = Field(example=1.0)
    device_id: str = Field(example="device123")
    new_device: bool = Field(example=False)
    ip_address: str = Field(example="192.168.1.1")
    ip_country: str = Field(example="us")
    location_mismatch: bool = Field(example=False)
    ip_risk_score: float = Field(example=0.5)
    kyc_tier: int = Field(example=1)
    account_age_days: int = Field(example=295)
    device_trust_score: int = Field(example=295)
    chargeback_history_count: int = Field(example=0)

@app.post("/predict")
def predict_fraud(data: TransactionData):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    y_proba = rf_model.predict_proba(X)[0][1]
    return {"fraud_probability": float(y_proba)}
