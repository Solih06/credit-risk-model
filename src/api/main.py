from fastapi import FastAPI
import uvicorn
import pandas as pd
import joblib
import mlflow.sklearn
from src.data_processing import build_production_pipeline
from pydantic_models import CreditScoringRequest, CreditScoringResponse

app = FastAPI(title="Bati Bank Automated Credit Scoring Engine API", version="1.0.0")

# 1. Define the production pipeline using the shared processing logic
# This ensures training and serving are mathematically consistent
#
categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory']
pipeline = build_production_pipeline(categorical_cols=categorical_cols)

# 2. Load the trained model
try:
    # Attempt to load from MLflow registry
    model_uri = "models:/BatiRiskCoreLR/1"
    scoring_model = mlflow.sklearn.load_model(model_uri)
except Exception:
    # Fallback to local serialization
    scoring_model = joblib.load("models/fallback_model.pkl")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": scoring_model is not None}

@app.post("/predict", response_model=CreditScoringResponse)
def evaluate_applicant_credit_risk(payload: CreditScoringRequest):
    # 3. Convert Pydantic request to DataFrame
    df = pd.DataFrame([payload.dict()])
    
    # 4. Transform raw data using the production pipeline
    # This automatically applies Imputation, Scaling, and WoE Encoding
    # matching the exact process used in train.py
    processed_data = pipeline.transform(df)
    
    # 5. Execute prediction
    prob = float(scoring_model.predict_proba(processed_data)[0][1]) if scoring_model else 0.85
    decision = "DENY" if prob > 0.50 else "APPROVE"
    
    return {
        "risk_probability": round(prob, 4),
        "credit_decision": decision
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
