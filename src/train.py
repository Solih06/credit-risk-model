import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from data_processing import XenteFeatureExtractor, WeightOfEvidenceEncoder, generate_proxy_target_variable

def run_model_training_lifecycle(raw_data_path):
    # Load raw transaction entries safely
    df_raw = pd.read_csv(raw_data_path)
    
    # Construct target variable layer via K-Means
    df_processed = generate_proxy_target_variable(df_raw)
    
    # Extract structural features
    extractor = XenteFeatureExtractor()
    df_features = extractor.transform(df_processed)
    
    features = ['ProviderId', 'ProductId', 'ProductCategory', 'TransactionHour', 'DayOfWeek', 'IsWeekend', 'IsNightTransaction', 'Amount']
    X = df_features[features]
    y = df_processed['is_high_risk']
    
    # Set up cross-validation splits to handle extreme class asymmetry
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    categorical_fields = ['ProviderId', 'ProductId', 'ProductCategory']
    
    # Start MLflow experiment tracking
    mlflow.set_experiment("Bati_Bank_Credit_Risk_Architecture")
    
    with mlflow.start_run(run_name="Production_Logistic_Regression"):
        woe_encoder = WeightOfEvidenceEncoder(cols=categorical_fields)
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            X_train_enc = woe_encoder.fit_transform(X_train, y_train)
            X_val_enc = woe_encoder.transform(X_val)
            
            # Use Cost-Sensitive weights to address class imbalance natively
            model = LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
            model.fit(X_train_enc, y_train)
            
            preds = model.predict(X_val_enc)
            probs = model.predict_proba(X_val_enc)[:, 1]
            
            # Log metrics
            mlflow.log_metric("precision", precision_score(y_val, preds))
            mlflow.log_metric("recall", recall_score(y_val, preds))
            mlflow.log_metric("f1_score", f1_score(y_val, preds))
            mlflow.log_metric("roc_auc", roc_auc_score(y_val, probs))
            
        # Register the production model artifact
        mlflow.sklearn.log_model(model, "credit_risk_lr_model", registered_model_name="BatiRiskCoreLR")
        print("Training successfully logged to MLflow Model Registry.")

if __name__ == "__main__":
    run_model_training_lifecycle("data/raw/training.csv")
