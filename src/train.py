import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from data_processing import generate_proxy_target_variable, build_production_pipeline

def run_model_training_lifecycle(raw_data_path):
    df_raw = pd.read_csv(raw_data_path)
    
    # 1. Synthesize proxy target variable via clustering
    df_processed = generate_proxy_target_variable(df_raw)
    
    # Columns required by the pipeline's feature extractor and encoders
    features = [
        'ProviderId', 'ProductId', 'ProductCategory', 'Amount',
        'CustomerId', 'TransactionStartTime' 
    ]
    
    X = df_processed[features]
    y = df_processed['is_high_risk']
    
    # 2. Clean Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Instantiate the production pipeline object
    categorical_fields = ['ProviderId', 'ProductId', 'ProductCategory']
    preprocessing_pipeline = build_production_pipeline(categorical_cols=categorical_fields)
    
    # 4. FIX: Explicitly call .fit() first to store the fitted states across all transformers,
    # then call .transform() sequentially. This guarantees the pipeline objects are marked as fitted.
    preprocessing_pipeline.fit(X_train, y_train)
    
    X_train_processed = preprocessing_pipeline.transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    mlflow.set_experiment("Bati_Bank_Credit_Risk_Architecture")
    
    # --- MODEL 1: LOGISTIC REGRESSION ---
    with mlflow.start_run(run_name="Logistic_Regression_Optimized"):
        lr_base = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        param_grid_lr = {'C': [0.1, 1.0]}
        grid_lr = GridSearchCV(lr_base, param_grid_lr, cv=3, scoring='f1', n_jobs=-1)
        grid_lr.fit(X_train_processed, y_train)
        best_lr = grid_lr.best_estimator_
        
        preds_lr = best_lr.predict(X_test_processed)
        mlflow.log_metric("f1_score", f1_score(y_test, preds_lr))
        mlflow.sklearn.log_model(best_lr, "logistic_regression_model")
        
    # --- MODEL 2: RANDOM FOREST ---
    with mlflow.start_run(run_name="Random_Forest_Optimized"):
        rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        grid_rf = GridSearchCV(rf_base, param_grid_rf, cv=3, scoring='f1', n_jobs=-1)
        grid_rf.fit(X_train_processed, y_train)
        best_rf = grid_rf.best_estimator_
        
        preds_rf = best_rf.predict(X_test_processed)
        mlflow.log_metric("f1_score", f1_score(y_test, preds_rf))
        
        os.makedirs("models", exist_ok=True)
        with open("models/fallback_model.pkl", "wb") as f:
            pickle.dump(best_rf, f)
            
        mlflow.sklearn.log_model(best_rf, "credit_risk_rf_model", registered_model_name="BatiRiskCoreRF")
        print("Both structural models evaluated, optimized, and logged via the Pipeline successfully.")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists("data/raw/training.csv"):
        mock_df = pd.DataFrame({
            'CustomerId': [f'CustomerId_{i}' for i in range(20)],
            'TransactionId': [f'Tx_{i}' for i in range(20)],
            'TransactionStartTime': ['2026-06-01T10:00:00Z'] * 20,
            'ProviderId': ['Prov_1'] * 20, 'ProductId': ['Prod_1'] * 20, 'ProductCategory': ['Utility'] * 20,
            'Amount': [10000, 12000, 15000, 200, 150] * 4
        })
        mock_df.to_csv("data/raw/training.csv", index=False)
        
    run_model_training_lifecycle("data/raw/training.csv")
