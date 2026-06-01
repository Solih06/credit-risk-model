"""
Model Training Module for Credit Risk Probability Underwriting.
Implements cost-sensitive classification architectures addressing class imbalances.
"""

import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_credit_model(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Splits the transactional data, configures balanced class weights to 
    counteract the 0.20% imbalance constraint, and trains a baseline model.
    """
    try:
        if target_col not in df.columns:
            raise ValueError(f"Target flag '{target_col}' not found in active frame arrays.")
            
        X = df[feature_cols]
        y = df[target_col]
        
        # Train-Test Segmentation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Data stratified. Training size: {X_train.shape}, Test size: {X_test.shape}")
        
        # Configure class_weight='balanced' to handle the severe 0.20% imbalance
        logger.info("Initializing Cost-Sensitive Logistic Regression Framework...")
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        
        model.fit(X_train, y_train)
        logger.info("Model optimization routine finalized successfully.")
        
        # Evaluate model performance using metrics resilient to class imbalance
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        logger.info("\n=== MODEL PERFORMANCE METRICS ===")
        logger.info(f"\n{classification_report(y_test, predictions)}")
        logger.info(f"Receiver Operating Characteristic (ROC-AUC) Score: {roc_auc_score(y_test, probabilities):.4f}")
        
        return model
    except Exception as e:
        logger.error(f"Fatal error encountered during model training sequence: {str(e)}")
        raise e 
