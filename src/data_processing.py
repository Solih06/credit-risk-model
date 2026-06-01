"""
Data Processing Module for Credit Risk Probability Model.
Handles robust scaling, feature engineering, and log transformations for skewed financial arrays.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_transaction_data(file_path: str) -> pd.DataFrame:
    """
    Safely loads raw transaction data from a CSV file path.
    """
    if not os.path.exists(file_path):
        logger.error(f"Target data file not found at path: {file_path}")
        raise FileNotFoundError(f"Missing file: {file_path}")
    
    try:
        logger.info(f"Loading raw transactional dataset from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to parse CSV transaction stream: {str(e)}")
        raise e

def transform_and_scale_features(df: pd.DataFrame, continuous_cols: list) -> pd.DataFrame:
    """
    Applies symmetric log transformations to fix monetary skewness and handles 
    negative numbers (reversals) gracefully, then scales data using RobustScaler.
    """
    df_processed = df.copy()
    
    try:
        for col in continuous_cols:
            if col not in df_processed.columns:
                raise KeyError(f"Target continuous column '{col}' missing from DataFrame.")
            
            # Apply Symmetric Log Transform to handle negative transaction values/reversals safely without making NaNs
            logger.info(f"Applying robust symmetric log transformation to column: {col}")
            df_processed[f'{col}_log'] = np.sign(df_processed[col]) * np.log1p(np.abs(df_processed[col]))
            
            # Defensive check to guarantee no NaNs leaked through
            nan_count = df_processed[f'{col}_log'].isna().sum()
            if nan_count > 0:
                logger.warning(f"Detected {nan_count} NaNs in {col}_log. Filling with baseline 0.0.")
                df_processed[f'{col}_log'] = df_processed[f'{col}_log'].fillna(0.0)
            
        # Instantiate RobustScaler to protect model from extreme outlier pulling forces
        scaler = RobustScaler()
        log_cols = [f'{col}_log' for col in continuous_cols]
        
        logger.info("Executing Robust Scaler transformation across engineered arrays...")
        df_processed[log_cols] = scaler.fit_transform(df_processed[log_cols])
        
        return df_processed
    except Exception as e:
        logger.error(f"Error encountered during feature scaling routine: {str(e)}")
        raise e