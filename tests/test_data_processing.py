import pytest
import pandas as pd
import numpy as np
from src.data_processing import XenteFeatureExtractor, generate_proxy_target_variable

def test_feature_extractor_adds_correct_columns():
    # Setup simple mock data mimicking structural input data records
    mock_data = pd.DataFrame({
        'CustomerId': ['Cust_1', 'Cust_1', 'Cust_2'],
        'TransactionStartTime': ['2026-06-01T10:00:00Z', '2026-06-01T23:30:00Z', '2026-06-01T14:15:00Z'],
        'Amount': [5000, 12000, 1500]
    })
    
    extractor = XenteFeatureExtractor()
    processed_df = extractor.fit_transform(mock_data)
    
    # Assert that all required Task 3 features are created
    assert 'TransactionHour' in processed_df.columns
    assert 'total_amount' in processed_df.columns
    assert 'avg_amount' in processed_df.columns
    assert 'transaction_count' in processed_df.columns

def test_proxy_target_generation_bounds():
    # Setup robust sample space including the missing 'TransactionId' column to satisfy grouping operations
    mock_data = pd.DataFrame({
        'CustomerId': [f'Cust_{i}' for i in range(15)],
        'TransactionId': [f'Tx_{i}' for i in range(15)],
        'TransactionStartTime': ['2026-06-01T10:00:00Z'] * 15,
        'Amount': [10000, 12000, 15000, 11000, 13000,  # High Value Cluster
                   5000, 6000, 5500, 4800, 5200,       # Mid Value Cluster
                   500, 600, 400, 300, 700]            # Low Value (High Risk Proxy Cluster)
    })
    
    output_df = generate_proxy_target_variable(mock_data)
    
    assert 'is_high_risk' in output_df.columns
    assert output_df['is_high_risk'].nunique() <= 2
    assert set(output_df['is_high_risk'].unique()).issubset({0, 1})
