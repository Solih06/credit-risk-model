import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

class XenteFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        # Parse timestamp string to datetime object
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Datetime component extraction
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsNightTransaction'] = df['TransactionHour'].between(23, 5).astype(int)
        
        # Account aggregation primitives
        df['Amount'] = df['Amount'].astype(float)
        return df

class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.woe_maps = {}
        
    def fit(self, X, y):
        df = X.copy()
        df['target'] = y
        for col in self.cols:
            total_pos = df['target'].sum()
            total_neg = len(df) - total_pos
            
            # Avoid divide-by-zero adjustments using a Laplace-style smoothing regularizer
            stats = df.groupby(col)['target'].agg(['sum', 'count'])
            stats['pos_dist'] = (stats['sum'] + 0.5) / (total_pos + 1.0)
            stats['neg_dist'] = ((stats['count'] - stats['sum']) + 0.5) / (total_neg + 1.0)
            
            self.woe_maps[col] = np.log(stats['neg_dist'] / stats['pos_dist']).to_dict()
        return self
        
    def transform(self, X):
        df = X.copy()
        for col in self.cols:
            default_val = 0.0
            df[col] = df[col].map(self.woe_maps[col]).fillna(default_val)
        return df

def generate_proxy_target_variable(df):
    """
    Computes rolling RFM parameters and flags the highest risk cluster.
    """
    data = df.copy()
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    snapshot_date = data['TransactionStartTime'].max()
    
    # Calculate RFM metrics per Customer
    rfm = data.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    # Normalize continuous fields using Robust Scaler to neutralize extreme outliers
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(rfm)
    
    # Run deterministic 3-Cluster K-Means segmenting
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Isolate the high-risk proxy segment (Characterized by lowest frequency/engagement patterns)
    risk_cluster = rfm.groupby('Cluster')['Frequency'].mean().idxmin()
    rfm['is_high_risk'] = (rfm['Cluster'] == risk_cluster).astype(int)
    
    return data.merge(rfm[['is_high_risk']], on='CustomerId', how='left')