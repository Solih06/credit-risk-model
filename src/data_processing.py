import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

class XenteFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.customer_aggregates_ = None
        
    def fit(self, X, y=None):
        # Calculate per-customer historical aggregates safely during fit
        self.customer_aggregates_ = X.groupby('CustomerId')['Amount'].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_amount', 'std')
        ]).reset_index()
        self.is_fitted_ = True # Tells sklearn this transformer step is fitted
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Temporal feature extraction
        X_out['TransactionStartTime'] = pd.to_datetime(X_out['TransactionStartTime'])
        X_out['TransactionHour'] = X_out['TransactionStartTime'].dt.hour
        X_out['DayOfWeek'] = X_out['TransactionStartTime'].dt.dayofweek
        X_out['IsWeekend'] = X_out['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        X_out['IsNightTransaction'] = X_out['TransactionHour'].apply(lambda x: 1 if x >= 23 or x <= 5 else 0)
        
        # 2. Dynamically merge calculated aggregate features back
        X_out = pd.merge(X_out, self.customer_aggregates_, on='CustomerId', how='left')
        
        # Drop string context paths so downstream steps receive pure numeric vectors
        X_out = X_out.drop(columns=['CustomerId', 'TransactionStartTime'], errors='ignore')
        return X_out

class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.woe_maps_ = {}
        
    def fit(self, X, y=None):
        self.is_fitted_ = True # Prevents NotFittedError inside parent Pipeline
        if y is None:
            return self
        df = X.copy()
        df['target'] = y
        
        for col in self.cols:
            if col not in df.columns:
                continue
            event_total = df['target'].sum()
            non_event_total = len(df) - event_total
            
            stats = df.groupby(col)['target'].agg(['sum', 'count'])
            stats['non_events'] = stats['count'] - stats['sum']
            
            # Laplace smoothing (0.5) to avoid division by zero errors
            stats['woe'] = np.log(((stats['non_events'] + 0.5) / (non_event_total + 0.5)) / 
                                  ((stats['sum'] + 0.5) / (event_total + 0.5)))
            
            self.woe_maps_[col] = stats['woe'].to_dict()
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            if col in self.woe_maps_ and col in X_out.columns:
                X_out[col] = X_out[col].map(self.woe_maps_[col]).fillna(0)
        return X_out

class DenseTransformer(BaseEstimator, TransformerMixin):
    """Converts dataframes to standard numpy float arrays safely for sklearn steps."""
    def fit(self, X, y=None):
        self.is_fitted_ = True # Crucial: Tells the Pipeline container its final step is fitted
        return self
    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

def generate_proxy_target_variable(df_raw, random_state=42):
    """
    Executes Unsupervised K-Means clustering across RFM profiles
    to synthesize the binary credit default target vector layer 'is_high_risk'.
    """
    df = df_raw.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    risk_cluster = rfm.groupby('cluster')['Monetary'].mean().idxmin()
    rfm['is_high_risk'] = rfm['cluster'].apply(lambda x: 1 if x == risk_cluster else 0)
    
    return df.merge(rfm[['is_high_risk']], on='CustomerId', how='left')

def build_production_pipeline(categorical_cols):
    """
    Links Feature Extraction, Categorical WoE Encoding, Explicit Missing-Value Imputation, 
    and Full Feature Scaling into one reusable pipeline object.
    """
    return Pipeline([
        ('extractor', XenteFeatureExtractor()),
        ('encoder', WeightOfEvidenceEncoder(cols=categorical_cols)),
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', RobustScaler()),                   
        ('converter', DenseTransformer())
    ])