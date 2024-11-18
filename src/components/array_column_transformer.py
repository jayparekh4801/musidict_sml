import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ArrayColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        # Use StandardScaler by default if no scaler is provided
        self.scaler = scaler or StandardScaler()

    def fit(self, X, y=None):
        # No fitting necessary for transformation
        return self

    def transform(self, X):
        df_data = []
        for _, val in X.iterrows():
            cols_data = []
            for col in X.columns:
                transformed_row = self.scaler.fit_transform(val[col])
                cols_data.append(transformed_row)
            df_data.append(cols_data)
        
        df_data = np.array(df_data, dtype=object)
        return pd.DataFrame(df_data, columns=X.columns)