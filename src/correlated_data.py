import numpy as np
import pandas as pd


def remove_correlated_features(X: pd.DataFrame, threshold=0.8):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    return X.drop(to_drop, axis=1)