import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_data(file_path, index_col=0):
    """Load CSV data and set index column."""
    return pd.read_csv(file_path, index_col=index_col)

def pivot_approach(df, aggfunc='mean'):
    """Pivot the dataframe and flatten column names."""
    values = df.columns.drop(['POSITION'])
    result = pd.pivot_table(df, values=values, index=df.index, columns='POSITION', aggfunc=aggfunc, fill_value=np.nan)
    result.columns = [f'{stat}_{pos}' for stat, pos in result.columns]
    return result

def prepare_player_data(df, prefix):
    """Prepare and pivot player statistics."""
    df = df.drop(["LEAGUE", "TEAM_NAME", "PLAYER_NAME"], axis=1, errors='ignore')
    pivoted = pivot_approach(df)
    pivoted.columns = f'{prefix}_' + pivoted.columns
    return pivoted

def prepare_team_data(df, prefix):
    """Prepare team statistics."""
    df = df.iloc[:, 2:]
    df.columns = f'{prefix}_' + df.columns
    return df

def handle_nan_values(df, method='mean'):
    """Handle NaN values using specified method."""
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'mice':
        imputer = IterativeImputer(random_state=0)
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', 'knn', or 'mice'.")
    
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

