import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
'''
file_path = "../data/raw/EEG_Eye_State_Classification.csv"  # replace with your file name
df = pd.read_csv(file_path)
print(df.head())
'''

def load_data():
    project_path = Path(__file__).resolve().parent.parent # file is current then goes to parent and its parent, resolve is absolute
    raw_data_path = project_path/'data'/'raw'
    data_dict = {}
    for csv_files in raw_data_path.glob('*.csv'): #glob=pattern
        df = pd.read_csv(csv_files)
        key_name = csv_files.stem
        data_dict[key_name] = df
    return data_dict

# Test w/o executing when importing
if __name__ == "__main__":
    data = load_data()
    for name, df in data.items():
        print(f"Dataset: {name}, shape: {df.shape}")
        print(df.head())

def prep_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])
    return pipeline

def window_eeg(df, target, window_size=64):
    feature_cols = df.columns.drop(target)
    X = df[feature_cols].values
    y = df[target].values

    n_samples, n_features = X.shape
    n_windows = n_samples // window_size
    n_rows_trunc = n_windows * window_size

    # truncate to make divisible by window_size
    X = X[:n_rows_trunc]
    y = y[:n_rows_trunc]

    # reshape
    X_reshaped = X.reshape(n_windows, window_size, n_features)
    X_windows = X_reshaped.reshape(n_windows, window_size * n_features)

    y_reshaped = y.reshape(n_windows, window_size)
    # take mode across each window
    y_windows = np.array([np.bincount(row).argmax() for row in y_reshaped])
    print(f"X reshaped shape is {X_reshaped.shape}")
    return X_windows, y_windows




