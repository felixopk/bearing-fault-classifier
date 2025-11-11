import pandas as pd

def load_features(file_path="data/processed/features.csv"):
    """Load processed features CSV"""
    df = pd.read_csv(file_path)
    feature_cols = [c for c in df.columns if c not in ['label', 'label_original', 'fault']]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y, feature_cols, df['label'].unique()
