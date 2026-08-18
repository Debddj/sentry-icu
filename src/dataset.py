import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

KEY_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
KEY_LABS = ['Lactate', 'WBC', 'Creatinine', 'BUN', 'Glucose', 'Platelets', 'Hgb']

def extract_patient_features(psv_path):
    """
    Extracts summary statistics (mean, min, max, std, last) across an ICU stay for one patient.
    """
    try:
        df = pd.read_csv(psv_path, sep='|')
        if len(df) == 0:
            return None
        
        label = int(df['SepsisLabel'].max())
        features = {}
        
        for col in KEY_VITALS + KEY_LABS:
            series = df[col].ffill().bfill()
            if series.isna().all():
                features[f'{col}_mean'] = 0.0
                features[f'{col}_min'] = 0.0
                features[f'{col}_max'] = 0.0
                features[f'{col}_std'] = 0.0
                features[f'{col}_last'] = 0.0
            else:
                features[f'{col}_mean'] = float(series.mean())
                features[f'{col}_min'] = float(series.min())
                features[f'{col}_max'] = float(series.max())
                features[f'{col}_std'] = float(series.std()) if len(series) > 1 else 0.0
                features[f'{col}_last'] = float(series.iloc[-1])
                
        features['Age'] = float(df['Age'].iloc[0]) if not pd.isna(df['Age'].iloc[0]) else 60.0
        features['Gender'] = float(df['Gender'].iloc[0]) if not pd.isna(df['Gender'].iloc[0]) else 0.0
        features['Max_ICULOS'] = float(df['ICULOS'].max())
        features['Label'] = label
        return features
    except Exception:
        return None

def build_cohort_dataframe(folder_path, max_patients=2500):
    files = glob.glob(os.path.join(folder_path, '*.psv'))[:max_patients]
    rows = [extract_patient_features(f) for f in tqdm(files, desc="Building Cohort")]
    df = pd.DataFrame([r for r in rows if r is not None]).fillna(0)
    return df

class ICUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(cohort_df, batch_size=64, test_size=0.2, random_state=42):
    feature_cols = [c for c in cohort_df.columns if c != 'Label']
    X_raw = cohort_df[feature_cols].values
    y_raw = cohort_df['Label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_loader = DataLoader(ICUDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ICUDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = float(num_neg / max(num_pos, 1))

    return train_loader, val_loader, scaler, feature_cols, pos_weight
