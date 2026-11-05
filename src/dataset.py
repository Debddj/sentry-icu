"""
SentryICU Dataset & Preprocessing Pipeline
Supports:
1. Canonical static 73-D summary extraction (Module 1 / 2)
2. Resilient schema normalization & PatientID alias resolution (BUG-M3-00 fix)
3. Sliding window multimodal extraction for 1D-CNN (Module 3)
4. Physiological normal median imputation (BUG-M3-07 fix)
5. Stratified patient-level splitting without data leakage (BUG-M3-01 fix)
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Canonical Clinical Variable Sets
KEY_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
KEY_LABS = ['Lactate', 'WBC', 'Creatinine', 'BUN', 'Glucose', 'Platelets', 'Hgb']
FEATURE_COLUMNS_14 = KEY_VITALS + KEY_LABS

# Physiological Normal Medians (prevents extreme outlier z-scores from zero-imputation)
PHYSIOLOGICAL_NORMALS = {
    'HR': 80.0,
    'O2Sat': 98.0,
    'Temp': 37.0,
    'SBP': 120.0,
    'MAP': 80.0,
    'DBP': 80.0,
    'Resp': 16.0,
    'Lactate': 1.0,
    'WBC': 7.0,
    'Creatinine': 1.0,
    'BUN': 15.0,
    'Glucose': 100.0,
    'Platelets': 250.0,
    'Hgb': 14.0,
}


def normalize_icu_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column aliases, reconstructs patient IDs if missing,
    and validates time-series schema for Module 3 pipeline.
    
    Fixes BUG-M3-00: Real PhysioNet/Kaggle Dataset.csv uses 'Patient_ID',
    while module code previously checked strictly for 'PatientID'.
    
    Args:
        df: Input DataFrame with arbitrary ICU column casing/naming.
        
    Returns:
        pd.DataFrame: Normalized DataFrame guaranteed to have 'PatientID',
                      'Hour', and standardized feature names.
    """
    df = df.copy()

    # 1. Strip unnamed index columns
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c) or c == '']
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    # 2. Patient ID Aliasing
    patient_id_aliases = ['Patient_ID', 'patient_id', 'pid', 'SubjectID', 'subject_id', 'PATIENT_ID']
    for alias in patient_id_aliases:
        if alias in df.columns and 'PatientID' not in df.columns:
            df.rename(columns={alias: 'PatientID'}, inplace=True)
            break

    # 3. Fallback: Reconstruct patient boundaries if no patient ID column exists
    if 'PatientID' not in df.columns:
        if 'ICULOS' in df.columns:
            # ICU length of stay resets to 1 (or drops) at the start of each new patient stay
            new_patient_flags = (df['ICULOS'] == 1) | (df['ICULOS'].diff() < 0)
            if len(new_patient_flags) > 0:
                new_patient_flags.iloc[0] = True
            patient_ints = new_patient_flags.cumsum()
            df['PatientID'] = patient_ints.apply(lambda x: f"p{x:06d}")
        elif 'Hour' in df.columns:
            new_patient_flags = (df['Hour'] == 0) | (df['Hour'].diff() < 0)
            if len(new_patient_flags) > 0:
                new_patient_flags.iloc[0] = True
            patient_ints = new_patient_flags.cumsum()
            df['PatientID'] = patient_ints.apply(lambda x: f"p{x:06d}")
        else:
            raise ValueError("Cannot infer PatientID: dataset lacks PatientID, ICULOS, and Hour columns.")

    # 4. SepsisLabel Aliasing
    if 'SepsisLabel' not in df.columns:
        label_aliases = ['Label', 'label', 'sepsis_label', 'sepsis', 'SEPSISLABEL']
        for la in label_aliases:
            if la in df.columns:
                df.rename(columns={la: 'SepsisLabel'}, inplace=True)
                break

    # 5. Hour Aliasing
    if 'Hour' not in df.columns:
        if 'ICULOS' in df.columns:
            df['Hour'] = df.groupby('PatientID').cumcount()
        else:
            df['Hour'] = df.groupby('PatientID').cumcount()

    # 6. Type casting
    df['PatientID'] = df['PatientID'].astype(str)
    return df


def compute_73d_summary(window_df: pd.DataFrame) -> np.ndarray:
    """
    Computes canonical 73-D static summary vector matching Module 1 / 2 schema:
    - 14 continuous variables (7 vitals + 7 labs) x 5 stats = 70 features
      (mean, min, max, std, last)
    - 3 demographic/admin features: Age, Gender, Max_ICULOS = 73 total
    """
    features = []

    for col in KEY_VITALS + KEY_LABS:
        default_val = PHYSIOLOGICAL_NORMALS.get(col, 0.0)
        if col in window_df.columns:
            series = window_df[col].ffill().bfill().fillna(default_val)
        else:
            series = pd.Series(default_val, index=window_df.index)

        mean_val = float(series.mean())
        min_val = float(series.min())
        max_val = float(series.max())
        std_val = float(series.std()) if len(series) > 1 else 0.0
        last_val = float(series.iloc[-1])
        features.extend([mean_val, min_val, max_val, std_val, last_val])

    age = float(window_df['Age'].iloc[0]) if ('Age' in window_df.columns and not pd.isna(window_df['Age'].iloc[0])) else 60.0
    gender = float(window_df['Gender'].iloc[0]) if ('Gender' in window_df.columns and not pd.isna(window_df['Gender'].iloc[0])) else 0.0
    max_iculos = float(window_df['ICULOS'].max()) if ('ICULOS' in window_df.columns and not pd.isna(window_df['ICULOS'].max())) else 12.0
    features.extend([age, gender, max_iculos])

    return np.array(features, dtype=np.float32)


def extract_sliding_windows(df: pd.DataFrame, window_size=12, step=1, stride=None, target_pids=None):
    """
    Extracts sliding vital sign windows and corresponding 73-D static summaries across patient stays.
    
    Args:
        df: Normalized or raw ICU DataFrame.
        window_size: Length of temporal lookback window in hours (default: 12).
        step: Stride between successive sliding windows (default: 1).
        stride: Stride alias for step.
        target_pids: Optional list/set of PatientIDs to restrict extraction to.
        
    Returns:
        tuple: (X_73d, X_vitals, y)
            - X_73d: (N, 73) np.ndarray float32 static feature summaries
            - X_vitals: (N, 7, 12) np.ndarray float32 vital sign waveform tensors
            - y: (N,) np.ndarray float32 binary labels at window termination
    """
    if stride is not None:
        step = stride

    df = normalize_icu_dataframe(df)

    if target_pids is not None:
        target_set = set(map(str, target_pids))
        pids = [p for p in df['PatientID'].unique() if p in target_set]
    else:
        pids = list(df['PatientID'].unique())

    X_73d_list, X_ts_list, y_list = [], [], []

    for pid in pids:
        sort_col = 'Hour' if 'Hour' in df.columns else 'ICULOS'
        pdf = df[df['PatientID'] == pid].sort_values(sort_col).reset_index(drop=True)
        num_rows = len(pdf)
        if num_rows < window_size:
            continue

        # Ensure all vital and lab columns exist, with physiological imputation
        for col in KEY_VITALS + KEY_LABS:
            if col not in pdf.columns:
                pdf[col] = PHYSIOLOGICAL_NORMALS.get(col, 0.0)
            else:
                pdf[col] = pdf[col].ffill().bfill().fillna(PHYSIOLOGICAL_NORMALS.get(col, 0.0))

        for start_idx in range(0, num_rows - window_size + 1, step):
            window = pdf.iloc[start_idx : start_idx + window_size]

            # 73-D summary vector
            f73 = compute_73d_summary(window)

            # (7, 12) vital sign waveform matrix
            vitals_mat = window[KEY_VITALS].values.T.astype(np.float32)

            # Sepsis label at window termination (avoids target leakage)
            label = int(window['SepsisLabel'].iloc[-1]) if 'SepsisLabel' in window.columns else 0

            X_73d_list.append(f73)
            X_ts_list.append(vitals_mat)
            y_list.append(label)

    if len(X_73d_list) > 0:
        X_73d = np.array(X_73d_list, dtype=np.float32)
        X_vitals = np.array(X_ts_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
    else:
        X_73d = np.empty((0, 73), dtype=np.float32)
        X_vitals = np.empty((0, len(KEY_VITALS), window_size), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)

    return X_73d, X_vitals, y


class ICUSlidingWindowDataset(Dataset):
    """
    PyTorch Dataset yielding multimodal pairs:
    - x_73d: (73,) static tabular summary features
    - x_vitals: (7, 12) continuous vital sign temporal waveform
    - y: (1,) binary sepsis target label
    """
    def __init__(self, X_73d, X_vitals, y):
        self.X_73d = torch.as_tensor(X_73d, dtype=torch.float32)
        self.X_vitals = torch.as_tensor(X_vitals, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            self.y = y_tensor.unsqueeze(1)
        else:
            self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_73d[idx], self.X_vitals[idx], self.y[idx]

    def get_pos_weight(self):
        """Computes pos_weight directly from dataset labels to prevent global variable leakage."""
        num_neg = (self.y == 0).sum().item()
        num_pos = (self.y == 1).sum().item()
        return float(num_neg / max(num_pos, 1))


# Interface aliases
MultimodalICUDataset = ICUSlidingWindowDataset
SepsisDataset = ICUSlidingWindowDataset


def get_sliding_window_dataloaders(df: pd.DataFrame, batch_size=32, window_size=12,
                                   step=1, test_size=0.2, random_state=42):
    """
    Builds train and validation DataLoaders with stratified patient-level splitting
    and leak-free feature standardization.
    
    Args:
        df: Raw or normalized ICU DataFrame.
        batch_size: DataLoader mini-batch size (default: 32).
        window_size: Lookback window length in hours (default: 12).
        step: Stride for sliding window extraction (default: 1).
        test_size: Proportion of patients reserved for validation (default: 0.2).
        random_state: Reproducibility seed (default: 42).
        
    Returns:
        tuple: (train_loader, val_loader, scaler_73d, (ts_mean, ts_std), pos_weight)
    """
    df = normalize_icu_dataframe(df)

    # Patient-level stratified split (BUG-M3-01 fix: ensures proportional sepsis rate)
    if 'SepsisLabel' in df.columns:
        patient_labels = df.groupby('PatientID')['SepsisLabel'].max()
    else:
        patient_labels = pd.Series(0, index=df['PatientID'].unique())

    patient_ids = patient_labels.index.values
    labels_arr = patient_labels.values

    # Check if stratified split is possible (at least 2 classes and >= 2 samples per class)
    unique_classes, counts = np.unique(labels_arr, return_counts=True)
    if len(unique_classes) > 1 and np.min(counts) >= 2:
        train_pids, val_pids = train_test_split(
            patient_ids, test_size=test_size, random_state=random_state, stratify=labels_arr
        )
    else:
        train_pids, val_pids = train_test_split(
            patient_ids, test_size=test_size, random_state=random_state
        )

    # Extract windows separately for train and val splits
    X_train_73d_raw, X_train_ts_raw, y_train = extract_sliding_windows(
        df, window_size=window_size, step=step, target_pids=train_pids
    )
    X_val_73d_raw, X_val_ts_raw, y_val = extract_sliding_windows(
        df, window_size=window_size, step=step, target_pids=val_pids
    )

    # Fit StandardScaler strictly on training 73-D features
    scaler_73d = StandardScaler()
    if len(X_train_73d_raw) > 0:
        X_train_73d = scaler_73d.fit_transform(X_train_73d_raw)
        X_val_73d = scaler_73d.transform(X_val_73d_raw) if len(X_val_73d_raw) > 0 else X_val_73d_raw
    else:
        X_train_73d = X_train_73d_raw
        X_val_73d = X_val_73d_raw

    # Channel-wise time-series standardization fitted strictly on training cohort
    if len(X_train_ts_raw) > 0:
        ts_mean = X_train_ts_raw.mean(axis=(0, 2), keepdims=True)
        ts_std = X_train_ts_raw.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train_ts = (X_train_ts_raw - ts_mean) / ts_std
        X_val_ts = (X_val_ts_raw - ts_mean) / ts_std if len(X_val_ts_raw) > 0 else X_val_ts_raw
    else:
        ts_mean = np.zeros((1, len(KEY_VITALS), 1), dtype=np.float32)
        ts_std = np.ones((1, len(KEY_VITALS), 1), dtype=np.float32)
        X_train_ts = X_train_ts_raw
        X_val_ts = X_val_ts_raw

    train_dataset = ICUSlidingWindowDataset(X_train_73d, X_train_ts, y_train)
    val_dataset = ICUSlidingWindowDataset(X_val_73d, X_val_ts, y_val)

    # Avoid dropping batches if dataset is small
    drop_last_train = len(train_dataset) >= batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pos_weight = train_dataset.get_pos_weight()

    return train_loader, val_loader, scaler_73d, (ts_mean, ts_std), pos_weight


# Backwards compatibility alias
get_multimodal_dataloaders = get_sliding_window_dataloaders


# =========================================================================
# Canonical Module 1 / 2 Single-Stay Extraction Pipeline (Preserved)
# =========================================================================

def extract_patient_features(psv_path):
    """
    Extracts summary statistics (mean, min, max, std, last) across an ICU stay for one patient.
    """
    try:
        df = pd.read_csv(psv_path, sep='|')
        if len(df) == 0:
            return None
        
        label = int(df['SepsisLabel'].max()) if 'SepsisLabel' in df.columns else 0
        features = {}
        
        for col in KEY_VITALS + KEY_LABS:
            series = df[col].ffill().bfill() if col in df.columns else pd.Series(dtype=float)
            if series.isna().all() or len(series) == 0:
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
                
        features['Age'] = float(df['Age'].iloc[0]) if ('Age' in df.columns and not pd.isna(df['Age'].iloc[0])) else 60.0
        features['Gender'] = float(df['Gender'].iloc[0]) if ('Gender' in df.columns and not pd.isna(df['Gender'].iloc[0])) else 0.0
        features['Max_ICULOS'] = float(df['ICULOS'].max()) if ('ICULOS' in df.columns and not pd.isna(df['ICULOS'].max())) else 1.0
        features['Label'] = label
        return features
    except (OSError, ValueError, KeyError):
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


# =========================================================================
# Module 4: Longitudinal Temporal Sequence Pipeline & GRU-D Datasets
# =========================================================================

class ICUSequenceDataset(Dataset):
    """
    Longitudinal ICU Sequence Dataset over a 24-hour lookback horizon across 14 clinical features.
    
    Fixes BUG-M4-02: Enforces standard RIGHT-PADDING ((0, pad_len), (0, 0)) so that
    timesteps 0 .. seq_len-1 contain true observations and pad_len timesteps occupy seq_len .. 23.
    This ensures that:
    1. lengths - 1 indexes the most recent actual observation.
    2. arange(24) < lengths correctly attends to true observations and masks out padding zeros.
    
    Fixes BUG-M4-06 / Bug 6: Computes observation mask m_t and elapsed time delta delta_t.
    For variables missing at admission (t=0), initializes delta_t = 24.0 so decay factor
    gamma_x smoothly approaches 0.0 and reverts to population baseline.
    
    Fixes BUG-M4-11 / Bug 7: Imputes completely unobserved features with physiological medians
    or training set means before standardization to prevent extreme negative outlier spikes.
    """
    def __init__(
        self,
        df_or_tensors: pd.DataFrame | np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor | None = None,
        delta: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
        y: np.ndarray | torch.Tensor | None = None,
        X_raw: np.ndarray | torch.Tensor | None = None,
        feature_cols: list[str] = FEATURE_COLUMNS_14,
        id_col: str = 'PatientID',
        time_col: str = 'Hour',
        label_col: str = 'SepsisLabel',
        max_lookback: int = 24,
        feature_means: np.ndarray | None = None,
        feature_stds: np.ndarray | None = None
    ):
        super().__init__()
        self.feature_cols = feature_cols
        self.num_features = len(feature_cols)
        self.max_lookback = max_lookback
        self.id_col = id_col
        self.time_col = time_col
        self.label_col = label_col
        self.samples = []

        # Mode A: Initialized directly from pre-computed arrays or tensors
        if isinstance(df_or_tensors, (np.ndarray, torch.Tensor)):
            X_seq_tensor = torch.as_tensor(df_or_tensors, dtype=torch.float32)
            n_samples = X_seq_tensor.size(0)
            
            mask_tensor = (
                torch.as_tensor(mask, dtype=torch.float32)
                if mask is not None
                else torch.ones_like(X_seq_tensor)
            )
            delta_tensor = (
                torch.as_tensor(delta, dtype=torch.float32)
                if delta is not None
                else torch.zeros_like(X_seq_tensor)
            )
            lengths_tensor = (
                torch.as_tensor(lengths, dtype=torch.int64)
                if lengths is not None
                else torch.full((n_samples,), X_seq_tensor.size(1), dtype=torch.int64)
            )
            if y is not None:
                y_tensor = torch.as_tensor(y, dtype=torch.float32)
                if y_tensor.ndim == 1:
                    y_tensor = y_tensor.unsqueeze(1)
            else:
                y_tensor = torch.zeros((n_samples, 1), dtype=torch.float32)

            raw_tensor = (
                torch.as_tensor(X_raw, dtype=torch.float32)
                if X_raw is not None
                else X_seq_tensor.clone()
            )

            for i in range(n_samples):
                self.samples.append({
                    'X_seq': X_seq_tensor[i],
                    'X_raw': raw_tensor[i],
                    'mask': mask_tensor[i],
                    'm_t': mask_tensor[i],
                    'delta': delta_tensor[i],
                    'delta_t': delta_tensor[i],
                    'lengths': lengths_tensor[i],
                    'y': y_tensor[i]
                })

            if feature_means is not None:
                self.means = np.asarray(feature_means, dtype=np.float32)
            else:
                self.means = np.zeros(self.num_features, dtype=np.float32)
            if feature_stds is not None:
                self.stds = np.asarray(feature_stds, dtype=np.float32)
            else:
                self.stds = np.ones(self.num_features, dtype=np.float32)

        # Mode B: Initialized from a longitudinal ICU DataFrame
        elif isinstance(df_or_tensors, pd.DataFrame):
            df_norm = normalize_icu_dataframe(df_or_tensors)
            
            # Ensure all required 14 clinical features exist in df
            for col in self.feature_cols:
                if col not in df_norm.columns:
                    df_norm[col] = PHYSIOLOGICAL_NORMALS.get(col, 0.0)

            # Compute population statistics for normalization if not supplied
            if feature_means is None or feature_stds is None:
                raw_vals = df_norm[self.feature_cols].values
                self.means = np.nanmean(raw_vals, axis=0)
                self.stds = np.nanstd(raw_vals, axis=0)
                # Replace NaNs or zeros in statistics
                for idx, c in enumerate(self.feature_cols):
                    if np.isnan(self.means[idx]):
                        self.means[idx] = PHYSIOLOGICAL_NORMALS.get(c, 0.0)
                    if np.isnan(self.stds[idx]) or self.stds[idx] <= 1e-6:
                        self.stds[idx] = 1.0
            else:
                self.means = np.asarray(feature_means, dtype=np.float32)
                self.stds = np.asarray(feature_stds, dtype=np.float32)
                self.stds[self.stds <= 1e-6] = 1.0

            # Group by PatientID
            sort_col = 'Hour' if 'Hour' in df_norm.columns else 'ICULOS'
            grouped = dict(tuple(df_norm.groupby('PatientID')))
            patient_ids = list(grouped.keys())

            for pid in patient_ids:
                p_df = grouped[pid].sort_values(sort_col).reset_index(drop=True)
                raw_matrix = p_df[self.feature_cols].values.astype(np.float32)
                total_len = len(raw_matrix)
                if total_len == 0:
                    continue

                seq_len = min(total_len, self.max_lookback)
                # Select the most recent 24 hours of data
                raw_window = raw_matrix[-seq_len:]

                # 1. Observation mask: 1.0 if observed, 0.0 if NaN
                mask_window = (~np.isnan(raw_window)).astype(np.float32)

                # 2. Elapsed time delta matrix (delta_t)
                delta_window = np.zeros_like(raw_window, dtype=np.float32)
                last_observed = np.zeros(self.num_features, dtype=np.float32)

                for t in range(seq_len):
                    if t == 0:
                        for d in range(self.num_features):
                            if mask_window[0, d] == 1.0:
                                delta_window[0, d] = 0.0
                                last_observed[d] = 0.0
                            else:
                                # Feature missing at t=0: initialize with large delta (BUG-M4-06 fix)
                                delta_window[0, d] = float(self.max_lookback)
                                last_observed[d] = float(self.max_lookback)
                    else:
                        for d in range(self.num_features):
                            if mask_window[t - 1, d] == 1.0:
                                last_observed[d] = 1.0
                            else:
                                last_observed[d] += 1.0
                            delta_window[t, d] = last_observed[d] if mask_window[t, d] == 0.0 else 0.0

                # 3. Impute and standardize for recurrent inputs
                # Fill missing with physiological normals/means to avoid outlier spikes (BUG 7 fix)
                ffill_df = pd.DataFrame(raw_window, columns=self.feature_cols).ffill().bfill()
                for d_idx, c in enumerate(self.feature_cols):
                    default_c = PHYSIOLOGICAL_NORMALS.get(c, float(self.means[d_idx]))
                    if c in ffill_df.columns:
                        ffill_df[c] = ffill_df[c].fillna(default_c)
                
                ffill_window = ffill_df.values.astype(np.float32)
                norm_window = (ffill_window - self.means) / self.stds
                norm_window = np.nan_to_num(norm_window, nan=0.0)
                raw_window_clean = np.nan_to_num(raw_window, nan=0.0)

                # 4. Standard RIGHT-PADDING (BUG-M4-02 fix)
                pad_len = self.max_lookback - seq_len
                if pad_len > 0:
                    X_seq = np.pad(norm_window, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
                    X_raw_pad = np.pad(raw_window_clean, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
                    mask_pad = np.pad(mask_window, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
                    delta_pad = np.pad(delta_window, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
                else:
                    X_seq = norm_window
                    X_raw_pad = raw_window_clean
                    mask_pad = mask_window
                    delta_pad = delta_window

                # Patient label at end of lookback window
                if 'SepsisLabel' in p_df.columns:
                    label = float(p_df['SepsisLabel'].iloc[-1])
                else:
                    label = 0.0

                self.samples.append({
                    'X_seq': torch.as_tensor(X_seq, dtype=torch.float32),
                    'X_raw': torch.as_tensor(X_raw_pad, dtype=torch.float32),
                    'mask': torch.as_tensor(mask_pad, dtype=torch.float32),
                    'm_t': torch.as_tensor(mask_pad, dtype=torch.float32),
                    'delta': torch.as_tensor(delta_pad, dtype=torch.float32),
                    'delta_t': torch.as_tensor(delta_pad, dtype=torch.float32),
                    'lengths': torch.as_tensor(seq_len, dtype=torch.int64),
                    'y': torch.as_tensor([label], dtype=torch.float32)
                })
        else:
            raise TypeError(f"df_or_tensors must be pd.DataFrame, np.ndarray, or torch.Tensor, got {type(df_or_tensors)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]

    def get_pos_weight(self) -> float:
        """Computes pos_weight from training labels to avoid global leakage."""
        if len(self.samples) == 0:
            return 9.5
        all_y = torch.cat([s['y'] for s in self.samples], dim=0)
        num_neg = (all_y == 0).sum().item()
        num_pos = (all_y == 1).sum().item()
        return float(num_neg / max(num_pos, 1))


def get_sequence_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    max_lookback: int = 24,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: list[str] | None = None
) -> tuple[DataLoader, DataLoader, tuple[np.ndarray, np.ndarray], float]:
    """
    Builds patient-level stratified DataLoaders for ICUSequenceDataset without data leakage.
    
    1. Normalizes patient IDs and schema.
    2. Performs stratified train/val split strictly across patient stays.
    3. Computes standardization parameters (mean, std) strictly on train split.
    4. Right-pads sequences up to max_lookback (default: 24).
    5. Returns train_loader, val_loader, (means, stds), and pos_weight.
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS_14
    df_norm = normalize_icu_dataframe(df)

    if 'SepsisLabel' in df_norm.columns:
        patient_labels = df_norm.groupby('PatientID')['SepsisLabel'].max()
    else:
        patient_labels = pd.Series(0, index=df_norm['PatientID'].unique())

    patient_ids = patient_labels.index.values
    labels_arr = patient_labels.values

    unique_classes, counts = np.unique(labels_arr, return_counts=True)
    if len(unique_classes) > 1 and np.min(counts) >= 2:
        train_pids, val_pids = train_test_split(
            patient_ids, test_size=test_size, random_state=random_state, stratify=labels_arr
        )
    else:
        train_pids, val_pids = train_test_split(
            patient_ids, test_size=test_size, random_state=random_state
        )

    df_train = df_norm[df_norm['PatientID'].isin(set(train_pids))].copy()
    df_val = df_norm[df_norm['PatientID'].isin(set(val_pids))].copy()

    # Compute population statistics strictly on training patient split (BUG-M4-11 fix)
    raw_train = df_train[cols].values if len(df_train) > 0 else np.zeros((1, len(cols)))
    means = np.nanmean(raw_train, axis=0)
    stds = np.nanstd(raw_train, axis=0)
    for idx, c in enumerate(cols):
        if np.isnan(means[idx]):
            means[idx] = PHYSIOLOGICAL_NORMALS.get(c, 0.0)
        if np.isnan(stds[idx]) or stds[idx] <= 1e-6:
            stds[idx] = 1.0

    train_dataset = ICUSequenceDataset(
        df_train, feature_cols=cols, max_lookback=max_lookback,
        feature_means=means, feature_stds=stds
    )
    val_dataset = ICUSequenceDataset(
        df_val, feature_cols=cols, max_lookback=max_lookback,
        feature_means=means, feature_stds=stds
    )

    drop_last_train = len(train_dataset) >= batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pos_weight = train_dataset.get_pos_weight()

    return train_loader, val_loader, (means, stds), pos_weight


def create_sequence_dataloader(
    df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = True,
    max_lookback: int = 24
) -> DataLoader:
    """Helper factory for quick single-loader creation."""
    dataset = ICUSequenceDataset(df, max_lookback=max_lookback)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TriModalICUDataset(Dataset):
    """
    Multimodal Dataset unifying:
    1. Static tabular 73-D summary: (B, 73)
    2. Multi-scale 1D-CNN vital waveforms: (B, 7, 12)
    3. Longitudinal temporal 24-hr sequences: (B, 24, 14), mask (B, 24, 14), delta (B, 24, 14), lengths (B,)
    4. Target sepsis label: (B, 1)
    """
    def __init__(
        self,
        X_73d: np.ndarray | torch.Tensor,
        X_vitals: np.ndarray | torch.Tensor,
        X_seq: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        delta: np.ndarray | torch.Tensor,
        lengths: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor
    ):
        self.X_73d = torch.as_tensor(X_73d, dtype=torch.float32)
        self.X_vitals = torch.as_tensor(X_vitals, dtype=torch.float32)
        self.X_seq = torch.as_tensor(X_seq, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.delta = torch.as_tensor(delta, dtype=torch.float32)
        self.lengths = torch.as_tensor(lengths, dtype=torch.int64)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        self.y = y_t.unsqueeze(1) if y_t.ndim == 1 else y_t

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.X_73d[idx],
            self.X_vitals[idx],
            self.X_seq[idx],
            self.mask[idx],
            self.delta[idx],
            self.lengths[idx],
            self.y[idx]
        )
