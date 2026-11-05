"""
tests/conftest.py
=================
Pytest configuration and authoritative real-data fixtures for SentryICU.

CRITICAL MANDATE:
- All fixtures MUST supply ACTUAL REAL PATIENT DATA from the PhysioNet ICU cohort.
- Real data source: `data/sample/real_patients_sample.csv` or `C:\\Users\\debnil\\Downloads\\archive.zip` (Dataset.csv).
- NO SYNTHETIC DATA IS USED FOR VERIFICATION.
"""

import os
import sys
import zipfile

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import (
    ICUSequenceDataset,
    ICUSlidingWindowDataset,
    extract_sliding_windows,
    get_sequence_dataloaders,
    get_sliding_window_dataloaders,
    normalize_icu_dataframe,
)

SAMPLE_PATH = os.path.join("data", "sample", "real_patients_sample.csv")
ARCHIVE_ZIP = r"C:\Users\debnil\Downloads\archive.zip"


@pytest.fixture(scope="session")
def real_icu_data() -> pd.DataFrame:
    """
    Authoritative fixture loading 100 genuine ICU patient stays.
    Checks `data/sample/real_patients_sample.csv` first. If missing,
    extracts the first 100 complete patient stays directly from `C:\\Users\\debnil\\Downloads\\archive.zip`.
    """
    if os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
        return df

    if os.path.exists(ARCHIVE_ZIP):
        with zipfile.ZipFile(ARCHIVE_ZIP, "r") as z, z.open("Dataset.csv") as f:
            # Read chunks to get 100 complete patients
            df_raw = pd.read_csv(f, nrows=5000)
            patient_col = "Patient_ID" if "Patient_ID" in df_raw.columns else "PatientID"
            pids = df_raw[patient_col].unique()[:100]
            df_sample = df_raw[df_raw[patient_col].isin(pids)].copy()

            os.makedirs(os.path.dirname(SAMPLE_PATH), exist_ok=True)
            df_sample.to_csv(SAMPLE_PATH, index=False)
            return df_sample

    raise FileNotFoundError(
        f"Neither sample cache '{SAMPLE_PATH}' nor source archive '{ARCHIVE_ZIP}' found."
    )


@pytest.fixture(scope="session")
def raw_unnormalized_icu_data(real_icu_data: pd.DataFrame) -> pd.DataFrame:
    """Returns un-normalized real patient data preserving 'Patient_ID' casing for schema validation."""
    return real_icu_data.copy()


@pytest.fixture(scope="session")
def normalized_icu_data(real_icu_data: pd.DataFrame) -> pd.DataFrame:
    """Returns real patient data after schema normalization (PatientID, Hour, SepsisLabel)."""
    return normalize_icu_dataframe(real_icu_data)


@pytest.fixture(scope="session")
def real_sliding_windows_data(normalized_icu_data: pd.DataFrame):
    """
    Extracts sliding windows (12-hour lookback, step=4) from real patient stays.
    Returns:
        tuple: (X_73d, X_vitals, y) as numpy arrays.
    """
    X_73d, X_vitals, y = extract_sliding_windows(
        normalized_icu_data, window_size=12, step=4
    )
    return X_73d, X_vitals, y


@pytest.fixture(scope="session")
def real_sliding_window_dataset(real_sliding_windows_data) -> ICUSlidingWindowDataset:
    """PyTorch Dataset with real 73-D static summary, (7, 12) vital waveforms, and labels."""
    X_73d, X_vitals, y = real_sliding_windows_data
    return ICUSlidingWindowDataset(X_73d, X_vitals, y)


@pytest.fixture(scope="session")
def real_sliding_window_loaders(normalized_icu_data: pd.DataFrame):
    """Stratified train and validation DataLoaders on real patient stays."""
    return get_sliding_window_dataloaders(
        normalized_icu_data, batch_size=16, window_size=12, step=4, test_size=0.25, random_state=42
    )


@pytest.fixture(scope="session")
def real_sequence_dataset(normalized_icu_data: pd.DataFrame) -> ICUSequenceDataset:
    """PyTorch Dataset with real 24-hour right-padded longitudinal clinical trajectories."""
    return ICUSequenceDataset(normalized_icu_data, max_lookback=24)


@pytest.fixture(scope="session")
def real_sequence_loaders(normalized_icu_data: pd.DataFrame):
    """Stratified sequence DataLoaders on real patient trajectories."""
    return get_sequence_dataloaders(
        normalized_icu_data, batch_size=16, max_lookback=24, test_size=0.25, random_state=42
    )


@pytest.fixture(scope="session")
def real_multimodal_batch(
    real_sliding_windows_data, real_sequence_dataset
) -> dict[str, torch.Tensor]:
    """
    Provides a synchronized batch of B=16 real patient records across all modalities:
    - x_tab: (16, 73)
    - x_vitals: (16, 7, 12)
    - x_seq: (16, 24, 14)
    - mask: (16, 24, 14)
    - delta: (16, 24, 14)
    - lengths: (16,)
    - y: (16, 1)
    """
    X_73d, X_vitals, y = real_sliding_windows_data
    batch_size = 16

    # Select B samples from sliding window data
    indices = np.arange(min(batch_size, len(X_73d)))
    x_tab = torch.as_tensor(X_73d[indices], dtype=torch.float32)
    x_vitals = torch.as_tensor(X_vitals[indices], dtype=torch.float32)
    y_target = torch.as_tensor(y[indices], dtype=torch.float32).unsqueeze(1)

    # Select B samples from sequence dataset
    seq_samples = [real_sequence_dataset[i] for i in range(len(indices))]
    x_seq = torch.stack([s["X_seq"] for s in seq_samples], dim=0)
    mask = torch.stack([s["mask"] for s in seq_samples], dim=0)
    delta = torch.stack([s["delta"] for s in seq_samples], dim=0)
    lengths = torch.stack([s["lengths"] for s in seq_samples], dim=0)

    return {
        "x_tab": x_tab,
        "x_vitals": x_vitals,
        "x_seq": x_seq,
        "mask": mask,
        "delta": delta,
        "lengths": lengths,
        "y": y_target,
    }
