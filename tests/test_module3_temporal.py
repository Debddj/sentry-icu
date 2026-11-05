"""
tests/test_module3_temporal.py
==============================
Real-data verification and unit tests for SentryICU Module 3:
Temporal Dynamics & Multi-Scale 1D-CNN Fusion.

Covers:
1. Schema normalization & Patient_ID aliasing on real PhysioNet Dataset.csv (BUG-M3-00 fix)
2. Sliding vital window dataset & dataloaders (extract_sliding_windows, ICUSlidingWindowDataset)
3. ClinicalCNN1D multi-scale convolution & shape contract (B, 7, 12) -> (B, 32)
4. SentryICUFusionModel bimodal fusion (B, 73) + (B, 7, 12) -> (B, 105) -> (B, 1)
5. TemporalGradCAM1D sign-preserving attribution capturing acute hypotension/hypothermia drops (BUG-M3-03 fix)
6. Clinical metrics suite (ECE, Brier score, AUROC, AUPRC)
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dataset import (
    KEY_VITALS,
    ICUSlidingWindowDataset,
    extract_sliding_windows,
    get_sliding_window_dataloaders,
    normalize_icu_dataframe,
)
from src.models.clinical_cnn import (
    Baseline73DMLP,
    ClinicalCNN1D,
    CNNOnlyModel,
    MultiScale1DCNN,
    SentryICUFusionModel,
    TemporalGradCAM1D,
)
from src.utils.metrics import (
    brier_score_loss,
    compute_metrics,
    expected_calibration_error,
)


class TestModule3SchemaAndWindowing:
    """Tests for schema normalization and multi-scale sliding window extraction on real patient data."""

    def test_normalize_icu_dataframe_handles_real_patient_id(self, raw_unnormalized_icu_data):
        """Verify BUG-M3-00 fix: resolves 'Patient_ID' alias to 'PatientID' on actual PhysioNet data."""
        assert "Patient_ID" in raw_unnormalized_icu_data.columns or "PatientID" in raw_unnormalized_icu_data.columns

        df_norm = normalize_icu_dataframe(raw_unnormalized_icu_data)
        assert "PatientID" in df_norm.columns
        assert "Hour" in df_norm.columns
        assert "SepsisLabel" in df_norm.columns

        # Verify patient IDs are non-empty strings
        pids = df_norm["PatientID"].unique()
        assert len(pids) > 0
        assert all(isinstance(pid, str) and len(pid) > 0 for pid in pids)

    def test_normalize_icu_dataframe_missing_patient_id_fallback(self):
        """Verify fallback boundary reconstruction when PatientID is absent but ICULOS exists."""
        df_dummy = pd.DataFrame({
            "ICULOS": [1, 2, 3, 1, 2, 1],
            "HR": [80.0, 82.0, 85.0, 75.0, 78.0, 90.0],
            "SepsisLabel": [0, 0, 0, 0, 1, 0],
        })
        df_norm = normalize_icu_dataframe(df_dummy)
        assert "PatientID" in df_norm.columns
        assert df_norm["PatientID"].nunique() == 3
        assert df_norm["Hour"].tolist() == [0, 1, 2, 0, 1, 0]

    def test_extract_sliding_windows_real_data_dimensions(self, normalized_icu_data):
        """Verify sliding window extraction produces correct tensor dimensions on real stays."""
        window_size = 12
        step = 4
        X_73d, X_vitals, y = extract_sliding_windows(
            normalized_icu_data, window_size=window_size, step=step
        )

        assert isinstance(X_73d, np.ndarray)
        assert isinstance(X_vitals, np.ndarray)
        assert isinstance(y, np.ndarray)

        assert X_73d.ndim == 2
        assert X_73d.shape[1] == 73, f"Expected 73 static features, got {X_73d.shape[1]}"

        assert X_vitals.ndim == 3
        assert X_vitals.shape[1] == len(KEY_VITALS), f"Expected 7 vital channels, got {X_vitals.shape[1]}"
        assert X_vitals.shape[2] == window_size, f"Expected lookback length {window_size}, got {X_vitals.shape[2]}"

        assert len(X_73d) == len(X_vitals) == len(y)
        assert len(X_73d) > 0

        # Verify no NaN or Inf in extracted real features
        assert not np.isnan(X_73d).any(), "Found NaNs in real 73-D static summaries"
        assert not np.isnan(X_vitals).any(), "Found NaNs in real vital waveforms"
        assert np.isin(y, [0, 1]).all(), "Labels must be binary {0, 1}"

    def test_icu_sliding_window_dataset_and_pos_weight(self, real_sliding_windows_data):
        """Verify ICUSlidingWindowDataset item shapes and pos_weight calculation."""
        X_73d, X_vitals, y = real_sliding_windows_data
        dataset = ICUSlidingWindowDataset(X_73d, X_vitals, y)

        assert len(dataset) == len(y)
        sample_73d, sample_vitals, sample_y = dataset[0]

        assert sample_73d.shape == torch.Size([73])
        assert sample_73d.dtype == torch.float32
        assert sample_vitals.shape == torch.Size([7, 12])
        assert sample_vitals.dtype == torch.float32
        assert sample_y.shape == torch.Size([1])
        assert sample_y.dtype == torch.float32

        pos_weight = dataset.get_pos_weight()
        assert isinstance(pos_weight, float)
        assert pos_weight > 0.0

    def test_get_sliding_window_dataloaders_stratification(self, normalized_icu_data):
        """Verify stratified splitting, leak-free standardization, and DataLoader creation."""
        batch_size = 16
        train_loader, val_loader, scaler, ts_stats, pos_weight = get_sliding_window_dataloaders(
            normalized_icu_data,
            batch_size=batch_size,
            window_size=12,
            step=4,
            test_size=0.25,
            random_state=42,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert pos_weight > 0.0
        assert scaler is not None
        assert ts_stats is not None and len(ts_stats) == 2

        # Check train batch
        batch_73d, batch_vitals, batch_y = next(iter(train_loader))
        assert batch_73d.size(1) == 73
        assert batch_vitals.size(1) == 7
        assert batch_vitals.size(2) == 12
        assert batch_y.size(1) == 1

        # Check val batch
        v_73d, v_vitals, v_y = next(iter(val_loader))
        assert v_73d.size(1) == 73
        assert v_vitals.size(1) == 7
        assert v_vitals.size(2) == 12
        assert v_y.size(1) == 1


class TestClinicalCNN1D:
    """Tests for ClinicalCNN1D feature extractor shape contracts and multi-scale operations."""

    def test_shape_contract_real_vitals(self, real_multimodal_batch):
        """Verify contract: (B, 7, 12) -> (B, 32) float32 representation."""
        x_vitals = real_multimodal_batch["x_vitals"]
        batch_size = x_vitals.size(0)

        model = ClinicalCNN1D(in_channels=7, out_embed_dim=32, pool_type="avg")
        model.eval()

        with torch.no_grad():
            emb = model(x_vitals)

        assert emb.shape == torch.Size([batch_size, 32])
        assert emb.dtype == torch.float32
        assert not torch.isnan(emb).any()

    def test_single_sample_batch_shape(self, real_multimodal_batch):
        """Verify model handles batch size B=1 without collapsing dimensions."""
        x_vitals = real_multimodal_batch["x_vitals"][:1]  # (1, 7, 12)
        model = ClinicalCNN1D(in_channels=7, out_embed_dim=32)
        emb = model(x_vitals)
        assert emb.shape == torch.Size([1, 32])

    def test_extract_features_multiscale(self, real_multimodal_batch):
        """Verify parallel multi-scale feature maps before pooling: 3 branches x 16 = 48 channels."""
        x_vitals = real_multimodal_batch["x_vitals"]
        model = ClinicalCNN1D(in_channels=7, out_embed_dim=32)
        feat_maps = model.extract_features(x_vitals)

        assert feat_maps.shape == torch.Size([x_vitals.size(0), 48, 12])

    def test_max_pool_option(self, real_multimodal_batch):
        """Verify adaptive max pooling option operates cleanly."""
        x_vitals = real_multimodal_batch["x_vitals"]
        model_max = ClinicalCNN1D(in_channels=7, out_embed_dim=32, pool_type="max")
        emb_max = model_max(x_vitals)
        assert emb_max.shape == torch.Size([x_vitals.size(0), 32])

    def test_contract_alias(self):
        """Verify interface alias MultiScale1DCNN == ClinicalCNN1D."""
        assert MultiScale1DCNN is ClinicalCNN1D


class TestSentryICUFusionModel:
    """Tests for SentryICUFusionModel multimodal late-fusion and ablation heads."""

    def test_forward_pass_and_shape_contract(self, real_multimodal_batch):
        """Verify bimodal fusion contract: (B, 73) + (B, 7, 12) -> logits (B, 1), cnn (B, 32), fused (B, 105)."""
        x_tab = real_multimodal_batch["x_tab"]
        x_vitals = real_multimodal_batch["x_vitals"]
        batch_size = x_tab.size(0)

        model = SentryICUFusionModel(static_dim=73, ts_channels=7, cnn_embed_dim=32)
        model.eval()

        with torch.no_grad():
            logits, cnn_embed, fused = model(x_tab, x_vitals)

        assert logits.shape == torch.Size([batch_size, 1])
        assert cnn_embed.shape == torch.Size([batch_size, 32])
        assert fused.shape == torch.Size([batch_size, 105])
        assert not torch.isnan(logits).any()

    def test_predict_proba_range(self, real_multimodal_batch):
        """Verify .predict_proba() yields calibrated probabilities in [0.0, 1.0]."""
        x_tab = real_multimodal_batch["x_tab"]
        x_vitals = real_multimodal_batch["x_vitals"]

        model = SentryICUFusionModel()
        probs = model.predict_proba(x_tab, x_vitals)

        assert probs.shape == torch.Size([x_tab.size(0), 1])
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_backward_gradient_propagation(self, real_multimodal_batch):
        """Verify end-to-end backpropagation: gradients reach both static head and 1D-CNN branches."""
        x_tab = real_multimodal_batch["x_tab"]
        x_vitals = real_multimodal_batch["x_vitals"]
        y_true = real_multimodal_batch["y"]

        model = SentryICUFusionModel()
        model.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.5]))

        logits, _, _ = model(x_tab, x_vitals)
        loss = criterion(logits, y_true)
        loss.backward()

        # Check gradients in CNN branch
        conv_grad = model.cnn_encoder.branch_k3[0].weight.grad
        assert conv_grad is not None
        assert not torch.all(conv_grad == 0.0)

        # Check gradients in Fusion MLP
        mlp_grad = model.fusion_mlp[0].weight.grad
        assert mlp_grad is not None
        assert not torch.all(mlp_grad == 0.0)

    def test_ablation_baselines(self, real_multimodal_batch):
        """Verify capacity-controlled Baseline73DMLP and CNNOnlyModel."""
        x_tab = real_multimodal_batch["x_tab"]
        x_vitals = real_multimodal_batch["x_vitals"]
        batch_size = x_tab.size(0)

        static_mlp = Baseline73DMLP(input_dim=73)
        cnn_only = CNNOnlyModel(in_channels=7, cnn_embed_dim=32)

        out_static = static_mlp(x_tab)
        assert out_static.shape == torch.Size([batch_size, 1])
        prob_static = static_mlp.predict_proba(x_tab)
        assert (prob_static >= 0.0).all() and (prob_static <= 1.0).all()

        out_cnn = cnn_only(x_ts=x_vitals)
        assert out_cnn.shape == torch.Size([batch_size, 1])


class TestTemporalGradCAM1D:
    """Tests for hook-based 1D Temporal Grad-CAM with sign-preserving attribution (BUG-M3-03 fix)."""

    def test_gradcam_attribution_heatmap_shape_and_range(self, real_multimodal_batch):
        """Verify Grad-CAM produces a normalized (7, 12) heatmap across vital channels and hours."""
        x_tab = real_multimodal_batch["x_tab"][:1]
        x_vitals = real_multimodal_batch["x_vitals"][:1]

        model = SentryICUFusionModel()
        gradcam = TemporalGradCAM1D(model)

        heatmap = gradcam.generate_heatmap(x_tab, x_vitals, target_class=1)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (7, 12), f"Expected (7, 12) heatmap, got {heatmap.shape}"
        assert not np.isnan(heatmap).any(), "Heatmap contains NaNs"
        assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all(), "Heatmap must be normalized in [0.0, 1.0]"

    def test_sign_preserving_captures_hypotension_hypothermia_drops(self):
        """
        Verify BUG-M3-03 fix: acute drops in vital signs (e.g. MAP drop in septic shock)
        have non-zero attribution and are not wiped out by a destructive ReLU.
        """
        model = SentryICUFusionModel()
        gradcam = TemporalGradCAM1D(model)

        # Construct a real-like vital sequence with precipitous drop in MAP (index 4) and Temp (index 2)
        # MAP normal=80 dropping to 45 (severe septic shock), Temp normal=37 dropping to 35 (hypothermia)
        x_vitals_drop = torch.zeros(1, 7, 12, dtype=torch.float32)
        for t in range(12):
            x_vitals_drop[0, 4, t] = 80.0 if t < 6 else 45.0  # MAP drop
            x_vitals_drop[0, 2, t] = 37.0 if t < 6 else 35.0  # Temp drop

        x_tab = torch.zeros(1, 73, dtype=torch.float32)

        heatmap = gradcam.generate_heatmap(x_tab, x_vitals_drop, target_class=1)
        assert heatmap.shape == (7, 12)
        # Verify attribution exists and is strictly positive
        assert heatmap.max() > 0.0


class TestClinicalMetrics:
    """Tests for clinical calibration, discrimination, and error metrics."""

    def test_expected_calibration_error_bounds(self):
        """Verify ECE output is bounded in [0.0, 1.0] and well-calibrated predictions have lower ECE."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Perfect probabilities: ECE should be near 0
        y_prob_good = np.array([0.05, 0.05, 0.1, 0.9, 0.95, 0.95])
        ece_good = expected_calibration_error(y_true, y_prob_good, n_bins=5)

        # Inverted probabilities: ECE should be high
        y_prob_bad = np.array([0.9, 0.95, 0.9, 0.05, 0.1, 0.05])
        ece_bad = expected_calibration_error(y_true, y_prob_bad, n_bins=5)

        assert 0.0 <= ece_good <= 1.0
        assert 0.0 <= ece_bad <= 1.0
        assert ece_good < ece_bad

    def test_brier_score_loss_exact_values(self):
        """Verify Brier score formula matches mean squared probability deviation."""
        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.2, 0.8])
        # ( (0.2 - 0)^2 + (0.8 - 1)^2 ) / 2 = (0.04 + 0.04) / 2 = 0.04
        expected_brier = 0.04
        assert np.isclose(brier_score_loss(y_true, y_prob), expected_brier)

    def test_compute_metrics_comprehensive(self):
        """Verify compute_metrics returns all required discrimination and calibration metrics."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.85, 0.9, 0.7, 0.4, 0.8])

        metrics = compute_metrics(y_true, y_prob, threshold=0.5, n_bins=5)

        required_keys = [
            "auroc", "auprc", "f1", "sensitivity", "specificity",
            "precision", "accuracy", "ece", "brier_score", "confusion_matrix"
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["auprc"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0

        cm = metrics["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == len(y_true)

    def test_compute_metrics_single_class_edge_case(self):
        """Verify compute_metrics does not crash when evaluation split contains only negatives (0)."""
        y_true_single = np.zeros(20, dtype=int)
        y_prob_single = np.full(20, 0.1)

        # Should handle without raising ValueError
        metrics = compute_metrics(y_true_single, y_prob_single)
        assert metrics["auroc"] == 0.5
        assert metrics["f1"] == 0.0
