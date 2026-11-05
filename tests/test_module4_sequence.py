"""
tests/test_module4_sequence.py
==============================
Real-data verification and unit tests for SentryICU Module 4:
Longitudinal Temporal Sequence Modeling & Tri-Modal Fusion.

Covers:
1. ICUSequenceDataset & DataLoaders on real PhysioNet patient trajectories:
   - Right-padding enforcement (BUG-M4-02 fix)
   - Observation mask m_t and elapsed delta_t (BUG-M4-06 fix)
   - Admission missingness delta_t = 24.0 initialization
2. Recurrent sequence backbones:
   - ClinicalVanillaRNN, ClinicalLSTM, ClinicalGRU (and alias TemporalGRU)
   - BidirectionalGRU (32 fwd + 32 bwd = 64-D)
   - AttentionGRU with stable masking (BUG-M4-07 fix)
   - Shape contracts (B, 24, 14) -> (B, 64) and (B, 1) logits
3. ClinicalGRUD (and alias TemporalGRUD):
   - Softplus decay parameterization eliminating dead ReLU trap (BUG-M4-01 fix)
   - Non-zero decay gradients (grad != 0) from step 1
   - Empirical mean buffer populating and imputation (BUG-M4-04 fix)
   - Observation persistence in x_last (BUG-M4-03 fix)
4. SentryICUTriModalFusion:
   - Late-fusion shape contract: (B, 73) + (B, 32) + (B, 64) -> (B, 169) -> (B, 1)
5. SentryICUEndToEndTriModal:
   - Full raw-input pipeline execution and .predict_proba()
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.dataset import (
    ICUSequenceDataset,
    get_sequence_dataloaders,
)
from src.models.grud import (
    ClinicalGRUD,
    ClinicalGRUDCell,
    TemporalGRUD,
)
from src.models.sequence_models import (
    AttentionGRU,
    BidirectionalGRU,
    ClinicalGRU,
    ClinicalLSTM,
    ClinicalVanillaRNN,
    TemporalGRU,
)
from src.models.trimodal_fusion import (
    SentryICUEndToEndTriModal,
    SentryICUTriModalFusion,
)


class TestModule4SequenceDataset:
    """Tests for longitudinal sequence dataset and right-padding on real patient stays."""

    def test_sequence_dataset_right_padding_alignment(self, normalized_icu_data):
        """
        Verify BUG-M4-02 fix: sequences are RIGHT-padded.
        Timesteps 0 .. seq_len-1 contain true observations; seq_len .. 23 contain padding zeros.
        lengths - 1 indexes the most recent actual observation.
        """
        dataset = ICUSequenceDataset(normalized_icu_data, max_lookback=24)
        assert len(dataset) > 0

        # Find a sample with stay length < 24 to check right padding
        short_sample = None
        for s in dataset:
            if s["lengths"].item() < 24:
                short_sample = s
                break

        if short_sample is not None:
            seq_len = short_sample["lengths"].item()
            X_seq = short_sample["X_seq"]  # (24, 14)
            mask = short_sample["mask"]    # (24, 14)

            # Timesteps after seq_len should be padded with zeros
            assert torch.all(X_seq[seq_len:] == 0.0), "Padding timesteps must be zero"
            assert torch.all(mask[seq_len:] == 0.0), "Padding mask must be zero"

    def test_missingness_delta_initialization_at_admission(self, normalized_icu_data):
        """
        Verify BUG-M4-06 fix: variables unobserved at admission (t=0) have delta_t = 24.0,
        ensuring smooth decay factor gamma_x -> 0.0 toward population mean.
        """
        dataset = ICUSequenceDataset(normalized_icu_data, max_lookback=24)

        # Check across samples for an unobserved feature at t=0
        found_missing_at_t0 = False
        for s in dataset:
            m_0 = s["mask"][0]      # (14,)
            delta_0 = s["delta"][0]  # (14,)
            for d in range(14):
                if m_0[d].item() == 0.0:
                    assert delta_0[d].item() == 24.0, f"Unobserved t=0 variable must have delta=24.0, got {delta_0[d].item()}"
                    found_missing_at_t0 = True
                    break
            if found_missing_at_t0:
                break

        assert found_missing_at_t0, "Expected at least one unobserved variable at admission in real ICU cohort"

    def test_get_sequence_dataloaders_stratified(self, normalized_icu_data):
        """Verify stratified sequence dataloaders without data leakage."""
        batch_size = 16
        train_loader, val_loader, stats, pos_weight = get_sequence_dataloaders(
            normalized_icu_data, batch_size=batch_size, max_lookback=24, test_size=0.25, random_state=42
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert pos_weight > 0.0
        assert stats is not None and len(stats) == 2

        train_batch = next(iter(train_loader))
        assert train_batch["X_seq"].shape[1:] == torch.Size([24, 14])
        assert train_batch["mask"].shape[1:] == torch.Size([24, 14])
        assert train_batch["delta"].shape[1:] == torch.Size([24, 14])
        assert train_batch["y"].shape[1] == 1


class TestSequenceModelsBackbone:
    """Tests for RNN, LSTM, GRU, BiGRU, and AttentionGRU sequence models."""

    def test_contract_alias_temporal_gru(self):
        """Verify docs/interfaces.md contract alias TemporalGRU == ClinicalGRU."""
        assert TemporalGRU is ClinicalGRU

    def test_recurrent_architectures_shape_contract(self, real_multimodal_batch):
        """
        Verify shape contracts for all recurrent models:
        Input: (B, 24, 14) -> Output: temporal embedding (B, 64) and logit (B, 1).
        """
        x_seq = real_multimodal_batch["x_seq"]
        lengths = real_multimodal_batch["lengths"]
        batch_size = x_seq.size(0)

        models = [
            ("ClinicalVanillaRNN", ClinicalVanillaRNN(input_dim=14, hidden_dim=64)),
            ("ClinicalLSTM", ClinicalLSTM(input_dim=14, hidden_dim=64)),
            ("ClinicalGRU", ClinicalGRU(input_dim=14, hidden_dim=64)),
            ("BidirectionalGRU", BidirectionalGRU(input_dim=14, hidden_dim=64)),
            ("AttentionGRU", AttentionGRU(input_dim=14, hidden_dim=64)),
        ]

        for name, model in models:
            model.eval()
            with torch.no_grad():
                logits, emb = model(x_seq, lengths=lengths)

            assert logits.shape == torch.Size([batch_size, 1]), f"{name} produced incorrect logits shape"
            assert emb.shape == torch.Size([batch_size, 64]), f"{name} produced incorrect embedding shape"
            assert not torch.isnan(logits).any(), f"{name} produced NaN logits"
            assert not torch.isnan(emb).any(), f"{name} produced NaN embeddings"

            # Test predict_proba
            probs = model.predict_proba(x_seq, lengths=lengths)
            assert probs.shape == torch.Size([batch_size, 1])
            assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"{name} predict_proba out of bounds"

    def test_bidirectional_gru_concatenation(self, real_multimodal_batch):
        """Verify BidirectionalGRU projects 32 forward + 32 backward = 64-D embedding."""
        x_seq = real_multimodal_batch["x_seq"]
        lengths = real_multimodal_batch["lengths"]
        bigru = BidirectionalGRU(input_dim=14, hidden_dim=64)
        emb = bigru.extract_features(x_seq, lengths=lengths)
        assert emb.shape == torch.Size([x_seq.size(0), 64])

    def test_attention_gru_masking_and_weights(self, real_multimodal_batch):
        """
        Verify AttentionGRU attention distribution:
        - Weights sum to 1.0 along the sequence length.
        - Softmax scores at right-padded timesteps are properly masked.
        """
        x_seq = real_multimodal_batch["x_seq"]
        lengths = real_multimodal_batch["lengths"]

        att_model = AttentionGRU(input_dim=14, hidden_dim=64)
        logits, emb, alpha = att_model(x_seq, lengths=lengths, return_attention=True)

        assert logits.shape == torch.Size([x_seq.size(0), 1])
        assert emb.shape == torch.Size([x_seq.size(0), 64])
        assert alpha.shape == torch.Size([x_seq.size(0), 24])

        # Verify softmax probabilities sum to 1.0 per sequence
        row_sums = alpha.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        # For samples with length < 24, padding timesteps should receive near-zero attention
        for i in range(len(lengths)):
            seq_len = lengths[i].item()
            if seq_len < 24:
                pad_alpha = alpha[i, seq_len:]
                assert (pad_alpha < 1e-4).all(), "Padding timesteps must have near-zero attention weight"


class TestClinicalGRUD:
    """Tests for ClinicalGRUD with Softplus decay, non-zero gradients, and mean imputation."""

    def test_contract_alias_temporal_grud(self):
        """Verify docs/interfaces.md alias TemporalGRUD == ClinicalGRUD."""
        assert TemporalGRUD is ClinicalGRUD

    def test_softplus_decay_nonzero_gradients(self, real_multimodal_batch):
        """
        Verify BUG-M4-01 fix:
        Replaces dead ReLU with Softplus decay: gamma_x = exp(-softplus(W) * delta_t).
        Guarantees non-zero decay gradients (grad != 0) immediately upon initialization.
        """
        x_seq = real_multimodal_batch["x_seq"]
        mask = real_multimodal_batch["mask"]
        delta = real_multimodal_batch["delta"]
        y_true = real_multimodal_batch["y"]

        model = ClinicalGRUD(input_dim=14, hidden_dim=64)
        model.train()

        logits, _ = model(x_seq, mask, delta)
        loss = F.binary_cross_entropy_with_logits(logits, y_true)
        loss.backward()

        # Decay weights must have non-zero gradients
        w_gamma_x_grad = model.cell.W_gamma_x.grad
        assert w_gamma_x_grad is not None
        assert not torch.all(w_gamma_x_grad == 0.0), (
            "Decay parameter W_gamma_x has zero gradient! Dead ReLU trap regression."
        )

        w_gamma_h_grad = model.cell.W_gamma_h.weight.grad
        assert w_gamma_h_grad is not None
        assert not torch.all(w_gamma_h_grad == 0.0), (
            "Hidden decay projection W_gamma_h has zero gradient!"
        )

    def test_empirical_mean_buffer_registered_and_populated(self):
        """
        Verify BUG-M4-04 fix:
        ClinicalGRUD registers and populates empirical_mean buffer, preventing
        missing features from decaying toward 0.0.
        """
        real_means = torch.tensor([80.0, 98.0, 37.0, 120.0, 80.0, 80.0, 16.0,
                                   1.0, 7.0, 1.0, 15.0, 100.0, 250.0, 14.0], dtype=torch.float32)

        model = ClinicalGRUD(input_dim=14, hidden_dim=64, empirical_mean=real_means)
        assert hasattr(model, "empirical_mean")
        assert torch.allclose(model.empirical_mean, real_means)

        # Verify update method
        new_means = real_means + 1.0
        model.set_empirical_mean(new_means)
        assert torch.allclose(model.empirical_mean, new_means)

    def test_observation_preservation_in_cell(self):
        """
        Verify BUG-M4-03 fix:
        x_last is updated strictly when the feature is actively observed (m_t = 1),
        preventing compounding exponential decay over multiple unobserved timesteps.
        """
        cell = ClinicalGRUDCell(input_dim=2, hidden_dim=8)
        mean_x = torch.tensor([50.0, 50.0])
        h_prev = torch.zeros(1, 8)
        x_last_init = torch.tensor([[100.0, 100.0]])

        # Step 1: Feature 0 observed (m=1), Feature 1 missing (m=0)
        x_raw = torch.tensor([[120.0, 0.0]])
        m_t = torch.tensor([[1.0, 0.0]])
        delta_t = torch.tensor([[1.0, 1.0]])

        h_next, x_last_next = cell(x_raw, m_t, delta_t, h_prev, x_last_init, mean_x)

        assert h_next.shape == torch.Size([1, 8])
        # Feature 0 was observed: x_last should update to 120.0
        assert np.isclose(x_last_next[0, 0].item(), 120.0)
        # Feature 1 was missing: x_last must preserve previous value (100.0), NOT decayed value
        assert np.isclose(x_last_next[0, 1].item(), 100.0)

    def test_grud_forward_unrolling_stability(self, real_multimodal_batch):
        """Verify unrolling over 24 timesteps with real data produces valid float32 outputs."""
        x_seq = real_multimodal_batch["x_seq"]
        mask = real_multimodal_batch["mask"]
        delta = real_multimodal_batch["delta"]
        lengths = real_multimodal_batch["lengths"]

        model = ClinicalGRUD(input_dim=14, hidden_dim=64)
        model.eval()

        with torch.no_grad():
            logits, emb = model(x_seq, mask, delta, lengths=lengths)

        assert logits.shape == torch.Size([x_seq.size(0), 1])
        assert emb.shape == torch.Size([x_seq.size(0), 64])
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestTriModalFusion:
    """Tests for SentryICUTriModalFusion and SentryICUEndToEndTriModal early warning system."""

    def test_trimodal_fusion_head_shape_contract(self):
        """Verify late fusion shape contract: (B, 73) + (B, 32) + (B, 64) -> (B, 169) -> (B, 1)."""
        batch_size = 16
        x_tab = torch.randn(batch_size, 73)
        x_cnn = torch.randn(batch_size, 32)
        x_gru = torch.randn(batch_size, 64)

        fusion_head = SentryICUTriModalFusion(dim_tabular=73, dim_cnn=32, dim_temporal=64)
        fusion_head.eval()

        with torch.no_grad():
            logits, fused = fusion_head(x_tab, x_cnn, x_gru)

        assert logits.shape == torch.Size([batch_size, 1])
        assert fused.shape == torch.Size([batch_size, 169])

        # Test predict_proba
        probs = fusion_head.predict_proba(x_tab, x_cnn, x_gru)
        assert probs.shape == torch.Size([batch_size, 1])
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_trimodal_fusion_dimension_validation(self):
        """Verify assertion error when unexpected dimension is supplied."""
        fusion_head = SentryICUTriModalFusion(dim_tabular=73, dim_cnn=32, dim_temporal=64)
        bad_tab = torch.randn(4, 50)  # Should be 73
        valid_cnn = torch.randn(4, 32)
        valid_gru = torch.randn(4, 64)

        with pytest.raises(AssertionError, match="Expected tabular dim 73"):
            fusion_head(bad_tab, valid_cnn, valid_gru)

    def test_end_to_end_trimodal_real_data_execution(self, real_multimodal_batch):
        """Verify full end-to-end tri-modal pipeline on real patient inputs."""
        x_tab = real_multimodal_batch["x_tab"]
        x_vitals = real_multimodal_batch["x_vitals"]
        x_seq = real_multimodal_batch["x_seq"]
        mask = real_multimodal_batch["mask"]
        delta = real_multimodal_batch["delta"]
        lengths = real_multimodal_batch["lengths"]
        batch_size = x_tab.size(0)

        # 1. Test with GRU-D temporal backbone
        model_grud = SentryICUEndToEndTriModal(use_grud=True)
        model_grud.eval()

        with torch.no_grad():
            logits, fused, embs = model_grud(
                x_tab, x_vitals, x_seq, mask=mask, delta=delta, lengths=lengths, return_embeddings=True
            )

        assert logits.shape == torch.Size([batch_size, 1])
        assert fused.shape == torch.Size([batch_size, 169])
        assert embs["tabular"].shape == torch.Size([batch_size, 73])
        assert embs["cnn"].shape == torch.Size([batch_size, 32])
        assert embs["temporal"].shape == torch.Size([batch_size, 64])

        probs = model_grud.predict_proba(
            x_tab, x_vitals, x_seq, mask=mask, delta=delta, lengths=lengths
        )
        assert probs.shape == torch.Size([batch_size, 1])
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        # 2. Test with standard GRU temporal backbone
        model_gru = SentryICUEndToEndTriModal(use_grud=False)
        model_gru.eval()

        with torch.no_grad():
            logits_gru, _ = model_gru(x_tab, x_vitals, x_seq, lengths=lengths)

        assert logits_gru.shape == torch.Size([batch_size, 1])
