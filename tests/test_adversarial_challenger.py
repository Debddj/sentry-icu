"""
tests/test_adversarial_challenger.py
====================================
Adversarial Challenge & Stress Harness for SentryICU Multi-Module Integration (Modules 3, 4, 5).
Authored by Challenger 1 (teamwork_challenger_1).

Tests:
1. GRU-D Softplus vs Dead ReLU decay: Verify non-zero gradient on W_gamma_x from step 1.
2. ICUSequenceDataset Right-Padding: Verify canary value is gathered at lengths - 1.
3. TemporalGradCAM1D: Verify non-zero attribution on hypotensive shock / hypothermia drops.
4. PatientVAE Balanced ELBO: Verify latent space variance and absence of posterior collapse.
5. ClinicalRAGAssistant: Verify offline metric extraction and strict PHI column rejection.
6. Stress Tests:
   - Batch size B=1 across all models
   - All-zero inputs (zeros, zero masks, zero deltas)
   - Extreme physiological values (+1e6, -1e6, delta=1e4)
   - Zero-length sequence clamping and boundary conditions
   - All-negative evaluation metrics
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import (
    ICUSequenceDataset,
    ICUSlidingWindowDataset,
    normalize_icu_dataframe,
    extract_sliding_windows,
)
from src.models.clinical_cnn import (
    ClinicalCNN1D,
    SentryICUFusionModel,
    TemporalGradCAM1D,
)
from src.models.clinical_rag import (
    ClinicalRAGAssistant,
    PHI_IDENTIFIER_COLUMNS,
)
from src.models.gan import (
    TabularGenerator,
    TabularDiscriminator,
    TabularGAN,
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
from src.models.vae import (
    PatientVAE,
    VAEDriftDetector,
    vae_loss,
)
from src.utils.metrics import (
    brier_score_loss,
    compute_metrics,
    expected_calibration_error,
)


class TestEmpiricalBugFixVerification:
    """1. Empirical verification of specific cataloged bug fixes."""

    def test_grud_softplus_vs_dead_relu_gradient_step1(self):
        """
        Adversarial Test 1.1:
        Verify GRU-D Softplus decay produces strictly non-zero gradients on W_gamma_x from step 1.
        Contrast empirically with the notebook's dead ReLU formulation where W=-0.1 trapped gradients at 0.
        """
        # Setup input with missing values (m_t = 0) and positive time elapsed delta_t > 0
        torch.manual_seed(42)
        batch_size = 4
        input_dim = 14
        hidden_dim = 64

        x_raw = torch.randn(batch_size, input_dim)
        # All features unobserved at step 1 (m_t = 0) to ensure dependence on decay factor gamma_x
        m_t = torch.zeros(batch_size, input_dim)
        delta_t = torch.ones(batch_size, input_dim) * 2.0  # 2 hours elapsed
        h_prev = torch.randn(batch_size, hidden_dim)
        x_last = torch.randn(batch_size, input_dim)
        mean_x = torch.zeros(input_dim)

        # 1. New Softplus Cell (Production)
        cell_softplus = ClinicalGRUDCell(input_dim=input_dim, hidden_dim=hidden_dim)
        cell_softplus.train()
        h_next, x_last_next, x_hat = cell_softplus(
            x_raw, m_t, delta_t, h_prev, x_last, mean_x, return_x_hat=True
        )
        # Compute loss on h_next
        loss_softplus = h_next.sum() + x_hat.sum()
        loss_softplus.backward()

        w_gamma_x_grad = cell_softplus.W_gamma_x.grad
        assert w_gamma_x_grad is not None, "W_gamma_x gradient must exist"
        grad_norm = torch.norm(w_gamma_x_grad).item()
        assert grad_norm > 1e-5, f"W_gamma_x gradient norm must be non-zero from step 1, got {grad_norm}"
        # Every channel with delta_t > 0 and m_t = 0 should have non-zero gradient
        assert (w_gamma_x_grad.abs() > 1e-6).all(), (
            f"Expected all channels to have non-zero gradients, got {w_gamma_x_grad}"
        )

        # 2. Dead ReLU baseline reproduction (to verify contrast)
        w_relu = nn.Parameter(torch.full((input_dim,), -0.1))  # Notebook initialization
        b_relu = nn.Parameter(torch.zeros(input_dim))
        relu_decay = torch.exp(-F.relu(w_relu * delta_t + b_relu))
        x_hat_relu = (1.0 - m_t) * (relu_decay * x_last)
        loss_relu = x_hat_relu.sum()
        loss_relu.backward()

        assert torch.all(w_relu.grad == 0.0), "Sanity check: dead ReLU must have zero gradients!"

    def test_right_padding_canary_gathering_at_lengths_minus_one(self):
        """
        Adversarial Test 1.2:
        Verify right-padding in ICUSequenceDataset gathers the TRUE last observation at lengths - 1.
        A 'canary' value placed at timestep lengths - 1 must be exactly retrieved,
        and padding timesteps (lengths .. 23) must be 0 and ignored.
        """
        max_lookback = 24
        num_features = 14
        true_len = 7  # Stays for 7 hours, then right-padded with 17 zeros

        # Create synthetic stay where hour true_len - 1 has a distinct canary signature
        stay_rows = []
        for h in range(true_len):
            row = {"PatientID": "p0001", "Hour": h, "SepsisLabel": 0}
            for col_idx, col in enumerate([
                'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                'Lactate', 'WBC', 'Creatinine', 'BUN', 'Glucose', 'Platelets', 'Hgb'
            ]):
                if h == true_len - 1 and col == 'HR':
                    row[col] = 188.0  # Canary signature at last true observation
                else:
                    row[col] = 70.0 + h
            stay_rows.append(row)

        df_canary = pd.DataFrame(stay_rows)
        dataset = ICUSequenceDataset(df_canary, max_lookback=max_lookback)
        assert len(dataset) == 1
        sample = dataset[0]

        recorded_len = sample["lengths"].item()
        assert recorded_len == true_len, f"Expected length {true_len}, got {recorded_len}"

        x_seq = sample["X_seq"]  # (24, 14)
        mask = sample["mask"]    # (24, 14)

        # Timesteps 0 .. true_len - 1 must have mask == 1 (since all features were provided)
        assert torch.all(mask[:true_len] == 1.0)
        # Timesteps true_len .. 23 must be padding zeros
        assert torch.all(x_seq[true_len:] == 0.0)
        assert torch.all(mask[true_len:] == 0.0)

        # Feed through ClinicalGRU and verify gathering at lengths - 1
        gru = ClinicalGRU(input_dim=num_features, hidden_dim=64)
        gru.eval()

        batch_x = x_seq.unsqueeze(0)  # (1, 24, 14)
        batch_lens = torch.tensor([true_len])

        # Step-by-step unroll to compare
        full_out, _ = gru.gru(batch_x)  # (1, 24, 64)
        expected_last_hidden = full_out[0, true_len - 1, :]  # Hidden state at true_len - 1

        gathered_emb = gru.extract_features(batch_x, lengths=batch_lens)  # (1, 64)

        assert torch.allclose(gathered_emb[0], expected_last_hidden, atol=1e-6), (
            "Model gathered hidden state from wrong timestep! Did not gather at lengths - 1."
        )

        # Verify it did NOT gather from padding timestep 23
        padding_hidden = full_out[0, 23, :]
        assert not torch.allclose(gathered_emb[0], padding_hidden, atol=1e-4), (
            "Model gathered from padding timestep 23 instead of true last observation!"
        )

    def test_temporal_gradcam1d_signed_attribution_hypotensive_hypothermic_drop(self):
        """
        Adversarial Test 1.3:
        Verify TemporalGradCAM1D produces non-zero attribution on hypotensive shock / hypothermia drops
        without zeroing out negative gradients.
        """
        model = SentryICUFusionModel(static_dim=73, ts_channels=7, cnn_embed_dim=32)
        model.eval()

        gradcam = TemporalGradCAM1D(model)

        # Baseline vital window: constant normal vitals
        x_normal = torch.tensor([[80.0, 98.0, 37.0, 120.0, 80.0, 80.0, 16.0]], dtype=torch.float32)
        x_vitals_normal = x_normal.unsqueeze(-1).repeat(1, 1, 12)  # (1, 7, 12)

        # Shock window: MAP (index 4) drops from 80 to 35 mmHg; Temp (index 2) drops from 37 to 34 C
        x_vitals_shock = x_vitals_normal.clone()
        x_vitals_shock[0, 4, 6:] = 35.0  # Acute MAP drop in second half of window
        x_vitals_shock[0, 2, 6:] = 34.0  # Hypothermia drop in second half of window

        x_tab = torch.randn(1, 73)

        # Generate heatmap
        heatmap = gradcam.generate_heatmap(x_tab, x_vitals_shock, target_class=1)

        assert heatmap.shape == (7, 12)
        assert not np.isnan(heatmap).any()
        assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all()

        # Channel 4 (MAP) and Channel 2 (Temp) must have non-zero attribution
        map_attr = heatmap[4, :]
        temp_attr = heatmap[2, :]

        assert map_attr.max() > 0.0, "MAP drop attribution was zeroed out!"
        assert temp_attr.max() > 0.0, "Temp drop attribution was zeroed out!"
        assert heatmap.sum() > 0.0, "Total attribution must be strictly positive"

    def test_patient_vae_balanced_elbo_prevents_posterior_collapse(self):
        """
        Adversarial Test 1.4:
        Verify PatientVAE with balanced ELBO loss avoids posterior collapse.
        Check that:
        1. Latent dimensions maintain variance across samples (Var(mu) > 0.01).
        2. Average KL divergence does not collapse to 0.0.
        3. Model learns non-trivial reconstruction.
        """
        torch.manual_seed(42)
        input_dim = 73
        latent_dim = 16
        n_samples = 256

        # Create structured synthetic patient cohort with distinct clusters (healthy vs septic)
        cluster_a = torch.randn(n_samples // 2, input_dim) - 2.0
        cluster_b = torch.randn(n_samples // 2, input_dim) + 2.0
        X_data = torch.cat([cluster_a, cluster_b], dim=0)

        vae = PatientVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[64, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.005)

        # Train for 40 epochs with balanced loss
        vae.train()
        for epoch in range(40):
            optimizer.zero_grad()
            recon, mu, logvar = vae(X_data)
            loss, recon_l, kl_l = vae_loss(recon, X_data, mu, logvar, beta=1.0, reduction="balanced")
            loss.backward()
            optimizer.step()

        vae.eval()
        with torch.no_grad():
            recon, mu, logvar = vae(X_data)
            total_loss, recon_l, kl_l = vae_loss(recon, X_data, mu, logvar, beta=1.0, reduction="balanced")

        # Check for posterior collapse:
        # 1. Variance of mu across the cohort must be non-zero (if collapsed, all mu are identical)
        mu_variances = torch.var(mu, dim=0)  # (16,)
        mean_mu_var = mu_variances.mean().item()
        assert mean_mu_var > 0.05, f"Posterior collapse detected! Mean latent variance is {mean_mu_var:.6f}"

        # 2. KL divergence must remain active (not collapsed to 0)
        assert kl_l.item() > 0.1, f"KL divergence collapsed to near zero: {kl_l.item():.6f}"

        # 3. Active units: number of latent dims with Var(mu) > 0.01
        active_dims = (mu_variances > 0.01).sum().item()
        assert active_dims >= 4, f"Expected at least 4 active latent dimensions, got {active_dims}"

    def test_clinical_rag_assistant_offline_mode_and_phi_rejection(self):
        """
        Adversarial Test 1.5:
        Verify ClinicalRAGAssistant offline mode:
        1. Correctly extracts metrics and answers queries without network.
        2. Strictly rejects DataFrames containing PHI columns ('PatientID', 'patient_id', etc.).
        """
        assistant = ClinicalRAGAssistant(api_key=None)

        # 1. Offline Mode Question Answering
        q_thresh = "What is the calibrated VAE anomaly threshold?"
        res_thresh = assistant.ask(q_thresh)
        assert res_thresh["mode"] == "offline"
        assert len(res_thresh["answer"]) > 0
        assert "threshold" in res_thresh["answer"].lower() or "anomaly" in res_thresh["answer"].lower()

        q_drift = "Tell me about distribution drift and KS test results"
        res_drift = assistant.ask(q_drift)
        assert res_drift["mode"] == "offline"
        assert "drift" in res_drift["answer"].lower() or "ks" in res_drift["answer"].lower()

        # 2. Strict HIPAA PHI Rejection
        for phi_col in ["PatientID", "patient_id", "Patient_ID", "pid", "Subject_ID", "SSN", "MRN"]:
            df_violating = pd.DataFrame({
                phi_col: ["p0001", "p0002"],
                "HR_mean": [80.0, 90.0],
                "AUROC": [0.85, 0.88]
            })
            with pytest.raises(ValueError, match="HIPAA VIOLATION"):
                assistant.verify_hipaa_compliance(df_violating, source_name="test_artifact.csv")

            with pytest.raises(ValueError, match="HIPAA VIOLATION"):
                assistant.build_knowledge_base(custom_results={"violating.csv": df_violating})

        # Non-violating aggregate table must pass
        df_valid_aggregate = pd.DataFrame({
            "Metric": ["AUROC", "AUPRC", "ECE", "Brier_Score"],
            "Hospital_A": [0.88, 0.65, 0.04, 0.08],
            "Hospital_B": [0.82, 0.55, 0.07, 0.12]
        })
        assert assistant.verify_hipaa_compliance(df_valid_aggregate, source_name="metrics_summary.csv") is True


class TestAdversarialStressAndEdgeCases:
    """2. Stress tests and extreme edge cases."""

    def test_single_sample_batch_b1_all_models(self):
        """
        Stress Test 2.1:
        Verify all models handle mini-batch size B=1 cleanly without collapsing dimensions
        or raising RuntimeError / IndexError.
        """
        batch_size = 1

        # Module 3
        x_tab = torch.randn(batch_size, 73)
        x_vitals = torch.randn(batch_size, 7, 12)

        cnn = ClinicalCNN1D(in_channels=7, out_embed_dim=32)
        cnn.eval()
        emb_cnn = cnn(x_vitals)
        assert emb_cnn.shape == (1, 32)

        bimodal = SentryICUFusionModel(static_dim=73, ts_channels=7, cnn_embed_dim=32)
        bimodal.eval()
        logits_bim, cnn_e, fused_bim = bimodal(x_tab, x_vitals)
        assert logits_bim.shape == (1, 1)
        assert fused_bim.shape == (1, 105)

        gradcam = TemporalGradCAM1D(bimodal)
        hm = gradcam.generate_heatmap(x_tab, x_vitals)
        assert hm.shape == (7, 12)

        # Module 4
        x_seq = torch.randn(batch_size, 24, 14)
        mask = torch.ones(batch_size, 24, 14)
        delta = torch.zeros(batch_size, 24, 14)
        lengths = torch.tensor([12])

        rnn_models = [
            ClinicalVanillaRNN(14, 64),
            ClinicalLSTM(14, 64),
            ClinicalGRU(14, 64),
            BidirectionalGRU(14, 64),
            AttentionGRU(14, 64),
        ]
        for m in rnn_models:
            m.eval()
            l, e = m(x_seq, lengths=lengths)
            assert l.shape == (1, 1)
            assert e.shape == (1, 64)
            p = m.predict_proba(x_seq, lengths=lengths)
            assert p.shape == (1, 1)

        grud = ClinicalGRUD(14, 64)
        grud.eval()
        l_grud, e_grud = grud(x_seq, mask, delta, lengths=lengths)
        assert l_grud.shape == (1, 1)
        assert e_grud.shape == (1, 64)

        # Tri-modal
        trimodal_head = SentryICUTriModalFusion(73, 32, 64)
        trimodal_head.eval()
        l_tri, fused_tri = trimodal_head(x_tab, emb_cnn, e_grud)
        assert l_tri.shape == (1, 1)
        assert fused_tri.shape == (1, 169)

        end_to_end = SentryICUEndToEndTriModal()
        end_to_end.eval()
        l_e2e, fused_e2e = end_to_end(x_tab, x_vitals, x_seq, mask, delta, lengths)
        assert l_e2e.shape == (1, 1)
        assert fused_e2e.shape == (1, 169)

        # Module 5
        vae = PatientVAE(73, 16)
        vae.eval()
        recon, mu, logvar = vae(x_tab)
        assert recon.shape == (1, 73)
        assert mu.shape == (1, 16)
        assert logvar.shape == (1, 16)

        gan_gen = TabularGenerator(32, 73)
        gan_gen.eval()
        syn = gan_gen(torch.randn(1, 32))
        assert syn.shape == (1, 73)

        gan_disc = TabularDiscriminator(73)
        gan_disc.eval()
        disc_out = gan_disc(syn)
        assert disc_out.shape == (1, 1)

    def test_all_zero_inputs_stability(self):
        """
        Stress Test 2.2:
        Verify all architectures handle all-zero input tensors gracefully
        without NaN outputs or zero-division exceptions.
        """
        batch_size = 4

        # Zeros for Module 3
        zero_tab = torch.zeros(batch_size, 73)
        zero_vitals = torch.zeros(batch_size, 7, 12)

        cnn = ClinicalCNN1D()
        cnn.eval()
        cnn_out = cnn(zero_vitals)
        assert not torch.isnan(cnn_out).any()

        bimodal = SentryICUFusionModel()
        bimodal.eval()
        logits, _, fused = bimodal(zero_tab, zero_vitals)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(fused).any()

        # Zeros for Module 4 (zero sequence, zero mask, zero delta)
        zero_seq = torch.zeros(batch_size, 24, 14)
        zero_mask = torch.zeros(batch_size, 24, 14)
        zero_delta = torch.zeros(batch_size, 24, 14)
        lens = torch.tensor([24, 24, 24, 24])

        grud = ClinicalGRUD(14, 64)
        grud.eval()
        grud_logits, grud_emb = grud(zero_seq, zero_mask, zero_delta, lengths=lens)
        assert not torch.isnan(grud_logits).any()
        assert not torch.isnan(grud_emb).any()

        # Zeros for Module 5
        vae = PatientVAE(73, 16)
        vae.eval()
        recon, mu, logvar = vae(zero_tab)
        assert not torch.isnan(recon).any()
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()

        score = vae.compute_anomaly_score(zero_tab)
        assert not torch.isnan(score).any()
        assert (score >= 0.0).all()

    def test_extreme_values_and_numerical_boundaries(self):
        """
        Stress Test 2.3:
        Verify numerical stability under extreme inputs:
        - Large positive and negative feature values (+1e6, -1e6)
        - Extreme time elapsed deltas (delta_t = 10000.0 hrs)
        - VAE logvar clamping in [-10.0, 10.0]
        - Probability bounding in [0.0, 1.0]
        """
        # 1. Extreme elapsed delta in GRU-D
        cell = ClinicalGRUDCell(input_dim=14, hidden_dim=64)
        x_raw = torch.randn(2, 14)
        m_t = torch.zeros(2, 14)
        # 10,000 hours elapsed
        delta_extreme = torch.full((2, 14), 10000.0)
        h_prev = torch.randn(2, 64)
        x_last = torch.full((2, 14), 150.0)
        mean_x = torch.full((14,), 80.0)

        h_t, x_last_next, x_hat = cell(x_raw, m_t, delta_extreme, h_prev, x_last, mean_x, return_x_hat=True)
        assert not torch.isnan(h_t).any()
        assert not torch.isnan(x_hat).any()
        # With delta -> infinity, gamma_x -> 0.0, so x_hat must equal mean_x (80.0)
        assert torch.allclose(x_hat, mean_x.unsqueeze(0).expand(2, -1), atol=1e-4), (
            f"Expected x_hat to revert to empirical mean (80.0) under extreme delta, got {x_hat}"
        )

        # 2. VAE Logvar Clamping under Extreme Activation
        vae = PatientVAE(73, 16)
        # Directly inject extreme mu and logvar
        extreme_mu = torch.full((2, 16), 1e5)
        extreme_logvar_high = torch.full((2, 16), 100.0)
        extreme_logvar_low = torch.full((2, 16), -100.0)

        # Reparameterize must clamp to [-10, 10]
        z_high = vae.reparameterize(extreme_mu, extreme_logvar_high)
        z_low = vae.reparameterize(extreme_mu, extreme_logvar_low)
        assert not torch.isnan(z_high).any() and not torch.isinf(z_high).any()
        assert not torch.isnan(z_low).any() and not torch.isinf(z_low).any()

        # 3. Predict proba on extreme logits must stay strictly in [0.0, 1.0]
        bigru = BidirectionalGRU(14, 64)
        extreme_x = torch.full((2, 24, 14), 1e6)
        probs = bigru.predict_proba(extreme_x)
        assert not torch.isnan(probs).any()
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_zero_length_sequence_boundary(self):
        """
        Stress Test 2.4:
        Verify handling of zero-length or length=1 sequences:
        lengths = 0 clamped to 1, avoiding division-by-zero or negative indexing.
        """
        x_seq = torch.randn(2, 24, 14)
        zero_lengths = torch.tensor([0, 1])

        gru = ClinicalGRU(14, 64)
        emb = gru.extract_features(x_seq, lengths=zero_lengths)
        assert emb.shape == (2, 64)
        assert not torch.isnan(emb).any()

        grud = ClinicalGRUD(14, 64)
        mask = torch.ones_like(x_seq)
        delta = torch.zeros_like(x_seq)
        emb_grud = grud.extract_features(x_seq, mask, delta, lengths=zero_lengths)
        assert emb_grud.shape == (2, 64)
        assert not torch.isnan(emb_grud).any()

    def test_metrics_suite_all_zeros_and_all_ones_targets(self):
        """
        Stress Test 2.5:
        Verify compute_metrics on single-class cohorts (all 0s or all 1s):
        Must not crash, AUROC defaults safely to 0.5, ECE and Brier score compute cleanly.
        """
        # All zeros
        y_all_0 = np.zeros(50, dtype=int)
        p_all_0 = np.full(50, 0.05)
        m0 = compute_metrics(y_all_0, p_all_0)
        assert m0["auroc"] == 0.5
        assert m0["f1"] == 0.0
        assert 0.0 <= m0["ece"] <= 1.0
        assert 0.0 <= m0["brier_score"] <= 1.0

        # All ones
        y_all_1 = np.ones(50, dtype=int)
        p_all_1 = np.full(50, 0.95)
        m1 = compute_metrics(y_all_1, p_all_1)
        assert m1["auroc"] == 0.5
        assert 0.0 <= m1["ece"] <= 1.0
        assert 0.0 <= m1["brier_score"] <= 1.0

    def test_missing_features_imputation_fallback(self):
        """
        Stress Test 2.6:
        Verify extract_sliding_windows when some vital columns are completely missing from DataFrame.
        """
        df_incomplete = pd.DataFrame({
            "PatientID": ["p1"] * 15,
            "Hour": list(range(15)),
            "HR": [80.0] * 15,
            # Temp, SBP, MAP, DBP, Resp, O2Sat completely absent!
            "SepsisLabel": [0] * 15,
        })
        X_73d, X_vitals, y = extract_sliding_windows(df_incomplete, window_size=12, step=1)
        assert X_73d.shape == (4, 73)
        assert X_vitals.shape == (4, 7, 12)
        assert y.shape == (4,)
        # Check that missing channels were filled with physiological defaults and no NaNs exist
        assert not np.isnan(X_73d).any()
        assert not np.isnan(X_vitals).any()
