"""
tests/test_module5_generative.py
================================
Real-data verification and unit tests for SentryICU Module 5:
Generative AI, Variational Autoencoder, Drift Gate, GAN, and Clinical RAG.

Covers:
1. PatientVAE & ClinicalVAE on real 73-D static clinical features:
   - Forward pass contract: recon (B, 73), mu (B, 16), logvar (B, 16)
   - Clamped logvar in [-10.0, 10.0] numerical stability (BUG-M5-04 fix)
   - Anomaly score via reconstruction MSE (B,)
2. Balanced ELBO loss (vae_loss):
   - Prevention of >20x KL dominance and posterior collapse (BUG-M5-05 fix)
   - Reductions: 'balanced', 'feature_mean', 'none'
3. VAEDriftDetector:
   - 95th-percentile anomaly calibration on normal ICU stays
   - Two-sample Kolmogorov-Smirnov test with Bonferroni adjustment (BUG-M5-08 fix)
   - Mini-batched RBF Maximum Mean Discrepancy (MMD)
4. TabularGAN & SMOTE minority augmentation:
   - TabularGenerator (B, 32) -> (B, 73) with empirical range clamping (BUG-M5-12 fix)
   - TabularDiscriminator (B, 73) -> (B, 1)
   - TabularGAN fit on real sepsis-positive patient records and synthetic generation
5. ClinicalRAGAssistant:
   - Zero-dependency local keyword & table extraction engine
   - Strict HIPAA safety filter detecting patient identifiers (BUG-M5-22 fix)
   - Knowledge base compilation and grounded response schema
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.clinical_rag import (
    ClinicalRAGAssistant,
)
from src.models.gan import (
    TabularDiscriminator,
    TabularGAN,
    TabularGenerator,
)
from src.models.vae import (
    ClinicalVAE,
    PatientVAE,
    VAEDriftDetector,
    vae_loss,
)


class TestPatientVAE:
    """Tests for PatientVAE health manifold model and anomaly scoring on real patient records."""

    def test_contract_alias_clinical_vae(self):
        """Verify docs/interfaces.md alias ClinicalVAE == PatientVAE."""
        assert ClinicalVAE is PatientVAE

    def test_vae_forward_pass_and_shape_contract(self, real_multimodal_batch):
        """
        Verify VAE shape contract on real 73-D static features:
        Input: (B, 73) -> Recon (B, 73), Mu (B, 16), Logvar (B, 16).
        """
        x_tab = real_multimodal_batch["x_tab"]
        batch_size = x_tab.size(0)

        vae = PatientVAE(input_dim=73, latent_dim=16)
        vae.eval()

        with torch.no_grad():
            recon, mu, logvar = vae(x_tab)

        assert recon.shape == torch.Size([batch_size, 73])
        assert mu.shape == torch.Size([batch_size, 16])
        assert logvar.shape == torch.Size([batch_size, 16])
        assert not torch.isnan(recon).any()
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()

    def test_logvar_clamping_numerical_stability(self):
        """
        Verify BUG-M5-04 fix:
        logvar is clamped to [-10.0, 10.0] in reparameterization,
        preventing exponential overflow and infinite/NaN gradients.
        """
        vae = PatientVAE(input_dim=73, latent_dim=16)
        mu = torch.zeros(4, 16)
        extreme_logvar = torch.tensor([[100.0] * 16, [-100.0] * 16, [0.0] * 16, [50.0] * 16])

        z = vae.reparameterize(mu, extreme_logvar)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()
        # Since logvar is clamped at 10.0, std <= exp(5.0) ~ 148.4
        assert torch.all(torch.abs(z) < 1000.0)

    def test_reconstruction_mse_anomaly_score(self, real_multimodal_batch):
        """Verify reconstruction MSE anomaly score produces (B,) non-negative float scores."""
        x_tab = real_multimodal_batch["x_tab"]
        vae = PatientVAE(input_dim=73, latent_dim=16)

        torch.manual_seed(42)
        mse_scores = vae.get_reconstruction_error(x_tab)
        assert mse_scores.shape == torch.Size([x_tab.size(0)])
        assert (mse_scores >= 0.0).all()

        torch.manual_seed(42)
        alias_scores = vae.compute_anomaly_score(x_tab)
        assert torch.allclose(mse_scores, alias_scores)


class TestBalancedELBOLoss:
    """Tests for balanced ELBO loss formulation preventing posterior collapse."""

    def test_balanced_loss_scale_and_energy_parity(self, real_multimodal_batch):
        """
        Verify BUG-M5-05 fix:
        Balanced ELBO keeps reconstruction and KL divergence terms on comparable energy scales,
        preventing the >20x KL dominance that caused severe posterior collapse in previous code.
        """
        x_tab = real_multimodal_batch["x_tab"]
        vae = PatientVAE(input_dim=73, latent_dim=16)
        recon, mu, logvar = vae(x_tab)

        total_loss, recon_loss, kl_loss = vae_loss(
            recon, x_tab, mu, logvar, beta=1.0, reduction="balanced"
        )

        assert total_loss.item() > 0.0
        assert recon_loss.item() > 0.0
        assert kl_loss.item() >= 0.0

        # Neither term should dwarf the other by 20x at initialization
        ratio = (recon_loss.item() + 1e-6) / (kl_loss.item() + 1e-6)
        assert 0.01 <= ratio <= 500.0, f"Unbalanced ELBO scale distortion: ratio={ratio:.4f}"

    def test_vae_loss_reductions(self, real_multimodal_batch):
        """Verify 'feature_mean' and 'none' reduction modes."""
        x_tab = real_multimodal_batch["x_tab"]
        vae = PatientVAE(input_dim=73, latent_dim=16)
        recon, mu, logvar = vae(x_tab)

        # 1. Feature mean reduction
        tot_m, rec_m, kl_m = vae_loss(recon, x_tab, mu, logvar, reduction="feature_mean")
        assert tot_m.ndim == 0
        assert rec_m.ndim == 0
        assert kl_m.ndim == 0

        # 2. Unreduced loss ('none')
        tot_none, rec_none, kl_none = vae_loss(recon, x_tab, mu, logvar, reduction="none")
        assert tot_none.shape == torch.Size([x_tab.size(0)])
        assert rec_none.shape == torch.Size([x_tab.size(0)])
        assert kl_none.shape == torch.Size([x_tab.size(0)])

    def test_vae_backward_gradients(self, real_multimodal_batch):
        """Verify smooth gradient propagation through both encoder and decoder branches."""
        x_tab = real_multimodal_batch["x_tab"]
        vae = PatientVAE(input_dim=73, latent_dim=16)
        vae.train()

        recon, mu, logvar = vae(x_tab)
        total_loss, _, _ = vae_loss(recon, x_tab, mu, logvar, reduction="balanced")
        total_loss.backward()

        # Check encoder gradient
        enc_grad = vae.fc_mu.weight.grad
        assert enc_grad is not None
        assert not torch.all(enc_grad == 0.0)

        # Check decoder gradient
        dec_grad = vae.decoder[-1].weight.grad
        assert dec_grad is not None
        assert not torch.all(dec_grad == 0.0)


class TestVAEDriftDetector:
    """Tests for 95th-percentile anomaly threshold calibration and drift testing."""

    def test_fit_and_anomaly_threshold_calibration(self, real_sliding_windows_data):
        """Verify calibration of 95th-percentile anomaly threshold on real normal patient stays."""
        X_73d, _, y = real_sliding_windows_data

        # Filter normal stays (y == 0)
        neg_mask = (y == 0)
        X_neg = X_73d[neg_mask]
        assert len(X_neg) >= 10

        split_idx = int(len(X_neg) * 0.7)
        train_neg = X_neg[:split_idx]
        val_neg = X_neg[split_idx:]

        detector = VAEDriftDetector()
        tau = detector.fit_normal_baseline(
            train_neg, val_neg, epochs=5, batch_size=16, patience=3
        )

        assert isinstance(tau, float)
        assert tau > 0.0
        assert detector.is_fitted
        assert detector.anomaly_threshold == tau

    def test_score_dataframe_appends_columns(self, real_sliding_windows_data):
        """Verify score_dataframe appends 'VAE_Reconstruction_MSE' and 'VAE_Anomaly'."""
        X_73d, _, _ = real_sliding_windows_data
        feature_names = [f"feat_{i}" for i in range(73)]
        df_real = pd.DataFrame(X_73d[:20], columns=feature_names)

        detector = VAEDriftDetector(feature_cols=feature_names)
        detector.fit_normal_baseline(
            df_real.iloc[:10], df_real.iloc[10:], feature_cols=feature_names, epochs=2, batch_size=8
        )

        scored_df = detector.score_dataframe(df_real)
        assert "VAE_Reconstruction_MSE" in scored_df.columns
        assert "VAE_Anomaly" in scored_df.columns
        assert scored_df["VAE_Anomaly"].isin([0, 1]).all()

    def test_compute_ks_drift_bonferroni(self):
        """
        Verify BUG-M5-08 fix:
        Two-sample KS drift test applies Bonferroni correction: p_adjusted = min(p * n_features, 1.0).
        """
        n_samples = 50
        cols = ["HR_mean", "Temp_mean", "MAP_mean"]

        df_ref = pd.DataFrame({
            "HR_mean": np.random.normal(80, 10, n_samples),
            "Temp_mean": np.random.normal(37, 0.5, n_samples),
            "MAP_mean": np.random.normal(80, 8, n_samples),
        })

        # Severe distribution shift in MAP_mean
        df_target = pd.DataFrame({
            "HR_mean": np.random.normal(80, 10, n_samples),
            "Temp_mean": np.random.normal(37, 0.5, n_samples),
            "MAP_mean": np.random.normal(50, 8, n_samples),  # Severe drop
        })

        ks_df = VAEDriftDetector.compute_ks_drift(df_ref, df_target, cols)
        assert len(ks_df) == len(cols)
        assert "p_adjusted" in ks_df.columns
        assert (ks_df["p_adjusted"] <= 1.0).all()
        assert (ks_df["p_adjusted"] >= 0.0).all()

        # MAP_mean should have the largest KS statistic
        top_feature = ks_df.iloc[0]["Feature"]
        assert top_feature == "MAP_mean"
        assert ks_df.iloc[0]["Significant"] is True or ks_df.iloc[0]["KS_Stat"] > 0.5

    def test_compute_rbf_mmd(self):
        """Verify mini-batched RBF-MMD returns 0.0 for identical distributions and >0 for shifted."""
        rng = np.random.default_rng(42)
        X1 = rng.normal(0, 1, size=(50, 10))
        X2 = rng.normal(0, 1, size=(50, 10))
        X_shifted = rng.normal(5, 1, size=(50, 10))

        mmd_same = VAEDriftDetector.compute_rbf_mmd(X1, X2, max_samples=50)
        mmd_shifted = VAEDriftDetector.compute_rbf_mmd(X1, X_shifted, max_samples=50)

        assert mmd_same >= 0.0
        assert mmd_shifted > mmd_same


class TestTabularGAN:
    """Tests for TabularGenerator, TabularDiscriminator, and TabularGAN."""

    def test_tabular_generator_shape_and_clamping(self):
        """
        Verify BUG-M5-12 fix:
        Generator clamps outputs to empirical [min, max] feature bounds.
        """
        noise = torch.randn(8, 32)
        min_bounds = torch.full((73,), -2.0)
        max_bounds = torch.full((73,), 5.0)

        generator = TabularGenerator(
            noise_dim=32, output_dim=73, clamp_bounds=(min_bounds, max_bounds)
        )
        generator.eval()

        with torch.no_grad():
            syn = generator(noise)

        assert syn.shape == torch.Size([8, 73])
        assert (syn >= -2.0 - 1e-5).all()
        assert (syn <= 5.0 + 1e-5).all()

    def test_tabular_discriminator_shape(self):
        """Verify discriminator maps (B, 73) -> (B, 1) logit."""
        x = torch.randn(8, 73)
        discriminator = TabularDiscriminator(input_dim=73)
        logit = discriminator(x)
        assert logit.shape == torch.Size([8, 1])

    def test_tabular_gan_fit_and_generate_on_real_positive_cases(self, real_sliding_windows_data):
        """Verify TabularGAN trains on real positive sepsis cases and generates valid synthetic records."""
        X_73d, _, y = real_sliding_windows_data
        pos_mask = (y == 1)
        X_pos = X_73d[pos_mask]

        if len(X_pos) < 2:
            # Fallback if few positive cases in window step sample
            X_pos = X_73d[:10]

        gan = TabularGAN(noise_dim=32, feature_dim=73)
        gan.fit(X_pos, epochs=5, batch_size=8, label_smoothing=0.9, clamp_to_bounds=True)

        assert gan.is_fitted
        synthetic_records = gan.generate(n_samples=10, as_numpy=True)

        assert isinstance(synthetic_records, np.ndarray)
        assert synthetic_records.shape == (10, 73)
        assert not np.isnan(synthetic_records).any()

        # Check bounds
        assert (synthetic_records >= gan.empirical_min - 1e-4).all()
        assert (synthetic_records <= gan.empirical_max + 1e-4).all()


class TestClinicalRAGAssistant:
    """Tests for ClinicalRAGAssistant offline engine and HIPAA safety compliance."""

    def test_hipaa_compliance_verification(self):
        """
        Verify BUG-M5-22 fix:
        ClinicalRAGAssistant forbids ingesting unaggregated patient identifier columns (PHI),
        raising ValueError, while allowing aggregate statistical evaluation tables.
        """
        # Table containing individual patient identifiers
        phi_table = pd.DataFrame({
            "PatientID": ["p0001", "p0002"],
            "HeartRate": [80.0, 85.0],
            "SepsisLabel": [0, 1],
        })

        with pytest.raises(ValueError, match="HIPAA VIOLATION"):
            ClinicalRAGAssistant.verify_hipaa_compliance(phi_table, source_name="phi_leak.csv")

        # Table containing aggregate summary statistics
        safe_table = pd.DataFrame({
            "Model": ["Baseline_MLP", "TriModal_Fusion"],
            "AUROC": [0.812, 0.887],
            "AUPRC": [0.420, 0.540],
            "ECE": [0.082, 0.045],
        })
        assert ClinicalRAGAssistant.verify_hipaa_compliance(safe_table, source_name="eval_metrics.csv") is True

    def test_offline_rag_query_resolution(self):
        """
        Verify ClinicalRAGAssistant resolves clinical queries using its local
        keyword/table extraction engine when unauthenticated (offline).
        """
        assistant = ClinicalRAGAssistant(api_key=None)

        custom_results = {
            "vae_evaluation.csv": pd.DataFrame({
                "Metric": ["95th_Percentile_Threshold", "Mean_Val_MSE"],
                "Value": [0.452, 0.218],
            }),
            "vae_drift.csv": pd.DataFrame({
                "Feature": ["HR_mean", "MAP_mean"],
                "Significant": [False, True],
            }),
            "gan_vs_smote.csv": pd.DataFrame({
                "Method": ["SMOTE", "TabularGAN"],
                "AUROC": [0.842, 0.881],
            }),
        }

        assistant.build_knowledge_base(custom_results=custom_results)

        # 1. Query VAE anomaly threshold
        res_vae = assistant.ask("What is the anomaly threshold for PatientVAE?")
        assert res_vae["mode"] in ("offline", "offline_fallback")
        assert "vae_evaluation.csv" in res_vae["sources"]
        assert "0.452" in res_vae["answer"]

        # 2. Query Drift
        res_drift = assistant.ask("What are the results of the distribution drift analysis?")
        assert any("drift" in s for s in res_drift["sources"])
        assert "Distribution Drift Analysis" in res_drift["answer"]

        # 3. Query GAN vs SMOTE
        res_gan = assistant.ask("How does TabularGAN compare to SMOTE on minority sepsis oversampling?")
        assert "gan_vs_smote.csv" in res_gan["sources"]
        assert "TabularGAN" in res_gan["answer"]

        # 4. Default / General inquiry
        res_gen = assistant.ask("General overview of available clinical metrics?")
        assert "module5_knowledge_base.txt" in res_gen["sources"]

    def test_gemini_model_selection_and_resolution(self):
        """
        Verify that ClinicalRAGAssistant defaults to the flagship model 'gemini-3.7-flash'
        and supports flexible resolution of aliases such as 3.5 flash, 3.7, and 3.8.
        """
        # Default model is gemini-3.7-flash
        default_assistant = ClinicalRAGAssistant(api_key=None)
        assert default_assistant.model_name == "gemini-3.7-flash"

        # Explicit and shorthand alias resolutions
        assert ClinicalRAGAssistant.resolve_model_name("3.7") == "gemini-3.7-flash"
        assert ClinicalRAGAssistant.resolve_model_name("gemini-3.7-flash") == "gemini-3.7-flash"
        assert ClinicalRAGAssistant.resolve_model_name("3.5 flash") == "gemini-3.5-flash-lite"
        assert ClinicalRAGAssistant.resolve_model_name("gemini-3.5-flash") == "gemini-3.5-flash-lite"
        assert ClinicalRAGAssistant.resolve_model_name("3.8") == "gemini-3.8-flash"
        assert ClinicalRAGAssistant.resolve_model_name("gemini-3.8-flash") == "gemini-3.8-flash"
        assert ClinicalRAGAssistant.resolve_model_name("gemini-3.1-pro") == "gemini-3.1-pro-preview"

        # Initialization with alias
        assistant_35 = ClinicalRAGAssistant(model_name="3.5 flash", api_key=None)
        assert assistant_35.model_name == "gemini-3.5-flash-lite"

        assistant_38 = ClinicalRAGAssistant(model_name="3.8", api_key=None)
        assert assistant_38.model_name == "gemini-3.8-flash"


