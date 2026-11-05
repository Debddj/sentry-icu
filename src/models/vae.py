"""
src/models/vae.py
=================
Variational Autoencoder (PatientVAE), Health Manifold Modeling, and Cross-Hospital Drift Gate.

Adheres strictly to docs/interfaces.md contract:
- Class: PatientVAE(nn.Module) [Alias: ClinicalVAE = PatientVAE]
- Input: (Batch_Size, 73)
- Outputs:
    - Reconstructed input: (Batch_Size, 73)
    - Latent distribution parameters: mu (Batch_Size, 16), logvar (Batch_Size, 16)
    - Anomaly / Drift Score: Mean Squared Error (MSE) reconstruction error.

Key Improvements & Bug Fixes:
- Clamped logvar in [-10.0, 10.0] preventing numerical overflow in reparameterization.
- Balanced ELBO loss formulation eliminating >20x KL dominance and posterior collapse.
- Mini-batched evaluation preventing CUDA out-of-memory errors on large cohorts.
- Robust VAEDriftDetector with non-parametric 95th-percentile anomaly calibration,
  Bonferroni-adjusted two-sample Kolmogorov-Smirnov test, and RBF Maximum Mean Discrepancy (MMD).
"""

from typing import List, Optional, Tuple, Union
import copy
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PatientVAE(nn.Module):
    """
    Variational Autoencoder for ICU Health Manifold Modeling and Anomaly Gating.

    Parameters
    ----------
    input_dim : int, default=73
        Number of input clinical features.
    latent_dim : int, default=16
        Dimensionality of the bottleneck latent representation.
    hidden_dims : list of int, default=[64, 32]
        Hidden layer sizes for encoder and decoder.
    """

    def __init__(
        self,
        input_dim: int = 73,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32]

        # Encoder: input_dim -> hidden_0 -> hidden_1
        encoder_layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder: latent_dim -> hidden_1 -> hidden_0 -> input_dim
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(self.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps input tensor to latent Gaussian parameters (mu, logvar).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples latent representation z = mu + sigma * eps using the reparameterization trick.
        Clamps logvar to [-10.0, 10.0] for guaranteed numerical stability.
        """
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vector z back into the 73-dimensional feature space.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (reconstruction, mu, logvar).
        Shapes:
            reconstruction: (Batch_Size, 73)
            mu: (Batch_Size, 16)
            logvar: (Batch_Size, 16)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes deterministic or mean reconstruction without backpropagation.
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            return self.decode(mu)

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-record Mean Squared Error across all features.
        Output shape: (Batch_Size,)
        """
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            mse = (x - reconstruction).pow(2).mean(dim=1)
            return mse

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for get_reconstruction_error adhering to system anomaly interfaces.
        """
        return self.get_reconstruction_error(x)


# Interface alias per docs/interfaces.md
ClinicalVAE = PatientVAE


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "balanced",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evidence Lower Bound (ELBO) Loss for Variational Autoencoder.

    Fixes the severe >20x scale distortion in baseline implementations where
    reconstruction was averaged per feature while KL divergence was summed across
    latent dimensions.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Decoded reconstruction of shape (Batch_Size, input_dim).
    x : torch.Tensor
        Target input tensor of shape (Batch_Size, input_dim).
    mu : torch.Tensor
        Latent mean parameter of shape (Batch_Size, latent_dim).
    logvar : torch.Tensor
        Latent log-variance parameter of shape (Batch_Size, latent_dim).
    beta : float, default=1.0
        KL divergence regularization weight.
    reduction : str, default='balanced'
        'balanced': Sums reconstruction error across features and KL divergence
                    across latent dimensions, maintaining equal total energy scale.
        'feature_mean': Averages both terms per-feature dimension.
        'none': Returns unreduced per-sample losses.

    Returns
    -------
    total_loss, recon_loss, kl_loss : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)

    # Per-sample squared reconstruction error summed over features
    recon_per_sample = torch.sum((reconstruction - x).pow(2), dim=1)

    # Per-sample KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1
    )

    if reduction == "balanced":
        recon_loss = recon_per_sample.mean()
        kl_loss = kl_per_sample.mean()
    elif reduction in ("mean", "feature_mean"):
        # Normalize by feature dimension to keep loss in variance units
        recon_loss = (reconstruction - x).pow(2).mean(dim=1).mean()
        kl_loss = (kl_per_sample / x.size(1)).mean()
    elif reduction == "none":
        return recon_per_sample + beta * kl_per_sample, recon_per_sample, kl_per_sample
    else:
        raise ValueError(f"Unknown reduction mode: {reduction}. Use 'balanced' or 'feature_mean'.")

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


class VAEDriftDetector:
    """
    High-level clinical distribution drift detector and anomaly scoring engine.

    Calibrates on non-septic ICU stays from Hospital A, establishing a 95th-percentile
    anomaly threshold. Evaluates univariate drift via Kolmogorov-Smirnov tests with
    Bonferroni adjustment and multivariate drift via RBF Maximum Mean Discrepancy.
    """

    def __init__(
        self,
        model: Optional[PatientVAE] = None,
        scaler: Optional[StandardScaler] = None,
        anomaly_threshold: Optional[float] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.model = model if model is not None else PatientVAE()
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.anomaly_threshold = anomaly_threshold
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.is_fitted = False

    def fit_normal_baseline(
        self,
        normal_train: Union[pd.DataFrame, np.ndarray],
        normal_val: Union[pd.DataFrame, np.ndarray],
        feature_cols: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 12,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: str = "cpu",
        verbose: bool = False,
    ) -> float:
        """
        Fits StandardScaler on normal training data, trains PatientVAE with early stopping,
        and calibrates the 95th-percentile anomaly threshold on the normal validation set.
        """
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)

        if isinstance(normal_train, pd.DataFrame):
            if self.feature_cols is None:
                raise ValueError("feature_cols must be specified when passing DataFrames.")
            X_train_raw = normal_train[self.feature_cols].to_numpy(dtype=np.float32)
        else:
            X_train_raw = np.asarray(normal_train, dtype=np.float32)

        if isinstance(normal_val, pd.DataFrame):
            if self.feature_cols is None:
                raise ValueError("feature_cols must be specified when passing DataFrames.")
            X_val_raw = normal_val[self.feature_cols].to_numpy(dtype=np.float32)
        else:
            X_val_raw = np.asarray(normal_val, dtype=np.float32)

        # Fit scaler on normal training cohort only (prevent leakage)
        X_train_scaled = self.scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val_raw).astype(np.float32)

        train_dataset = TensorDataset(torch.from_numpy(X_train_scaled))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        val_dataset = TensorDataset(torch.from_numpy(X_val_scaled))
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss_acc = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch_x)
                loss, _, _ = vae_loss(recon, batch_x, mu, logvar, beta=beta, reduction="balanced")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss_acc += loss.item() * len(batch_x)

            # Validation loss evaluation
            self.model.eval()
            val_loss_acc = 0.0
            with torch.no_grad():
                for (batch_val_x,) in val_loader:
                    batch_val_x = batch_val_x.to(device)
                    v_recon, v_mu, v_logvar = self.model(batch_val_x)
                    v_loss, _, _ = vae_loss(
                        v_recon, batch_val_x, v_mu, v_logvar, beta=beta, reduction="balanced"
                    )
                    val_loss_acc += v_loss.item() * len(batch_val_x)

            avg_val_loss = val_loss_acc / max(len(X_val_scaled), 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.4f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Calibrate 95th-percentile anomaly threshold on normal validation set
        val_errors = self.score_array(X_val_scaled, batch_size=256, device=device)
        self.anomaly_threshold = float(np.quantile(val_errors, 0.95))
        self.is_fitted = True

        return self.anomaly_threshold

    def score_array(
        self, X: np.ndarray, batch_size: int = 512, device: str = "cpu"
    ) -> np.ndarray:
        """
        Computes per-sample reconstruction MSE in minibatches to prevent OOM.
        """
        self.model.eval()
        self.model.to(device)
        X_arr = np.asarray(X, dtype=np.float32)
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_arr), batch_size):
                batch_x = torch.as_tensor(X_arr[i : i + batch_size], device=device)
                err = self.model.get_reconstruction_error(batch_x)
                scores.extend(err.cpu().numpy())

        return np.asarray(scores, dtype=np.float64)

    def score_dataframe(
        self, df: pd.DataFrame, device: str = "cpu"
    ) -> pd.DataFrame:
        """
        Scores input DataFrame using fitted scaler and calibrated anomaly threshold.
        Appends 'VAE_Reconstruction_MSE' and 'VAE_Anomaly' columns.
        """
        if self.feature_cols is None:
            raise ValueError("VAEDriftDetector has no feature_cols configured.")
        if self.anomaly_threshold is None:
            raise ValueError("Anomaly threshold not calibrated. Call fit_normal_baseline first.")

        X_raw = df[self.feature_cols].to_numpy(dtype=np.float32)
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)
        scores = self.score_array(X_scaled, device=device)

        result_df = df.copy()
        result_df["VAE_Reconstruction_MSE"] = scores
        result_df["VAE_Anomaly"] = (scores > self.anomaly_threshold).astype(int)
        return result_df

    @staticmethod
    def compute_ks_drift(
        df_ref: pd.DataFrame, df_target: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Performs two-sample Kolmogorov-Smirnov test across all features with Bonferroni correction.
        """
        rows = []
        n_features = len(feature_cols)
        for f in feature_cols:
            stat, p_val = ks_2samp(df_ref[f].to_numpy(), df_target[f].to_numpy())
            p_adj = min(float(p_val) * n_features, 1.0)
            rows.append({
                "Feature": f,
                "KS_Stat": float(stat),
                "p_value": float(p_val),
                "p_adjusted": float(p_adj),
                "Significant": bool(p_adj < 0.05),
            })

        res_df = pd.DataFrame(rows)
        return res_df.sort_values("KS_Stat", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_rbf_mmd(
        X_ref: Union[np.ndarray, pd.DataFrame],
        X_target: Union[np.ndarray, pd.DataFrame],
        max_samples: int = 1000,
        seed: int = 42,
    ) -> float:
        """
        Calculates mini-batched Maximum Mean Discrepancy (MMD) with RBF kernel and median heuristic.
        Subsamples up to max_samples to eliminate O(N^2) memory bottlenecks.
        """
        rng = np.random.default_rng(seed)

        if isinstance(X_ref, pd.DataFrame):
            X = X_ref.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(X_ref, dtype=np.float64)

        if isinstance(X_target, pd.DataFrame):
            Y = X_target.to_numpy(dtype=np.float64)
        else:
            Y = np.asarray(X_target, dtype=np.float64)

        if len(X) > max_samples:
            X = X[rng.choice(len(X), max_samples, replace=False)]
        if len(Y) > max_samples:
            Y = Y[rng.choice(len(Y), max_samples, replace=False)]

        pooled = np.vstack([X, Y])
        dist_matrix = cdist(pooled, pooled, metric="sqeuclidean")
        pos_dist = dist_matrix[dist_matrix > 0]
        if len(pos_dist) == 0:
            return 0.0

        median_dist = float(np.median(pos_dist))
        gamma = 1.0 / max(median_dist, 1e-12)

        Kxx = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
        Kyy = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
        Kxy = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

        mmd_sq = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
        return max(0.0, mmd_sq)

    def save_checkpoint(self, path: str):
        """Serializes VAE weights, scaler parameters, threshold, and feature schema."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "latent_dim": self.model.latent_dim,
            "hidden_dims": self.model.hidden_dims,
            "anomaly_threshold": self.anomaly_threshold,
            "feature_cols": self.feature_cols,
            "scaler_mean": getattr(self.scaler, "mean_", None),
            "scaler_scale": getattr(self.scaler, "scale_", None),
            "is_fitted": self.is_fitted,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device: str = "cpu"):
        """Restores detector state from saved checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model = PatientVAE(
            input_dim=checkpoint.get("input_dim", 73),
            latent_dim=checkpoint.get("latent_dim", 16),
            hidden_dims=checkpoint.get("hidden_dims", [64, 32]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)

        self.anomaly_threshold = checkpoint.get("anomaly_threshold")
        self.feature_cols = checkpoint.get("feature_cols")
        self.is_fitted = checkpoint.get("is_fitted", True)

        mean = checkpoint.get("scaler_mean")
        scale = checkpoint.get("scaler_scale")
        if mean is not None and scale is not None:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.asarray(mean)
            self.scaler.scale_ = np.asarray(scale)
            self.scaler.var_ = np.asarray(scale) ** 2
            self.scaler.n_features_in_ = len(mean)
