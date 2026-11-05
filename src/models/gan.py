"""
src/models/gan.py
=================
Tabular Generative Adversarial Network (TabularGAN) for ICU Sepsis Minority Augmentation.

Components:
- TabularGenerator: Deep MLP (noise_dim -> 64 -> 128 -> output_dim) with LayerNorm, ReLU,
  and empirical range clamping to prevent non-physiological outputs.
- TabularDiscriminator: Deep MLP (input_dim -> 128 -> 64 -> 1) with LeakyReLU(0.2) and Dropout.
- TabularGAN: GAN manager implementing one-sided label smoothing (0.9), stable Adam optimization,
  checkpoint persistence, and safe generation.
- benchmark_smote_vs_gan: Rigorous empirical benchmark comparing unweighted baseline,
  cost-sensitive weighting, SMOTE, and GAN augmentation using compute_metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.metrics import compute_metrics


class TabularGenerator(nn.Module):
    """
    Generator network synthesizing continuous tabular clinical vectors from Gaussian noise.

    Parameters
    ----------
    noise_dim : int, default=32
        Dimensionality of the input Gaussian latent vector z.
    output_dim : int, default=73
        Dimensionality of the synthesized clinical feature vector.
    hidden_dims : list of int, default=[64, 128]
        Capacities of hidden layers.
    clamp_bounds : tuple of (torch.Tensor, torch.Tensor), optional
        (min_bounds, max_bounds) empirical bounds for output clamping.
    """

    def __init__(
        self,
        noise_dim: int = 32,
        output_dim: int = 73,
        hidden_dims: Optional[List[int]] = None,
        clamp_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 128]

        layers = []
        in_dim = noise_dim
        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.min_bounds: Optional[torch.Tensor] = None
        self.max_bounds: Optional[torch.Tensor] = None
        if clamp_bounds is not None:
            self.set_clamp_bounds(clamp_bounds[0], clamp_bounds[1])

    def set_clamp_bounds(
        self, min_bounds: Union[torch.Tensor, np.ndarray], max_bounds: Union[torch.Tensor, np.ndarray]
    ):
        """Sets empirical feature clipping bounds to avoid non-physiological values."""
        self.min_bounds = (
            torch.as_tensor(min_bounds, dtype=torch.float32)
            if not isinstance(min_bounds, torch.Tensor)
            else min_bounds.float()
        )
        self.max_bounds = (
            torch.as_tensor(max_bounds, dtype=torch.float32)
            if not isinstance(max_bounds, torch.Tensor)
            else max_bounds.float()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes tabular features from noise vector z.
        Shape: (Batch_Size, noise_dim) -> (Batch_Size, output_dim)
        """
        x = self.net(z)
        if not self.training and self.min_bounds is not None and self.max_bounds is not None:
            min_b = self.min_bounds.to(x.device)
            max_b = self.max_bounds.to(x.device)
            x = torch.max(torch.min(x, max_b), min_b)
        return x


class TabularDiscriminator(nn.Module):
    """
    Discriminator network evaluating authenticity of tabular clinical vectors.

    Parameters
    ----------
    input_dim : int, default=73
        Number of clinical features.
    hidden_dims : list of int, default=[128, 64]
        Capacities of hidden layers.
    negative_slope : float, default=0.2
        Negative slope coefficient for LeakyReLU activations.
    dropout_rate : float, default=0.2
        Dropout probability to prevent discriminator memorization.
    """

    def __init__(
        self,
        input_dim: int = 73,
        hidden_dims: Optional[List[int]] = None,
        negative_slope: float = 0.2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]

        layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(negative_slope),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes discrimination logit for input feature vector x.
        Shape: (Batch_Size, input_dim) -> (Batch_Size, 1)
        """
        return self.net(x)


class TabularGAN:
    """
    Tabular Generative Adversarial Network with one-sided label smoothing for ICU minority oversampling.

    Parameters
    ----------
    noise_dim : int, default=32
        Latent noise dimension.
    feature_dim : int, default=73
        Number of tabular clinical features.
    lr : float, default=2e-4
        Adam learning rate for both generator and discriminator.
    device : str, default='cpu'
        Target compute device.
    """

    def __init__(
        self,
        noise_dim: int = 32,
        feature_dim: int = 73,
        lr: float = 2e-4,
        device: str = "cpu",
    ):
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.device = device

        self.generator = TabularGenerator(noise_dim=noise_dim, output_dim=feature_dim).to(device)
        self.discriminator = TabularDiscriminator(input_dim=feature_dim).to(device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.bce = nn.BCEWithLogitsLoss()

        self.empirical_min: Optional[np.ndarray] = None
        self.empirical_max: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(
        self,
        X_positive: Union[np.ndarray, pd.DataFrame],
        epochs: int = 300,
        batch_size: int = 32,
        label_smoothing: float = 0.9,
        clamp_to_bounds: bool = True,
        verbose: bool = False,
    ):
        """
        Trains TabularGAN on minority positive sepsis records.

        Parameters
        ----------
        X_positive : np.ndarray or pd.DataFrame
            Minority class feature matrix of shape (N_pos, feature_dim).
        epochs : int, default=300
            Number of adversarial training epochs.
        batch_size : int, default=32
            Minibatch size.
        label_smoothing : float, default=0.9
            One-sided label smoothing value for real samples.
        clamp_to_bounds : bool, default=True
            Whether to clamp synthetic outputs to empirical min/max bounds.
        verbose : bool, default=False
            Whether to print training loss progress.
        """
        if isinstance(X_positive, pd.DataFrame):
            X_arr = X_positive.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_positive, dtype=np.float32)

        if len(X_arr) < 2:
            raise ValueError(
                f"Insufficient positive cases ({len(X_arr)}) to train TabularGAN. Minimum 2 required."
            )

        if clamp_to_bounds:
            self.empirical_min = np.min(X_arr, axis=0)
            self.empirical_max = np.max(X_arr, axis=0)
            self.generator.set_clamp_bounds(self.empirical_min, self.empirical_max)

        safe_batch_size = max(1, min(batch_size, len(X_arr)))
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X_arr, dtype=torch.float32)),
            batch_size=safe_batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.generator.train()
        self.discriminator.train()
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        for epoch in range(epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            batch_count = 0

            for (real_batch,) in loader:
                real_batch = real_batch.to(self.device)
                b_sz = real_batch.size(0)

                # =========================================================
                # 1. Discriminator Step: Maximize log(D(x)) + log(1 - D(G(z)))
                # =========================================================
                noise = torch.randn(b_sz, self.noise_dim, device=self.device)
                fake_batch = self.generator(noise).detach()

                self.d_optimizer.zero_grad()
                real_logits = self.discriminator(real_batch)
                fake_logits = self.discriminator(fake_batch)

                # One-sided label smoothing on real targets to prevent discriminator overconfidence
                real_targets = torch.full_like(real_logits, label_smoothing)
                fake_targets = torch.zeros_like(fake_logits)

                d_loss = self.bce(real_logits, real_targets) + self.bce(fake_logits, fake_targets)
                d_loss.backward()
                self.d_optimizer.step()

                # =========================================================
                # 2. Generator Step: Maximize log(D(G(z)))
                # =========================================================
                noise = torch.randn(b_sz, self.noise_dim, device=self.device)
                gen_data = self.generator(noise)
                gen_logits = self.discriminator(gen_data)

                self.g_optimizer.zero_grad()
                g_loss = self.bce(gen_logits, torch.ones_like(gen_logits))
                g_loss.backward()
                self.g_optimizer.step()

                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                batch_count += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"D-Loss: {d_loss_epoch / max(batch_count, 1):.4f} "
                    f"G-Loss: {g_loss_epoch / max(batch_count, 1):.4f}"
                )

        self.is_fitted = True

    def generate(
        self, n_samples: int, as_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generates synthetic patient records from the trained generator.

        Parameters
        ----------
        n_samples : int
            Number of synthetic patient records to generate.
        as_numpy : bool, default=True
            Whether to return a numpy array (True) or torch.Tensor (False).

        Returns
        -------
        synthetic_records : np.ndarray or torch.Tensor
            Array of shape (n_samples, feature_dim).
        """
        if n_samples <= 0:
            if as_numpy:
                return np.empty((0, self.feature_dim), dtype=np.float32)
            return torch.empty((0, self.feature_dim), dtype=torch.float32, device=self.device)

        self.generator.eval()
        self.generator.to(self.device)

        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim, device=self.device)
            synthetic = self.generator(noise)

            if as_numpy:
                return synthetic.cpu().numpy().astype(np.float32)
            return synthetic

    def save(self, path: str):
        """Serializes generator and discriminator checkpoints to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "noise_dim": self.noise_dim,
            "feature_dim": self.feature_dim,
            "empirical_min": self.empirical_min,
            "empirical_max": self.empirical_max,
            "is_fitted": self.is_fitted,
        }
        torch.save(checkpoint, path)

    def load(self, path: str, device: str = "cpu"):
        """Restores TabularGAN state from saved checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.noise_dim = checkpoint.get("noise_dim", 32)
        self.feature_dim = checkpoint.get("feature_dim", 73)
        self.device = device

        self.generator = TabularGenerator(
            noise_dim=self.noise_dim, output_dim=self.feature_dim
        ).to(device)
        self.discriminator = TabularDiscriminator(input_dim=self.feature_dim).to(device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        self.empirical_min = checkpoint.get("empirical_min")
        self.empirical_max = checkpoint.get("empirical_max")
        if self.empirical_min is not None and self.empirical_max is not None:
            self.generator.set_clamp_bounds(self.empirical_min, self.empirical_max)

        self.is_fitted = checkpoint.get("is_fitted", True)


def benchmark_smote_vs_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: Optional[List[str]] = None,
    n_gan_epochs: int = 150,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Executes a head-to-head empirical comparison across imbalance remediation techniques:
    1. No Oversampling (unweighted baseline)
    2. Cost-Sensitive Baseline (class_weight='balanced')
    3. SMOTE (k-nearest neighbors convex hull interpolation)
    4. TabularGAN Augmentation (adversarial minority generation)

    Evaluates downstream performance using compute_metrics on X_test.
    """
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=int).ravel()
    X_te = np.asarray(X_test, dtype=np.float32)
    y_te = np.asarray(y_test, dtype=int).ravel()

    num_pos = int((y_tr == 1).sum())
    num_neg = int((y_tr == 0).sum())

    training_variants: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # 1. Baseline: Unweighted Natural Distribution
    training_variants["No Oversampling (Unweighted)"] = (X_tr, y_tr)

    # 2. Cost-Sensitive Baseline: Trained with class_weight='balanced'
    training_variants["Cost-Sensitive Baseline"] = (X_tr, y_tr)

    # 3. SMOTE Oversampling
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=seed, k_neighbors=min(5, max(1, num_pos - 1)))
        X_smote, y_smote = smote.fit_resample(X_tr, y_tr)
        training_variants["SMOTE"] = (X_smote, y_smote)
    except Exception:
        # Fallback synthetic linear interpolation if imbalanced-learn is unavailable
        if num_pos >= 2:
            rng = np.random.default_rng(seed)
            pos_indices = np.where(y_tr == 1)[0]
            n_needed = max(0, num_neg - num_pos)
            syn_samples = []
            for _ in range(n_needed):
                idx1, idx2 = rng.choice(pos_indices, size=2, replace=True)
                lam = rng.uniform(0.0, 1.0)
                syn_sample = X_tr[idx1] + lam * (X_tr[idx2] - X_tr[idx1])
                syn_samples.append(syn_sample)
            if syn_samples:
                X_smote = np.vstack([X_tr, np.asarray(syn_samples, dtype=np.float32)])
                y_smote = np.concatenate([y_tr, np.ones(len(syn_samples), dtype=int)])
            else:
                X_smote, y_smote = X_tr, y_tr
            training_variants["SMOTE (Interpolation Fallback)"] = (X_smote, y_smote)

    # 4. Tabular GAN Augmentation
    if num_pos >= 2:
        X_positive = X_tr[y_tr == 1]
        gan = TabularGAN(
            noise_dim=32, feature_dim=X_tr.shape[1], lr=2e-4, device=device
        )
        gan.fit(X_positive, epochs=n_gan_epochs, batch_size=32, verbose=False)
        n_needed = max(0, num_neg - num_pos)
        syn_positive = gan.generate(n_needed, as_numpy=True)
        if len(syn_positive) > 0:
            X_gan = np.vstack([X_tr, syn_positive])
            y_gan = np.concatenate([y_tr, np.ones(len(syn_positive), dtype=int)])
        else:
            X_gan, y_gan = X_tr, y_tr
        training_variants["Tabular GAN Augmentation"] = (X_gan, y_gan)

    # Fit LogisticRegression and evaluate via compute_metrics
    results = []
    for method, (X_fit, y_fit) in training_variants.items():
        is_cost_sensitive = "Cost-Sensitive" in method
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if is_cost_sensitive else None,
            random_state=seed,
        )
        clf.fit(X_fit, y_fit)

        test_probs = clf.predict_proba(X_te)[:, 1]
        metrics = compute_metrics(y_te, test_probs, threshold=0.5)

        results.append({
            "Method": method,
            "AUROC": metrics["auroc"],
            "AUPRC": metrics["auprc"],
            "Sensitivity": metrics["sensitivity"],
            "Specificity": metrics["specificity"],
            "F1": metrics["f1"],
            "Brier_Score": metrics["brier_score"],
        })

    return pd.DataFrame(results)
