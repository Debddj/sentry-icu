"""
SentryICU Tri-Modal Physiological Fusion Architecture (Module 4)
================================================================
Unifies multi-source ICU telemetry into a single comprehensive risk score:
1. Static Tabular Features & Demographics (Module 1/2) -> 73-D
2. Multi-Scale 1D-CNN Vital Waveforms   (Module 3)   -> 32-D
3. Longitudinal GRU / GRU-D Trajectory   (Module 4)   -> 64-D
-----------------------------------------------------------
Total Fused Latent Physiological Representation      = 169-D

Components:
- SentryICUTriModalFusion: Multimodal late-fusion classification head
- SentryICUEndToEndTriModal: Complete end-to-end model managing raw inputs across all branches

Bug Fixes Applied:
- BUG 8 (Hardcoded dimensions): Stores dim_tabular, dim_cnn, dim_temporal as instance variables
        and performs flexible validation.
- BUG 10 (Predict Proba): Implements predict_proba() returning calibrated sigmoid probabilities in [0.0, 1.0].
- BUG 14 (End-to-End wrapper): Implements SentryICUEndToEndTriModal executing encoders jointly from raw inputs.
"""


import torch
from torch import nn

from .clinical_cnn import ClinicalCNN1D
from .grud import ClinicalGRUD, TemporalGRUD
from .sequence_models import ClinicalGRU


class SentryICUTriModalFusion(nn.Module):
    """
    Tri-Modal Fusion Classification Head.
    Fuses:
    - Tabular Demographics/Summary: (B, 73)
    - Multi-Scale CNN Waveforms:     (B, 32)
    - Longitudinal GRU / GRU-D:     (B, 64)
    Total representation:           (B, 169) -> (B, 1) logit
    """
    def __init__(
        self,
        dim_tabular: int = 73,
        dim_cnn: int = 32,
        dim_temporal: int = 64,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.3
    ):
        super().__init__()
        self.dim_tabular = dim_tabular
        self.dim_cnn = dim_cnn
        self.dim_temporal = dim_temporal
        self.dim_fused = dim_tabular + dim_cnn + dim_temporal  # 73 + 32 + 64 = 169

        # Fusion Classification Head: 169 -> 128 -> 64 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_fused, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),

            nn.Linear(hidden_dims[1], 1)
        )

    def forward(
        self,
        tabular_emb: torch.Tensor,
        cnn_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass concatenating the three branches.
        Args:
            tabular_emb: Static summary features (B, 73)
            cnn_emb: Vital sign waveform embedding (B, 32)
            temporal_emb: Longitudinal trajectory embedding (B, 64)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits: Classification prediction logit (B, 1)
                - fused_representation: Concatenated features (B, 169)
        """
        # Validate input embedding shapes dynamically (BUG 8 fix)
        assert tabular_emb.size(1) == self.dim_tabular, (
            f"Expected tabular dim {self.dim_tabular}, got {tabular_emb.size(1)}"
        )
        assert cnn_emb.size(1) == self.dim_cnn, (
            f"Expected CNN dim {self.dim_cnn}, got {cnn_emb.size(1)}"
        )
        assert temporal_emb.size(1) == self.dim_temporal, (
            f"Expected temporal dim {self.dim_temporal}, got {temporal_emb.size(1)}"
        )

        fused_representation = torch.cat(
            [tabular_emb, cnn_emb, temporal_emb],
            dim=1
        )  # (B, 169)

        logits = self.classifier(fused_representation)  # (B, 1)
        return logits, fused_representation

    def predict_proba(
        self,
        tabular_emb: torch.Tensor,
        cnn_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> torch.Tensor:
        """Returns calibrated probability via sigmoid in [0.0, 1.0]."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(tabular_emb, cnn_emb, temporal_emb)
            return torch.sigmoid(logits)


class SentryICUEndToEndTriModal(nn.Module):
    """
    End-to-End Clinical Deterioration Early Warning System.
    Jointly encodes:
    1. Static tabular 73-D summary vector
    2. Vital sign waveform tensor (B, 7, 12) via ClinicalCNN1D -> (B, 32)
    3. Longitudinal sequence tensor (B, 24, 14) via ClinicalGRUD or ClinicalGRU -> (B, 64)
    4. SentryICUTriModalFusion head (169 -> 128 -> 64 -> 1)
    """
    def __init__(
        self,
        cnn_encoder: nn.Module | None = None,
        temporal_encoder: nn.Module | None = None,
        fusion_head: SentryICUTriModalFusion | None = None,
        dim_tabular: int = 73,
        dim_cnn: int = 32,
        dim_temporal: int = 64,
        use_grud: bool = True,
        empirical_mean: torch.Tensor | None = None
    ):
        super().__init__()
        self.dim_tabular = dim_tabular
        self.dim_cnn = dim_cnn
        self.dim_temporal = dim_temporal

        # 1. 1D-CNN vital sign extractor
        if cnn_encoder is not None:
            self.cnn_encoder = cnn_encoder
        else:
            self.cnn_encoder = ClinicalCNN1D(in_channels=7, out_embed_dim=dim_cnn)

        # 2. Longitudinal temporal sequence extractor (GRU or GRU-D)
        if temporal_encoder is not None:
            self.temporal_encoder = temporal_encoder
        elif use_grud:
            self.temporal_encoder = ClinicalGRUD(
                input_dim=14, hidden_dim=dim_temporal, empirical_mean=empirical_mean
            )
        else:
            self.temporal_encoder = ClinicalGRU(
                input_dim=14, hidden_dim=dim_temporal
            )

        # 3. Tri-modal fusion head
        if fusion_head is not None:
            self.fusion_head = fusion_head
        else:
            self.fusion_head = SentryICUTriModalFusion(
                dim_tabular=dim_tabular,
                dim_cnn=dim_cnn,
                dim_temporal=dim_temporal
            )

    def forward(
        self,
        x_tab: torch.Tensor,
        x_vitals: torch.Tensor,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_embeddings: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        End-to-end forward execution across raw clinical modalities.
        Args:
            x_tab: (B, 73) static tabular summary features
            x_vitals: (B, 7, 12) vital sign waveform matrix
            x_seq: (B, 24, 14) longitudinal clinical records
            mask: Optional (B, 24, 14) observation indicator mask
            delta: Optional (B, 24, 14) elapsed time deltas
            lengths: Optional (B,) sequence lengths before right-padding
            return_embeddings: If True, returns dict of intermediate representations
        Returns:
            logits: (B, 1) classification logit
            fused: (B, 169) fused latent representation
            embeddings (opt): Dict containing individual branch embeddings
        """
        # 1. Vital waveform branch: (B, 7, 12) -> (B, 32)
        cnn_emb = self.cnn_encoder(x_vitals)

        # 2. Longitudinal sequence branch: (B, 24, 14) -> (B, 64)
        if isinstance(self.temporal_encoder, (ClinicalGRUD, TemporalGRUD)):
            if mask is None:
                mask = torch.ones_like(x_seq)
            if delta is None:
                delta = torch.zeros_like(x_seq)
            temporal_emb = self.temporal_encoder.extract_features(x_seq, mask, delta, lengths=lengths)
        else:
            temporal_emb = self.temporal_encoder.extract_features(x_seq, lengths=lengths)

        # 3. Tri-modal fusion: (B, 73) + (B, 32) + (B, 64) -> (B, 169) -> (B, 1)
        logits, fused = self.fusion_head(x_tab, cnn_emb, temporal_emb)

        if return_embeddings:
            embs = {
                'tabular': x_tab,
                'cnn': cnn_emb,
                'temporal': temporal_emb,
                'fused': fused
            }
            return logits, fused, embs

        return logits, fused

    def predict_proba(
        self,
        x_tab: torch.Tensor,
        x_vitals: torch.Tensor,
        x_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns calibrated probability in [0.0, 1.0] via sigmoid."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(
                x_tab, x_vitals, x_seq, mask=mask, delta=delta, lengths=lengths
            )
            return torch.sigmoid(logits)
