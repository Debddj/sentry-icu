"""
SentryICU Longitudinal Sequence Models (Module 4)
==================================================
Provides production-grade PyTorch implementations of:
- ClinicalVanillaRNN
- ClinicalLSTM
- ClinicalGRU (and alias TemporalGRU per docs/interfaces.md)
- BidirectionalGRU
- AttentionGRU

Contracts:
- Input shape: (Batch_Size, Seq_Len=24, Num_Features=14) with optional lengths (Batch_Size,)
- Temporal Embedding Output: (Batch_Size, 64) float32 representation
- Logit Output: (Batch_Size, 1) float32
- Probability Output: .predict_proba() -> (Batch_Size, 1) in [0.0, 1.0]

Bug Fixes Applied:
- BUG-M4-02: Aligned with standard right-padding, extracting final temporal representation
             at (lengths - 1) instead of fixed index 23 or left-padded 0.
- BUG-M4-07: AttentionGRU uses torch.finfo(scores.dtype).min for masking, avoiding underflow.
- BUG-M4-09: Provided TemporalGRU = ClinicalGRU alias matching docs/interfaces.md.
- BUG-M4-12: Clamped lengths to min=1 to prevent division by zero or empty sequence NaNs.
- BUG-M4-14: Explicit squeeze dimension squeeze(1) avoiding batch dimension collapse when batch_size=1.
- BUG 9: Polymorphic forward returns (logits, emb) by default, and (logits, emb, alpha) only
         when return_attention=True.
- BUG 10: Added predict_proba() across all architectures.
"""


import torch
import torch.nn.functional as F
from torch import nn


class ClinicalSequenceBase(nn.Module):
    """Base class providing calibrated probability prediction interface."""

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns calibrated probability in [0.0, 1.0] via sigmoid.
        Args:
            x: Input sequence tensor (B, T, D)
            lengths: Optional sequence length tensor (B,)
        Returns:
            torch.Tensor: (B, 1) probability values
        """
        self.eval()
        with torch.no_grad():
            res = self.forward(x, lengths)
            logits = res[0] if isinstance(res, tuple) else res
            return torch.sigmoid(logits)


class ClinicalVanillaRNN(ClinicalSequenceBase):
    """
    2-Layer Vanilla Elman RNN with tanh activations.
    Maps (B, 24, 14) -> (B, 64) temporal embedding and (B, 1) classification logit.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Extracts (B, 64) temporal representation."""
        out, _ = self.rnn(x)  # (B, T, 64)

        if lengths is not None:
            # BUG-M4-02: Gather at right-padded true final timestep lengths - 1
            safe_lengths = torch.clamp(lengths, min=1, max=out.size(1))
            idx = (safe_lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2)).to(x.device)
            return out.gather(1, idx).squeeze(1)

        return out[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.extract_features(x, lengths)  # (B, 64)
        logits = self.classifier(emb)            # (B, 1)
        return logits, emb


class ClinicalLSTM(ClinicalSequenceBase):
    """
    2-Layer Long Short-Term Memory network.
    Maps (B, 24, 14) -> (B, 64) temporal embedding and (B, 1) classification logit.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, 64)

        if lengths is not None:
            safe_lengths = torch.clamp(lengths, min=1, max=out.size(1))
            idx = (safe_lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2)).to(x.device)
            return out.gather(1, idx).squeeze(1)

        return out[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.extract_features(x, lengths)  # (B, 64)
        logits = self.classifier(emb)            # (B, 1)
        return logits, emb


class ClinicalGRU(ClinicalSequenceBase):
    """
    Canonical 2-Layer Gated Recurrent Unit network.
    Maps (B, 24, 14) -> (B, 64) temporal embedding and (B, 1) classification logit.
    Adheres strictly to docs/interfaces.md Pair D contract.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gru(x)  # (B, T, 64)

        if lengths is not None:
            safe_lengths = torch.clamp(lengths, min=1, max=out.size(1))
            idx = (safe_lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2)).to(x.device)
            return out.gather(1, idx).squeeze(1)

        return out[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.extract_features(x, lengths)  # (B, 64)
        logits = self.classifier(emb)            # (B, 1)
        return logits, emb


# Contract Alias per docs/interfaces.md
TemporalGRU = ClinicalGRU


class BidirectionalGRU(ClinicalSequenceBase):
    """
    Bidirectional GRU for retrospective trajectory analysis.
    Uses hidden_dim=32 per direction, concatenating forward final state and
    backward final state to output exactly (B, 64).
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_per_dir = hidden_dim // 2  # 32 per direction

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_per_dir,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            safe_lengths = torch.clamp(lengths, min=1, max=x.size(1)).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, safe_lengths, batch_first=True, enforce_sorted=False
            )
            _, h_n = self.bigru(x_packed)
            # h_n shape: (num_layers * 2, B, hidden_per_dir)
            fwd_last = h_n[-2]  # (B, 32)
            bwd_last = h_n[-1]  # (B, 32)
            return torch.cat([fwd_last, bwd_last], dim=1)  # (B, 64)

        out, h_n = self.bigru(x)
        # out shape: (B, T, 64) where [:32] is forward and [32:] is backward
        fwd_last = out[:, -1, :self.hidden_per_dir]
        bwd_first = out[:, 0, self.hidden_per_dir:]
        return torch.cat([fwd_last, bwd_first], dim=1)  # (B, 64)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.extract_features(x, lengths)  # (B, 64)
        logits = self.classifier(emb)            # (B, 1)
        return logits, emb


class AttentionGRU(ClinicalSequenceBase):
    """
    Temporal Attention GRU. Computes additive attention scores across timesteps:
    u_t = tanh(W_a h_t)
    scores = v_a^T u_t
    alpha = softmax(scores)
    context = sum(alpha_t * h_t) in R^64
    
    Fixes BUG-M4-07: Applies masking with torch.finfo(scores.dtype).min for numerical stability.
    Fixes BUG 9: Returns (logits, emb) by default, and (logits, emb, alpha) when return_attention=True.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x)  # (B, T, 64)

        u_t = torch.tanh(self.W_a(out))      # (B, T, 64)
        scores = self.v_a(u_t).squeeze(-1)   # (B, T)

        if lengths is not None:
            safe_lengths = torch.clamp(lengths, min=1, max=x.size(1))
            mask = torch.arange(x.size(1), device=x.device)[None, :] < safe_lengths[:, None]
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, min_val)

        alpha = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)  # (B, 64)
        return context, alpha

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_attention: bool = False
    ):
        context, alpha = self.extract_features(x, lengths)  # (B, 64), (B, T)
        logits = self.classifier(context)                   # (B, 1)

        if return_attention:
            return logits, context, alpha
        return logits, context
