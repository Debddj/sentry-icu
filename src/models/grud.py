"""
SentryICU Clinical GRU-D Architecture (Module 4)
=================================================
Gated Recurrent Unit with Dual Exponential Decay for Irregularly Sampled Clinical Sequences
(Che et al., Nature Scientific Reports 2018).

Components:
- ClinicalGRUDCell: Single-timestep recurrent cell with dual exponential decay for inputs and hidden states
- ClinicalGRUD: Full longitudinal GRU-D model unrolling over 24-hr clinical trajectories
- TemporalGRUD = ClinicalGRUD: Alias per docs/interfaces.md Pair D contract

Bug Fixes Applied:
- BUG-M4-01 (Dead ReLU at initialization): Parameterized input decay strictly positively via
             gamma_x = exp(-F.softplus(W_gamma_x) * delta_t) and hidden decay via
             gamma_h = exp(-F.softplus(W_gamma_h(delta_t))). This eliminates the dead ReLU trap
             and guarantees smooth, non-zero gradients (grad != 0) from step 1.
- BUG-M4-03 (Compounding decay error): Updates x_last strictly when observed:
             x_last_next = m_t * x_raw + (1.0 - m_t) * x_last. Prevents compounding decay
             on already-decayed x_hat values over multiple missing intervals.
- BUG-M4-04 (Uninitialized empirical mean): Accepts empirical training feature means at init
             via empirical_mean argument and populates empirical_mean buffer, preventing missing
             vital signs from decaying to 0.0 (simulating cardiac arrest).
- BUG-M4-10 (Device placement): Ensures empirical mean and intermediate tensors match device of inputs.
- BUG-M4-14 (Batch dimension squeezing): Uses explicit squeeze(1) indexing.
- BUG 10 (Predict Proba): Implements predict_proba() on ClinicalGRUD.
"""


import torch
import torch.nn.functional as F
from torch import nn


class ClinicalGRUDCell(nn.Module):
    """
    Custom GRU-D Cell implementing dual exponential decay:
    1. Input decay gamma_x towards empirical feature mean:
       gamma_x = exp(-softplus(W_gamma_x) * delta_t)
       x_hat = m_t * x_raw + (1 - m_t) * (gamma_x * x_last + (1 - gamma_x) * mean_x)
    2. Hidden state decay gamma_h:
       gamma_h = exp(-softplus(W_gamma_h(delta_t)))
       h_hat = gamma_h * h_prev
    3. Gated recurrent transition over concatenated [x_hat, h_hat, m_t].
    4. Observation preservation (BUG-M4-03 fix):
       x_last_next = m_t * x_raw + (1 - m_t) * x_last
    """
    def __init__(self, input_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Learned input decay parameter vector (BUG-M4-01 fix: positive initialization)
        self.W_gamma_x = nn.Parameter(torch.empty(input_dim))

        # Hidden state decay projection from delta_t
        self.W_gamma_h = nn.Linear(input_dim, hidden_dim)

        # Standard GRU gates taking concatenated [x_hat, h_hat, m_t]
        gate_input_dim = input_dim * 2 + hidden_dim
        self.w_z = nn.Linear(gate_input_dim, hidden_dim)
        self.w_r = nn.Linear(gate_input_dim, hidden_dim)
        self.w_h = nn.Linear(gate_input_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # BUG-M4-01 fix: Initialize W_gamma_x positively (0.1) so softplus(0.1) > 0 and gradients exist
        nn.init.constant_(self.W_gamma_x, 0.1)

        nn.init.xavier_normal_(self.W_gamma_h.weight)
        nn.init.zeros_(self.W_gamma_h.bias)

        nn.init.xavier_normal_(self.w_z.weight)
        nn.init.zeros_(self.w_z.bias)
        nn.init.xavier_normal_(self.w_r.weight)
        nn.init.zeros_(self.w_r.bias)
        nn.init.xavier_normal_(self.w_h.weight)
        nn.init.zeros_(self.w_h.bias)

    def forward(
        self,
        x_raw: torch.Tensor,
        m_t: torch.Tensor,
        delta_t: torch.Tensor,
        h_prev: torch.Tensor,
        x_last: torch.Tensor,
        mean_x: torch.Tensor,
        return_x_hat: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.
        Args:
            x_raw: Raw input tensor (B, 14)
            m_t: Binary observation mask (B, 14), 1 if observed, 0 if missing
            delta_t: Elapsed time delta tensor (B, 14)
            h_prev: Previous hidden state (B, 64)
            x_last: Value at last actual observation (B, 14)
            mean_x: Empirical population mean vector (14,) or (B, 14)
            return_x_hat: If True, also returns imputed x_hat as third element
        Returns:
            (h_t, x_last_next) or (h_t, x_last_next, x_hat)
        """
        # Ensure mean_x matches device and dimensions
        if mean_x.device != x_raw.device:
            mean_x = mean_x.to(x_raw.device)
        if mean_x.dim() == 1:
            mean_x = mean_x.unsqueeze(0)

        # BUG-M4-01 fix: Strictly positive parameterization via softplus
        gamma_x = torch.exp(-F.softplus(self.W_gamma_x) * delta_t)
        gamma_h = torch.exp(-F.softplus(self.W_gamma_h(delta_t)))

        # Decayed input imputation toward empirical mean
        x_hat = m_t * x_raw + (1.0 - m_t) * (gamma_x * x_last + (1.0 - gamma_x) * mean_x)

        # Decayed hidden state
        h_hat = gamma_h * h_prev

        # Concatenate inputs for update and reset gates: [x_hat, h_hat, m_t]
        inputs = torch.cat([x_hat, h_hat, m_t], dim=-1)
        z_t = torch.sigmoid(self.w_z(inputs))
        r_t = torch.sigmoid(self.w_r(inputs))

        # Candidate hidden state
        inputs_candidate = torch.cat([x_hat, r_t * h_hat, m_t], dim=-1)
        h_tilde = torch.tanh(self.w_h(inputs_candidate))

        # New hidden state
        h_t = (1.0 - z_t) * h_hat + z_t * h_tilde

        # BUG-M4-03 fix: Update x_last strictly when feature is actively observed (m_t = 1)
        x_last_next = m_t * x_raw + (1.0 - m_t) * x_last

        if return_x_hat:
            return h_t, x_last_next, x_hat
        return h_t, x_last_next


class ClinicalGRUD(nn.Module):
    """
    Full Clinical GRU-D Model unrolled over 24-hour multivariate clinical time-series.
    Input: (B, 24, 14) with observation mask (B, 24, 14) and elapsed delta (B, 24, 14).
    Output: (B, 64) temporal representation and (B, 1) classification logit.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 1,
        empirical_mean: torch.Tensor | list | None = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cell = ClinicalGRUDCell(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

        # BUG-M4-04 fix: Register buffer populated with empirical feature means
        if empirical_mean is not None:
            mean_tensor = torch.as_tensor(empirical_mean, dtype=torch.float32)
            self.register_buffer('empirical_mean', mean_tensor.clone().detach())
        else:
            self.register_buffer('empirical_mean', torch.zeros(input_dim, dtype=torch.float32))

    def set_empirical_mean(self, mean: torch.Tensor | list):
        """Updates empirical mean buffer with training cohort statistics."""
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=self.empirical_mean.device)
        self.empirical_mean.copy_(mean_tensor)

    def extract_features(
        self,
        x_raw: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Unrolls ClinicalGRUDCell across timesteps t = 0 .. seq_len-1.
        Args:
            x_raw: (B, T, 14) raw/imputed clinical values
            mask: (B, T, 14) binary observation indicators
            delta: (B, T, 14) elapsed time deltas
            lengths: Optional (B,) true sequence lengths before right-padding
        Returns:
            torch.Tensor: (B, 64) temporal embedding
        """
        batch_size, seq_len, _ = x_raw.size()
        device = x_raw.device

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        # Initialize x_last with population mean for unobserved variables
        x_last = self.empirical_mean.unsqueeze(0).expand(batch_size, -1).clone().to(device)

        hidden_states = []

        for t in range(seq_len):
            x_t = x_raw[:, t, :]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]

            h_t, x_last = self.cell(
                x_t,
                m_t,
                d_t,
                h_t,
                x_last,
                self.empirical_mean
            )
            hidden_states.append(h_t.unsqueeze(1))

        all_hidden = torch.cat(hidden_states, dim=1)  # (B, T, 64)

        if lengths is not None:
            # BUG-M4-02: Gather at right-padded true final timestep lengths - 1
            safe_lengths = torch.clamp(lengths, min=1, max=seq_len)
            idx = (safe_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_dim).to(device)
            return all_hidden.gather(1, idx).squeeze(1)

        return h_t  # (B, 64)

    def forward(
        self,
        x_raw: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.extract_features(x_raw, mask, delta, lengths)  # (B, 64)
        logits = self.classifier(emb)                             # (B, 1)
        return logits, emb

    def predict_proba(
        self,
        x_raw: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns calibrated probability via sigmoid in [0.0, 1.0]."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x_raw, mask, delta, lengths)
            return torch.sigmoid(logits)


# Contract Alias per docs/interfaces.md
TemporalGRUD = ClinicalGRUD
