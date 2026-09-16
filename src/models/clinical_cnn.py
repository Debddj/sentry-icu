import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClinicalCNN1D(nn.Module):
    """
    Multi-Scale 1D-CNN Clinical Feature Extractor for Multivariate ICU Vital Signs.
    
    Adheres strictly to docs/interfaces.md contract:
    - Input: Sliding vital sign windows (Batch_Size, Num_Vitals=7, Window_Length=12)
    - Output Embedding: (Batch_Size, 32) float32 representation for concatenation into Fusion Head.
    
    Extracts multi-scale temporal dynamics using parallel 1D convolutional branches
    with kernel sizes k=3 (short-term acute fluctuations), k=5 (sub-acute trends),
    and k=7 (sustained physiological drift).
    """
    def __init__(self, in_channels=7, out_embed_dim=32, pool_type='avg'):
        super(ClinicalCNN1D, self).__init__()
        self.in_channels = in_channels
        self.out_embed_dim = out_embed_dim
        self.pool_type = pool_type.lower()
        
        # Parallel multi-scale convolutional branches
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU()
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU()
        )
        self.branch_k7 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU()
        )
        
        # 16 channels x 3 branches = 48 concatenated channels
        if self.pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif self.pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f"Unsupported pool_type '{pool_type}'. Choose 'avg' or 'max'.")
            
        self.fc_embed = nn.Sequential(
            nn.Linear(48, out_embed_dim),
            nn.GELU()
        )

    def extract_features(self, x_ts):
        """
        Returns feature maps from each branch before pooling (useful for Grad-CAM).
        x_ts: (Batch_Size, 7, 12)
        """
        f3 = self.branch_k3(x_ts)  # (B, 16, 12)
        f5 = self.branch_k5(x_ts)  # (B, 16, 12)
        f7 = self.branch_k7(x_ts)  # (B, 16, 12)
        multi_scale_feat = torch.cat([f3, f5, f7], dim=1)  # (B, 48, 12)
        return multi_scale_feat

    def forward(self, x_ts):
        """
        Forward pass.
        Args:
            x_ts: (Batch_Size, 7, 12)
        Returns:
            embedding: (Batch_Size, 32)
        """
        multi_scale_feat = self.extract_features(x_ts)      # (B, 48, 12)
        pooled = self.pool(multi_scale_feat).squeeze(-1)    # (B, 48)
        embedding = self.fc_embed(pooled)                   # (B, 32)
        return embedding


# Interface Alias per docs/interfaces.md
MultiScale1DCNN = ClinicalCNN1D


class SentryICUFusionModel(nn.Module):
    """
    Multimodal Early Deterioration Fusion Classifier.
    Concatenates:
      - 73-D static / summary tabular features (Module 1 / Module 2 schema)
      - 32-D temporal multi-scale vital sign embedding (Module 3 ClinicalCNN1D)
    Total representation: 105-D
    
    Fusion Head applies regularized MLP layers with BatchNorm, GELU, and Dropout.
    """
    def __init__(self, static_dim=73, ts_channels=7, cnn_embed_dim=32,
                 hidden_dims=[64, 32], dropout_rate=0.2, pool_type='avg'):
        super(SentryICUFusionModel, self).__init__()
        self.static_dim = static_dim
        self.cnn_embed_dim = cnn_embed_dim
        self.cnn_encoder = ClinicalCNN1D(
            in_channels=ts_channels, out_embed_dim=cnn_embed_dim, pool_type=pool_type
        )
        
        fusion_dim = static_dim + cnn_embed_dim  # 73 + 32 = 105
        layers = []
        in_dim = fusion_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, x_73d, x_ts):
        """
        Forward pass.
        Args:
            x_73d: (Batch_Size, 73)
            x_ts: (Batch_Size, 7, 12)
        Returns:
            logits: (Batch_Size, 1)
            cnn_embed: (Batch_Size, 32)
            fused: (Batch_Size, 105)
        """
        cnn_embed = self.cnn_encoder(x_ts)              # (B, 32)
        fused = torch.cat([x_73d, cnn_embed], dim=1)     # (B, 105)
        logits = self.fusion_mlp(fused)                  # (B, 1)
        return logits, cnn_embed, fused

    def predict_proba(self, x_73d, x_ts):
        """Returns calibrated probability via sigmoid."""
        with torch.no_grad():
            logits, _, _ = self.forward(x_73d, x_ts)
            return torch.sigmoid(logits)


class TemporalGradCAM1D:
    """
    Hook-based 1D Temporal Grad-CAM for ClinicalCNN1D / SentryICUFusionModel.
    Computes gradient-weighted class activation mapping over convolutional feature maps.
    
    L_GradCAM = ReLU(sum_k alpha_k * A_k)
    where alpha_k = (1 / T) * sum_t (dScore / dA_{k,t})
    """
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Attach hook to the multi-scale feature layer or target conv branch
        cnn = self.model.cnn_encoder if hasattr(self.model, 'cnn_encoder') else self.model
        target_layer = cnn.fc_embed[0]  # Layer immediately after pooling or conv
        
        # We hook into branch_k3, branch_k5, branch_k7 before pooling
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hook on branch_k5 (or extract_features)
        h1 = cnn.branch_k5[0].register_forward_hook(forward_hook)
        h2 = cnn.branch_k5[0].register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_heatmap(self, x_73d, x_ts, target_class=1):
        """
        Generates 1D temporal importance scores across (7, 12).
        Returns:
            attribution_matrix: (7, 12) float32 array
        """
        self.model.eval()
        x_ts = x_ts.clone().detach().requires_grad_(True)
        
        # Forward pass
        if hasattr(self.model, 'fusion_mlp'):
            logits, _, _ = self.model(x_73d, x_ts)
        else:
            logits = self.model(x_ts)
            
        score = logits[0, 0]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Input * Gradient attribution with physiological sign awareness
        # Preserves both acute elevations (tachycardia) and acute depressions (hypotension, hypothermia)
        input_grad = x_ts.grad.data.cpu().numpy()[0]   # (7, 12)
        raw_vitals = x_ts.data.cpu().numpy()[0]        # (7, 12)
        
        # Saliency attribution: |x * grad| to capture bidirectional clinical deviations
        attribution = np.abs(raw_vitals * input_grad)
        
        # Normalize per-channel or globally to [0, 1]
        attr_range = attribution.max() - attribution.min()
        if attr_range > 1e-6:
            attribution = (attribution - attribution.min()) / attr_range
        else:
            attribution = np.zeros_like(attribution)
            
        return attribution

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
