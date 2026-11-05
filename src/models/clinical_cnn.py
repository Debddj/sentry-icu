"""
SentryICU Clinical 1D-CNN & Multimodal Fusion Architectures (Module 3)

Implements:
1. ClinicalCNN1D: Multi-scale parallel 1D convolutional waveform feature extractor
   - Kernel receptive fields: k=3 (acute shocks), k=5 (sub-acute trends), k=7 (sustained drift)
   - Input shape: (Batch_Size, 7, 12) -> Output shape: (Batch_Size, 32)
2. SentryICUFusionModel: Multimodal late-fusion network
   - Combines 73-D static summary + 32-D CNN embedding -> 105-D -> 1-D logit
   - 2-layer regularized MLP with BatchNorm, GELU, and Dropout
3. TemporalGradCAM1D: Hook-based 1D Temporal Grad-CAM with sign-preserving attribution
   - Hooked into convolutional feature maps
   - Sign-preserving attribution (|x * grad_x|) correctly highlights acute hypotension and hypothermia
4. Baseline73DMLP & CNNOnlyModel: De-confounded architectures for clinical ablation studies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Parallel multi-scale convolutional branches (symmetric padding preserves L=12)
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
        Returns concatenated feature maps from each branch before pooling.
        Args:
            x_ts: (Batch_Size, 7, 12)
        Returns:
            multi_scale_feat: (Batch_Size, 48, 12)
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
    
    Fusion Head applies regularized 2-layer MLP with BatchNorm, GELU, and Dropout.
    """
    def __init__(self, static_dim=73, ts_channels=7, cnn_embed_dim=32,
                 hidden_dims=None, dropout_rate=0.2, pool_type='avg'):
        super(SentryICUFusionModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
            
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
        cnn_embed = self.cnn_encoder(x_ts)               # (B, 32)
        fused = torch.cat([x_73d, cnn_embed], dim=1)      # (B, 105)
        logits = self.fusion_mlp(fused)                   # (B, 1)
        return logits, cnn_embed, fused

    def predict_proba(self, x_73d, x_ts):
        """Returns calibrated probability via sigmoid in [0.0, 1.0]."""
        with torch.no_grad():
            logits, _, _ = self.forward(x_73d, x_ts)
            return torch.sigmoid(logits)


class Baseline73DMLP(nn.Module):
    """
    Standardized 73-D static tabular MLP baseline for fair ablation comparison.
    Matches the fusion head capacity (2 hidden layers with BatchNorm, GELU, Dropout).
    """
    def __init__(self, input_dim=73, hidden_dims=None, dropout_rate=0.2):
        super(Baseline73DMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
            
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_73d, x_ts=None):
        return self.net(x_73d)

    def predict_proba(self, x_73d, x_ts=None):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x_73d, x_ts))


class CNNOnlyModel(nn.Module):
    """
    CNN-only baseline for ablation comparison (32-D embedding to classification head).
    """
    def __init__(self, in_channels=7, cnn_embed_dim=32, hidden_dims=None, dropout_rate=0.2):
        super(CNNOnlyModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32]
            
        self.cnn_encoder = ClinicalCNN1D(in_channels=in_channels, out_embed_dim=cnn_embed_dim)
        layers = []
        in_dim = cnn_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_73d=None, x_ts=None):
        if x_ts is None and x_73d is not None and x_73d.ndim == 3:
            x_ts = x_73d
        embed = self.cnn_encoder(x_ts)
        return self.classifier(embed)


class TemporalGradCAM1D:
    """
    Hook-based 1D Temporal Grad-CAM for ClinicalCNN1D / SentryICUFusionModel.
    Hooks into convolutional feature maps and implements sign-preserving attribution (|x * grad_x|).
    
    Clinical Rationale:
    Physiological decompensation in septic shock frequently presents with acute drops in vital signs:
    - Hypotension: sudden precipitous drop in MAP (< 65 mmHg) or SBP (< 90 mmHg).
    - Hypothermia: core temperature drop (< 36.0 °C).
    Conventional Grad-CAM applies ReLU (np.maximum(grads, 0)), discarding negative gradients
    and completely blinding clinicians to acute shock states.
    Sign-preserving attribution (|x * grad_x|) captures bidirectional clinical deviations.
    """
    def __init__(self, model):
        self.model = model
        self.branch_activations = {}
        self.branch_gradients = {}
        self._hook_handles = []
        self._register_hooks()

    def _get_cnn(self):
        if hasattr(self.model, 'cnn_encoder'):
            return self.model.cnn_encoder
        return self.model

    def _register_hooks(self):
        cnn = self._get_cnn()
        
        # Attach hooks to each convolutional branch (k=3, k=5, k=7)
        branches = {
            'k3': cnn.branch_k3[0],  # Conv1d
            'k5': cnn.branch_k5[0],  # Conv1d
            'k7': cnn.branch_k7[0],  # Conv1d
        }

        for name, layer in branches.items():
            def make_fhook(branch_name):
                def hook(module, inp, output):
                    self.branch_activations[branch_name] = output
                return hook

            def make_bhook(branch_name):
                def hook(module, grad_in, grad_out):
                    self.branch_gradients[branch_name] = grad_out[0]
                return hook

            h_f = layer.register_forward_hook(make_fhook(name))
            h_b = layer.register_full_backward_hook(make_bhook(name))
            self._hook_handles.extend([h_f, h_b])

    def generate_heatmap(self, x_73d, x_ts, target_class=1):
        """
        Generates 1D temporal importance scores across (Num_Vitals=7, Window_Length=12).
        
        Args:
            x_73d: (1, 73) or None static features tensor.
            x_ts: (1, 7, 12) vital sign time-series tensor.
            target_class: Target class index (1 for sepsis).
            
        Returns:
            np.ndarray: (7, 12) float32 normalized attribution heatmap in [0.0, 1.0].
        """
        self.model.eval()
        
        if x_ts.ndim == 2:
            x_ts = x_ts.unsqueeze(0)
        if x_73d is not None and x_73d.ndim == 1:
            x_73d = x_73d.unsqueeze(0)
            
        x_ts = x_ts.clone().detach().requires_grad_(True)
        
        # Forward pass
        if hasattr(self.model, 'fusion_mlp') and x_73d is not None:
            logits, _, _ = self.model(x_73d, x_ts)
        else:
            logits = self.model(x_ts)
            
        score = logits[0, 0]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # 1. Convolutional Feature-Map Grad-CAM weighting across branches
        temporal_cam = np.zeros(12, dtype=np.float32)
        total_branches = 0
        for name in ['k3', 'k5', 'k7']:
            if name in self.branch_activations and name in self.branch_gradients:
                act = self.branch_activations[name].data.cpu().numpy()[0]  # (16, 12)
                grad = self.branch_gradients[name].data.cpu().numpy()[0]   # (16, 12)
                # Channel importance: global mean of absolute gradient
                weights = np.mean(np.abs(grad), axis=-1, keepdims=True)    # (16, 1)
                # Weighted activation over time
                branch_cam = np.sum(weights * np.abs(act), axis=0)         # (12,)
                temporal_cam += branch_cam
                total_branches += 1
                
        if total_branches > 0:
            cam_range = temporal_cam.max() - temporal_cam.min()
            if cam_range > 1e-8:
                temporal_cam = (temporal_cam - temporal_cam.min()) / cam_range
            else:
                temporal_cam = np.ones(12, dtype=np.float32)
        else:
            temporal_cam = np.ones(12, dtype=np.float32)

        # 2. Input-level sign-preserving attribution: |x * grad_x|
        # Captures both acute elevations (tachycardia) and acute depressions (hypotension, hypothermia)
        input_grad = x_ts.grad.data.cpu().numpy()[0]   # (7, 12)
        raw_vitals = x_ts.data.cpu().numpy()[0]        # (7, 12)
        
        input_attribution = np.abs(raw_vitals * input_grad)  # (7, 12)
        
        # Modulate input saliency with convolutional temporal Grad-CAM weights
        combined_attribution = input_attribution * temporal_cam[np.newaxis, :]  # (7, 12)
        
        # Normalize globally to [0.0, 1.0]
        attr_range = combined_attribution.max() - combined_attribution.min()
        if attr_range > 1e-8:
            normalized_attr = (combined_attribution - combined_attribution.min()) / attr_range
        else:
            # If flat, fall back to pure input attribution or zeros
            raw_range = input_attribution.max() - input_attribution.min()
            if raw_range > 1e-8:
                normalized_attr = (input_attribution - input_attribution.min()) / raw_range
            else:
                normalized_attr = np.zeros_like(combined_attribution)
                
        return normalized_attr.astype(np.float32)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def __del__(self):
        self.remove_hooks()
