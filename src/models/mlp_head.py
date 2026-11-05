from typing import List, Optional
import torch
import torch.nn as nn


class ClinicalRiskMLP(nn.Module):
    """
    Modular Multi-Layer Perceptron (MLP) for Clinical Deterioration Prediction.
    Supports dynamic activations: 'relu', 'leaky_relu', 'gelu', 'elu'.
    """
    def __init__(
        self,
        input_dim: int = 73,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout_rate: float = 0.2
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        
        act_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.1),
            'gelu': nn.GELU(),
            'elu': nn.ELU(alpha=1.0)
        }
        act_layer = act_map.get(activation.lower(), nn.ReLU())
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_layer)
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
