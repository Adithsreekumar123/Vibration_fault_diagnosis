# src/models/domain_discriminator.py
"""
Domain Discriminator with Gradient Reversal Layer (GRL)
Used in DANN (Domain Adversarial Neural Network) for domain adaptation

The GRL reverses gradients during backpropagation, causing the encoder
to learn domain-invariant features that confuse the discriminator.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward: Pass input unchanged
    Backward: Reverse (negate) the gradients and scale by alpha
    
    This tricks the encoder into learning features that CONFUSE
    the domain discriminator instead of helping it.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient!
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """
    Wrapper module for Gradient Reversal
    
    Usage:
        grl = GradientReversalLayer(alpha=1.0)
        reversed_features = grl(features)
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        """Update alpha during training (for scheduling)"""
        self.alpha = alpha


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator Network
    
    Tries to classify whether features come from:
    - Source domain (CWRU) -> 0
    - Target domain (Paderborn) -> 1
    
    The encoder learns to FOOL this discriminator, making features
    indistinguishable between domains.
    """
    
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(alpha=1.0)
        
        # Discriminator network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, alpha=None):
        """
        Forward pass with optional alpha update
        
        Args:
            x: Features from encoder (batch_size, 128)
            alpha: GRL strength (increases during training)
        
        Returns:
            Domain prediction (batch_size, 1) in range [0, 1]
            0 = CWRU (source), 1 = Paderborn (target)
        """
        if alpha is not None:
            self.grl.set_alpha(alpha)
        
        # Apply gradient reversal
        x = self.grl(x)
        
        # Predict domain
        return self.net(x)


def compute_alpha(epoch, max_epochs, gamma=10):
    """
    Compute alpha for GRL scheduling
    
    Alpha starts small and increases during training.
    This helps stabilize early training.
    
    Formula: alpha = 2 / (1 + exp(-gamma * p)) - 1
    where p = epoch / max_epochs
    
    Args:
        epoch: Current epoch
        max_epochs: Total epochs
        gamma: Scheduling parameter (default 10)
    
    Returns:
        alpha value in range [0, 1]
    """
    import math
    p = epoch / max_epochs
    alpha = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
    return alpha
