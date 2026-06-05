"""
models/losses.py  —  SE3AF v3.7 Loss Functions
================================================
V37-BUG-04 FIX: Previous models/losses.py had an INCORRECT implementation of
DynamicLossBalancer using `precision * loss + log_vars[i]` instead of the
correct Kendall (2018) formula: `loss / (2*exp(s)) + s/2`.

This file now simply re-exports the CANONICAL loss functions from core/model.py
to eliminate the duplication and incorrect implementation.

Loss Architecture
-----------------
SE3AFLoss:
    Combined multi-task loss using DynamicLossBalancer:
    - Focal loss (main classification, FIXED: alpha=0.5 for balanced data)
    - Stability auxiliary loss (BCE on auxiliary head)
    - Interaction auxiliary loss (BCE on auxiliary head)
    - SSL/Contrastive (disabled, weight=0.0)

DynamicLossBalancer:
    Kendall (2018) uncertainty-based multi-task weighting.
    L_total = Σᵢ [ Lᵢ / (2σᵢ²) + log(σᵢ) ]
    σᵢ are learned uncertainty parameters.
    VALID to produce negative total loss when σᵢ > 1 (high uncertainty tasks).

focal_loss():
    Binary focal loss with label smoothing.
    gamma=2.0 down-weights easy examples.
    alpha=0.5 (FIXED from 0.25 — v3.7 change for balanced datasets).
"""

# V37-BUG-04 FIX: Import canonical implementations from core/model.py
from core.model import (
    SE3AFLoss,
    DynamicLossBalancer,
    focal_loss,
    autocast_off,
)

__all__ = [
    "SE3AFLoss",
    "DynamicLossBalancer",
    "focal_loss",
    "autocast_off",
]
