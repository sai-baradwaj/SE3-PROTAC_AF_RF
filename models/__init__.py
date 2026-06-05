"""
models/  —  SE3AF v3.7 Model Components Package
================================================
Proper architecture modules (not empty shims).

V37-BUG-05 FIX: Previous versions had models/encoders.py, fusion.py, se3af.py
as empty shims that just re-imported from core/model.py. This was misleading.

The canonical implementation lives in core/model.py (single source of truth).
This package provides clean, properly-named re-exports for external use.

V37-BUG-04 FIX: models/losses.py previously had WRONG DynamicLossBalancer formula.
Removed. All loss functions now come from core/model.py.
"""

from core.model import (
    # Loss functions (canonical implementation)
    SE3AFLoss,
    DynamicLossBalancer,
    focal_loss,

    # Encoders
    Lite3DEncoder,
    SE3GraphTransformer,
    DropPath,
    DistanceBiasRBF,
    EdgeConvBlock,
    GlobalAttentionPooling,

    # Fusion
    CrossInteractionFusion,
    CrossAttentionBlock,
    AuxHead,

    # Model
    SE3AFModel,
    build_from_config,
    get_encoder_display_name,
)

__all__ = [
    # Loss
    "SE3AFLoss",
    "DynamicLossBalancer",
    "focal_loss",
    # Encoders
    "Lite3DEncoder",
    "SE3GraphTransformer",
    "DropPath",
    "DistanceBiasRBF",
    "EdgeConvBlock",
    "GlobalAttentionPooling",
    # Fusion
    "CrossInteractionFusion",
    "CrossAttentionBlock",
    "AuxHead",
    # Model
    "SE3AFModel",
    "build_from_config",
    "get_encoder_display_name",
]
