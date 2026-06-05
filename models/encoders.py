"""
models/encoders.py  —  SE3AF v3.7 Graph Encoder Backends
=========================================================
V37-BUG-05 FIX: This file now documents the encoder API clearly
instead of being an empty shim.

The canonical implementations are in core/model.py.
This module re-exports them with proper documentation.

Graph Encoder Backends
----------------------
Lite3DEncoder:
    Distance-bias multi-head attention graph encoder.
    Faster than SE3, suitable for CPU/limited VRAM.
    Uses DistanceBiasRBF to add 3D distance information to attention.
    Architecture: EdgeConvBlock × N → GlobalAttentionPooling → (graph_hidden,)

SE3GraphTransformer:
    SE(3)-equivariant message passing graph transformer.
    Respects rotational and translational symmetry of molecular geometry.
    More expressive than Lite for 3D-sensitive properties.
    Architecture: SE3MessageLayer × N → _SE3AttentionPool → (graph_hidden,)

Backend Selection (from config.py)
-----------------------------------
    BACKEND = "se3"   → SE3GraphTransformer
    BACKEND = "lite"  → Lite3DEncoder

Changing BACKEND in config.py automatically selects the correct encoder
without any other code changes.
"""

from core.model import (
    DropPath,
    DistanceBiasRBF,
    EdgeConvBlock,
    GlobalAttentionPooling,
    Lite3DEncoder,
    GaussianRBF,
    SE3MessageLayer,
    SE3GraphTransformer,
)

__all__ = [
    "DropPath",
    "DistanceBiasRBF",
    "EdgeConvBlock",
    "GlobalAttentionPooling",
    "Lite3DEncoder",
    "GaussianRBF",
    "SE3MessageLayer",
    "SE3GraphTransformer",
]
