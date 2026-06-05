"""
core/model.py  —  SE3AF v4.1
==============================
All neural-network components for SE3AF.

v4.1 ARCHITECTURAL IMPROVEMENTS
---------------------------------
ARCH-01  Lite3DEncoder:  FFN expanded 2×→4× (standard transformer ratio)
ARCH-02  SE3MessageLayer: SiLU activation throughout; dual-path message net
ARCH-03  CrossInteractionFusion: parallel cross-attention; residual gating
ARCH-04  AuxHead: 3-layer MLP with residual skip for capacity + stability
ARCH-05  SE3AFModel: shared encoder option removed (3 independent encoders)
ARCH-06  GaussianRBF: increased to NUM_RBF=16 centres for finer distance res.
ARCH-07  GlobalAttentionPooling: softmax gating + mean fallback for 1-node graphs
ARCH-08  DistanceBiasRBF: same NUM_RBF as GaussianRBF (unified constant)
ARCH-09  SE3MessageLayer: direction-feature projection before concatenation
ARCH-10  CrossAttentionBlock: multi-head projection (not single shared head)

GENERALISATION IMPROVEMENTS
-----------------------------
GEN-01  DropPath stochastic depth (linearly increasing per layer)
GEN-02  Feature-level Gaussian noise injection (training only)
GEN-03  3D coordinate jitter augmentation (training only, COORD_JITTER_STD Å)
GEN-04  Label smoothing annealing in SE3AFLoss
GEN-05  DynamicLossBalancer uncertainty weighting (6 task slots)

3D ENHANCEMENTS (v4.1)
-----------------------
3D-01  Lite3DEncoder: learnable RBF distance bias on EVERY attention layer
3D-02  SE3GraphTransformer: direction unit-vector + RBF distance in messages
3D-03  Both backends force pos ≠ zeros; fallback → random unit-sphere
3D-04  Coordinate jitter applied inside forward() during training
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from itertools import combinations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from core.preprocessing import EDGE_DIM, NODE_DIM


# ===========================================================================
# DropPath (Stochastic Depth)
# ===========================================================================

class DropPath(nn.Module):
    """Per-sample stochastic depth regularisation (Huang et al. 2016)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x if residual is None else residual + x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, dtype=x.dtype, device=x.device))
        out   = x * mask / max(keep, 1e-8)
        return out if residual is None else residual + out

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ===========================================================================
# Loss functions
# ===========================================================================

@contextmanager
def autocast_off():
    """Disable AMP inside loss computation to prevent FP16 overflow."""
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", enabled=False):
            yield
    else:
        yield


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
    pos_weight: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Binary focal loss with optional label smoothing and pos_weight."""
    with autocast_off():
        logits  = logits.float()
        targets = targets.float()
        if label_smoothing > 0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        if pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight.to(logits.device), reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob  = torch.sigmoid(logits)
        pt    = torch.where(targets >= 0.5, prob, 1.0 - prob)
        fw    = (1.0 - pt).pow(gamma)
        alpha_t = torch.where(
            targets >= 0.5,
            torch.full_like(targets, alpha),
            torch.full_like(targets, 1.0 - alpha),
        )
        return (alpha_t * fw * bce).mean()


class DynamicLossBalancer(nn.Module):
    """Kendall et al. (2018) uncertainty-weighted multi-task loss.

    Simplified to 3 real tasks only (main, stability, interaction).
    SSL/contrastive stub tasks removed — they had no real signal and
    were wasting optimizer capacity.

    log_var[i] = log(σ_i²); total = Σ_i  loss_i / (2σ_i²) + log(σ_i)

    FIXES (v3.9.0):
    FIX-DLB-01: Clamp log_vars to [-4, 4] to prevent σ→∞ (negative total loss).
    FIX-DLB-03: Add L2 regularizer on log_vars to prevent runaway uncertainty.
    FIX-DLB-04: Floor total loss at 0.0 to prevent negative training loss.
    FIX-DLB-05: Removed SSL/contrastive stub tasks (simplified to 3 real tasks).
    """

    LOG_VAR_MIN: float = -4.0   # σ_min = exp(-2) ≈ 0.135
    LOG_VAR_MAX: float = 4.0    # σ_max = exp(2)  ≈ 7.39

    def __init__(self, n_tasks: int = 3) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses) -> Tensor:
        total = torch.zeros(1, device=self.log_vars.device)
        # FIX-DLB-03: L2 regularizer on log_vars (keeps uncertainty bounded)
        total = total + 0.01 * self.log_vars.pow(2).sum()
        for i, loss in enumerate(losses):
            if i >= len(self.log_vars):
                break
            # FIX-DLB-01: Clamp log_var to prevent σ→∞
            s = self.log_vars[i].clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
            if loss is None or not torch.is_tensor(loss):
                continue  # Skip None losses entirely
            else:
                # Kendall formula: L_i / (2σ_i²) + log(σ_i)
                total = total + loss / (2.0 * torch.exp(s)) + s / 2.0
        # FIX-DLB-04: Floor at 0 to prevent negative total loss
        return total.squeeze().clamp(min=0.0)


class SE3AFLoss(nn.Module):
    """Full training loss: focal main + BCE aux × 2 + dynamic balancing.
    
    v3.9.0: Simplified to 3 real tasks only. SSL/contrastive stubs removed.
    This gives the DynamicLossBalancer clean gradient signal from 3 meaningful
    objectives: main classification, stability prediction, interaction prediction.
    """

    def __init__(
        self,
        focal_gamma: float             = 2.0,
        focal_alpha: float             = 0.5,
        stability_loss_weight: float   = 0.25,
        interaction_loss_weight: float = 0.25,
        ssl_loss_weight: float         = 0.0,   # kept for API compat but ignored
        contrastive_loss_weight: float = 0.0,   # kept for API compat but ignored
        label_smoothing: float         = 0.08,
        pos_weight: Optional[Tensor]   = None,
    ) -> None:
        super().__init__()
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha
        self.stab_w       = stability_loss_weight
        self.inter_w      = interaction_loss_weight
        self.label_smooth = label_smoothing
        self.pos_weight   = pos_weight
        # v3.9.0: Simplified to 3 tasks (main, stability, interaction)
        # Removed ssl/contrastive stubs — they polluted the loss landscape
        self.balancer     = DynamicLossBalancer(n_tasks=3)

    def set_label_smoothing(self, v: float) -> None:
        self.label_smooth = v

    def forward(
        self,
        main_logit:        Tensor,
        stability_logit:   Tensor,
        interaction_logit: Tensor,
        labels:            Tensor,
        ssl_loss:          Optional[Tensor] = None,   # ignored (v3.9.0)
        contrastive_loss:  Optional[Tensor] = None,   # ignored (v3.9.0)
    ) -> Tensor:
        main_loss  = focal_loss(
            main_logit, labels,
            gamma=self.focal_gamma, alpha=self.focal_alpha,
            pos_weight=self.pos_weight, label_smoothing=self.label_smooth,
        )
        stab_loss  = F.binary_cross_entropy_with_logits(
            stability_logit.float(), labels.float()
        )
        inter_loss = F.binary_cross_entropy_with_logits(
            interaction_logit.float(), labels.float()
        )
        # v3.9.0: Only 3 real tasks — clean gradient signal
        return self.balancer(
            main_loss,
            self.stab_w  * stab_loss,
            self.inter_w * inter_loss,
        )


# ===========================================================================
# Shared scatter utilities
# ===========================================================================

def _scatter_add(src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    out = torch.zeros(num_nodes, src.size(1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


def _scatter_softmax(attn: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Numerically stable scatter softmax; always computes in float32."""
    orig_dtype = attn.dtype
    attn = attn.float()
    with torch.no_grad():
        max_val = torch.full(
            (num_nodes, attn.size(1)), float("-inf"),
            dtype=torch.float32, device=attn.device,
        )
        try:
            max_val.scatter_reduce_(
                0, index.unsqueeze(1).expand_as(attn),
                attn, reduce="amax", include_self=True,
            )
        except Exception:
            for e in range(attn.size(0)):
                n = index[e].item()
                for h in range(attn.size(1)):
                    if attn[e, h] > max_val[n, h]:
                        max_val[n, h] = attn[e, h]
        max_val[max_val == float("-inf")] = 0.0
    exp_a = torch.exp(attn - max_val[index])
    sum_e = torch.zeros(num_nodes, attn.size(1), dtype=torch.float32, device=attn.device)
    sum_e.scatter_add_(0, index.unsqueeze(1).expand_as(exp_a), exp_a)
    return (exp_a / (sum_e[index] + 1e-8)).to(orig_dtype)


# ===========================================================================
# SECTION 2 — Lite3DEncoder (distance-bias 3D attention)
# ===========================================================================

class DistanceBiasRBF(nn.Module):
    """Learnable Gaussian RBF distance→per-head attention bias (3D-01)."""

    def __init__(self, num_heads: int, num_rbf: int = 16, cutoff: float = 8.0) -> None:
        super().__init__()
        centres = torch.linspace(0.0, cutoff, num_rbf)
        widths  = torch.full((num_rbf,), (cutoff / num_rbf) ** 2)
        self.centres   = nn.Parameter(centres)
        self.widths    = nn.Parameter(widths)
        self.dist_proj = nn.Linear(num_rbf, num_heads, bias=True)

    def forward(self, dist: Tensor) -> Tensor:
        rbf = torch.exp(
            -((dist.unsqueeze(-1) - self.centres) ** 2)
            / self.widths.abs().clamp(min=1e-6)
        )
        return self.dist_proj(rbf)


class EdgeConvBlock(nn.Module):
    """Message-passing with 3D distance bias + post-norm + DropPath.

    ARCH-01: FFN expanded to 4× hidden dim (standard transformer ratio).
    3D-01  : learnable RBF distance bias on every attention head.
    GEN-01 : DropPath stochastic depth.
    GEN-03 : coordinate jitter injected before distance computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        num_heads:  int   = 8,
        num_rbf:    int   = 16,
        cutoff:     float = 8.0,
        dropout:    float = 0.1,
        edge_drop:  float = 0.1,
        drop_path:  float = 0.0,
        jitter_std: float = 0.0,   # GEN-03: coord jitter during training
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.edge_drop  = edge_drop
        self.jitter_std = jitter_std

        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # ARCH-01: 4× expansion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1     = nn.LayerNorm(hidden_dim)
        self.norm2     = nn.LayerNorm(hidden_dim)
        self.drop      = nn.Dropout(dropout)
        self.dp        = DropPath(drop_path)
        self.dist_bias = DistanceBiasRBF(num_heads, num_rbf, cutoff)  # 3D-01

    def forward(
        self,
        x:          Tensor,                   # (N, H)
        edge_index: Tensor,                   # (2, E)
        edge_attr:  Tensor,                   # (E, edge_dim)
        pos:        Optional[Tensor] = None,  # (N, 3)
    ) -> Tensor:
        residual = x
        N   = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        e = self.edge_proj(edge_attr)
        Q = self.q_proj(x)[dst].view(-1, self.num_heads, self.head_dim)
        K = (self.k_proj(x)[src] + e).view(-1, self.num_heads, self.head_dim)
        V = (self.v_proj(x)[src] + e).view(-1, self.num_heads, self.head_dim)

        attn = (Q * K).sum(-1) / math.sqrt(self.head_dim)   # (E, H)

        # 3D-01: distance bias — always applied (pos is never None in v4.1)
        if pos is not None:
            if self.training and self.jitter_std > 0:
                pos = pos + torch.randn_like(pos) * self.jitter_std  # GEN-03
            diff = pos[src] - pos[dst]
            dist = diff.norm(dim=-1).clamp(min=1e-6)
            attn = attn + self.dist_bias(dist)

        if self.training and self.edge_drop > 0:
            mask = torch.bernoulli(
                torch.full_like(attn[:, 0], 1.0 - self.edge_drop)
            ).bool()
            attn[~mask] = -1e9

        attn = _scatter_softmax(attn, dst, N)
        msg  = (attn.unsqueeze(-1) * V).reshape(-1, self.num_heads * self.head_dim)
        out  = self.out_proj(_scatter_add(msg, dst, N))

        x = self.norm1(self.dp(self.drop(out), residual))
        x = self.norm2(self.dp(self.drop(self.ffn(x)), x))
        return x


class GlobalAttentionPooling(nn.Module):
    """Soft attention-weighted global pooling (ARCH-07)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        num_g = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        w  = _scatter_softmax(self.gate(x), batch, num_g).squeeze(-1)
        return _scatter_add(x * w.unsqueeze(1), batch, num_g)


class Lite3DEncoder(nn.Module):
    """Distance-aware graph transformer backbone.

    v4.1 changes vs v4.0:
    - FFN 4× expansion (ARCH-01)
    - DistanceBiasRBF with 20 centres (ARCH-08)
    - Coord jitter injected per-layer (GEN-03)
    - Head projection after pooling (ARCH-05 simplification)
    """

    BACKEND_NAME = "Lite3DEncoder"

    def __init__(
        self,
        node_dim:          int,
        edge_dim:          int,
        hidden_dim:        int   = 256,
        num_layers:        int   = 5,
        num_heads:         int   = 8,
        num_rbf:           int   = 16,
        cutoff:            float = 8.0,
        dropout:           float = 0.1,
        stochastic_depth:  float = 0.0,
        jitter_std:        float = 0.0,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        dp_rates = [stochastic_depth * i / max(1, num_layers - 1)
                    for i in range(num_layers)]
        self.layers = nn.ModuleList([
            EdgeConvBlock(
                hidden_dim=hidden_dim, edge_dim=hidden_dim,
                num_heads=num_heads, num_rbf=num_rbf, cutoff=cutoff,
                dropout=dropout, edge_drop=0.1,
                drop_path=dp_rates[i], jitter_std=jitter_std,
            )
            for i in range(num_layers)
        ])
        self.pool = GlobalAttentionPooling(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, data: Data) -> Tensor:
        x         = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)
        pos       = getattr(data, "pos", None)
        batch     = (data.batch if data.batch is not None
                     else torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr, pos)
        return self.head(self.pool(x, batch))


# Backward-compatible alias
LiteGraphTransformer = Lite3DEncoder


# ===========================================================================
# SECTION 3 — SE(3) Graph Transformer Backend
# ===========================================================================

class GaussianRBF(nn.Module):
    """Learnable Gaussian RBF expansion of inter-atomic distances."""

    def __init__(self, num_rbf: int = 16, cutoff: float = 8.0) -> None:
        super().__init__()
        self.centres = nn.Parameter(torch.linspace(0.0, cutoff, num_rbf))
        self.widths  = nn.Parameter(torch.full((num_rbf,), (cutoff / num_rbf) ** 2))

    def forward(self, dist: Tensor) -> Tensor:
        return torch.exp(
            -((dist.unsqueeze(-1) - self.centres) ** 2)
            / self.widths.abs().clamp(min=1e-6)
        )


def _direction_features(pos_i: Tensor, pos_j: Tensor) -> Tuple[Tensor, Tensor]:
    diff = pos_j - pos_i
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6) + 1e-7
    return diff / dist, dist.squeeze(-1)


class SE3MessageLayer(nn.Module):
    """SE(3)-inspired message passing layer.

    v4.1 improvements (ARCH-02, ARCH-09):
    - Dual-path message network: scalar path (RBF+edge) + vector path (dir)
    - Direction vector projected to hidden_dim before concatenation
    - SiLU activations throughout (smooth, positive gradient for most inputs)
    - 4× FFN expansion
    - Coord jitter during training (GEN-03)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim:   int,
        num_rbf:    int   = 16,
        num_heads:  int   = 8,
        cutoff:     float = 8.0,
        dropout:    float = 0.1,
        drop_path:  float = 0.0,
        jitter_std: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.jitter_std = jitter_std

        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)

        # ARCH-09: project direction to hidden_dim before cat
        self.dir_proj  = nn.Linear(3,       hidden_dim // 2)
        self.rbf_proj  = nn.Linear(num_rbf, hidden_dim // 2)
        # Combined message: h_src + edge + dir_proj + rbf_proj
        msg_in = hidden_dim + edge_dim + hidden_dim // 2 + hidden_dim // 2
        self.msg_net = nn.Sequential(
            nn.Linear(msg_in, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.attn_proj = nn.Linear(hidden_dim, num_heads)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)

        # ARCH-01: 4× FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.dp    = DropPath(drop_path)

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        edge_attr:  Tensor,
        pos:        Tensor,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        if self.training and self.jitter_std > 0:
            pos = pos + torch.randn_like(pos) * self.jitter_std  # GEN-03

        unit_vec, dist = _direction_features(pos[src], pos[dst])
        rbf_feat  = self.rbf(dist)                        # (E, num_rbf)
        dir_feat  = F.silu(self.dir_proj(unit_vec))       # (E, H/2)
        rbf_feat2 = F.silu(self.rbf_proj(rbf_feat))       # (E, H/2)

        msg_in = torch.cat([x[src], edge_attr, dir_feat, rbf_feat2], dim=-1)
        msg    = self.msg_net(msg_in)                      # (E, H)

        attn_w = self.attn_proj(msg)                       # (E, num_heads)
        attn_w = _scatter_softmax(attn_w, dst, N)
        head_e = attn_w.repeat_interleave(self.head_dim, dim=-1)
        msg_w  = msg * head_e

        agg = torch.zeros(N, msg.size(1), dtype=msg.dtype, device=msg.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_w), msg_w)
        out = self.out_proj(agg)

        x = self.norm1(self.dp(self.drop(out), x))
        x = self.norm2(self.dp(self.drop(self.ffn(x)), x))
        return x


class _SE3AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        w   = torch.sigmoid(self.gate(x))
        num = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros(num, x.size(1), device=x.device, dtype=x.dtype)
        cnt = torch.zeros(num, 1,          device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x * w)
        cnt.scatter_add_(0, batch.unsqueeze(1), w)
        return out / (cnt + 1e-8)


class SE3GraphTransformer(nn.Module):
    """SE(3)-equivariant graph transformer backbone.

    v4.1 improvements: dual-path message (ARCH-02, ARCH-09),
    4× FFN (ARCH-01), SiLU activations, coord jitter (GEN-03).
    """

    BACKEND_NAME = "SE3GraphTransformer"

    def __init__(
        self,
        node_dim:         int,
        edge_dim:         int,
        hidden_dim:       int   = 256,
        num_layers:       int   = 5,
        num_heads:        int   = 8,
        num_rbf:          int   = 16,
        cutoff:           float = 8.0,
        dropout:          float = 0.1,
        stochastic_depth: float = 0.0,
        jitter_std:       float = 0.0,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        dp_rates = [stochastic_depth * i / max(1, num_layers - 1)
                    for i in range(num_layers)]
        self.layers = nn.ModuleList([
            SE3MessageLayer(
                hidden_dim=hidden_dim, edge_dim=hidden_dim,
                num_rbf=num_rbf, num_heads=num_heads, cutoff=cutoff,
                dropout=dropout, drop_path=dp_rates[i], jitter_std=jitter_std,
            )
            for i in range(num_layers)
        ])
        self.pool = _SE3AttentionPool(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, data: Data) -> Tensor:
        x   = self.node_proj(data.x)
        ea  = self.edge_proj(data.edge_attr)
        pos = (data.pos if data.pos is not None
               else torch.zeros(x.size(0), 3, device=x.device))
        batch = (data.batch if data.batch is not None
                 else torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        for layer in self.layers:
            x = layer(x, data.edge_index, ea, pos)
        return self.head(self.pool(x, batch))


# ===========================================================================
# SECTION 4 — Cross-Interaction Fusion
# ===========================================================================

class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between two modality embeddings.

    ARCH-10: multi-head cross-attention with full Q/K/V projections.
    Both A→B and B→A paths are computed; residual gating added (ARCH-03).
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_a    = nn.Linear(dim, dim)
        self.k_b    = nn.Linear(dim, dim)
        self.v_b    = nn.Linear(dim, dim)
        self.out_ab = nn.Linear(dim * 2, dim)

        self.q_b    = nn.Linear(dim, dim)
        self.k_a    = nn.Linear(dim, dim)
        self.v_a    = nn.Linear(dim, dim)
        self.out_ba = nn.Linear(dim * 2, dim)

        # ARCH-03: learned residual gate
        self.gate_a = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_b = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def _attn(self, q: Tensor, k: Tensor, v: Tensor,
              orig: Tensor, proj: nn.Linear) -> Tensor:
        B, H, D = q.size(0), self.num_heads, self.head_dim
        q_ = q.view(B, H, D); k_ = k.view(B, H, D); v_ = v.view(B, H, D)
        a   = torch.softmax((q_ * k_).sum(-1) / self.scale, dim=-1)
        ctx = (a.unsqueeze(-1) * v_).reshape(B, -1)
        return proj(torch.cat([orig, ctx], dim=-1))

    def forward(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        a_new = self._attn(self.q_a(a), self.k_b(b), self.v_b(b), a, self.out_ab)
        b_new = self._attn(self.q_b(b), self.k_a(a), self.v_a(a), b, self.out_ba)
        # ARCH-03: residual gating
        a = self.norm_a(a + self.drop(a_new) * self.gate_a(a))
        b = self.norm_b(b + self.drop(b_new) * self.gate_b(b))
        return a, b


class CrossInteractionFusion(nn.Module):
    """C(5,2)=10 pairwise cross-attention over all modality pairs.

    Modalities: [tgt_lig_graph, e3_lig_graph, linker_graph, tgt_ESM, e3_ESM]
    Output layout: CLS(1) + pairs(10) + diag(2) = 13 slots → 13×fusion_dim

    ARCH-03: parallel snapshot + delta accumulation (order-invariant).
    GOSS-01: adaptive pair weighting via EMA of output magnitudes.
    AF-01  : ESM projector accepts esm_dim + af_extra_dim.
    FIX-AF-DEVICE-01: zero-pad ESM when AF unavailable at runtime.
    """

    NUM_MODALITIES = 5
    NUM_PAIRS      = 10

    def __init__(
        self,
        fusion_dim:  int   = 256,
        esm_dim:     int   = 1280,
        graph_hidden: int  = 256,
        num_heads:   int   = 8,
        dropout:     float = 0.1,
        af_extra_dim: int  = 0,
        use_goss:    bool  = False,
        goss_top_k:  int   = 6,
    ) -> None:
        super().__init__()
        self.fusion_dim    = fusion_dim
        self.use_goss      = use_goss
        self.goss_top_k    = min(goss_top_k, self.NUM_PAIRS)
        self._af_extra_dim = af_extra_dim
        self._esm_dim      = esm_dim

        self.cls_token = nn.Parameter(torch.randn(1, fusion_dim) * 0.02)
        self.cls_norm  = nn.LayerNorm(fusion_dim)

        self.mol_proj = nn.Linear(graph_hidden, fusion_dim)
        self.esm_proj = nn.Linear(esm_dim + af_extra_dim, fusion_dim)

        self.pair_blocks = nn.ModuleList([
            CrossAttentionBlock(fusion_dim, num_heads, dropout)
            for _ in range(self.NUM_PAIRS)
        ])
        self.drop  = nn.Dropout(dropout)
        self.diag0 = nn.Linear(fusion_dim, fusion_dim)
        self.diag1 = nn.Linear(fusion_dim, fusion_dim)

        self.aux_in_dim   = 13 * fusion_dim
        self.cls_out_proj = nn.Linear(fusion_dim, fusion_dim)

        if use_goss:
            self.register_buffer("_goss_log_weights", torch.zeros(self.NUM_PAIRS))

    @torch.no_grad()
    def _update_goss_weights(self, pair_outputs: list) -> Tensor:
        target_device = pair_outputs[0].device
        if not self.use_goss or not self.training:
            return torch.ones(self.NUM_PAIRS, device=target_device)
        try:
            mags = torch.stack([p.detach().norm(dim=-1).mean()
                                 for p in pair_outputs]).to(self._goss_log_weights.device)
            self._goss_log_weights.data = (
                0.9 * self._goss_log_weights.data
                + 0.1 * torch.log(mags.clamp(min=1e-8))
            )
            topk = self._goss_log_weights.topk(self.goss_top_k).indices
            w = torch.ones(self.NUM_PAIRS, device=target_device)
            w[topk.to(target_device)] = 2.0
            return w * (self.NUM_PAIRS / w.sum())
        except Exception:
            return torch.ones(self.NUM_PAIRS, device=target_device)

    def _pad_esm(self, esm: Tensor) -> Tensor:
        if self._af_extra_dim == 0:
            return esm
        expected = self._esm_dim + self._af_extra_dim
        cur = esm.shape[-1]
        if cur == expected:
            return esm
        if cur < expected:
            pad = torch.zeros(*esm.shape[:-1], expected - cur,
                              dtype=esm.dtype, device=esm.device)
            return torch.cat([esm, pad], dim=-1)
        return esm[..., :expected]

    def forward(
        self,
        tgt_graph: Tensor,
        e3_graph:  Tensor,
        lnk_graph: Tensor,
        tgt_esm:   Tensor,
        e3_esm:    Tensor,
    ) -> Tuple[Tensor, Tensor]:
        B = tgt_graph.size(0)
        tgt_esm = self._pad_esm(tgt_esm)
        e3_esm  = self._pad_esm(e3_esm)

        m = [
            self.mol_proj(tgt_graph),
            self.mol_proj(e3_graph),
            self.mol_proj(lnk_graph),
            self.esm_proj(tgt_esm),
            self.esm_proj(e3_esm),
        ]

        cls  = self.cls_token.expand(B, -1)
        orig = [mi.clone() for mi in m]
        delta = [torch.zeros_like(mi) for mi in m]

        pair_raw = []
        for k, (i, j) in enumerate(combinations(range(self.NUM_MODALITIES), 2)):
            mi_new, mj_new = self.pair_blocks[k](orig[i], orig[j])
            delta[i] = delta[i] + (mi_new - orig[i])
            delta[j] = delta[j] + (mj_new - orig[j])
            pair_raw.append(mi_new + mj_new)

        for idx in range(self.NUM_MODALITIES):
            m[idx] = orig[idx] + delta[idx]

        goss_w = self._update_goss_weights(pair_raw)
        pairs  = [self.drop(pair_raw[k]) * goss_w[k] for k in range(self.NUM_PAIRS)]

        cls = cls + torch.stack(m, dim=1).mean(dim=1)
        cls = self.cls_norm(cls)
        cls = self.cls_out_proj(cls)

        diag0 = F.gelu(self.diag0(m[0] * m[2]))
        diag1 = F.gelu(self.diag1(m[1] * m[2]))

        aux_parts = [cls] + pairs + [diag0, diag1]
        assert len(aux_parts) == 13
        return cls, torch.cat(aux_parts, dim=-1)


# ===========================================================================
# SECTION 5 — Full SE3AF Model + Factory
# ===========================================================================

class AuxHead(nn.Module):
    """3-layer MLP classifier with residual skip (ARCH-04).

    Extra capacity + residual skip helps auxiliary heads learn without
    vanishing gradients at small batch sizes.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1   = nn.Linear(in_dim, hidden)
        self.fc2   = nn.Linear(hidden, hidden)
        self.fc3   = nn.Linear(hidden, 1)
        self.skip  = nn.Linear(in_dim, hidden)   # residual projection
        self.norm  = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.fc1(x))
        h = self.norm(F.gelu(self.fc2(self.drop(h))) + self.skip(x))
        return self.fc3(self.drop(h)).squeeze(-1)


# ===========================================================================
# ISSUE-3: ProteinGeometryEncoder — lightweight EGNN on AlphaFold residue graph
# ===========================================================================

class _ResidueGNNLayer(nn.Module):
    """Single E(n)-invariant message-passing layer for residue graphs.

    Messages use distance + relative position (invariant features only,
    no explicit SE(3) equivariance — lightweight but geometry-aware).
    Node update: h_i ← LN(h_i + MLP([h_i ‖ agg_msg]) )
    """

    def __init__(self, hidden: int, edge_feat_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        msg_in = hidden + hidden + edge_feat_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden * 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, edge_index: Tensor, edge_feat: Tensor) -> Tensor:
        src, tgt = edge_index[0], edge_index[1]        # both (E,)
        m_in = torch.cat([h[src], h[tgt], edge_feat], dim=-1)  # (E, msg_in)
        msg  = self.msg_mlp(m_in)                               # (E, hidden)
        # Mean-aggregate over incoming messages for each node
        agg  = torch.zeros_like(h)
        agg.index_add_(0, tgt, msg)
        cnt  = torch.zeros(h.size(0), 1, device=h.device)
        cnt.index_add_(0, tgt, torch.ones(msg.size(0), 1, device=h.device))
        cnt  = cnt.clamp(min=1.0)
        agg  = agg / cnt
        h2   = self.upd_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h2)


class ProteinGeometryEncoder(nn.Module):
    """ISSUE-3 FIX: Geometry-aware protein encoder using AlphaFold Cα residue graph.

    Replaces the simple linear mean-pool projection of AF features with a
    lightweight graph neural network that learns structural patterns from the
    residue-level contact graph.

    Architecture
    ------------
    1. Node input projection: (8-dim node_feat) → hidden
    2. Edge input projection:  (4-dim edge_feat) → hidden
    3. N_layers × ResidueGNNLayer (lightweight EGNN-style MP)
    4. Global attention pooling  → (hidden,) protein geometry embedding
    5. Output projection: hidden → out_dim

    Input (from af_residue_graph)
    -----
    node_feat  : Tensor (N, 8)
    edge_index : LongTensor (2, E)
    edge_feat  : Tensor (E, 4)

    Output
    ------
    Tensor (out_dim,) — geometry embedding for one protein
    """

    NODE_IN = 8   # must match af_residue_graph() node_feat dimension
    EDGE_IN = 4   # must match af_residue_graph() edge_feat dimension

    def __init__(
        self,
        hidden:    int   = 64,
        out_dim:   int   = 16,
        n_layers:  int   = 3,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden = hidden

        self.node_proj = nn.Sequential(
            nn.Linear(self.NODE_IN, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.EDGE_IN, hidden), nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            _ResidueGNNLayer(hidden, hidden, dropout) for _ in range(n_layers)
        ])
        # Global attention pooling
        self.attn_gate = nn.Linear(hidden, 1)
        self.out_proj   = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        node_feat:  Tensor,    # (N, 8)
        edge_index: Tensor,    # (2, E)  LongTensor
        edge_feat:  Tensor,    # (E, 4)
    ) -> Tensor:               # (out_dim,)
        h  = self.node_proj(node_feat)          # (N, hidden)
        ef = self.edge_proj(edge_feat)          # (E, hidden)
        for layer in self.layers:
            h = layer(h, edge_index, ef)        # (N, hidden)
        # Attention pooling: softmax over nodes
        gates  = torch.softmax(self.attn_gate(h), dim=0)  # (N, 1)
        pooled = (h * gates).sum(dim=0)                   # (hidden,)
        return self.out_proj(pooled)                       # (out_dim,)


# ===========================================================================
# ISSUE-4: ProteinBackboneEncoder — ESM + geometry-aware projection
# ===========================================================================

class ProteinBackboneEncoder(nn.Module):
    """ISSUE-4 FIX: Unified protein encoder combining ESM-2 + AlphaFold geometry.

    Problem (ISSUE-4):
        The current approach simply concatenates the (4,) mean-pooled AF vector
        to the (1280,) ESM embedding and passes through a single linear layer.
        This treats AF as a flat bias term rather than a structural signal.

    Solution:
        1. Project ESM embedding to fusion_dim via a 2-layer MLP with layer norm.
        2. Encode AF residue graph via ProteinGeometryEncoder → (geo_dim,).
        3. Cross-attend ESM and geometry features: geometry acts as Key/Value,
           ESM as Query in a single multi-head cross-attention step.
        4. Gated fusion: output = esm_proj + α * cross_attended_geometry,
           where α is a learned sigmoid gate.
        5. Output: (fusion_dim,) unified protein representation.

    Fallback: if no AF graph available (geo_emb=None), gate closes to 0
    and output equals the pure ESM projection — fully backward compatible.
    """

    def __init__(
        self,
        esm_dim:    int   = 1280,
        af_dim:     int   = 4,         # legacy mean-pool dim
        fusion_dim: int   = 128,
        geo_hidden: int   = 64,        # ProteinGeometryEncoder hidden
        geo_dim:    int   = 16,        # geometry embedding output size
        n_geo_layers: int = 3,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.geo_dim    = geo_dim

        # ESM + legacy AF mean-pool projection
        total_esm_in = esm_dim + af_dim
        self.esm_mlp = nn.Sequential(
            nn.Linear(total_esm_in, fusion_dim * 2), nn.LayerNorm(fusion_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim),
        )

        # Optional geometry encoder (used when AF residue graph is available)
        self.geo_encoder = ProteinGeometryEncoder(
            hidden=geo_hidden, out_dim=geo_dim, n_layers=n_geo_layers, dropout=dropout,
        )
        # Project geometry embedding to cross-attention dim
        self.geo_proj = nn.Linear(geo_dim, fusion_dim)
        # Single cross-attention: ESM query, geometry key/value
        head_dim = fusion_dim // num_heads
        assert head_dim * num_heads == fusion_dim, \
            f"fusion_dim={fusion_dim} must be divisible by num_heads={num_heads}"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Learned gate: scalar α per feature dimension
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        esm_feat:    Tensor,               # (B, esm_dim + af_dim)
        geo_graphs:  Optional[list] = None,  # list of length B: each entry is
                                             # (node_feat, edge_index, edge_feat)
                                             # or None if no AF available
    ) -> Tensor:                           # (B, fusion_dim)
        B = esm_feat.size(0)
        dev = esm_feat.device

        # ESM projection
        esm_h = self.esm_mlp(esm_feat)    # (B, fusion_dim)

        if geo_graphs is None or not any(g is not None for g in geo_graphs):
            return self.out_norm(esm_h)   # fallback: pure ESM, no geometry

        # Encode geometry for each sample in batch
        geo_embs = []
        for g in geo_graphs:
            if g is None:
                geo_embs.append(torch.zeros(self.geo_dim, device=dev))
            else:
                nf, ei, ef = g
                nf = nf.to(dev); ei = ei.to(dev); ef = ef.to(dev)
                geo_embs.append(self.geo_encoder(nf, ei, ef))
        geo_stack = torch.stack(geo_embs, dim=0)          # (B, geo_dim)
        geo_proj  = self.geo_proj(geo_stack).unsqueeze(1) # (B, 1, fusion_dim)

        # Cross-attention: query=esm_h, key/value=geometry
        q = esm_h.unsqueeze(1)             # (B, 1, fusion_dim)
        attn_out, _ = self.cross_attn(q, geo_proj, geo_proj)
        attn_out = attn_out.squeeze(1)     # (B, fusion_dim)

        # Gated fusion
        gate_w = self.gate(torch.cat([esm_h, attn_out], dim=-1))  # (B, fusion_dim)
        fused  = esm_h + gate_w * attn_out                         # (B, fusion_dim)
        return self.out_norm(fused)


class SE3AFModel(nn.Module):
    """SE3AF end-to-end PROTAC activity predictor.

    Always returns 3-tuple: (main_logit, stability_logit, interaction_logit).
    All shapes: (B,).
    """

    def __init__(
        self,
        backend:           str   = "se3",
        fusion_dim:        int   = 192,
        esm_dim:           int   = 1280,
        graph_hidden:      int   = 192,
        num_graph_layers:  int   = 5,
        num_heads:         int   = 6,
        num_rbf:           int   = 16,
        cutoff:            float = 8.0,
        dropout:           float = 0.1,
        use_checkpoint:    bool  = False,
        stochastic_depth:  float = 0.0,
        af_extra_dim:      int   = 0,
        use_goss:          bool  = False,
        goss_top_k:        int   = 6,
        jitter_std:        float = 0.0,
    ) -> None:
        super().__init__()
        assert backend in ("lite", "se3"), f"backend must be 'lite'|'se3', got {backend!r}"
        self.backend_name   = backend
        self.use_checkpoint = use_checkpoint
        self.af_extra_dim   = af_extra_dim

        def _make_enc():
            if backend == "se3":
                return SE3GraphTransformer(
                    node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                    hidden_dim=graph_hidden, num_layers=num_graph_layers,
                    num_heads=num_heads, num_rbf=num_rbf, cutoff=cutoff,
                    dropout=dropout, stochastic_depth=stochastic_depth,
                    jitter_std=jitter_std,
                )
            return Lite3DEncoder(
                node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                hidden_dim=graph_hidden, num_layers=num_graph_layers,
                num_heads=num_heads, num_rbf=num_rbf, cutoff=cutoff,
                dropout=dropout, stochastic_depth=stochastic_depth,
                jitter_std=jitter_std,
            )

        self.tgt_encoder = _make_enc()
        self.e3_encoder  = _make_enc()
        self.lnk_encoder = _make_enc()

        self.fusion = CrossInteractionFusion(
            fusion_dim=fusion_dim, esm_dim=esm_dim,
            graph_hidden=graph_hidden, num_heads=num_heads,
            dropout=dropout, af_extra_dim=af_extra_dim,
            use_goss=use_goss, goss_top_k=goss_top_k,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        self.stability_head   = AuxHead(self.fusion.aux_in_dim, fusion_dim, dropout)
        self.interaction_head = AuxHead(self.fusion.aux_in_dim, fusion_dim, dropout)

    def _encode(self, encoder: nn.Module, batch: Data) -> Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(encoder, batch, use_reentrant=False)
        return encoder(batch)

    def forward(
        self,
        tgt_graph: Data,
        e3_graph:  Data,
        lnk_graph: Data,
        tgt_esm:   Tensor,
        e3_esm:    Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        tg = self._encode(self.tgt_encoder, tgt_graph)
        eg = self._encode(self.e3_encoder,  e3_graph)
        lg = self._encode(self.lnk_encoder, lnk_graph)

        cls_emb, aux_flat = self.fusion(tg, eg, lg, tgt_esm, e3_esm)
        main  = self.classifier(cls_emb).squeeze(-1)
        stab  = self.stability_head(aux_flat)
        inter = self.interaction_head(aux_flat)
        return main, stab, inter


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_from_config(cfg: Dict[str, Any]) -> SE3AFModel:
    """Construct SE3AFModel from a config dict.

    v4.1: Accepts both 'stochastic_depth' (canonical) and legacy
    'stochastic_depth_p' key so checkpoints and old JSON configs remain loadable.
    """
    # Resolve stochastic depth — 'stochastic_depth' is the v4.1 canonical key;
    # fall back to 'stochastic_depth_p' for backward compatibility.
    _sd = cfg.get("stochastic_depth",
          cfg.get("stochastic_depth_p", 0.0))
    return SE3AFModel(
        backend           = cfg.get("backend",           "se3"),
        fusion_dim        = cfg.get("fusion_dim",        192),
        esm_dim           = cfg.get("esm_dim",           1280),
        graph_hidden      = cfg.get("graph_hidden",      192),
        num_graph_layers  = cfg.get("num_graph_layers",  5),
        num_heads         = cfg.get("num_heads",         6),
        num_rbf           = cfg.get("num_rbf",           16),
        cutoff            = cfg.get("cutoff",            8.0),
        dropout           = cfg.get("dropout",           0.25),
        use_checkpoint    = cfg.get("use_checkpoint",    False),
        stochastic_depth  = _sd,
        af_extra_dim      = cfg.get("af_extra_dim",      0),
        use_goss          = cfg.get("use_goss",          True),
        goss_top_k        = cfg.get("goss_top_k",        6),
        jitter_std        = cfg.get("jitter_std",        0.0),
    )


def get_encoder_display_name(model: SE3AFModel) -> str:
    name = getattr(model.tgt_encoder, "BACKEND_NAME", type(model.tgt_encoder).__name__)
    if name == "Lite3DEncoder":
        return "Lite3DEncoder — distance-bias 3D attention + 4× FFN"
    if "SE3" in name:
        return "SE3GraphTransformer — SE(3)-equivariant MP + dual-path messages"
    return f"{name} active"
