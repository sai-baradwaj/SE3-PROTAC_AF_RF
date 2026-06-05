"""
models/fusion.py  —  SE3AF v3.7 Cross-Interaction Fusion
=========================================================
CrossInteractionFusion computes all C(5,2)=10 pairwise cross-attention
combinations between the 5 modalities:
    - tgt_graph embedding (graph_hidden,)
    - e3_graph embedding  (graph_hidden,)
    - lnk_graph embedding (graph_hidden,)
    - tgt_esm embedding   (esm_dim,) → projected to (fusion_dim,)
    - e3_esm embedding    (esm_dim,) → projected to (fusion_dim,)

The 10 pairs are:
    tgt_graph × e3_graph    (target ligand ↔ E3 ligand)
    tgt_graph × lnk_graph   (target ligand ↔ linker)
    tgt_graph × tgt_esm     (target ligand ↔ target protein)
    tgt_graph × e3_esm      (target ligand ↔ E3 protein)
    e3_graph  × lnk_graph   (E3 ligand ↔ linker)
    e3_graph  × tgt_esm     (E3 ligand ↔ target protein)
    e3_graph  × e3_esm      (E3 ligand ↔ E3 protein)
    lnk_graph × tgt_esm     (linker ↔ target protein)
    lnk_graph × e3_esm      (linker ↔ E3 protein)
    tgt_esm   × e3_esm      (target protein ↔ E3 protein)

This comprehensive cross-attention models ternary complex formation
by capturing all pairwise interactions that determine PROTAC activity.
"""

from core.model import CrossAttentionBlock, CrossInteractionFusion, AuxHead

__all__ = ["CrossAttentionBlock", "CrossInteractionFusion", "AuxHead"]
