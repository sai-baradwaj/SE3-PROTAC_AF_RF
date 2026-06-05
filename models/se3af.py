"""
models/se3af.py  —  SE3AF v3.7 Main Model + Factory
====================================================
SE3AFModel:
    Full multimodal PROTAC activity predictor.
    Combines SE3GraphTransformer (or Lite3DEncoder) + ESM + CrossInteractionFusion
    + optional AlphaFold structural features.

    Forward signature:
        model(tgt_graph, e3_graph, lnk_graph, tgt_esm, e3_esm)
        → (main_logit, stability_logit, interaction_logit)

    All three outputs are raw logits (before sigmoid).
    Apply torch.sigmoid() for probabilities.

build_from_config(cfg_dict):
    Factory function. Creates SE3AFModel from a configuration dict.
    Keys: backend, fusion_dim, esm_dim, graph_hidden, num_graph_layers,
          num_heads, dropout, stochastic_depth_p, af_extra_dim,
          use_goss, goss_top_k

get_encoder_display_name(backend):
    Returns human-readable encoder name for UI display.
"""

from core.model import SE3AFModel, build_from_config, get_encoder_display_name

__all__ = ["SE3AFModel", "build_from_config", "get_encoder_display_name"]
