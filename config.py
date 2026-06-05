"""
config.py  —  SE3AF v4.0 (backward-compatibility shim)
=======================================================
v4.0: All configuration is managed by GLOBAL_CONFIG.py.

EDIT GLOBAL_CONFIG.py to change any setting.
This file imports from GLOBAL_CONFIG.py for backward compatibility.

CONFIG SOURCE + HASH are displayed at startup (V40-02).
"""

from __future__ import annotations

# ── v4.0: import everything from GLOBAL_CONFIG (single source of truth) ──────
from GLOBAL_CONFIG import (
    SE3AF_VERSION,
    BACKEND, USE_SE3, USE_LITE,
    USE_ESM, USE_RF, USE_ALPHAFOLD, USE_GOSS, USE_EMA, USE_SWA,
    TRAINING_MODE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP,
    WARMUP_EPOCHS, SEED,
    DROPOUT, LABEL_SMOOTHING, LABEL_SMOOTHING_END, FEATURE_NOISE_STD,
    MIXUP_ALPHA, STOCHASTIC_DEPTH_P,
    EMA_DECAY, SWA_START_FRACTION, SWA_LR, USE_AMP, AMP_DTYPE,
    FOCAL_GAMMA, FOCAL_ALPHA, STABILITY_LOSS_WEIGHT, INTERACTION_LOSS_WEIGHT,
    SSL_LOSS_WEIGHT, CONTRASTIVE_LOSS_WEIGHT,
    FUSION_DIM, ESM_DIM, GRAPH_HIDDEN, NUM_GRAPH_LAYERS, NUM_HEADS, GOSS_TOP_K,
    USE_PLDDT_ATTENTION, PLDDT_CONFIDENCE_THRESHOLD, PLDDT_LOW_CONFIDENCE_WEIGHT,
    CALIBRATION_ENABLED, TEMPERATURE_INIT,
    VAL_FRACTION, TEST_FRACTION, NUM_WORKERS, PIN_MEMORY,
    USE_SMILES_AUGMENT, SMILES_AUGMENT_P,
    MAX_SCAFFOLD_OVERLAP_FRACTION, MAX_SMILES_OVERLAP_FRACTION,
    CHECKPOINT_DIR, SAVE_EVERY_N_EPOCHS, KEEP_LAST_N,
    EARLY_STOPPING_PATIENCE, OVERFIT_WINDOW, GRAD_ACCUM_STEPS,
    CHECKPOINT_METRIC_WEIGHTS, CHECKPOINT_ARCH_KEYS,
    RF_TREES, RF_DEPTH, RF_FP_BITS, RF_MIN_SAMPLES_LEAF,
    ALPHAFOLD_DIR, AF_EXTRA_DIM,
    GPU_MEMORY_LIMIT_GB,
    DATA_DIR, DATA_FILE, CACHE_DIR, LOG_DIR, REPORTS_DIR,
    # Functions
    get_af_extra_dim, get_config_dict, get_config_hash,
    validate_config as validate,
    print_startup_info, generate_dynamic_flowchart,
    verify_backend, verify_checkpoint_arch,
    CONFIG_SOURCE,
)


def print_config_summary() -> None:
    """Print the v4.0 config summary (delegates to GLOBAL_CONFIG)."""
    print_startup_info()


if __name__ == "__main__":
    print_config_summary()
    print(f"Config hash: {get_config_hash()}")
    print(f"AF extra dim: {get_af_extra_dim()}")
