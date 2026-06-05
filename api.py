#!/usr/bin/env python3
"""
api.py
======
SE3AF REST API server  —  v3.8.0

A lightweight JSON REST API for programmatic PROTAC activity prediction.
Runs on http://localhost:5001 by default (different port from app.py / 5000).

Endpoints
---------
GET  /api/v1/health          — health check + model status
POST /api/v1/predict         — single PROTAC prediction (JSON body)
POST /api/v1/predict_batch   — batch prediction (JSON array body)
GET  /api/v1/model_info      — model metadata (version, backend, config)
GET  /api/v1/version         — API version string

Example: single prediction
--------------------------
curl -s -X POST http://localhost:5001/api/v1/predict \
     -H "Content-Type: application/json" \
     -d '{
       "smiles_target_lig": "CC(=O)Nc1ccc(Cl)cc1",
       "smiles_e3_lig":     "O=C1CN(c2ccc(F)cc2)C(=O)c2ccccc21",
       "smiles_linker":     "CCOCCOCCOCCN",
       "target_seq":        "",
       "ligase_seq":        ""
     }'

Expected response:
{
  "status":      "ok",
  "probability": 0.7234,
  "active":      true,
  "confidence":  "high"
}

Example: batch prediction
--------------------------
curl -s -X POST http://localhost:5001/api/v1/predict_batch \
     -H "Content-Type: application/json" \
     -d '[
       {"smiles_target_lig": "CCO", "smiles_e3_lig": "c1ccccc1", "smiles_linker": "CCCC"},
       {"smiles_target_lig": "CCN", "smiles_e3_lig": "c1ccccc1", "smiles_linker": "CCCC"}
     ]'
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, jsonify, request

# ─── App ─────────────────────────────────────────────────────────────────────
api_app = Flask(__name__)

# ─── Globals ─────────────────────────────────────────────────────────────────
_model        = None
_device       = None
_cfg          = None
_model_loaded = False

CHECKPOINT = Path("checkpoints/best_model.pt")
CACHE_DIR  = ".cache/"
API_VERSION = "3.8.0"
API_PORT    = int(os.environ.get("SE3AF_API_PORT", 5001))


def _init_model() -> None:
    """Load SE3AF model exactly once."""
    global _model, _device, _cfg, _model_loaded
    if _model_loaded:
        return
    if not CHECKPOINT.exists():
        return
    try:
        from core.inference import load_model_for_inference
        _model, _device = load_model_for_inference(
            checkpoint_path=str(CHECKPOINT),
            model_cfg=None,
        )
        _model_loaded = True
    except Exception as exc:
        print(f"[api.py] Model load failed: {exc}", file=sys.stderr)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _predict_single(smiles_tgt: str, smiles_e3: str, smiles_lnk: str,
                    tgt_seq: str = "", e3_seq: str = "") -> dict:
    """Run inference on one PROTAC triplet. Returns dict with probability."""
    import pandas as pd
    from core.dataset import PROTACDataset
    from core.trainer import SE3AFTrainer, TrainerConfig
    from core.utils import COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES

    if not _model_loaded:
        raise RuntimeError("Model not loaded — train first or ensure checkpoint exists.")

    row = {
        COL_TGT_SMILES:  smiles_tgt,
        COL_E3_SMILES:   smiles_e3,
        COL_LNK_SMILES:  smiles_lnk,
        "target_seq":    tgt_seq or "",
        "ligase_seq":    e3_seq or "",
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        pd.DataFrame([row]).to_csv(tmp_path, index=False)
        os.close(tmp_fd)

        ds = PROTACDataset(data_root=tmp_path, cache_dir=CACHE_DIR)
        from core.inference import predict_from_dataset
        probs = predict_from_dataset(_model, ds, _device)
        prob  = float(probs[0]) if len(probs) > 0 else 0.5
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    confidence = "high" if abs(prob - 0.5) > 0.3 else ("medium" if abs(prob - 0.5) > 0.15 else "low")
    return {
        "probability": round(prob, 6),
        "active":      prob >= 0.5,
        "confidence":  confidence,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@api_app.route("/api/v1/health", methods=["GET"])
def health():
    _init_model()
    return jsonify({
        "status":       "ok",
        "model_ready":  _model_loaded,
        "checkpoint":   str(CHECKPOINT),
        "api_version":  API_VERSION,
    })


@api_app.route("/api/v1/version", methods=["GET"])
def version():
    return jsonify({
        "version":     API_VERSION,
        "description": "SE3AF REST API",
    })


@api_app.route("/api/v1/model_info", methods=["GET"])
def model_info():
    _init_model()
    info: dict = {
        "api_version":  API_VERSION,
        "model_ready":  _model_loaded,
        "checkpoint":   str(CHECKPOINT),
    }
    if _model_loaded and _model is not None:
        try:
            info["model_class"] = type(_model).__name__
            info["num_params"]  = sum(p.numel() for p in _model.parameters())
        except Exception:
            pass
    return jsonify(info)


@api_app.route("/api/v1/predict", methods=["POST"])
def predict():
    _init_model()
    if not _model_loaded:
        return jsonify({"status": "error", "message": "Model not loaded. Train first."}), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON body."}), 400

    smiles_tgt = data.get("smiles_target_lig", "").strip()
    smiles_e3  = data.get("smiles_e3_lig",     "").strip()
    smiles_lnk = data.get("smiles_linker",     "").strip()
    tgt_seq    = data.get("target_seq",        "") or ""
    e3_seq     = data.get("ligase_seq",        "") or ""

    if not smiles_tgt or not smiles_e3 or not smiles_lnk:
        return jsonify({
            "status":  "error",
            "message": "Required fields: smiles_target_lig, smiles_e3_lig, smiles_linker"
        }), 400

    try:
        result = _predict_single(smiles_tgt, smiles_e3, smiles_lnk, tgt_seq, e3_seq)
        return jsonify({"status": "ok", **result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc),
                        "trace": traceback.format_exc()[-500:]}), 500


@api_app.route("/api/v1/predict_batch", methods=["POST"])
def predict_batch():
    _init_model()
    if not _model_loaded:
        return jsonify({"status": "error", "message": "Model not loaded. Train first."}), 503

    data = request.get_json(force=True, silent=True)
    if not isinstance(data, list):
        return jsonify({"status": "error", "message": "Body must be a JSON array."}), 400

    results = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            results.append({"index": idx, "status": "error", "message": "Not a dict"})
            continue
        smiles_tgt = item.get("smiles_target_lig", "").strip()
        smiles_e3  = item.get("smiles_e3_lig",     "").strip()
        smiles_lnk = item.get("smiles_linker",     "").strip()
        tgt_seq    = item.get("target_seq",        "") or ""
        e3_seq     = item.get("ligase_seq",        "") or ""
        try:
            result = _predict_single(smiles_tgt, smiles_e3, smiles_lnk, tgt_seq, e3_seq)
            results.append({"index": idx, "status": "ok", **result})
        except Exception as exc:
            results.append({"index": idx, "status": "error", "message": str(exc)})

    return jsonify({"status": "ok", "count": len(results), "results": results})


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║   SE3AF REST API  v{API_VERSION}                      ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║   Listening on: http://localhost:{API_PORT}          ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║   Endpoints:                                     ║")
    print("║     GET  /api/v1/health                          ║")
    print("║     GET  /api/v1/version                         ║")
    print("║     GET  /api/v1/model_info                      ║")
    print("║     POST /api/v1/predict                         ║")
    print("║     POST /api/v1/predict_batch                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    _init_model()
    if _model_loaded:
        print(f"  Model loaded: {CHECKPOINT}")
    else:
        print(f"  ⚠  Model NOT loaded (checkpoint not found: {CHECKPOINT})")
        print("     Train the model first: python main.py train --config configs/train_config.json --data data/")
    print()
    api_app.run(host="0.0.0.0", port=API_PORT, debug=False)


if __name__ == "__main__":
    main()
