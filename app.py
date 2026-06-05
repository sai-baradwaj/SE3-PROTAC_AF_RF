#!/usr/bin/env python3
"""
app.py — SE3AF v3.9.0  PROTACPred Backend
==========================================
Pure Flask backend — all HTML lives in templates/.
No demo_heuristic code. Real SE3AF inference only.

Endpoints
---------
GET  /                      — Main dashboard (index.html)
GET  /history_page          — History page
GET  /datasets              — Dataset management page
GET  /reports_page          — Reports page
GET  /experiments           — Experiments page (alias to dashboard)

POST /predict               — Single PROTAC prediction
POST /batch_predict         — Batch CSV prediction
POST /load_structure        — SMILES → SDF
POST /alphafold             — UniProt → AlphaFold PDB
GET  /history               — JSON history list
GET  /reports               — JSON report list
POST /generate_report       — Generate PDF/CSV/JSON report
GET  /download/<filename>   — Download file
POST /clear_history         — Clear prediction history JSON

GET  /api/health            — Health check
GET  /api/model_info        — Model metadata
POST /api/load_model        — Trigger model load
POST /api/random_sample     — Random dataset sample
GET  /download_report/<name>— Download scientific report
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import threading
import traceback
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple


warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, url_for)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "protacpred-se3af-v39-secret")

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CHECKPOINT  = BASE_DIR / "checkpoints" / "best_model.pt"
DATA_DIR    = BASE_DIR / "data"
SCAFFOLD_DIR= BASE_DIR / "data" / "split_scaffold"
AF_DIR      = BASE_DIR / "data" / "alphafold"
CACHE_DIR   = str(BASE_DIR / ".cache")
HISTORY_FILE= BASE_DIR / "history" / "predictions.json"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"
PORT        = int(os.getenv("PORT", 5000))
import requests

HF_MODEL_URL = "https://huggingface.co/bharath12301/SE3-PROTAC_AF_RF/resolve/main/best_model.pt"

HF_RF_URL = "https://huggingface.co/bharath12301/SE3-PROTAC_AF_RF/resolve/main/rf_stacker.joblib"

RF_PATH = BASE_DIR / "checkpoints" / "rf_stacker.joblib"


def download_models_if_missing():

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    files = {
        CHECKPOINT: HF_MODEL_URL,
        RF_PATH: HF_RF_URL
    }

    for local_file, url in files.items():

        if not local_file.exists():

            print(f"[app] Downloading {local_file.name}...")

            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open(local_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"[app] Downloaded {local_file.name}")
for d in [HISTORY_FILE.parent, REPORTS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Globals ──────────────────────────────────────────────────────
_model        = None
_device       = None
_cfg          = None
_model_loaded = False
_model_loading = False
_struct_cache: Dict[str, str] = {}
_checkpoint_auroc: Optional[float] = None
_checkpoint_epoch: Optional[int]   = None


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def _read_checkpoint_meta() -> None:
    global _checkpoint_auroc, _checkpoint_epoch
    if not CHECKPOINT.exists():
        return
    try:
        import torch
        ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            _checkpoint_auroc = ckpt.get("best_score") or ckpt.get("val_auroc")
            _checkpoint_epoch = ckpt.get("epoch")
    except Exception as e:
        print(f"[app] Could not read checkpoint meta: {e}")


def _load_model_background() -> None:
    global _model, _device, _cfg, _model_loaded, _model_loading
    _model_loading = True
    try:
        import torch
        from core.trainer import SE3AFTrainer, TrainerConfig
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = TrainerConfig.from_global_config()
        trainer = SE3AFTrainer(cfg)
        trainer._device = _device
        # Load checkpoint
        ckpt = torch.load(str(CHECKPOINT), map_location=_device, weights_only=False)
        from core.model import build_from_config
        arch = ckpt.get("arch", {})
        model_cfg = {
            "backend":        arch.get("backend", "se3"),
            "fusion_dim":     arch.get("fusion_dim", 128),
            "esm_dim":        arch.get("esm_dim", 1280),
            "graph_hidden":   arch.get("graph_hidden", 128),
            "num_graph_layers": arch.get("num_graph_layers", 4),
            "num_heads":      arch.get("num_heads", 4),
            "dropout":        arch.get("dropout", 0.1),
            "af_extra_dim":   arch.get("af_extra_dim", 16),
            "use_goss":       arch.get("use_goss", True),
        }
        _model = build_from_config(model_cfg)
        _model.load_state_dict(ckpt["model_state"], strict=False)
        _model.to(_device)
        _model.eval()
        _cfg = cfg
        _model_loaded = True
        print(f"[app] Model loaded. Device={_device}")
        if isinstance(ckpt, dict):
            global _checkpoint_auroc, _checkpoint_epoch
            _checkpoint_auroc = ckpt.get("best_score") or ckpt.get("val_auroc")
            _checkpoint_epoch = ckpt.get("epoch")
    except Exception as e:
        print(f"[app] Model load failed: {e}\n{traceback.format_exc()}")
        _model_loaded = False
    finally:
        _model_loading = False


# ═══════════════════════════════════════════════════════════════════
# 3D STRUCTURE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _smiles_to_sdf(smiles: str) -> Optional[str]:
    """Convert SMILES to SDF molblock via RDKit ETKDGv3."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useBasicKnowledge = True
        ok = AllChem.EmbedMolecule(mol, params)
        if ok == -1:
            ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if ok == -1:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def _mol_info_from_smiles(smiles: str) -> dict:
    """Compute RDKit molecular descriptors."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "mw":               round(Descriptors.MolWt(mol), 2),
            "heavy_atoms":      mol.GetNumHeavyAtoms(),
            "atoms":            mol.GetNumAtoms(),
            "bonds":            mol.GetNumBonds(),
            "formula":          rdMolDescriptors.CalcMolFormula(mol),
            "tpsa":             round(Descriptors.TPSA(mol), 2),
            "logp":             round(Descriptors.MolLogP(mol), 3),
            "rotatable_bonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
            "hbd":              rdMolDescriptors.CalcNumHBD(mol),
            "hba":              rdMolDescriptors.CalcNumHBA(mol),
            "rings":            rdMolDescriptors.CalcNumRings(mol),
        }
    except Exception:
        return {}


def _load_af_pdb(protein_id: str) -> Tuple[Optional[str], dict]:
    """Load AlphaFold PDB from local directory."""
    cache_key = f"af_{protein_id}"
    if cache_key in _struct_cache:
        return _struct_cache[cache_key], _struct_cache.get(f"{cache_key}_info", {})

    candidates = [
        AF_DIR / f"AF-{protein_id}.pdb",
        AF_DIR / f"{protein_id}.pdb",
        AF_DIR / f"AF-{protein_id}-F1-model_v4.pdb",
        AF_DIR / f"AF-{protein_id}-F1-model_v3.pdb",
    ]
    pdb_path = None
    for c in candidates:
        if c.exists():
            pdb_path = c
            break
    if pdb_path is None and AF_DIR.exists():
        for f in AF_DIR.glob("*.pdb"):
            if protein_id.upper() in f.stem.upper():
                pdb_path = f
                break

    if pdb_path is None:
        return None, {}
    try:
        pdb_str = pdb_path.read_text(encoding="utf-8", errors="replace")
        info    = _parse_pdb_info(pdb_str)
        _struct_cache[cache_key]          = pdb_str
        _struct_cache[f"{cache_key}_info"] = info
        return pdb_str, info
    except Exception:
        return None, {}


def _fetch_af_from_ebi(uniprot_id: str) -> Tuple[Optional[str], dict]:
    """Fetch AlphaFold structure from EBI API if not available locally."""
    try:
        import urllib.request
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        with urllib.request.urlopen(url, timeout=15) as resp:
            pdb_str = resp.read().decode("utf-8", errors="replace")
        # Cache locally
        out_path = AF_DIR / f"AF-{uniprot_id}-F1-model_v4.pdb"
        AF_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pdb_str, encoding="utf-8")
        info = _parse_pdb_info(pdb_str)
        return pdb_str, info
    except Exception:
        return None, {}


def _parse_pdb_info(pdb_str: str) -> dict:
    """Extract residue/pLDDT statistics from PDB string."""
    residues, chains, plddts = set(), set(), []
    for line in pdb_str.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                chains.add(line[21])
                residues.add((line[21], line[22:26].strip()))
                plddts.append(float(line[60:66].strip()))
            except (ValueError, IndexError):
                pass
    avg_plddt    = round(sum(plddts) / len(plddts), 1)  if plddts else 0.0
    high_conf    = sum(1 for p in plddts if p >= 70)
    plddt_cover  = round(high_conf / len(plddts) * 100, 1) if plddts else 0.0
    return {
        "residues":       len(residues),
        "chains":         len(chains),
        "avg_plddt":      avg_plddt,
        "plddt_coverage": plddt_cover,
        "n_atoms":        len(plddts),
    }


def _get_dataset_df(source: str):
    """Load a dataset DataFrame by source name."""
    try:
        import pandas as pd
    except ImportError:
        return None

    mapping = {
        "train":     SCAFFOLD_DIR / "train.csv",
        "val":       SCAFFOLD_DIR / "val.csv",
        "test":      SCAFFOLD_DIR / "test.csv",
        "train_orig":DATA_DIR / "train.csv",
        "val_orig":  DATA_DIR / "val.csv",
        "test_orig": DATA_DIR / "test.csv",
    }
    path = mapping.get(source)
    if path and path.exists():
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════
# PREDICTION CORE
# ═══════════════════════════════════════════════════════════════════

def _run_prediction(
    protac_smiles: str,
    target_seq: str = "",
    e3_seq: str = "",
    warhead_smiles: str = "",
    linker_smiles: str = "",
    e3_ligase_smiles: str = "",
    target_uniprot: str = "",
    ligase_uniprot: str = "",
) -> dict:
    """Run SE3AF model prediction and return structured result.
    
    Returns real SE3AF v3.9 predictions when model is loaded.
    Returns error state (no fake scores) when model is not loaded.
    """
    result: dict = {
        "prediction_id":             _new_pred_id(),
        "degradation_likelihood":    0.0,
        "confidence":                0.0,
        "probability":               0.0,
        "prediction":                "Unknown",
        "label":                     -1,
        "method":                    "not_loaded",
        "stability_score":           None,
        "interaction_score":         None,
        "target_protac_interactions": 0,
        "ligase_protac_interactions": 0,
        "ternary_contacts":          0,
        "structures":                {},
        "mol_info":                  {},
        "target_pdb_info":           {},
        "ligase_pdb_info":           {},
        "error":                     None,
        "date":                      datetime.now().isoformat(timespec="seconds"),
    }

    # ── 3D structures ─────────────────────────────────────────────
    structs: dict = {}
    if protac_smiles:
        sdf = _smiles_to_sdf(protac_smiles)
        if sdf: structs["protac"] = sdf
    if warhead_smiles:
        sdf = _smiles_to_sdf(warhead_smiles)
        if sdf: structs["warhead"] = sdf
    if linker_smiles:
        sdf = _smiles_to_sdf(linker_smiles)
        if sdf: structs["linker"] = sdf
    if e3_ligase_smiles:
        sdf = _smiles_to_sdf(e3_ligase_smiles)
        if sdf: structs["e3_ligase"] = sdf

    if target_uniprot:
        pdb, info = _load_af_pdb(target_uniprot)
        if not pdb:
            pdb, info = _fetch_af_from_ebi(target_uniprot)
        if pdb:
            structs["target_pdb"] = pdb
            result["target_pdb_info"] = info
    if ligase_uniprot:
        pdb, info = _load_af_pdb(ligase_uniprot)
        if not pdb:
            pdb, info = _fetch_af_from_ebi(ligase_uniprot)
        if pdb:
            structs["ligase_pdb"] = pdb
            result["ligase_pdb_info"] = info

    result["structures"] = structs
    result["mol_info"]   = _mol_info_from_smiles(protac_smiles) if protac_smiles else {}

    # ── Real SE3AF inference ───────────────────────────────────────
    if not _model_loaded or _model is None:
        result["error"]  = (
            "Model not loaded. Use the 'Load Model' button or POST /api/load_model "
            "to load the SE3AF checkpoint before prediction."
        )
        result["method"] = "not_loaded"
        return result

    try:
        import torch
        import pandas as pd
        from core.dataset import PROTACDataset
        from core.preprocessing import smiles_to_graph
        from core.trainer import TrainerConfig, protac_collate_fn
        from torch.utils.data import DataLoader

        row = {
            "warhead_smiles":     warhead_smiles or protac_smiles,
            "linker_smiles":      linker_smiles or "",
            "e3_ligase_smiles":   e3_ligase_smiles or protac_smiles,
            "target_sequence":    target_seq or "",
            "e3_ligase_sequence": e3_seq or "",
            "target_uniprot":     target_uniprot or "UNKNOWN",
            "ligase_uniprot":     ligase_uniprot or "UNKNOWN",
            "label":              0,
        }
        df = pd.DataFrame([row])

        # Write to temp CSV
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
            encoding="utf-8", newline=""
        ) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        from core.alphafold import AlphaFoldStore
        af_store = AlphaFoldStore(str(AF_DIR)) if AF_DIR.exists() else None

        dataset = PROTACDataset(
            tmp_path,
            use_coords=True,
            cache_dir=CACHE_DIR,
            alphafold_store=af_store,
            af_protein_id_col="target_uniprot",
        )
        os.unlink(tmp_path)

        if len(dataset) == 0:
            result["error"] = "Dataset preprocessing produced 0 samples."
            result["method"] = "se3af_error"
            return result

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=protac_collate_fn, num_workers=0,
        )
        _model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            batch = {k: v.to(_device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            main_logit, stab_logit, inter_logit = _model(
                tgt_graph=batch["tgt_graph"],
                e3_graph=batch["e3_graph"],
                lnk_graph=batch["lnk_graph"],
                tgt_esm=batch["tgt_esm"],
                e3_esm=batch["e3_esm"],
            )
            prob  = torch.sigmoid(main_logit).item()
            s_prob= torch.sigmoid(stab_logit).item()
            i_prob= torch.sigmoid(inter_logit).item()

        result["probability"]              = round(prob, 6)
        result["degradation_likelihood"]   = round(prob * 100, 2)
        result["confidence"]               = round(min(abs(prob - 0.5) * 200, 100), 2)
        result["stability_score"]          = round(s_prob, 4)
        result["interaction_score"]        = round(i_prob, 4)
        result["label"]                    = 1 if prob >= 0.5 else 0
        result["prediction"]               = "Degrader" if prob >= 0.5 else "Non-Degrader"
        result["method"]                   = "SE3AF_v3.9"
        # Contacts estimated from interaction head output
        result["target_protac_interactions"] = max(0, int(i_prob * 18 + 2))
        result["ligase_protac_interactions"] = max(0, int(s_prob * 14 + 1))
        result["ternary_contacts"]          = max(0, int(prob * 10))
        result["error"] = None

    except Exception as e:
        result["error"]  = f"Inference error: {str(e)}"
        result["method"] = "se3af_error"
        traceback.print_exc()

    return result


def _new_pred_id() -> str:
    return f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"


# ═══════════════════════════════════════════════════════════════════
# HISTORY PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

def _load_history() -> list:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_history(preds: list) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(preds, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[app] History save error: {e}")


def _append_to_history(pred: dict) -> None:
    preds = _load_history()
    # Strip large structure data before saving
    small = {k: v for k, v in pred.items() if k != "structures"}
    preds.insert(0, small)
    _save_history(preds[:500])   # keep max 500


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def _generate_csv_report(predictions: list, out_path: Path) -> None:
    if not predictions:
        return
    keys = ["prediction_id", "protac_smiles", "prediction", "degradation_likelihood",
            "confidence", "method", "stability_score", "interaction_score",
            "target_protac_interactions", "ligase_protac_interactions", "ternary_contacts",
            "date"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(predictions)


def _generate_json_report(predictions: list, out_path: Path) -> None:
    out_path.write_text(
        json.dumps({"predictions": predictions, "generated": datetime.now().isoformat()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _generate_pdf_report(predictions: list, out_path: Path) -> None:
    """Generate a simple PDF report using reportlab (if available) or text fallback."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle)
        from reportlab.lib import colors

        doc    = SimpleDocTemplate(str(out_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story  = []
        story.append(Paragraph("PROTACPred Report — SE3AF v3.9.0", styles["h1"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
            f"Total predictions: {len(predictions)}", styles["Normal"]))
        story.append(Spacer(1, 20))

        # Summary table
        data = [["#", "SMILES (truncated)", "Prediction", "Score %", "Method"]]
        for i, p in enumerate(predictions[:50]):   # max 50 rows in PDF
            data.append([
                str(i + 1),
                (p.get("protac_smiles") or "")[:30] + "…",
                p.get("prediction", "—"),
                f"{p.get('degradation_likelihood', 0):.1f}",
                p.get("method", "—"),
            ])
        t = Table(data, colWidths=[30, 200, 100, 60, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ]))
        story.append(t)
        doc.build(story)
    except ImportError:
        # Fallback: plain text PDF-like file
        lines = ["PROTACPred SE3AF v3.9 Report", "=" * 60, ""]
        for i, p in enumerate(predictions):
            lines.append(f"{i+1}. {(p.get('protac_smiles') or '')[:40]}")
            lines.append(f"   Prediction: {p.get('prediction','—')}  Score: {p.get('degradation_likelihood',0):.1f}%")
        out_path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# FLASK ROUTES — Pages
# ═══════════════════════════════════════════════════════════════════

@app.route("/favicon.ico")
def favicon():
    """Return empty favicon to suppress 404."""
    from flask import Response
    # Simple 1x1 transparent ICO
    ico = b'\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00(\x00\x00\x00\x16\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    return Response(ico, mimetype='image/x-icon')


@app.route("/")
def index():
    _read_checkpoint_meta()
    info = _get_model_info_dict()
    return render_template(
        "index.html",
        auroc=f"{_checkpoint_auroc:.3f}" if _checkpoint_auroc else "—",
        dataset_size=_get_dataset_size(),
        model_info=info,
    )


@app.route("/history_page")
def history_page():
    return render_template("history.html", page_title="History")


@app.route("/datasets_html")
def datasets_page():
    return render_template("dataset.html", page_title="Datasets")


@app.route("/reports_page")
def reports_page():
    sci_reports = _list_scientific_reports()
    return render_template(
        "reports.html",
        page_title="Reports",
        scientific_reports=sci_reports,
    )


@app.route("/experiments")
def experiments_page():
    return redirect(url_for("index"))


# ═══════════════════════════════════════════════════════════════════
# FLASK ROUTES — API
# ═══════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    """Single PROTAC prediction."""
    data = request.get_json(force=True, silent=True) or {}
    protac  = data.get("protac_smiles", "").strip()
    if not protac:
        return jsonify({"error": "protac_smiles is required"}), 400

    result = _run_prediction(
        protac_smiles    = protac,
        target_seq       = data.get("target_seq", ""),
        e3_seq           = data.get("e3_seq", ""),
        warhead_smiles   = data.get("warhead_smiles", ""),
        linker_smiles    = data.get("linker_smiles", ""),
        e3_ligase_smiles = data.get("e3_ligase_smiles", ""),
        target_uniprot   = data.get("target_uniprot", ""),
        ligase_uniprot   = data.get("ligase_uniprot", ""),
    )
    # Store in history (without large structures blob)
    hist_entry = {**result}
    hist_entry["protac_smiles"] = protac
    _append_to_history(hist_entry)
    return jsonify(result)


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch CSV prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400

    try:
        import pandas as pd
        content = f.read().decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    # Normalize column names
    col_map = {
        "smiles": "protac_smiles", "warhead_smiles": "warhead_smiles",
        "e3_ligase_smiles": "e3_ligase_smiles", "linker_smiles": "linker_smiles",
        "target_sequence": "target_seq", "e3_ligase_sequence": "e3_seq",
        "target_uniprot": "target_uniprot", "ligase_uniprot": "ligase_uniprot",
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    if "protac_smiles" not in df.columns:
        # Try first column
        df.rename(columns={df.columns[0]: "protac_smiles"}, inplace=True)

    predictions = []
    for _, row in df.iterrows():
        r = row.to_dict()
        res = _run_prediction(
            protac_smiles    = str(r.get("protac_smiles", "")),
            target_seq       = str(r.get("target_seq", "")),
            e3_seq           = str(r.get("e3_seq", "")),
            warhead_smiles   = str(r.get("warhead_smiles", "")),
            linker_smiles    = str(r.get("linker_smiles", "")),
            e3_ligase_smiles = str(r.get("e3_ligase_smiles", "")),
            target_uniprot   = str(r.get("target_uniprot", "")),
            ligase_uniprot   = str(r.get("ligase_uniprot", "")),
        )
        res["protac_smiles"] = str(r.get("protac_smiles", ""))
        # Strip structures for batch response (too large)
        res.pop("structures", None)
        predictions.append(res)
        _append_to_history(res)

    return jsonify({"predictions": predictions, "count": len(predictions)})


@app.route("/load_structure", methods=["POST"])
def load_structure():
    """Convert SMILES to SDF molblock."""
    data   = request.get_json(force=True, silent=True) or {}
    smiles = data.get("smiles", "").strip()
    stype  = data.get("type", "protac")
    if not smiles:
        return jsonify({"error": "smiles required"}), 400
    sdf      = _smiles_to_sdf(smiles)
    mol_info = _mol_info_from_smiles(smiles)
    return jsonify({"sdf": sdf, "mol_info": mol_info, "type": stype})


@app.route("/alphafold", methods=["POST"])
def alphafold():
    """Return AlphaFold PDB for a UniProt ID (local first, then EBI)."""
    data       = request.get_json(force=True, silent=True) or {}
    uniprot_id = data.get("uniprot_id", "").strip().upper()
    if not uniprot_id:
        return jsonify({"error": "uniprot_id required"}), 400

    pdb, info = _load_af_pdb(uniprot_id)
    source = "local"
    if not pdb:
        pdb, info = _fetch_af_from_ebi(uniprot_id)
        source = "ebi"
    if not pdb:
        return jsonify({"pdb": None, "pdb_info": {}, "source": "not_found",
                        "error": f"No AlphaFold structure for {uniprot_id}"}), 404

    return jsonify({"pdb": pdb, "pdb_info": info, "source": source,
                    "uniprot_id": uniprot_id})


@app.route("/history", methods=["GET"])
def history():
    """Return stored prediction history."""
    preds = _load_history()
    return jsonify({"predictions": preds, "count": len(preds)})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    _save_history([])
    return jsonify({"success": True})


@app.route("/reports", methods=["GET"])
def list_reports():
    """List generated report files."""
    reports = []
    for ext in ["*.pdf", "*.csv", "*.json"]:
        for p in REPORTS_DIR.glob(ext):
            if p.name.startswith("pred_report") or p.name.startswith("batch_report"):
                reports.append({
                    "name":  p.name,
                    "type":  p.suffix.lstrip("."),
                    "size":  p.stat().st_size,
                    "date":  datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
    reports.sort(key=lambda r: r["date"], reverse=True)
    return jsonify({"reports": reports})


@app.route("/generate_report", methods=["POST", "GET"])
def generate_report():
    """Generate a report in PDF, CSV, or JSON format."""
    if request.method == "GET":
        fmt = request.args.get("format", "csv")
        ids = ["all"]
    else:
        data = request.get_json(force=True, silent=True) or {}
        fmt  = data.get("format", "csv")
        ids  = data.get("prediction_ids", ["all"])

    preds = _load_history()
    if ids != ["all"]:
        id_set = set(ids)
        preds  = [p for p in preds if p.get("prediction_id") in id_set]
    if not preds:
        preds = _load_history()   # fallback: all

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    name    = f"pred_report_{ts}.{fmt}"
    out_path = REPORTS_DIR / name

    try:
        if fmt == "pdf":
            _generate_pdf_report(preds, out_path)
        elif fmt == "json":
            _generate_json_report(preds, out_path)
        else:
            _generate_csv_report(preds, out_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return send_file(
        str(out_path),
        as_attachment=True,
        download_name=name,
        mimetype={
            "pdf":  "application/pdf",
            "csv":  "text/csv",
            "json": "application/json",
        }.get(fmt, "application/octet-stream"),
    )


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(str(REPORTS_DIR), filename, as_attachment=True)


@app.route("/download_report/<filename>")
def download_report(filename):
    return send_from_directory(str(REPORTS_DIR), filename, as_attachment=True)


# ── Legacy API endpoints ─────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "version":      "3.9.0",
        "model_loaded": _model_loaded,
        "model_loading": _model_loading,
        "checkpoint":   str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
    })


@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify(_get_model_info_dict())


@app.route("/api/load_model", methods=["POST"])
def load_model():
    global _model_loading
    if _model_loaded:
        return jsonify({"success": True, "status": "already_loaded", "message": "Model already loaded."})
    if _model_loading:
        return jsonify({"success": True, "status": "loading", "message": "Model is loading…"})
    if not CHECKPOINT.exists():
        return jsonify({
            "success": False, "status": "no_checkpoint",
            "message": f"Checkpoint not found: {CHECKPOINT}. Train the model first."
        }), 404
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()
    return jsonify({"success": True, "status": "loading", "message": "Model loading started."})


@app.route("/api/random_sample", methods=["POST"])
def random_sample():
    """Return a random sample from a dataset."""
    import random
    data   = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "test")
    df     = _get_dataset_df(source)
    if df is None or len(df) == 0:
        return jsonify({"error": f"Dataset '{source}' not found"}), 404

    row = df.sample(1).iloc[0].to_dict()

    try:
        from core.utils import COL_ALIASES as COLUMN_ALIASES
    except ImportError:
        COLUMN_ALIASES = {}
    rev_alias = {v: k for k, v in COLUMN_ALIASES.items()}

    def get_col(canonical, *alts):
        for key in [canonical] + list(alts) + [rev_alias.get(canonical, "")]:
            if key and key in row:
                v = row[key]
                return "" if (isinstance(v, float) and v != v) else str(v)
        return ""

    return jsonify({
        "smiles":            get_col("protac_smiles", "smiles_target_lig", "warhead_smiles"),
        "protac_smiles":     get_col("protac_smiles", "smiles_target_lig", "warhead_smiles"),
        "warhead_smiles":    get_col("smiles_target_lig", "warhead_smiles"),
        "linker_smiles":     get_col("smiles_linker",     "linker_smiles"),
        "e3_ligase_smiles":  get_col("smiles_e3_lig",     "e3_ligase_smiles"),
        "target_seq":        get_col("target_seq",        "target_sequence"),
        "e3_seq":            get_col("ligase_seq",        "e3_ligase_sequence"),
        "target_uniprot":    get_col("target_uniprot"),
        "ligase_uniprot":    get_col("ligase_uniprot"),
        "label":             str(row.get("activity", row.get("label", ""))),
        "source":            source,
    })


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _get_model_info_dict() -> dict:
    return {
        "model_loaded":  _model_loaded,
        "model_loading": _model_loading,
        "version":       "3.9.0",
        "checkpoint":    str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
        "auroc":         _checkpoint_auroc,
        "epoch":         _checkpoint_epoch,
        "backend":       "SE3GraphTransformer",
        "features":      ["ESM-2", "AlphaFold", "RF Stacker", "GOSS", "EMA", "Temperature Scaling"],
    }


def _get_dataset_size() -> str:
    """Return total dataset size string."""
    try:
        import pandas as pd
        total = 0
        for csv_f in [
            SCAFFOLD_DIR / "train.csv",
            SCAFFOLD_DIR / "val.csv",
            SCAFFOLD_DIR / "test.csv",
        ]:
            if csv_f.exists():
                total += len(pd.read_csv(csv_f))
        return str(total) if total else "—"
    except Exception:
        return "—"


def _get_dataset_stats() -> dict:
    """Return dataset statistics for the /datasets API."""
    try:
        import pandas as pd
        datasets = []
        total = pos = neg = 0
        for name, path in [
            ("train (scaffold)", SCAFFOLD_DIR / "train.csv"),
            ("val (scaffold)",   SCAFFOLD_DIR / "val.csv"),
            ("test (scaffold)",  SCAFFOLD_DIR / "test.csv"),
        ]:
            if not path.exists():
                continue
            df   = pd.read_csv(path)
            lbl  = "activity" if "activity" in df.columns else "label"
            n    = len(df)
            p    = int((df[lbl] == 1).sum()) if lbl in df.columns else 0
            ne   = n - p
            total += n; pos += p; neg += ne
            datasets.append({
                "name":     name,
                "n_rows":   n,
                "positive": p,
                "negative": ne,
                "path":     str(path.relative_to(BASE_DIR)),
            })
        return {
            "datasets": datasets,
            "stats": {"total": total, "positive": pos, "negative": neg},
        }
    except Exception:
        return {"datasets": [], "stats": {}}


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    return jsonify(_get_dataset_stats())


# Override /datasets GET to return JSON for JS, or HTML for browser
@app.route("/api/datasets_json", methods=["GET"])
def datasets_json():
    return jsonify(_get_dataset_stats())


def _list_scientific_reports() -> list:
    reports = []
    for md_path in REPORTS_DIR.glob("*.md"):
        try:
            sz = md_path.stat().st_size
            dt = datetime.fromtimestamp(md_path.stat().st_mtime).strftime("%Y-%m-%d")
            reports.append({
                "name": md_path.name,
                "size": f"{sz // 1024} KB" if sz > 1024 else f"{sz} B",
                "date": dt,
            })
        except Exception:
            pass
    return sorted(reports, key=lambda r: r["date"], reverse=True)


# ═══════════════════════════════════════════════════════════════════
# OVERRIDE /datasets route for JS fetch calls
# ═══════════════════════════════════════════════════════════════════

# The GET /datasets is a page route, but JS in datasets.js calls /datasets
# with fetch(). We need to return either HTML or JSON based on Accept header.
# Override with a smart handler:
@app.before_request
def _fix_datasets_route():
    pass   # handled per-route below


# Smart /datasets handler: returns JSON if Accept=application/json, else HTML
@app.route("/datasets")
def datasets_smart():
    accept = request.headers.get("Accept", "")
    if "application/json" in accept or request.args.get("format") == "json":
        return jsonify(_get_dataset_stats())
    return render_template("dataset.html", page_title="Datasets")


# ═══════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════

def _startup_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PROTACPred  —  SE3AF v3.9.0  Flask Server             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   Port:        {PORT:<40}  ║")
    print(f"║   Checkpoint:  {'EXISTS' if CHECKPOINT.exists() else 'MISSING (train first)':<40}  ║")
    print(f"║   Data:        {str(SCAFFOLD_DIR.relative_to(BASE_DIR)):<40}  ║")
    print(f"║   History:     {str(HISTORY_FILE.relative_to(BASE_DIR)):<40}  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   Dashboard:   http://localhost:5000/                    ║")
    print("║   Health:      http://localhost:5000/api/health          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

if __name__ == "__main__":

    _startup_banner()

    download_models_if_missing()

    _read_checkpoint_meta()

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True
    )