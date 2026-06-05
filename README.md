# PROTACPred — SE3AF v3.9.0

## Quick Start (Single Command)
```bash
cd SE3AF_v381_FIXED
pip install -r requirements.txt
python app.py
# → open http://localhost:5000
```
Or use the helper script:
```bash
chmod +x run.sh && ./run.sh
```

## Project Overview
- **Name**: PROTACPred / SE3AF v3.9.0
- **Goal**: SE(3)-equivariant Graph Transformer for PROTAC ternary complex degradation activity prediction
- **Flask Version**: 3.1.3
- **Python**: 3.9+

## Features (All Working)
- ✅ Single prediction (SMILES + UniProt → degradation score)
- ✅ Batch CSV prediction with progress bar
- ✅ AlphaFold auto-fetch from EBI API (cached locally)
- ✅ 3D molecular viewer (3Dmol.js: Surface/Cartoon/Stick/Ball-and-Stick)
- ✅ View modes: Complex / PROTAC / Target / Ligase
- ✅ Compound Explorer with sortable table
- ✅ Prediction history (JSON persistence, max 500 entries)
- ✅ Report generation: PDF / CSV / JSON
- ✅ Dataset management page with leakage audit (0% overlap)
- ✅ Model loading indicator + status polling
- ✅ RDKit molecular properties (MW, formula, TPSA, LogP, etc.)
- ✅ Scaffold-aware data split (train/val/test in `data/split_scaffold/`)

## URLs
| Page | URL |
|------|-----|
| Dashboard | http://localhost:5000/ |
| History | http://localhost:5000/history_page |
| Datasets | http://localhost:5000/datasets |
| Reports | http://localhost:5000/reports_page |
| Health | http://localhost:5000/api/health |

## API Reference (18 Routes)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Dashboard |
| GET | `/history_page` | History page |
| GET | `/datasets` | Dataset page (HTML) or JSON (on Accept: application/json) |
| GET | `/reports_page` | Reports page |
| POST | `/predict` | Single PROTAC prediction |
| POST | `/batch_predict` | Batch CSV prediction |
| POST | `/load_structure` | SMILES → SDF molblock |
| POST | `/alphafold` | UniProt → AlphaFold PDB |
| POST | `/generate_report` | Generate PDF/CSV/JSON report |
| GET | `/reports` | List generated reports |
| GET | `/download_report/<name>` | Download a report file |
| GET | `/history` | Get prediction history (JSON) |
| POST | `/clear_history` | Clear all history |
| GET | `/api/health` | Health check |
| GET | `/api/model_info` | Model metadata |
| POST | `/api/load_model` | Load SE3AF checkpoint |
| POST | `/api/random_sample` | Random sample from dataset |
| GET | `/api/datasets` | Dataset statistics (JSON) |

## Data Architecture
- **Data Models**: PROTAC ternary complex (PROTAC SMILES, target UniProt, E3 ligase UniProt → degradation label)
- **Storage**: JSON files (`history/predictions.json`), CSV (`data/split_scaffold/`), PDF/CSV reports (`reports/`)
- **Split**: Scaffold-aware (Bemis-Murcko) — 0% train/val/test overlap
  - `data/split_scaffold/train.csv` — 1,234 compounds
  - `data/split_scaffold/val.csv`   — 163 compounds
  - `data/split_scaffold/test.csv`  — 109 compounds

## Backend Scientific Fixes (v3.9.0)
| # | Fix | File | Status |
|---|-----|------|--------|
| 1 | Data paths → `data/split_scaffold/` | `GLOBAL_CONFIG.py` | ✅ |
| 2 | Gradient accumulation `total_steps` | `core/trainer.py` | ✅ |
| 3 | `af_residue_graph()` + `ProteinGeometryEncoder` | `core/alphafold.py`, `core/model.py` | ✅ |
| 4 | `ProteinBackboneEncoder` | `core/model.py` | ✅ |
| 5 | 5-stage coordinate fallback | `core/preprocessing.py` | ✅ |
| 9 | UTF-8 encoding fixes (9 files) | multiple | ✅ |
| 10 | `num_workers` OS guard | `GLOBAL_CONFIG.py` | ✅ |

## Frontend Architecture
```
templates/
  index.html     — 3-column dashboard (280px | 1fr | 320px)
  history.html   — sortable history table
  dataset.html   — dataset management + leakage audit
  reports.html   — report generation + analytics
static/
  css/style.css  — full design system (icon sidebar, dark #1E293B)
  js/
    prediction.js  — all prediction/batch/AF/history logic (ES5)
    viewer.js      — 3Dmol.js wrapper (Surface/Cartoon/Stick/BallStick)
    datasets.js    — dataset stats loader
    reports.js     — reports page JS
```

## Design System
```css
--sidebar-bg: #1E293B   (icon-only, 64px)
--primary:    #6366F1   (PROTAC color)
--c-target:   #818CF8   (Target protein color)
--c-ligase:   #34D399   (E3 Ligase color)
dashboard-grid: 280px | 1fr | 320px
```

## Deployment
- **Platform**: Local Flask (Python 3.9+)
- **Status**: ✅ Active, tested 20/20 routes
- **Tech Stack**: Flask 3.1.3 + RDKit + ReportLab + 3Dmol.js + TailwindCSS CDN
- **Process Manager**: PM2 (`ecosystem.config.cjs`) or plain `python app.py`
- **Last Updated**: 2026-06-04

## Documentation
All technical documentation is consolidated in:
`reports/SE3AF_v39_MASTER_DOCS.md` (2,895 lines)
Includes: root cause analysis, bug reports, architecture, test results, API reference, how-to-run.
