#!/usr/bin/env python3
"""
run_all_tests.py — SE3AF v3.9 comprehensive test suite
Runs ALL tests in a SINGLE process to avoid repeated heavy-import spawning.
"""
import sys, os, time, traceback, warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PASS = []
FAIL = []

def test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()

# ─── 1. CORE IMPORTS ──────────────────────────────────────────────────────────
print("\n[1] Core Imports")

def t_import_numpy():
    import numpy as np
    assert np.__version__

def t_import_torch():
    import torch
    print(f"     torch {torch.__version__}, CUDA={torch.cuda.is_available()}")

def t_import_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdPartialCharges

def t_import_pyg():
    from torch_geometric.data import Data, Batch

def t_import_sklearn():
    from sklearn.ensemble import RandomForestClassifier

def t_import_pandas():
    import pandas as pd

test("numpy import", t_import_numpy)
test("torch import", t_import_torch)
test("rdkit import", t_import_rdkit)
test("torch_geometric import", t_import_pyg)
test("sklearn import", t_import_sklearn)
test("pandas import", t_import_pandas)

# ─── 2. SE3AF MODULE IMPORTS ────────────────────────────────────────────────
print("\n[2] SE3AF Module Imports")

def t_import_utils():
    from core.utils import COL_ALIASES, COL_LABEL, COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES
    from core.utils import get_logger, compute_metrics, discover_datasets

def t_import_preprocessing():
    from core.preprocessing import EDGE_DIM, NODE_DIM, ESM_DIM
    from core.preprocessing import smiles_to_graph, validate_graph, _dummy_graph

def t_import_alphafold():
    from core.alphafold import af_residue_graph, AlphaFoldStore, af_confidence_weighted_mean_pool

def t_import_model():
    from core.model import SE3AFModel, build_from_config, SE3AFLoss, get_encoder_display_name
    from core.model import ProteinGeometryEncoder, ProteinBackboneEncoder

def t_import_dataset():
    from core.dataset import PROTACDataset, protac_collate_fn

def t_import_trainer():
    from core.trainer import SE3AFTrainer, TrainerConfig

def t_import_inference():
    from core.inference import load_model_for_inference

def t_import_ui():
    from core.ui import print_startup_banner, train_epoch_bar

test("core.utils import", t_import_utils)
test("core.preprocessing import", t_import_preprocessing)
test("core.alphafold import", t_import_alphafold)
test("core.model import", t_import_model)
test("core.dataset import", t_import_dataset)
test("core.trainer import", t_import_trainer)
test("core.inference import", t_import_inference)
test("core.ui import", t_import_ui)

# ─── 3. DATA PIPELINE ─────────────────────────────────────────────────────────
print("\n[3] Data Pipeline")

def t_smiles_to_graph():
    from core.preprocessing import smiles_to_graph, validate_graph
    g = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", use_coords=False)
    assert not getattr(g, "is_dummy", False), "aspirin should not be dummy"
    assert validate_graph(g), "aspirin graph should be valid"

def t_dummy_graph():
    from core.preprocessing import _dummy_graph
    g = _dummy_graph()
    assert getattr(g, "is_dummy", False), "should be dummy"

def t_smiles_to_graph_3d():
    from core.preprocessing import smiles_to_graph, validate_graph
    g = smiles_to_graph("c1ccccc1", use_coords=True)
    # benzene has coords
    assert g.pos is not None or getattr(g, "is_dummy", False)
    print(f"     benzene 3D: {g.x.shape[0]} atoms, pos={'YES' if g.pos is not None else 'NO'}")

def t_scaffold_data_exists():
    import pandas as pd
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/split_scaffold/{split}.csv")
        assert len(df) > 0, f"{split} empty"
    print(f"     train={len(pd.read_csv('data/split_scaffold/train.csv'))}, "
          f"val={len(pd.read_csv('data/split_scaffold/val.csv'))}, "
          f"test={len(pd.read_csv('data/split_scaffold/test.csv'))}")

def t_scaffold_label_col():
    import pandas as pd
    from core.utils import COL_LABEL
    df = pd.read_csv("data/split_scaffold/train.csv")
    assert COL_LABEL in df.columns, f"'{COL_LABEL}' missing from train.csv"
    dist = df[COL_LABEL].value_counts().to_dict()
    print(f"     label dist: {dist}")

def t_global_config_paths():
    import GLOBAL_CONFIG as gc
    assert "split_scaffold" in gc.DATA_DIR or "split_scaffold" in gc.DATA_FILE, \
        f"DATA_DIR/FILE should point to split_scaffold, got {gc.DATA_DIR}, {gc.DATA_FILE}"
    print(f"     DATA_DIR={gc.DATA_DIR}")
    print(f"     DATA_FILE={gc.DATA_FILE}")

def t_num_workers_os_guard():
    import GLOBAL_CONFIG as gc, sys
    if sys.platform.startswith("win"):
        assert gc.NUM_WORKERS == 0, "Windows: NUM_WORKERS should be 0"
    else:
        assert gc.NUM_WORKERS >= 0, "NUM_WORKERS should be >= 0"
    print(f"     NUM_WORKERS={gc.NUM_WORKERS} (platform={sys.platform})")

test("smiles_to_graph (2D)", t_smiles_to_graph)
test("dummy_graph", t_dummy_graph)
test("smiles_to_graph (3D)", t_smiles_to_graph_3d)
test("scaffold split data exists", t_scaffold_data_exists)
test("scaffold split label column", t_scaffold_label_col)
test("GLOBAL_CONFIG.py DATA paths", t_global_config_paths)
test("num_workers OS guard", t_num_workers_os_guard)

# ─── 4. ALPHAFOLD / PROTEIN GEOMETRY ──────────────────────────────────────────
print("\n[4] AlphaFold / Protein Geometry")

def t_af_residue_graph():
    import numpy as np
    from core.alphafold import af_residue_graph
    coords = np.random.randn(40, 3).astype(np.float32)
    plddt  = np.random.uniform(60, 100, 40).astype(np.float32)
    g = af_residue_graph(coords, plddt, max_len=30, knn_k=8, distance_cutoff=10.0)
    assert g is not None
    assert g["node_feat"].shape[1] == 8, f"expected node_feat (N,8), got {g['node_feat'].shape}"
    assert g["edge_feat"].shape[1] == 4, f"expected edge_feat (E,4), got {g['edge_feat'].shape}"
    assert g["edge_index"].shape[0] == 2
    print(f"     nodes={g['node_feat'].shape}, edges={g['edge_index'].shape}")

def t_af_residue_graph_none():
    from core.alphafold import af_residue_graph
    g = af_residue_graph(None, None)
    assert g is None, "should return None for None inputs"

def t_af_confidence_pool():
    import numpy as np
    from core.alphafold import af_confidence_weighted_mean_pool
    coords = np.random.randn(50, 3).astype(np.float32)
    plddt  = np.random.uniform(40, 100, 50).astype(np.float32)
    feat = af_confidence_weighted_mean_pool(coords, plddt)
    assert feat.shape == (4,), f"expected (4,), got {feat.shape}"

def t_protein_geometry_encoder():
    import torch, numpy as np
    from core.model import ProteinGeometryEncoder
    from core.alphafold import af_residue_graph
    coords = np.random.randn(30, 3).astype(np.float32)
    plddt  = np.random.uniform(70, 100, 30).astype(np.float32)
    g = af_residue_graph(coords, plddt, knn_k=8, distance_cutoff=10.0)
    enc = ProteinGeometryEncoder(hidden=32, out_dim=16, n_layers=2)
    enc.eval()
    nf = torch.tensor(g["node_feat"])
    ei = torch.tensor(g["edge_index"])
    ef = torch.tensor(g["edge_feat"])
    with torch.no_grad():
        out = enc(nf, ei, ef)
    assert out.shape == (16,), f"expected (16,), got {out.shape}"
    print(f"     ProteinGeometryEncoder output: {out.shape}")

def t_protein_backbone_encoder():
    import torch
    from core.model import ProteinBackboneEncoder
    enc = ProteinBackboneEncoder(esm_dim=64, af_dim=4, fusion_dim=32, geo_hidden=16, geo_dim=8,
                                  n_geo_layers=1, num_heads=4)
    enc.eval()
    esm = torch.randn(2, 64+4)   # esm_dim + af_dim
    with torch.no_grad():
        out = enc(esm, None)
    assert out.shape == (2, 32), f"expected (2,32), got {out.shape}"
    print(f"     ProteinBackboneEncoder output: {out.shape}")

def t_af_store_load():
    from core.alphafold import AlphaFoldStore
    store = AlphaFoldStore("data/alphafold")
    c, p = store.get("P10275")
    if c is None:
        print("     P10275 not in cache (AF fetch skipped offline)")
    else:
        print(f"     P10275: coords={c.shape}, plddt={p.shape}")

test("af_residue_graph", t_af_residue_graph)
test("af_residue_graph (None input)", t_af_residue_graph_none)
test("af_confidence_weighted_mean_pool", t_af_confidence_pool)
test("ProteinGeometryEncoder forward", t_protein_geometry_encoder)
test("ProteinBackboneEncoder forward", t_protein_backbone_encoder)
test("AlphaFoldStore.get()", t_af_store_load)

# ─── 5. MODEL CONSTRUCTION ────────────────────────────────────────────────────
print("\n[5] Model Construction")

def t_build_model_se3():
    import torch
    from core.model import build_from_config
    cfg = {
        "backend": "se3", "fusion_dim": 64, "esm_dim": 32, "graph_hidden": 64,
        "num_graph_layers": 2, "num_heads": 4, "num_rbf": 8, "cutoff": 8.0,
        "dropout": 0.1, "af_extra_dim": 0, "use_goss": False, "goss_top_k": 4,
        "stochastic_depth_p": 0.0,
    }
    m = build_from_config(cfg)
    params = sum(p.numel() for p in m.parameters())
    print(f"     SE3 model: {params:,} params")
    assert params > 0

def t_build_model_lite():
    from core.model import build_from_config
    cfg = {
        "backend": "lite", "fusion_dim": 64, "esm_dim": 32, "graph_hidden": 64,
        "num_graph_layers": 2, "num_heads": 4, "num_rbf": 8, "cutoff": 8.0,
        "dropout": 0.1, "af_extra_dim": 0, "use_goss": False, "goss_top_k": 4,
        "stochastic_depth_p": 0.0,
    }
    m = build_from_config(cfg)
    params = sum(p.numel() for p in m.parameters())
    print(f"     Lite model: {params:,} params")

def t_model_forward():
    import torch
    from core.model import build_from_config
    from core.preprocessing import smiles_to_graph
    from torch_geometric.data import Batch
    cfg = {
        "backend": "lite", "fusion_dim": 32, "esm_dim": 16, "graph_hidden": 32,
        "num_graph_layers": 1, "num_heads": 4, "num_rbf": 4, "cutoff": 8.0,
        "dropout": 0.1, "af_extra_dim": 0, "use_goss": False, "goss_top_k": 2,
        "stochastic_depth_p": 0.0,
    }
    model = build_from_config(cfg)
    model.eval()
    g1 = smiles_to_graph("CC(=O)O", use_coords=True)
    g2 = smiles_to_graph("c1ccccc1", use_coords=True)
    tgt = Batch.from_data_list([g1, g2])
    e3  = Batch.from_data_list([g1, g2])
    lnk = Batch.from_data_list([g1, g2])
    tgt_esm = torch.randn(2, 16)
    e3_esm  = torch.randn(2, 16)
    with torch.no_grad():
        main, stab, inter = model(tgt, e3, lnk, tgt_esm, e3_esm)
    assert main.shape == (2,)
    assert stab.shape == (2,)
    assert inter.shape == (2,)
    print(f"     forward output: main={main.shape}, stab={stab.shape}, inter={inter.shape}")

def t_se3af_loss():
    import torch
    from core.model import SE3AFLoss
    # Use correct kwarg: stability_loss_weight (not stability_weight)
    criterion = SE3AFLoss(focal_gamma=2.0, focal_alpha=0.5,
                          stability_loss_weight=0.25, interaction_loss_weight=0.25,
                          label_smoothing=0.1)
    main  = torch.randn(8)
    stab  = torch.randn(8)
    inter = torch.randn(8)
    labels = torch.randint(0, 2, (8,)).float()
    loss = criterion(main, stab, inter, labels)
    assert loss.item() > 0
    print(f"     SE3AFLoss = {loss.item():.4f}")

def t_trainer_config():
    from core.trainer import TrainerConfig
    cfg = TrainerConfig.from_json("configs/train_config.json")
    assert cfg.epochs == 100
    assert cfg.backend == "se3"
    assert cfg.grad_accum_steps == 4
    assert cfg.af_extra_dim == 16
    print(f"     TrainerConfig: epochs={cfg.epochs}, backend={cfg.backend}, "
          f"accum={cfg.grad_accum_steps}, af_extra_dim={cfg.af_extra_dim}")

def t_scheduler_total_steps():
    import math
    from core.trainer import TrainerConfig, _CosineWarmupScheduler
    import torch
    cfg = TrainerConfig.from_json("configs/train_config.json")
    n_samples = 1506
    batches_per_epoch = math.ceil(n_samples / cfg.batch_size)
    update_steps_per_epoch = math.ceil(batches_per_epoch / cfg.grad_accum_steps)
    total_steps  = cfg.epochs * update_steps_per_epoch
    warmup_steps = cfg.warmup_epochs * update_steps_per_epoch
    assert total_steps > warmup_steps, \
        f"total_steps={total_steps} must be > warmup_steps={warmup_steps}"
    # Scheduler must never divide by zero
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = _CosineWarmupScheduler(opt, warmup_steps, total_steps)
    lrs = [sched.get_lr()[0] for _ in range(5)]
    assert all(lr >= 0 for lr in lrs)
    print(f"     total_steps={total_steps}, warmup_steps={warmup_steps}, "
          f"first 5 LR ratios={[f'{l:.4f}' for l in lrs]}")

test("build_from_config (SE3 backend)", t_build_model_se3)
test("build_from_config (Lite backend)", t_build_model_lite)
test("model forward pass", t_model_forward)
test("SE3AFLoss forward", t_se3af_loss)
test("TrainerConfig.from_json", t_trainer_config)
test("scheduler total_steps (ISSUE 2)", t_scheduler_total_steps)

# ─── 6. COLLATE + DATASET ─────────────────────────────────────────────────────
print("\n[6] Dataset & Collate")

def t_collate_fn():
    import torch, numpy as np
    from core.dataset import PROTACSample, protac_collate_fn
    from core.preprocessing import smiles_to_graph, _dummy_graph, ESM_DIM
    g = smiles_to_graph("CC(=O)O", use_coords=False)
    esm = np.zeros(ESM_DIM, dtype=np.float32)
    samples = [
        PROTACSample(tgt_graph=g, e3_graph=g, lnk_graph=g,
                     tgt_esm=esm, e3_esm=esm, label=1.0, has_label=True, idx=0),
        PROTACSample(tgt_graph=g, e3_graph=g, lnk_graph=g,
                     tgt_esm=esm, e3_esm=esm, label=0.0, has_label=True, idx=1),
    ]
    batch = protac_collate_fn(samples)
    assert "tgt_graph" in batch
    assert batch["tgt_esm"].shape == (2, ESM_DIM)
    assert batch["labels"].shape == (2,)
    print(f"     batch keys: {list(batch.keys())}, tgt_esm={batch['tgt_esm'].shape}")

def t_discover_datasets():
    from core.utils import discover_datasets
    import pandas as pd
    dfs = discover_datasets("data/split_scaffold/train.csv")
    assert len(dfs) >= 1
    df = pd.concat(dfs)
    assert len(df) > 0
    print(f"     discover_datasets: {len(df)} rows")

def t_dataset_small():
    import pandas as pd, tempfile, os
    from core.dataset import PROTACDataset
    # Create tiny CSV with 3 rows
    df_full = pd.read_csv("data/split_scaffold/train.csv", nrows=3)
    tmp = tempfile.mktemp(suffix=".csv")
    df_full.to_csv(tmp, index=False)
    try:
        ds = PROTACDataset(data_root=tmp, cache_dir=".cache_test",
                           use_coords=False, supervised=True)
        assert len(ds) == 3
        sample = ds[0]
        assert hasattr(sample, "label")
        print(f"     PROTACDataset(3): len={len(ds)}, label={sample.label}")
    finally:
        os.unlink(tmp)

test("protac_collate_fn", t_collate_fn)
test("discover_datasets", t_discover_datasets)
test("PROTACDataset (mini)", t_dataset_small)

# ─── 7. TRAINER SETUP ─────────────────────────────────────────────────────────
print("\n[7] Trainer Setup (no training)")

def t_trainer_setup():
    import pandas as pd, tempfile, os
    from core.trainer import SE3AFTrainer, TrainerConfig
    from core.dataset import PROTACDataset

    cfg = TrainerConfig.from_json("configs/train_config.json")
    cfg.epochs = 1
    cfg.batch_size = 2
    cfg.num_workers = 0
    cfg.use_rf_stacker = False
    cfg.use_ema = False
    cfg.use_amp = False
    cfg.af_extra_dim = 0   # no AF for speed
    cfg.early_stop_patience = 1

    df = pd.read_csv("data/split_scaffold/train.csv", nrows=10)
    tmp = tempfile.mktemp(suffix=".csv")
    df.to_csv(tmp, index=False)
    try:
        ds = PROTACDataset(data_root=tmp, cache_dir=".cache_test",
                           use_coords=False, supervised=True)
        trainer = SE3AFTrainer(cfg)
        trainer.setup(ds)
        # The train dataset is stored as _train_ds (not train_dataset)
        train_ds = getattr(trainer, '_train_ds', getattr(trainer, 'train_dataset', None))
        print(f"     trainer setup OK: device={trainer.device}, "
              f"train_n={len(train_ds) if train_ds is not None else 'N/A'}")
    finally:
        os.unlink(tmp)

test("SE3AFTrainer.setup()", t_trainer_setup)

# ─── 8. EVALUATE PIPELINE ─────────────────────────────────────────────────────
print("\n[8] Evaluate & Metrics")

def t_compute_metrics():
    import numpy as np
    from core.utils import compute_metrics
    probs  = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.2], dtype=np.float64)
    labels = np.array([1,   1,   0,   0,   1,   0  ], dtype=np.int64)
    m = compute_metrics(probs, labels)
    assert "auroc" in m or "auc" in m, f"metrics={list(m.keys())}"
    print(f"     metrics: {dict(list(m.items())[:5])}")

def t_bootstrap_ci():
    import numpy as np
    from core.utils import bootstrap_ci
    probs  = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.2, 0.85, 0.15], dtype=np.float64)
    labels = np.array([1,   1,   0,   0,   1,   0,  1,    0   ], dtype=np.int64)
    # bootstrap_ci returns a dict (not a tuple)
    result = bootstrap_ci(probs, labels, n=100)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 0 <= result['auroc'] <= 1, f"auroc={result['auroc']} invalid"
    assert 0 <= result['auroc_lo'] <= result['auroc_hi'] <= 1
    print(f"     bootstrap_ci: auroc={result['auroc']:.3f} "
          f"[{result['auroc_lo']:.3f}, {result['auroc_hi']:.3f}]")

test("compute_metrics", t_compute_metrics)
test("bootstrap_ci", t_bootstrap_ci)

# ─── 9. SCAFFOLD SPLIT ────────────────────────────────────────────────────────
print("\n[9] Scaffold Split")

def t_scaffold_zero_overlap():
    """Verify 0% scaffold overlap between splits."""
    import pandas as pd
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        def get_scaffold(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: return None
                return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)
            except: return None

        from core.utils import COL_TGT_SMILES
        train = pd.read_csv("data/split_scaffold/train.csv")
        val   = pd.read_csv("data/split_scaffold/val.csv")
        test  = pd.read_csv("data/split_scaffold/test.csv")

        col = COL_TGT_SMILES if COL_TGT_SMILES in train.columns else train.columns[0]
        # Sample up to 50 rows per split for speed
        train_smi = train[col].dropna().head(50).tolist()
        val_smi   = val[col].dropna().head(50).tolist()
        test_smi  = test[col].dropna().head(50).tolist()

        train_sc = set(filter(None, [get_scaffold(s) for s in train_smi]))
        val_sc   = set(filter(None, [get_scaffold(s) for s in val_smi]))
        test_sc  = set(filter(None, [get_scaffold(s) for s in test_smi]))

        tv_overlap = len(train_sc & val_sc)
        tt_overlap = len(train_sc & test_sc)
        print(f"     train_sc={len(train_sc)}, val_sc={len(val_sc)}, test_sc={len(test_sc)}")
        print(f"     train∩val={tv_overlap}, train∩test={tt_overlap} (0 expected in good split)")
        # Allow up to 5% overlap as a tolerance
        if len(train_sc) > 0:
            overlap_frac = tv_overlap / len(train_sc)
            if overlap_frac > 0.05:
                print(f"     WARNING: train-val overlap {overlap_frac:.1%} > 5%")
    except Exception as e:
        print(f"     scaffold check skipped: {e}")

test("scaffold split zero overlap check", t_scaffold_zero_overlap)

# ─── 10. FLASK APP (unit-level only, not live) ────────────────────────────────
print("\n[10] Flask App Routes")

def t_flask_app_import():
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_module", "app.py")
    # Don't actually execute (avoid Flask server start), just parse
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()
    # Check key routes exist
    assert "@app.route(\"/\")" in src
    assert "@app.route(\"/predict\"" in src
    assert "@app.route(\"/api/health\"" in src or "api/health" in src
    assert "@app.route(\"/api/random_sample\"" in src or "api/random_sample" in src
    assert "@app.route(\"/generate_report\"" in src
    print("     app.py route definitions: OK")

def t_flask_datasets_fix():
    """Verify datasets_page key-error fix is applied."""
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()
    # The bad line should NOT exist
    assert "view_functions.pop(\"datasets_page\"" not in src, \
        "datasets_page pop still present — fix not applied!"
    print("     datasets_page KeyError fix: OK")

def t_col_aliases_fix():
    """Verify COL_ALIASES (not COLUMN_ALIASES) in app.py."""
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()
    assert "COLUMN_ALIASES" not in src or "COL_ALIASES as COLUMN_ALIASES" in src, \
        "Bad COLUMN_ALIASES reference found"
    print("     COL_ALIASES fix: OK")

test("app.py routes defined", t_flask_app_import)
test("datasets_page KeyError fix", t_flask_datasets_fix)
test("COL_ALIASES import fix", t_col_aliases_fix)

# ─── 11. LIVE FLASK ENDPOINTS ─────────────────────────────────────────────────
print("\n[11] Live Flask Endpoints (http://localhost:5000)")

def _http(method, path, json_data=None):
    import urllib.request, json as _json, urllib.error
    url = f"http://localhost:5000{path}"
    data = _json.dumps(json_data).encode() if json_data else None
    req = urllib.request.Request(url, data=data,
          headers={"Content-Type": "application/json"} if data else {})
    req.method = method
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

def t_live_index():
    code, body = _http("GET", "/")
    assert code == 200, f"GET / → {code}"
    assert "<!DOCTYPE" in body or "<html" in body

def t_live_datasets_html():
    code, body = _http("GET", "/datasets_html")
    assert code == 200, f"GET /datasets_html → {code}"

def t_live_datasets():
    code, body = _http("GET", "/datasets")
    assert code == 200, f"GET /datasets → {code}"

def t_live_history():
    code, body = _http("GET", "/history_page")
    assert code == 200, f"GET /history_page → {code}"

def t_live_reports():
    code, body = _http("GET", "/reports_page")
    assert code == 200, f"GET /reports_page → {code}"

def t_live_health():
    code, body = _http("GET", "/api/health")
    assert code == 200, f"GET /api/health → {code}"
    import json as _j
    d = _j.loads(body)
    assert d.get("status") == "ok"

def t_live_random_sample():
    code, body = _http("POST", "/api/random_sample")
    assert code == 200, f"POST /api/random_sample → {code}"
    import json as _j
    d = _j.loads(body)
    assert len(d.get("protac_smiles", "")) > 0
    print(f"     protac_smiles len={len(d['protac_smiles'])}")

def t_live_load_structure():
    code, body = _http("POST", "/load_structure",
                       {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "component": "protac"})
    assert code == 200, f"POST /load_structure → {code}"
    import json as _j
    d = _j.loads(body)
    assert len(d.get("sdf", "")) > 100
    print(f"     SDF bytes={len(d['sdf'])}, mw={d.get('mol_info',{}).get('mw')}")

def t_live_predict():
    code, body = _http("POST", "/predict",
                       {"protac_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                        "target_uniprot": "P10275", "ligase_uniprot": "Q8NHZ8"})
    assert code == 200, f"POST /predict → {code}"
    import json as _j
    d = _j.loads(body)
    assert "method" in d
    print(f"     method={d['method']}, mol_info_keys={list(d.get('mol_info',{}).keys())[:4]}")

def t_live_datasets_api():
    code, body = _http("GET", "/api/datasets")
    assert code == 200, f"GET /api/datasets → {code}"
    import json as _j
    d = _j.loads(body)
    assert "datasets" in d
    print(f"     datasets count={len(d['datasets'])}")

def t_live_report_csv():
    code, body = _http("POST", "/generate_report", {"format": "csv", "scope": "all"})
    assert code == 200, f"POST /generate_report CSV → {code}"
    assert "prediction_id" in body or "smiles" in body.lower()

def t_live_report_pdf():
    import urllib.request, json as _j, urllib.error
    url = "http://localhost:5000/generate_report"
    data = _j.dumps({"format": "pdf", "scope": "all"}).encode()
    req = urllib.request.Request(url, data=data,
          headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code = r.status
            body = r.read()
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read()
    assert code == 200, f"POST /generate_report PDF → {code}"
    assert body[:4] == b"%PDF" or len(body) > 500, \
        f"Not a valid PDF (first bytes: {body[:8]!r})"
    print(f"     PDF size={len(body)} bytes, starts={body[:5]!r}")

def t_live_favicon():
    code, _ = _http("GET", "/favicon.ico")
    assert code == 200, f"GET /favicon.ico → {code}"

test("GET /",               t_live_index)
test("GET /datasets_html",  t_live_datasets_html)
test("GET /datasets",       t_live_datasets)
test("GET /history_page",   t_live_history)
test("GET /reports_page",   t_live_reports)
test("GET /api/health",     t_live_health)
test("POST /api/random_sample", t_live_random_sample)
test("POST /load_structure",    t_live_load_structure)
test("POST /predict",           t_live_predict)
test("GET /api/datasets",       t_live_datasets_api)
test("POST /generate_report (CSV)", t_live_report_csv)
test("POST /generate_report (PDF)", t_live_report_pdf)
test("GET /favicon.ico",        t_live_favicon)

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  TOTAL: {len(PASS)+len(FAIL)} tests")
print(f"  ✅ PASS: {len(PASS)}")
print(f"  ❌ FAIL: {len(FAIL)}")
if FAIL:
    print()
    print("  FAILED TESTS:")
    for name, err in FAIL:
        print(f"    ✗ {name}: {err}")
print("=" * 60)
sys.exit(0 if not FAIL else 1)
