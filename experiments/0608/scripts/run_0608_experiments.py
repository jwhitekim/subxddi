#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
# experiments/0608/scripts/ → experiments/0608/ → experiments/ → subxddi/ (repo root)
PROJECT_ROOT = EXPERIMENT_ROOT.parent.parent
DRUGBANK_TEST_DIR = PROJECT_ROOT / "models" / "dsn_encoder"
if str(DRUGBANK_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(DRUGBANK_TEST_DIR))

import models_fusion as models  # noqa: E402


PHASE_ORDER = ["sanity", "core", "decoder_ablation", "batch_ablation", "top_seed_robustness"]
FUSION_ORDER = [
    "cross_attention",
    "concat",
    "mlp",
]
DECODER_ABLATIONS = ["binary_no_coattn_no_rescal", "coattn_binary_no_rescal"]
RESULT_COLUMNS = [
    "run_id", "phase", "model_family", "dataset", "fold", "k", "path_count", "fusion",
    "attention_direction", "decoder", "train_mode", "batch_size", "seed",
    "epoch", "stopped_epoch", "best_epoch", "early_stopped", "stop_reason",
    "best_metric_name", "best_metric_value", "best_threshold",
    "test_accuracy", "test_f1", "test_roc_auc", "test_auc", "test_pr_auc",
    "test_ap", "test_ap_at_50", "val_accuracy", "val_f1", "val_roc_auc",
    "val_auc", "val_pr_auc", "val_ap", "train_accuracy", "train_f1",
    "train_roc_auc", "train_loss", "val_loss", "test_loss",
    "total_params", "trainable_params", "frozen_params", "train_seconds",
    "eval_seconds", "total_seconds", "checkpoint_pt_path", "checkpoint_pkl_path",
    "history_path", "loss_plot_path", "accuracy_plot_path", "auc_plot_path",
    "f1_plot_path", "roc_curve_path", "pr_curve_path", "config_path",
    "reasoning_jsonl_path", "reasoning_csv_path",
    "status", "error_message",
]


@dataclass(frozen=True)
class RunConfig:
    phase: str
    model_family: str
    dataset: str
    fold: int
    k: object
    fusion: str
    attention_direction: str
    decoder: str
    train_mode: str
    batch_size: int
    seed: int
    epochs: int
    path_count: int = 16

    @property
    def key_tuple(self):
        return (
            self.phase, self.model_family, self.dataset, str(self.fold), self.k_label,
            str(self.path_count), self.fusion, self.decoder, self.train_mode,
            str(self.batch_size), str(self.seed),
        )

    @property
    def k_label(self):
        return "NA" if self.k is None else str(self.k)

    @property
    def run_id(self):
        parts = [
            self.phase, self.model_family, self.dataset, f"fold{self.fold}",
            f"k{self.k}" if self.k is not None else "kNA", f"p{self.path_count}", self.fusion,
            self.decoder, self.train_mode, f"bs{self.batch_size}", f"seed{self.seed}",
        ]
        return "__".join(str(p).replace("/", "-") for p in parts)


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class MolecularCache:
    def __init__(self, dataset):
        self.dataset = dataset
        ds_dir = PROJECT_ROOT / "dataset" / dataset
        smiles_df = pd.read_csv(ds_dir / "drug_smiles.csv")
        self.drug_to_mol = {}
        tuples = []
        for drug_id, smiles in zip(smiles_df["drug_id"], smiles_df["smiles"]):
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol is not None:
                self.drug_to_mol[str(drug_id)] = mol
                tuples.append((str(drug_id), mol))
        self.mol_features = {
            drug_id: self._mol_edge_list_and_feat_mtx(mol)
            for drug_id, mol in tuples
        }
        self.mol_features = {k: v for k, v in self.mol_features.items() if v is not None}
        self.total_atom_feats = next(iter(self.mol_features.values()))[1].shape[-1]

    def _atom_features(self, atom):
        def one_of_k_encoding_unk(x, allowable_set):
            if x not in allowable_set:
                x = allowable_set[-1]
            return [x == s for s in allowable_set]

        feats = one_of_k_encoding_unk(
            atom.GetSymbol(),
            ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
             "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
             "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
             "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown"],
        )
        feats += [
            atom.GetDegree() / 10,
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
        ]
        feats += one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        feats += [atom.GetIsAromatic(), atom.GetTotalNumHs()]
        return torch.tensor(np.asarray(feats, dtype=np.float32))

    def _mol_edge_list_and_feat_mtx(self, mol):
        node_features = [(atom.GetIdx(), self._atom_features(atom)) for atom in mol.GetAtoms()]
        if not node_features:
            return None
        node_features.sort()
        _, features = zip(*node_features)
        features = torch.stack(features)
        edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
        if len(edge_list):
            edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0)
            edge_index = edge_list.T
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index, features

    def make_graph_data(self, drug_id):
        edge_index, feats = self.mol_features[drug_id]
        return Data(x=feats, edge_index=edge_index)

    def make_bipartite(self, drug_a, drug_b, h_data, t_data):
        mol_a = self.drug_to_mol[drug_a]
        mol_b = self.drug_to_mol[drug_b]
        x1 = np.arange(0, len(mol_a.GetAtoms()))
        x2 = np.arange(0, len(mol_b.GetAtoms()))
        edge_index = torch.as_tensor(np.asarray(np.meshgrid(x1, x2)), dtype=torch.long)
        edge_index = torch.stack([edge_index[0].reshape(-1), edge_index[1].reshape(-1)])
        return BipartiteData(edge_index, h_data.x, t_data.x)


class PairDataset(Dataset):
    def __init__(self, rows, cache, shuffle=True):
        self.rows = []
        self.cache = cache
        for row in rows:
            h, t, rel, neg, neg_side = row
            if h in cache.mol_features and t in cache.mol_features and neg in cache.mol_features:
                self.rows.append((h, t, int(rel), neg, neg_side))
        if shuffle:
            random.shuffle(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def collate_fn(self, batch):
        pos_h, pos_t, pos_b, pos_r, pos_pairs = [], [], [], [], []
        neg_h, neg_t, neg_b, neg_r, neg_pairs = [], [], [], [], []

        for h, t, rel, neg, neg_side in batch:
            h_data = self.cache.make_graph_data(h)
            t_data = self.cache.make_graph_data(t)
            pos_h.append(h_data)
            pos_t.append(t_data)
            pos_b.append(self.cache.make_bipartite(h, t, h_data, t_data))
            pos_r.append(rel)
            pos_pairs.append((h, t))

            if neg_side == "h":
                nh, nt = neg, t
                nh_data = self.cache.make_graph_data(nh)
                nt_data = t_data
            else:
                nh, nt = h, neg
                nh_data = h_data
                nt_data = self.cache.make_graph_data(nt)
            neg_h.append(nh_data)
            neg_t.append(nt_data)
            neg_b.append(self.cache.make_bipartite(nh, nt, nh_data, nt_data))
            neg_r.append(rel)
            neg_pairs.append((nh, nt))

        pos_tri = (
            Batch.from_data_list(pos_h),
            Batch.from_data_list(pos_t),
            torch.LongTensor(pos_r).unsqueeze(0),
            Batch.from_data_list(pos_b),
            pos_pairs,
        )
        neg_tri = (
            Batch.from_data_list(neg_h),
            Batch.from_data_list(neg_t),
            torch.LongTensor(neg_r).unsqueeze(0),
            Batch.from_data_list(neg_b),
            neg_pairs,
        )
        return pos_tri, neg_tri


class SigmoidLoss(nn.Module):
    def forward(self, p_scores, n_scores):
        p_loss = -F.logsigmoid(p_scores).mean()
        n_loss = -F.logsigmoid(-n_scores).mean()
        return (p_loss + n_loss) / 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASE_ORDER + ["all"], default="core")
    parser.add_argument("--datasets", nargs="+", default=["drugbank"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--path-counts", nargs="+", type=int, default=[10])
    parser.add_argument("--fusions", nargs="+", choices=FUSION_ORDER, default=FUSION_ORDER)
    parser.add_argument("--train-modes", nargs="+", default=["full"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument(
        "--kg-only",
        action="store_true",
        help="For core phase, skip original_dsn baseline runs and run only KG fusion families.",
    )
    parser.add_argument("--stop-acc-threshold", type=float, default=0.90)
    parser.add_argument("--min-epoch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--best-metric",
        choices=["val_auc", "val_accuracy", "val_f1", "val_loss"],
        default="val_accuracy",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-pickle", type=str2bool, default=True)
    parser.add_argument("--save-curves", type=str2bool, default=True)
    parser.add_argument("--save-plots", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-log-interval", type=int, default=1)
    parser.add_argument("--attention-entropy-lambda", type=float, default=1e-3)
    parser.add_argument("--attention-min-paths", type=int, default=2)
    parser.add_argument("--attention-no-weight-decay", type=str2bool, default=True)
    parser.add_argument("--subxddi-kg-dir", default=str(PROJECT_ROOT / "models" / "kg_encoder"))
    parser.add_argument("--hkg-cache-dir", default=str(PROJECT_ROOT / "kg"))
    parser.add_argument("--prefer-hkg-cache", type=str2bool, default=True)
    parser.add_argument("--export-reasoning", type=str2bool, default=True)
    parser.add_argument("--reasoning-max-batches", type=int, default=0)
    parser.add_argument("--reasoning-split", choices=["test"], default="test")
    return parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


class TeeRunLog:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = TeeStream(self.stdout, self.file)
        sys.stderr = TeeStream(self.stderr, self.file)
        print(f"[RUN_LOG] path={self.path}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if self.file is not None:
            self.file.flush()
            self.file.close()
        return False


def normalize_neg(value):
    raw = str(value)
    if "$" in raw:
        drug, side = raw.split("$", 1)
        return drug, "h" if side.startswith("h") else "t"
    return raw, "t"


def load_rows(dataset, fold, seed):
    ds_dir = PROJECT_ROOT / "dataset" / dataset / f"fold{fold}"
    train_df = pd.read_csv(ds_dir / "train.csv")
    test_df = pd.read_csv(ds_dir / "test.csv")

    def rows_from_df(df):
        rows = []
        required = ["d1", "d2", "type", "Neg samples"]
        for _, row in df.dropna(subset=required).iterrows():
            neg, side = normalize_neg(row["Neg samples"])
            rows.append((str(row["d1"]), str(row["d2"]), int(row["type"]), neg, side))
        return rows

    train_rows = rows_from_df(train_df)
    test_rows = rows_from_df(test_df)
    labels = np.asarray([r[2] for r in train_rows])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
    train_idx, val_idx = next(splitter.split(np.zeros(len(train_rows)), labels))
    train_split = [train_rows[i] for i in train_idx]
    val_split = [train_rows[i] for i in val_idx]
    return train_split, val_split, test_rows


def rel_total(dataset):
    df = pd.read_csv(PROJECT_ROOT / "dataset" / dataset / "ddis.csv")
    return int(df["type"].max()) + 1


def build_runs_for_phase(phase, args, output_dir):
    if phase == "sanity":
        return [
            RunConfig("sanity", f"{fusion}_original_decoder", "drugbank", 0, k, fusion,
                      direction_for_fusion(fusion), "original_decoder", "full", 128, 0, 3)
            for k in [0, 2]
            for fusion in ["concat", "bidirect_cross_attention"]
        ]

    if phase == "core":
        runs = []
        for fusion in args.fusions:
            for k in args.ks:
                for path_count in args.path_counts:
                    for dataset in args.datasets:
                        for fold in args.folds:
                            for train_mode in args.train_modes:
                                for batch_size in args.batch_sizes:
                                    for seed in args.seeds:
                                        runs.append(RunConfig(
                                            "core", f"{fusion}_original_decoder_p{path_count}", dataset, fold, k,
                                            fusion, direction_for_fusion(fusion), "original_decoder",
                                            train_mode, batch_size, seed, args.epochs,
                                            path_count=path_count,
                                        ))
        return runs

    if phase == "decoder_ablation":
        best_k = select_best_k_by_dataset(output_dir, args)
        runs = []
        for decoder in DECODER_ABLATIONS:
            for fusion in ["one_direct_cross_attention", "bidirect_cross_attention"]:
                for dataset in args.datasets:
                    selected_k = best_k.get(dataset, args.ks[0])
                    for fold in args.folds:
                        for train_mode in args.train_modes:
                            for seed in args.seeds:
                                runs.append(RunConfig(
                                    "decoder_ablation", f"{fusion}_{decoder}", dataset, fold,
                                    selected_k, fusion, direction_for_fusion(fusion), decoder,
                                    train_mode, 128, seed, args.epochs,
                                ))
        return runs

    if phase == "batch_ablation":
        configs = select_best_config_by_dataset(output_dir, args)
        runs = []
        for dataset in args.datasets:
            cfg = configs.get(dataset)
            if cfg is None:
                cfg = {"fusion": "bidirect_cross_attention", "decoder": "original_decoder",
                       "k": args.ks[0], "train_mode": args.train_modes[0]}
            for fold in args.folds:
                for batch_size in args.batch_sizes:
                    for seed in args.seeds:
                        runs.append(RunConfig(
                            "batch_ablation", f"{cfg['fusion']}_{cfg['decoder']}", dataset, fold,
                            cfg["k"], cfg["fusion"], direction_for_fusion(cfg["fusion"]),
                            cfg["decoder"], cfg["train_mode"], batch_size, seed, args.epochs,
                        ))
        return runs

    if phase == "top_seed_robustness":
        configs = select_best_config_by_dataset(output_dir, args)
        runs = []
        for dataset in args.datasets:
            cfg = configs.get(dataset)
            if cfg is None:
                cfg = {"fusion": "bidirect_cross_attention", "decoder": "original_decoder",
                       "k": args.ks[0], "train_mode": args.train_modes[0], "batch_size": 128}
            for fold in args.folds:
                for seed in [1, 2]:
                    runs.append(RunConfig(
                        "top_seed_robustness", f"{cfg['fusion']}_{cfg['decoder']}", dataset, fold,
                        cfg["k"], cfg["fusion"], direction_for_fusion(cfg["fusion"]),
                        cfg["decoder"], cfg["train_mode"], int(cfg.get("batch_size", 128)), seed,
                        args.epochs,
                    ))
        return runs

    raise ValueError(f"Unsupported phase: {phase}")


def direction_for_fusion(fusion):
    if fusion in ("cross_attention", "one_direct_cross_attention"):
        return "dsn_to_kg"
    if fusion == "bidirect_cross_attention":
        return "bidirectional"
    return "none"


def read_results(path):
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(path)


def select_best_k_by_dataset(output_dir, args):
    df = read_results(output_dir / "all_results.csv")
    if df.empty:
        return {}
    df = df[
        (df["phase"] == "core")
        & (df["status"] == "completed")
        & (df["k"].notna())
        & (df["k"].astype(str) != "NA")
    ]
    if df.empty:
        return {}
    chosen = {}
    for dataset, group in df.groupby("dataset"):
        summary = group.groupby("k")[["val_auc", "val_f1", "val_accuracy"]].mean(numeric_only=True)
        summary = summary.sort_values(["val_auc", "val_f1", "val_accuracy"], ascending=False)
        chosen[dataset] = int(float(summary.index[0]))
    return chosen


def select_best_config_by_dataset(output_dir, args):
    df = read_results(output_dir / "all_results.csv")
    if df.empty:
        return {}
    df = df[df["status"] == "completed"].copy()
    if df.empty:
        return {}
    configs = {}
    for dataset, group in df.groupby("dataset"):
        group = group.sort_values(
            ["test_roc_auc", "test_f1", "test_accuracy", "test_ap"],
            ascending=False,
            na_position="last",
        )
        row = group.iloc[0].to_dict()
        configs[dataset] = {
            "fusion": row.get("fusion", "none"),
            "decoder": row.get("decoder", "original_decoder"),
            "k": None if pd.isna(row.get("k")) or str(row.get("k")) == "NA" else int(float(row.get("k"))),
            "train_mode": row.get("train_mode", "full"),
            "batch_size": int(row.get("batch_size", 128)),
        }
    return configs


def completed_and_failed(output_dir):
    results = read_results(output_dir / "all_results.csv")
    completed = set()
    if not results.empty:
        for _, row in results[results["status"] == "completed"].iterrows():
            completed.add(tuple(str(row.get(c, "")) for c in [
                "phase", "model_family", "dataset", "fold", "k", "path_count", "fusion",
                "decoder", "train_mode", "batch_size", "seed",
            ]))
    failed_counts = defaultdict(int)
    failed_path = output_dir / "failed_runs.csv"
    if failed_path.exists():
        failed = pd.read_csv(failed_path)
        for _, row in failed.iterrows():
            key = tuple(str(row.get(c, "")) for c in [
                "phase", "model_family", "dataset", "fold", "k", "path_count", "fusion",
                "decoder", "train_mode", "batch_size", "seed",
            ])
            failed_counts[key] += 1
    return completed, failed_counts


def run_phase(phase, args, output_dir):
    runs = build_runs_for_phase(phase, args, output_dir)
    print(f"[PHASE_START] phase={phase} total_runs={len(runs)} dry_run={args.dry_run}", flush=True)
    if args.dry_run:
        for idx, cfg in enumerate(runs, start=1):
            print(
                f"[DRY_RUN] index={idx}/{len(runs)} phase={cfg.phase} "
                f"model_family={cfg.model_family} dataset={cfg.dataset} fold={cfg.fold} "
                f"k={cfg.k if cfg.k is not None else 'NA'} fusion={cfg.fusion} "
                f"path_count={cfg.path_count} decoder={cfg.decoder} train_mode={cfg.train_mode} "
                f"batch_size={cfg.batch_size} seed={cfg.seed}",
                flush=True,
            )
        print(f"[DRY_RUN_DONE] phase={phase} total_runs={len(runs)}", flush=True)
        return {"completed": 0, "failed": 0, "total": len(runs)}

    completed_keys, failed_counts = completed_and_failed(output_dir)
    dirs = ensure_dirs(output_dir)
    completed = 0
    failed = 0
    for cfg in runs:
        key = cfg.key_tuple
        if args.resume and key in completed_keys:
            print(f"[SKIP] run_id={cfg.run_id} reason=already_completed", flush=True)
            completed += 1
            continue
        if failed_counts.get(key, 0) >= 2:
            print(f"[SKIP] run_id={cfg.run_id} reason=failed_twice", flush=True)
            failed += 1
            continue
        run_log_path = dirs["logs"] / f"{cfg.run_id}.log"
        with TeeRunLog(run_log_path):
            try:
                run_one(cfg, args, output_dir)
                completed += 1
            except Exception as exc:
                failed += 1
                traceback.print_exc()
                error = str(exc).replace("\n", " ")[:1000]
                print(f"[FAILED] run_id={cfg.run_id} error={error}", flush=True)
                append_failed(output_dir, cfg, error)
    write_summaries(output_dir)
    print(
        f"[PHASE_DONE] phase={phase} completed_runs={completed} failed_runs={failed} "
        f"results_csv={output_dir / 'all_results.csv'}",
        flush=True,
    )
    return {"completed": completed, "failed": failed, "total": len(runs)}


def run_one(cfg, args, output_dir):
    run_start = time.time()
    print(
        f"[RUN_START] run_id={cfg.run_id} phase={cfg.phase} model_family={cfg.model_family} "
        f"dataset={cfg.dataset} fold={cfg.fold} k={cfg.k_label} fusion={cfg.fusion} "
        f"path_count={cfg.path_count} decoder={cfg.decoder} train_mode={cfg.train_mode} "
        f"batch_size={cfg.batch_size} seed={cfg.seed}",
        flush=True,
    )
    set_seed(cfg.seed)
    dirs = ensure_dirs(output_dir)
    device = resolve_device(args.device)

    cache = MolecularCache(cfg.dataset)
    train_rows, val_rows, test_rows = load_rows(cfg.dataset, cfg.fold, cfg.seed)
    train_data = PairDataset(train_rows, cache, shuffle=True)
    val_data = PairDataset(val_rows, cache, shuffle=False)
    test_data = PairDataset(test_rows, cache, shuffle=False)
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise RuntimeError("Empty dataset after molecular graph filtering")

    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=train_data.collate_fn,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.batch_size * 3, shuffle=False,
        num_workers=args.num_workers, collate_fn=val_data.collate_fn,
    )
    test_loader = DataLoader(
        test_data, batch_size=cfg.batch_size * 3, shuffle=False,
        num_workers=args.num_workers, collate_fn=test_data.collate_fn,
    )

    model = models.MVN_DDI(
        cache.total_atom_feats, 128, 128, rel_total(cfg.dataset),
        heads_out_feat_params=[64, 64, 64, 64],
        blocks_params=[2, 2, 2, 2],
        fusion_mode=cfg.fusion,
        subxddi_kg_dir=args.subxddi_kg_dir,
        subxddi_kg_dim=384,
        subxddi_kg_max_paths=cfg.path_count,
        cross_attention_heads=4,
        decoder=cfg.decoder,
        kg_k=int(cfg.k or 0),
        hkg_cache_dir=args.hkg_cache_dir,
        prefer_hkg_cache=args.prefer_hkg_cache,
    ).to(device)

    configure_train_mode(model, cfg.train_mode, cfg)
    total_params, trainable_params, frozen_params = count_params(model)
    print(
        f"[PARAMS] run_id={cfg.run_id} total={total_params} "
        f"trainable={trainable_params} frozen={frozen_params}",
        flush=True,
    )
    optimizer = build_optimizer(model, args)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)
    loss_fn = SigmoidLoss()

    paths = paths_for_run(dirs, cfg)
    config_payload = asdict(cfg)
    config_payload.update({
        "device": str(device),
        "project_root": str(PROJECT_ROOT),
        "subxddi_kg_dir": args.subxddi_kg_dir,
        "hkg_cache_dir": args.hkg_cache_dir,
        "hkg_cache_file": "hkg_cache.pt" if int(cfg.k or 0) == 1 else f"hkg_cache_k{int(cfg.k)}.pt",
        "prefer_hkg_cache": args.prefer_hkg_cache,
        "best_metric": args.best_metric,
        "attention_entropy_lambda": args.attention_entropy_lambda,
        "attention_min_paths": args.attention_min_paths,
        "attention_no_weight_decay": args.attention_no_weight_decay,
    })
    write_json(paths["config_path"], config_payload)

    history = []
    best = None
    no_improve = 0
    start_epoch = 1
    early_stopped = False
    stop_reason = "max_epoch"
    train_seconds = 0.0

    if args.resume:
        start_epoch, history, best, no_improve = load_partial_run_state(
            paths, model, optimizer, cfg, args, device
        )
        if start_epoch > 1:
            align_lr_schedule(optimizer, scheduler, args.lr, start_epoch)
            print(
                f"[RESUME_EPOCH] run_id={cfg.run_id} start_epoch={start_epoch} "
                f"history_rows={len(history)} best_epoch={best['epoch'] if best else 'NA'}",
                flush=True,
            )

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        print(
            f"[EPOCH_START] run_id={cfg.run_id} epoch={epoch}/{cfg.epochs} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)}",
            flush=True,
        )
        train_loss, train_probs, train_targets, train_extra = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            cfg.run_id, epoch, args.progress_log_interval,
            args.attention_entropy_lambda, args.attention_min_paths,
        )
        train_seconds += time.time() - epoch_start
        train_metrics = compute_metrics(train_targets, train_probs, loss=train_loss)
        val_loss, val_probs, val_targets = evaluate_scores(
            model, val_loader, loss_fn, device,
            cfg.run_id, epoch, args.progress_log_interval,
        )
        val_metrics = compute_metrics(val_targets, val_probs, loss=val_loss)

        best_threshold = best_f1_threshold(val_targets, val_probs)
        lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_auc": train_metrics["roc_auc"],
            "val_auc": val_metrics["roc_auc"],
            "train_f1": train_metrics["f1_at_0_5"],
            "val_f1": val_metrics["f1_at_0_5"],
            "learning_rate": lr,
        }
        history_row.update(train_extra)
        history_row.update(cross_attention_qk_norms(model))
        history.append(history_row)
        save_history(paths, history, args)
        print(
            f"[EPOCH_SAVED] run_id={cfg.run_id} epoch={epoch} "
            f"history={paths['history_path']} elapsed={time.time() - epoch_start:.1f}s",
            flush=True,
        )

        improved, metric_name, metric_value = is_improved(best, val_metrics, val_loss, args.best_metric)
        if improved:
            best = {
                "epoch": epoch,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_threshold": best_threshold,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            no_improve = 0
            save_checkpoint(paths, model, optimizer, cfg, best, args)
            print(
                f"[BEST] run_id={cfg.run_id} epoch={epoch} metric={metric_name} "
                f"value={metric_value:.6f} checkpoint={paths['checkpoint_pt_path']}",
                flush=True,
            )
        else:
            no_improve += 1

        print(
            f"[EPOCH] run_id={cfg.run_id} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_metrics['accuracy']:.6f} "
            f"val_acc={val_metrics['accuracy']:.6f} "
            f"val_auc={val_metrics['roc_auc']:.6f} "
            f"val_f1={val_metrics['f1_at_0_5']:.6f}"
            f"{format_optional_metrics(history_row, ['train_ddi_loss', 'train_attention_entropy_loss', 'attention_entropy', 'attention_spread', 'cross_attention_q_weight_norm', 'cross_attention_k_weight_norm'])}",
            flush=True,
        )

        scheduler.step()
        save_last_checkpoint(paths, model, optimizer, cfg, epoch)
        if (
            epoch >= args.min_epoch
            and val_metrics["accuracy"] > args.stop_acc_threshold
            and no_improve >= args.patience
        ):
            early_stopped = True
            stop_reason = "val_acc_threshold"
            break

    if best is None:
        raise RuntimeError("No valid best checkpoint was produced")

    model.load_state_dict(best["state_dict"])
    model.to(device)
    eval_start = time.time()
    test_loss, test_probs, test_targets = evaluate_scores(model, test_loader, loss_fn, device)
    eval_seconds = time.time() - eval_start
    threshold = best["best_threshold"]
    test_metrics = compute_metrics(test_targets, test_probs, loss=test_loss, threshold=threshold)
    save_curves(paths, test_targets, test_probs, args)
    export_reasoning_outputs(model, test_loader, device, paths, cfg, args)

    stopped_epoch = history[-1]["epoch"]
    row = result_row(
        cfg, best, test_metrics, test_loss, paths, total_params, trainable_params,
        frozen_params, train_seconds, eval_seconds, time.time() - run_start,
        stopped_epoch, early_stopped, stop_reason,
    )
    append_result(output_dir, row)
    print(
        f"[RESULT] phase={cfg.phase} model_family={cfg.model_family} dataset={cfg.dataset} "
        f"fold={cfg.fold} k={cfg.k} path_count={cfg.path_count} fusion={cfg.fusion} decoder={cfg.decoder} "
        f"train_mode={cfg.train_mode} batch_size={cfg.batch_size} seed={cfg.seed} "
        f"best_epoch={best['epoch']} stopped_epoch={stopped_epoch} "
        f"early_stopped={early_stopped} best_threshold={threshold:.2f} "
        f"acc={test_metrics['accuracy']:.6f} f1={test_metrics['f1']:.6f} "
        f"roc_auc={test_metrics['roc_auc']:.6f} auc={test_metrics['roc_auc']:.6f}",
        flush=True,
    )


def move_tri_to_device(tri, device):
    return tuple(item.to(device=device) if hasattr(item, "to") else item for item in tri)


def get_scores(output):
    return output["scores"] if isinstance(output, dict) else output


def batch_scores(model, batch, device, return_attention=False):
    pos_tri, neg_tri = batch
    pos_tri = move_tri_to_device(pos_tri, device)
    neg_tri = move_tri_to_device(neg_tri, device)
    p_out = model(pos_tri, return_attention=return_attention)
    n_out = model(neg_tri, return_attention=return_attention)
    p_score = get_scores(p_out)
    n_score = get_scores(n_out)
    probs = np.concatenate([
        torch.sigmoid(p_score.detach()).cpu().numpy(),
        torch.sigmoid(n_score.detach()).cpu().numpy(),
    ])
    targets = np.concatenate([np.ones(len(p_score)), np.zeros(len(n_score))])
    if return_attention:
        return p_score, n_score, probs, targets, p_out, n_out
    return p_score, n_score, probs, targets


def train_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    run_id,
    epoch,
    progress_log_interval,
    attention_entropy_lambda=0.0,
    attention_min_paths=2,
):
    model.train()
    total_loss = 0.0
    total_ddi_loss = 0.0
    total_attention_loss = 0.0
    total_pos = 0
    probs_all, targets_all = [], []
    stat_sums = defaultdict(float)
    stat_counts = defaultdict(int)
    total_batches = len(loader)
    split_start = time.time()
    use_attention_loss = attention_entropy_lambda > 0.0
    for batch_index, batch in enumerate(loader, start=1):
        if use_attention_loss:
            p_score, n_score, probs, targets, p_out, n_out = batch_scores(
                model, batch, device, return_attention=True
            )
        else:
            p_score, n_score, probs, targets = batch_scores(model, batch, device)
            p_out, n_out = None, None
        ddi_loss = loss_fn(p_score, n_score)
        attention_loss, attention_stats = batch_attention_regularizer(
            [p_out, n_out], p_score, attention_min_paths
        )
        loss = ddi_loss + float(attention_entropy_lambda) * attention_loss
        optimizer.zero_grad()
        loss.backward()
        sync_optimizer_state_with_grads(optimizer)
        optimizer.step()
        total_loss += float(loss.item()) * len(p_score)
        total_ddi_loss += float(ddi_loss.item()) * len(p_score)
        total_attention_loss += float(attention_loss.item()) * len(p_score)
        total_pos += len(p_score)
        probs_all.append(probs)
        targets_all.append(targets)
        for key, value in attention_stats.items():
            stat_sums[key] += float(value)
            stat_counts[key] += 1
        log_progress(
            run_id, epoch, "train", batch_index, total_batches,
            total_pos, split_start, progress_log_interval,
        )
    print(
        f"[TRAIN_DONE] run_id={run_id} epoch={epoch} "
        f"batches={total_batches} examples={total_pos} elapsed={time.time() - split_start:.1f}s",
        flush=True,
    )
    extra = {
        "train_ddi_loss": total_ddi_loss / max(1, total_pos),
        "train_attention_entropy_loss": total_attention_loss / max(1, total_pos),
    }
    extra.update({key: stat_sums[key] / stat_counts[key] for key in stat_sums})
    return total_loss / max(1, total_pos), np.concatenate(probs_all), np.concatenate(targets_all), extra


def batch_attention_regularizer(outputs, reference_scores, attention_min_paths):
    losses = []
    stat_values = defaultdict(list)
    for output in outputs:
        loss, stats = attention_entropy_regularizer(output, attention_min_paths)
        if loss is not None:
            losses.append(loss)
        for key, value in stats.items():
            stat_values[key].append(value)
    if losses:
        regularizer = torch.stack(losses).mean()
    else:
        regularizer = reference_scores.new_zeros(())
    stats = {
        key: float(np.mean(values))
        for key, values in stat_values.items()
        if values
    }
    return regularizer, stats


def attention_entropy_regularizer(output, attention_min_paths):
    if not isinstance(output, dict):
        return None, {}
    attention = output.get("attention") or {}
    padding_mask = attention.get("kg_padding_mask")
    losses = []
    stat_values = defaultdict(list)
    for label in ("fusion_heads", "fusion_tails"):
        payload = attention.get(label) or {}
        weights = payload.get("dsn_to_kg")
        if weights is None:
            continue
        path_probs = weights.mean(dim=(1, 2))
        if padding_mask is not None:
            valid = (~padding_mask.to(device=path_probs.device)).to(dtype=path_probs.dtype)
        else:
            valid = torch.ones_like(path_probs)
        valid_counts = valid.sum(dim=-1)
        eligible = valid_counts >= max(2, int(attention_min_paths))
        if not bool(eligible.any()):
            continue

        path_probs = path_probs * valid
        path_probs = path_probs / path_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(path_probs.clamp_min(1e-12).log() * path_probs * valid).sum(dim=-1)
        normalized_entropy = entropy / valid_counts.clamp_min(2.0).log()
        eligible_probs = path_probs[eligible]
        eligible_valid = valid[eligible].bool()
        valid_max = eligible_probs.masked_fill(~eligible_valid, 0.0).max(dim=-1).values
        valid_min = eligible_probs.masked_fill(~eligible_valid, 1.0).min(dim=-1).values

        losses.append(normalized_entropy[eligible].mean())
        stat_values["attention_entropy"].append(float(normalized_entropy[eligible].detach().mean().item()))
        stat_values["attention_spread"].append(float((valid_max - valid_min).detach().mean().item()))
        stat_values["attention_max"].append(float(valid_max.detach().mean().item()))
        stat_values["attention_valid_paths"].append(float(valid_counts[eligible].detach().mean().item()))

    if not losses:
        return None, {}
    stats = {
        key: float(np.mean(values))
        for key, values in stat_values.items()
        if values
    }
    return torch.stack(losses).mean(), stats


@torch.no_grad()
def evaluate_scores(model, loader, loss_fn, device, run_id=None, epoch=None, progress_log_interval=0):
    model.eval()
    total_loss = 0.0
    total_pos = 0
    probs_all, targets_all = [], []
    total_batches = len(loader)
    split_start = time.time()
    for batch_index, batch in enumerate(loader, start=1):
        p_score, n_score, probs, targets = batch_scores(model, batch, device)
        loss = loss_fn(p_score, n_score)
        total_loss += float(loss.item()) * len(p_score)
        total_pos += len(p_score)
        probs_all.append(probs)
        targets_all.append(targets)
        if run_id is not None and epoch is not None:
            log_progress(
                run_id, epoch, "val", batch_index, total_batches,
                total_pos, split_start, progress_log_interval,
            )
    if run_id is not None and epoch is not None:
        print(
            f"[VAL_DONE] run_id={run_id} epoch={epoch} "
            f"batches={total_batches} examples={total_pos} elapsed={time.time() - split_start:.1f}s",
            flush=True,
        )
    return total_loss / max(1, total_pos), np.concatenate(probs_all), np.concatenate(targets_all)


def log_progress(run_id, epoch, split, batch_index, total_batches, examples, split_start, interval):
    if interval <= 0:
        return
    if batch_index != 1 and batch_index != total_batches and batch_index % interval != 0:
        return
    elapsed = time.time() - split_start
    pct = 100.0 * batch_index / max(1, total_batches)
    print(
        f"[{split.upper()}_PROGRESS] run_id={run_id} epoch={epoch} "
        f"batch={batch_index}/{total_batches} pct={pct:.1f} "
        f"examples={examples} elapsed={elapsed:.1f}s",
        flush=True,
    )


def compute_metrics(targets, probs, loss=np.nan, threshold=0.5):
    targets = np.asarray(targets).astype(int)
    probs = np.asarray(probs).astype(float)
    pred_05 = (probs >= 0.5).astype(int)
    pred = (probs >= threshold).astype(int)
    out = {
        "loss": float(loss),
        "accuracy": safe_metric(metrics.accuracy_score, targets, pred),
        "accuracy_at_0_5": safe_metric(metrics.accuracy_score, targets, pred_05),
        "f1": safe_metric(metrics.f1_score, targets, pred, zero_division=0),
        "f1_at_0_5": safe_metric(metrics.f1_score, targets, pred_05, zero_division=0),
        "roc_auc": safe_metric(metrics.roc_auc_score, targets, probs),
        "auc": safe_metric(metrics.roc_auc_score, targets, probs),
        "ap": safe_metric(metrics.average_precision_score, targets, probs),
        "ap_at_50": ap_at_k(targets, probs, 50),
    }
    try:
        precision, recall, _ = metrics.precision_recall_curve(targets, probs)
        out["pr_auc"] = float(metrics.auc(recall, precision))
    except Exception:
        out["pr_auc"] = np.nan
    return out


def safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return np.nan


def best_f1_threshold(targets, probs):
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        pred = (probs >= threshold).astype(int)
        score = safe_metric(metrics.f1_score, targets, pred, zero_division=0)
        if not np.isnan(score) and score > best_f1:
            best_t, best_f1 = float(threshold), score
    return best_t


def ap_at_k(targets, probs, k):
    targets = np.asarray(targets).astype(int)
    if targets.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(probs))[: min(k, len(probs))]
    hits = targets[order]
    if len(hits) == 0:
        return np.nan
    precisions = []
    positive_seen = 0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            positive_seen += 1
            precisions.append(positive_seen / idx)
    return float(np.mean(precisions)) if precisions else 0.0


def is_improved(best, val_metrics, val_loss, best_metric="val_auc"):
    metric_name, metric_value = primary_metric_value(val_metrics, val_loss, best_metric)
    if best is None:
        return True, metric_name, metric_value
    current_tuple = metric_tuple(val_metrics, val_loss, best_metric)
    best_tuple = metric_tuple(best["val_metrics"], best["val_loss"], best_metric)
    return current_tuple > best_tuple, metric_name, metric_value


def primary_metric_value(m, loss, best_metric):
    if best_metric == "val_accuracy":
        return "val_accuracy", m["accuracy"]
    if best_metric == "val_f1":
        return "val_f1", m["f1_at_0_5"]
    if best_metric == "val_loss":
        return "val_loss", -float(loss)
    return "val_auc", m["roc_auc"]


def metric_tuple(m, loss, best_metric="val_auc"):
    if best_metric == "val_accuracy":
        return (
            nan_to_neg(m["accuracy"]),
            nan_to_neg(m["roc_auc"]),
            nan_to_neg(m["f1_at_0_5"]),
            -float(loss),
        )
    if best_metric == "val_f1":
        return (
            nan_to_neg(m["f1_at_0_5"]),
            nan_to_neg(m["roc_auc"]),
            nan_to_neg(m["accuracy"]),
            -float(loss),
        )
    if best_metric == "val_loss":
        return (
            -float(loss),
            nan_to_neg(m["roc_auc"]),
            nan_to_neg(m["f1_at_0_5"]),
            nan_to_neg(m["accuracy"]),
        )
    return (
        nan_to_neg(m["roc_auc"]),
        nan_to_neg(m["f1_at_0_5"]),
        nan_to_neg(m["accuracy"]),
        -float(loss),
    )


def nan_to_neg(value):
    return -1e18 if value is None or np.isnan(value) else float(value)


def configure_train_mode(model, train_mode, cfg):
    if train_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return
    if train_mode != "decoder_only":
        raise ValueError(f"Unsupported train_mode={train_mode}")

    for p in model.parameters():
        p.requires_grad = False
    trainable_modules = [model.decoder_fusion]
    if cfg.decoder == "original_decoder":
        trainable_modules.extend([model.co_attention, model.KGE])
    elif cfg.decoder == "coattn_binary_no_rescal":
        trainable_modules.extend([model.co_attention, model.binary_relation, model.binary_classifier])
    else:
        trainable_modules.extend([model.binary_relation, model.binary_classifier])
    for module in trainable_modules:
        for p in module.parameters():
            p.requires_grad = True

    encoder_prefixes = ("initial_norm", "block", "net_norms", "sumgnn_encoder")
    leaks = [name for name, p in model.named_parameters() if p.requires_grad and name.startswith(encoder_prefixes)]
    if leaks:
        raise RuntimeError(f"decoder_only has trainable encoder parameters: {leaks[:5]}")


def build_optimizer(model, args):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_weight_decay_param(name, args.attention_no_weight_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": args.weight_decay, "group_name": "decay"})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "group_name": "no_decay"})
    print(
        f"[OPTIMIZER] decay_params={sum(p.numel() for p in decay_params)} "
        f"no_decay_params={sum(p.numel() for p in no_decay_params)} "
        f"weight_decay={args.weight_decay} attention_no_weight_decay={args.attention_no_weight_decay}",
        flush=True,
    )
    return optim.Adam(groups, lr=args.lr)


def is_no_weight_decay_param(name, attention_no_weight_decay):
    lowered = name.lower()
    if name.endswith(".bias") or "norm" in lowered:
        return True
    if not attention_no_weight_decay:
        return False
    attention_keywords = (
        "decoder_fusion.cross_attention",
        "decoder_fusion.sumgnn_proj",
        "sumgnn_encoder.path_token_proj",
    )
    return any(keyword in name for keyword in attention_keywords)


def cross_attention_qk_norms(model):
    module = getattr(getattr(model, "decoder_fusion", None), "cross_attention", None)
    if module is None or not hasattr(module, "in_proj_weight"):
        return {}
    embed_dim = int(module.embed_dim)
    weight = module.in_proj_weight.detach()
    out = {
        "cross_attention_q_weight_norm": float(weight[:embed_dim].norm().item()),
        "cross_attention_k_weight_norm": float(weight[embed_dim:2 * embed_dim].norm().item()),
        "cross_attention_v_weight_norm": float(weight[2 * embed_dim:].norm().item()),
    }
    bias = getattr(module, "in_proj_bias", None)
    if bias is not None:
        bias = bias.detach()
        out.update({
            "cross_attention_q_bias_norm": float(bias[:embed_dim].norm().item()),
            "cross_attention_k_bias_norm": float(bias[embed_dim:2 * embed_dim].norm().item()),
            "cross_attention_v_bias_norm": float(bias[2 * embed_dim:].norm().item()),
        })
    return out


def format_optional_metrics(row, keys):
    parts = []
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        parts.append(f"{key}={numeric:.6f}")
    return (" " + " ".join(parts)) if parts else ""


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def ensure_dirs(output_dir):
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "history": output_dir / "history",
        "plots": output_dir / "plots",
        "curves": output_dir / "curves",
        "configs": output_dir / "configs",
        "logs": output_dir / "logs",
        "reasoning": output_dir / "reasoning",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def paths_for_run(dirs, cfg):
    rid = cfg.run_id
    return {
        "checkpoint_pt_path": dirs["checkpoints"] / f"{rid}_best.pt",
        "checkpoint_pkl_path": dirs["checkpoints"] / f"{rid}_best.pkl",
        "last_checkpoint_pt_path": dirs["checkpoints"] / f"{rid}_last.pt",
        "best_config_path": dirs["checkpoints"] / f"{rid}_best_config.json",
        "best_metrics_path": dirs["checkpoints"] / f"{rid}_best_metrics.json",
        "history_path": dirs["history"] / f"{rid}_history.csv",
        "history_json_path": dirs["history"] / f"{rid}_history.json",
        "loss_plot_path": dirs["plots"] / f"{rid}_loss.png",
        "accuracy_plot_path": dirs["plots"] / f"{rid}_accuracy.png",
        "auc_plot_path": dirs["plots"] / f"{rid}_auc.png",
        "f1_plot_path": dirs["plots"] / f"{rid}_f1.png",
        "roc_curve_path": dirs["curves"] / f"{rid}_roc_curve.csv",
        "roc_curve_plot_path": dirs["curves"] / f"{rid}_roc_curve.png",
        "pr_curve_path": dirs["curves"] / f"{rid}_pr_curve.csv",
        "pr_curve_plot_path": dirs["curves"] / f"{rid}_pr_curve.png",
        "config_path": dirs["configs"] / f"{rid}_config.json",
        "reasoning_jsonl_path": dirs["reasoning"] / f"{rid}_test_reasoning_paths.jsonl",
        "reasoning_csv_path": dirs["reasoning"] / f"{rid}_test_reasoning_paths.csv",
    }


def load_partial_run_state(paths, model, optimizer, cfg, args, device):
    history = load_history_rows(paths["history_path"])
    if not history:
        return 1, [], None, 0

    last_epoch = int(history[-1]["epoch"])
    if last_epoch >= cfg.epochs:
        return 1, [], None, 0

    last_checkpoint_path = paths["last_checkpoint_pt_path"]
    resume_checkpoint_path = (
        last_checkpoint_path if last_checkpoint_path.exists() else paths["checkpoint_pt_path"]
    )
    if not resume_checkpoint_path.exists():
        raise RuntimeError(
            f"Cannot resume {cfg.run_id}: history exists through epoch {last_epoch}, "
            f"but no checkpoint exists at {resume_checkpoint_path}"
        )

    checkpoint = torch.load(str(resume_checkpoint_path), map_location="cpu")
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    if checkpoint_epoch != last_epoch:
        raise RuntimeError(
            f"Cannot resume {cfg.run_id}: history ends at epoch {last_epoch}, "
            f"but checkpoint epoch is {checkpoint_epoch}"
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state_dict = move_nested_to_device(checkpoint["optimizer_state_dict"], device)
    optimizer.load_state_dict(optimizer_state_dict)
    move_optimizer_state_to_device(optimizer, device)
    best = load_best_state(paths, history)
    start_epoch = last_epoch + 1
    no_improve = max(0, last_epoch - int(best["epoch"])) if best else 0
    return start_epoch, history, best, no_improve


def load_history_rows(path):
    if not path.exists():
        return []
    rows = []
    for row in pd.read_csv(path).to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                value = None
            clean[key] = value
        clean["epoch"] = int(clean["epoch"])
        rows.append(clean)
    return rows


def load_best_state(paths, history):
    checkpoint = torch.load(str(paths["checkpoint_pt_path"]), map_location="cpu")
    metrics_payload = read_json(paths["best_metrics_path"]) if paths["best_metrics_path"].exists() else {}
    best_epoch = int(checkpoint["epoch"])
    history_by_epoch = {int(row["epoch"]): row for row in history}
    best_row = history_by_epoch.get(best_epoch, {})
    train_metrics = metrics_payload.get("train_metrics") or metrics_from_history(best_row, "train")
    val_metrics = metrics_payload.get("val_metrics") or metrics_from_history(best_row, "val")
    return {
        "epoch": best_epoch,
        "metric_name": checkpoint.get("best_metric", metrics_payload.get("best_metric", "val_auc")),
        "metric_value": checkpoint.get(
            "best_metric_value", metrics_payload.get("best_metric_value", best_row.get("val_auc"))
        ),
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
        "train_loss": best_row.get("train_loss"),
        "val_loss": best_row.get("val_loss"),
        "best_threshold": metrics_payload.get("best_threshold", 0.5),
        "state_dict": checkpoint["model_state_dict"],
    }


def metrics_from_history(row, prefix):
    return {
        "loss": row.get(f"{prefix}_loss"),
        "accuracy": row.get(f"{prefix}_accuracy"),
        "roc_auc": row.get(f"{prefix}_auc"),
        "f1_at_0_5": row.get(f"{prefix}_f1"),
        "f1": row.get(f"{prefix}_f1"),
    }


def align_lr_schedule(optimizer, scheduler, base_lr, start_epoch):
    lr = base_lr * (0.96 ** max(0, start_epoch - 1))
    for group in optimizer.param_groups:
        group["lr"] = lr
    scheduler.last_epoch = start_epoch - 1


def move_optimizer_state_to_device(optimizer, device):
    for param, state in optimizer.state.items():
        target_device = param.device if isinstance(param, torch.Tensor) else device
        for key, value in list(state.items()):
            state[key] = move_nested_to_device(value, target_device)


def sync_optimizer_state_with_grads(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = optimizer.state.get(param)
            if not state:
                continue
            target_device = param.grad.device
            for key, value in list(state.items()):
                state[key] = move_nested_to_device(value, target_device)


def move_nested_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested_to_device(item, device) for item in value)
    return value


def runtime_config_payload(cfg, args):
    payload = asdict(cfg)
    payload.update({
        "best_metric": args.best_metric,
        "attention_entropy_lambda": args.attention_entropy_lambda,
        "attention_min_paths": args.attention_min_paths,
        "attention_no_weight_decay": args.attention_no_weight_decay,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "stop_acc_threshold": args.stop_acc_threshold,
        "progress_log_interval": args.progress_log_interval,
        "prefer_hkg_cache": args.prefer_hkg_cache,
    })
    return payload


def named_best_checkpoint_path(paths, cfg, epoch, suffix):
    return paths["checkpoint_pt_path"].parent / f"k{cfg.k_label}_{cfg.fusion}_epoch{epoch}{suffix}"


def save_checkpoint(paths, model, optimizer, cfg, best, args):
    payload = {
        "model_state_dict": best["state_dict"],
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": best["epoch"],
        "best_metric": best["metric_name"],
        "best_metric_value": best["metric_value"],
        "config": runtime_config_payload(cfg, args),
    }
    torch.save(payload, paths["checkpoint_pt_path"])
    named_pt_path = named_best_checkpoint_path(paths, cfg, best["epoch"], ".pt")
    torch.save(payload, named_pt_path)
    write_json(paths["best_config_path"], runtime_config_payload(cfg, args))
    write_json(paths["best_metrics_path"], {
        "epoch": best["epoch"],
        "best_metric": best["metric_name"],
        "best_metric_value": best["metric_value"],
        "val_metrics": best["val_metrics"],
        "train_metrics": best["train_metrics"],
        "best_threshold": best["best_threshold"],
        "named_checkpoint_pt_path": str(named_pt_path),
    })
    if args.save_pickle:
        torch.save(model, paths["checkpoint_pkl_path"])
        torch.save(model, named_best_checkpoint_path(paths, cfg, best["epoch"], ".pkl"))


def save_last_checkpoint(paths, model, optimizer, cfg, epoch):
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(cfg),
    }
    torch.save(payload, paths["last_checkpoint_pt_path"])


def save_history(paths, history, args):
    pd.DataFrame(history).to_csv(paths["history_path"], index=False)
    write_json(paths["history_json_path"], history)
    if args.save_plots:
        plot_line(history, "epoch", ["train_loss", "val_loss"], paths["loss_plot_path"])
        plot_line(history, "epoch", ["train_accuracy", "val_accuracy"], paths["accuracy_plot_path"])
        plot_line(history, "epoch", ["train_auc", "val_auc"], paths["auc_plot_path"])
        plot_line(history, "epoch", ["train_f1", "val_f1"], paths["f1_plot_path"])


def save_curves(paths, targets, probs, args):
    if not args.save_curves:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(targets, probs)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(paths["roc_curve_path"], index=False)
        plot_xy(fpr, tpr, paths["roc_curve_plot_path"], "ROC")
    except Exception:
        pass
    try:
        precision, recall, _ = metrics.precision_recall_curve(targets, probs)
        pd.DataFrame({"recall": recall, "precision": precision}).to_csv(paths["pr_curve_path"], index=False)
        plot_xy(recall, precision, paths["pr_curve_plot_path"], "PR")
    except Exception:
        pass


def normalize_reasoning(item):
    item = dict(item or {})
    return {
        "drug_a": item.get("drug_a"),
        "drug_b": item.get("drug_b"),
        "num_paths": item.get("num_paths", 0),
        "num_paths_total": item.get("num_paths_total", item.get("num_paths", 0)),
        "paths_truncated": item.get("paths_truncated", False),
        "subgraph_nodes": item.get("subgraph_nodes", 0),
        "subgraph_edges": item.get("subgraph_edges", 0),
        "source": item.get("source"),
        "fallback_reason": item.get("fallback_reason"),
        "load_error": item.get("load_error"),
        "fusion_head_path_attention": item.get("fusion_head_path_attention"),
        "fusion_tail_path_attention": item.get("fusion_tail_path_attention"),
        "fusion_path_attention": item.get("fusion_path_attention"),
        "path_attention_source": item.get("path_attention_source"),
        "hkg_cache_hit": item.get("hkg_cache_hit", False),
        "hkg_cache_used_as_primary": item.get("hkg_cache_used_as_primary", False),
        "hkg_cache_path": item.get("hkg_cache_path"),
        "paths": item.get("paths") or [],
    }


def sample_attention(attention, index):
    if not attention:
        return {}

    row = {}
    co_attention = attention.get("co_attention")
    if co_attention is not None:
        co_raw = co_attention[index].detach().cpu()
        co_softmax = torch.softmax(co_raw.reshape(-1), dim=0).view_as(co_raw)
        row.update({
            "co_attention_raw_mean": float(co_raw.mean().item()),
            "co_attention_raw_max": float(co_raw.max().item()),
            "co_attention_softmax_max": float(co_softmax.max().item()),
        })

    for prefix, key in (("fusion_head", "fusion_heads"), ("fusion_tail", "fusion_tails")):
        payload = attention.get(key) or {}
        dsn_to_kg = payload.get("dsn_to_kg")
        if dsn_to_kg is not None:
            weights = dsn_to_kg[index].detach().cpu()
            mean_over_heads = weights.mean(dim=0)
            row[f"{prefix}_dsn_to_kg_attention"] = mean_over_heads.tolist()
            row[f"{prefix}_dsn_to_kg_attention_mean"] = float(weights.mean().item())
            row[f"{prefix}_dsn_to_kg_attention_max"] = float(weights.max().item())
            row[f"{prefix}_kg_source_length"] = int(weights.size(-1))

    if row.get("fusion_head_kg_source_length") == 1:
        row["attention_note"] = (
            "Cross-attention source length is 1 for this sample; no non-empty path token was available."
        )
    return row


@torch.no_grad()
def export_reasoning_kind(model, tri, pair_kind, split, batch_index, global_offset, rows):
    out = model(tri, return_reasoning=True, return_attention=True)
    scores = get_scores(out).detach().cpu()
    probs = torch.sigmoid(scores).numpy().tolist()
    pair_ids = tri[-1]
    rels = tri[2].detach().cpu().view(-1).tolist()
    reasoning = out.get("reasoning_paths", []) if isinstance(out, dict) else []
    attention = out.get("attention", {}) if isinstance(out, dict) else {}

    for idx, pair in enumerate(pair_ids):
        item = normalize_reasoning(reasoning[idx] if idx < len(reasoning) else {})
        row = {
            "split": split,
            "batch_index": batch_index,
            "row_index": global_offset + idx,
            "pair_kind": pair_kind,
            "drug_a": pair[0],
            "drug_b": pair[1],
            "relation": rels[idx] if idx < len(rels) else None,
            "score": float(scores[idx].item()),
            "probability": float(probs[idx]),
            **item,
            **sample_attention(attention, idx),
        }
        rows.append(row)


def write_reasoning_outputs(rows, paths):
    fieldnames = [
        "split", "batch_index", "row_index", "pair_kind", "drug_a", "drug_b",
        "relation", "score", "probability", "num_paths", "num_paths_total",
        "paths_truncated", "subgraph_nodes", "subgraph_edges", "source",
        "fallback_reason", "load_error", "fusion_head_path_attention",
        "fusion_tail_path_attention", "fusion_path_attention", "path_attention_source",
        "hkg_cache_hit", "hkg_cache_used_as_primary", "hkg_cache_path",
        "paths", "fusion_head_dsn_to_kg_attention",
        "fusion_head_dsn_to_kg_attention_mean", "fusion_head_dsn_to_kg_attention_max",
        "fusion_head_kg_source_length", "fusion_tail_dsn_to_kg_attention",
        "fusion_tail_dsn_to_kg_attention_mean", "fusion_tail_dsn_to_kg_attention_max",
        "fusion_tail_kg_source_length", "co_attention_raw_mean",
        "co_attention_raw_max", "co_attention_softmax_max", "attention_note",
    ]

    with open(paths["reasoning_jsonl_path"], "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=json_default, ensure_ascii=False) + "\n")

    with open(paths["reasoning_csv_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key, value in list(csv_row.items()):
                if isinstance(value, (list, dict)):
                    csv_row[key] = json.dumps(value, default=json_default, ensure_ascii=False)
            writer.writerow({key: csv_row.get(key, "") for key in fieldnames})


@torch.no_grad()
def export_reasoning_outputs(model, test_loader, device, paths, cfg, args):
    if not args.export_reasoning:
        return
    model.eval()
    exported = []
    global_offset = 0
    max_batches = None if args.reasoning_max_batches == 0 else args.reasoning_max_batches
    for batch_index, batch in enumerate(test_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        pos_tri, neg_tri = batch
        pos_tri = move_tri_to_device(pos_tri, device)
        neg_tri = move_tri_to_device(neg_tri, device)
        export_reasoning_kind(
            model, pos_tri, "positive", args.reasoning_split, batch_index, global_offset, exported
        )
        export_reasoning_kind(
            model, neg_tri, "negative", args.reasoning_split, batch_index, global_offset, exported
        )
        global_offset += len(pos_tri[-1])

    write_reasoning_outputs(exported, paths)
    print(
        f"[REASONING_DONE] run_id={cfg.run_id} rows={len(exported)} "
        f"jsonl={paths['reasoning_jsonl_path']} csv={paths['reasoning_csv_path']}",
        flush=True,
    )


def plot_line(history, x_key, y_keys, path):
    if not history:
        return
    xs = [row[x_key] for row in history]
    series = [(key, [row.get(key, np.nan) for row in history]) for key in y_keys]
    draw_plot(
        xs,
        series,
        path,
        path.stem,
        x_label=axis_label(x_key),
        y_label=metric_axis_label(y_keys),
    )


def plot_xy(xs, ys, path, title):
    labels = {
        "ROC": ("False positive rate", "True positive rate"),
        "PR": ("Recall", "Precision"),
    }
    x_label, y_label = labels.get(title, ("X", "Y"))
    draw_plot(list(xs), [(title, list(ys))], path, title, x_label=x_label, y_label=y_label)


def draw_plot(xs, series, path, title, x_label="X", y_label="Y"):
    width, height = 760, 420
    left_margin = 88
    right_margin = 32
    top_margin = 55
    bottom_margin = 78
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left_margin, 18), title, fill=(20, 20, 20))
    x0, y0 = left_margin, height - bottom_margin
    x1, y1 = width - right_margin, top_margin
    draw.line((x0, y0, x1, y0), fill=(120, 120, 120))
    draw.line((x0, y0, x0, y1), fill=(120, 120, 120))
    values = [v for _, ys in series for v in ys if not np.isnan(v)]
    if not values:
        image.save(path)
        return
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def xp(x):
        return x0 + (x - x_min) * (x1 - x0) / (x_max - x_min)

    def yp(y):
        return y0 - (y - y_min) * (y0 - y1) / (y_max - y_min)

    def text_size(text):
        try:
            bbox = draw.textbbox((0, 0), text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            return draw.textsize(text)

    for tick in x_ticks(x_min, x_max):
        px = xp(tick)
        draw.line((px, y0, px, y0 + 5), fill=(120, 120, 120))
        label = format_tick(tick)
        tw, _ = text_size(label)
        draw.text((px - tw / 2, y0 + 9), label, fill=(70, 70, 70))

    for tick in y_ticks(y_min, y_max):
        py = yp(tick)
        draw.line((x0 - 5, py, x0, py), fill=(120, 120, 120))
        draw.line((x0, py, x1, py), fill=(230, 230, 230))
        label = format_tick(tick)
        tw, th = text_size(label)
        draw.text((x0 - tw - 9, py - th / 2), label, fill=(70, 70, 70))

    colors = [(37, 99, 235), (220, 38, 38), (5, 150, 105), (124, 58, 237)]
    x_label_width, _ = text_size(x_label)
    draw.text(((x0 + x1) / 2 - x_label_width / 2, height - 26), x_label, fill=(35, 35, 35))
    draw_vertical_label(image, y_label, 18, (y0 + y1) / 2)
    draw_legend(draw, series, colors, x1 - 136, y1 + 8)

    for idx, (name, ys) in enumerate(series):
        pts = [(xp(x), yp(y)) for x, y in zip(xs, ys) if not np.isnan(y)]
        if len(pts) >= 2:
            draw.line(pts, fill=colors[idx % len(colors)], width=3)
        elif len(pts) == 1:
            draw.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2),
                         fill=colors[idx % len(colors)])
    image.save(path)


def axis_label(key):
    labels = {
        "epoch": "Epoch",
    }
    return labels.get(str(key), str(key).replace("_", " ").title())


def metric_axis_label(y_keys):
    joined = " ".join(str(key).lower() for key in y_keys)
    if "loss" in joined:
        return "Loss"
    if "accuracy" in joined:
        return "Accuracy"
    if "auc" in joined:
        return "AUC"
    if "f1" in joined:
        return "F1"
    return "Metric"


def series_label(name):
    raw = str(name)
    if raw.startswith("train_"):
        return "train"
    if raw.startswith("val_"):
        return "val"
    return raw.replace("_", " ")


def color_name(index):
    names = ["blue", "red", "green", "purple"]
    return names[index % len(names)]


def draw_legend(draw, series, colors, x, y):
    for idx, (name, _) in enumerate(series):
        y_pos = y + idx * 18
        color = colors[idx % len(colors)]
        draw.line((x, y_pos + 6, x + 24, y_pos + 6), fill=color, width=3)
        draw.text(
            (x + 30, y_pos),
            f"{color_name(idx)} line: {series_label(name)}",
            fill=(35, 35, 35),
        )


def draw_vertical_label(image, text, x, center_y):
    text = str(text)
    probe = ImageDraw.Draw(image)
    try:
        bbox = probe.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width, text_height = probe.textsize(text)
    label = Image.new("RGBA", (text_width + 6, text_height + 6), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((3, 3), text, fill=(35, 35, 35))
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (int(x), int(center_y - rotated.size[1] / 2)), rotated)


def x_ticks(x_min, x_max):
    if 0.0 <= x_min and x_max <= 1.0:
        return [i / 10.0 for i in range(11)]
    start = int(math.ceil(x_min / 10.0) * 10)
    end = int(math.floor(x_max / 10.0) * 10)
    ticks = list(range(start, end + 1, 10)) if end >= start else []
    if not ticks:
        ticks = [x_min, x_max]
    else:
        if ticks[0] > x_min:
            ticks.insert(0, x_min)
        if ticks[-1] < x_max:
            ticks.append(x_max)
    return ticks


def y_ticks(y_min, y_max):
    if 0.0 <= y_min and y_max <= 1.0:
        return [i / 10.0 for i in range(11)]
    step = nice_tick_step((y_max - y_min) / 10.0)
    start = math.ceil(y_min / step) * step
    ticks = []
    current = start
    for _ in range(20):
        if current > y_max + 1e-9:
            break
        ticks.append(current)
        current += step
    if not ticks:
        ticks = [y_min, y_max]
    return ticks


def nice_tick_step(raw_step):
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def format_tick(value):
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def result_row(cfg, best, test_metrics, test_loss, paths, total_params, trainable_params,
               frozen_params, train_seconds, eval_seconds, total_seconds, stopped_epoch,
               early_stopped, stop_reason):
    val_metrics = best["val_metrics"]
    train_metrics = best["train_metrics"]
    return {
        "run_id": cfg.run_id,
        "phase": cfg.phase,
        "model_family": cfg.model_family,
        "dataset": cfg.dataset,
        "fold": cfg.fold,
        "k": cfg.k_label,
        "path_count": cfg.path_count,
        "fusion": cfg.fusion,
        "attention_direction": cfg.attention_direction,
        "decoder": cfg.decoder,
        "train_mode": cfg.train_mode,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "epoch": cfg.epochs,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best["epoch"],
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "best_metric_name": best["metric_name"],
        "best_metric_value": best["metric_value"],
        "best_threshold": best["best_threshold"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_ap": test_metrics["ap"],
        "test_ap_at_50": test_metrics["ap_at_50"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1_at_0_5"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_ap": val_metrics["ap"],
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1_at_0_5"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "test_loss": test_loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "total_seconds": total_seconds,
        "checkpoint_pt_path": str(paths["checkpoint_pt_path"]),
        "checkpoint_pkl_path": str(paths["checkpoint_pkl_path"]),
        "history_path": str(paths["history_path"]),
        "loss_plot_path": str(paths["loss_plot_path"]),
        "accuracy_plot_path": str(paths["accuracy_plot_path"]),
        "auc_plot_path": str(paths["auc_plot_path"]),
        "f1_plot_path": str(paths["f1_plot_path"]),
        "roc_curve_path": str(paths["roc_curve_path"]),
        "pr_curve_path": str(paths["pr_curve_path"]),
        "config_path": str(paths["config_path"]),
        "reasoning_jsonl_path": str(paths["reasoning_jsonl_path"]),
        "reasoning_csv_path": str(paths["reasoning_csv_path"]),
        "status": "completed",
        "error_message": "",
    }


def append_result(output_dir, row):
    csv_path = output_dir / "all_results.csv"
    jsonl_path = output_dir / "all_results.jsonl"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=json_default) + "\n")


def append_failed(output_dir, cfg, error):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "failed_runs.csv"
    columns = ["run_id", "phase", "model_family", "dataset", "fold", "k", "path_count", "fusion",
               "decoder", "train_mode", "batch_size", "seed", "error_message", "time"]
    write_header = not path.exists()
    row = asdict(cfg)
    row.update({"run_id": cfg.run_id, "k": cfg.k_label, "error_message": error, "time": time.time()})
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


def write_summaries(output_dir):
    path = output_dir / "all_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    group_cols = [
        "phase", "model_family", "dataset", "k", "path_count", "fusion",
        "decoder", "train_mode", "batch_size",
    ]
    metric_cols = [
        "test_accuracy", "test_f1", "test_roc_auc", "test_auc", "test_pr_auc",
        "test_ap", "test_ap_at_50", "best_epoch", "total_seconds",
    ]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    rankings = []
    for dataset, group in summary.groupby("dataset"):
        ranked = group.sort_values(
            ["test_roc_auc_mean", "test_f1_mean", "test_accuracy_mean", "test_ap_mean"],
            ascending=False,
            na_position="last",
        ).copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        rankings.append(ranked)
    if rankings:
        pd.concat(rankings).to_csv(output_dir / "best_ranking.csv", index=False)

    df.groupby(["phase", "status"]).size().reset_index(name="count").to_csv(
        output_dir / "phase_summary.csv", index=False
    )


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    if args.background:
        print("[BACKGROUND] Use: bash scripts/run_background.sh <phase>", flush=True)
        return 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    total_completed = 0
    total_failed = 0
    for phase in phases:
        stats = run_phase(phase, args, output_dir)
        total_completed += stats["completed"]
        total_failed += stats["failed"]
        if phase == "sanity" and not args.dry_run and stats["failed"] > 0 and args.phase == "all":
            print("[ALL_STOPPED] reason=sanity_failed", flush=True)
            return 1
    write_summaries(output_dir)
    print(
        f"[ALL_DONE] completed_runs={total_completed} failed_runs={total_failed} "
        f"results_csv={output_dir / 'all_results.csv'} "
        f"summary_csv={output_dir / 'summary_mean_std.csv'} "
        f"ranking_csv={output_dir / 'best_ranking.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
