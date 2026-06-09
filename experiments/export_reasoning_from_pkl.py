#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_0608_experiments as exp  # noqa: E402


class ReasoningArgs:
    def __init__(self, max_batches):
        self.export_reasoning = True
        self.reasoning_max_batches = max_batches
        self.reasoning_split = "test"


def load_config(path):
    with open(path) as f:
        payload = json.load(f)
    return exp.RunConfig(
        phase=payload["phase"],
        model_family=payload["model_family"],
        dataset=payload["dataset"],
        fold=int(payload["fold"]),
        k=payload["k"],
        fusion=payload["fusion"],
        attention_direction=payload.get("attention_direction", "dsn_to_kg"),
        decoder=payload["decoder"],
        train_mode=payload["train_mode"],
        batch_size=int(payload["batch_size"]),
        seed=int(payload["seed"]),
        epochs=int(payload.get("epochs", 200)),
        path_count=int(payload.get("path_count", 10)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-pkl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=str(exp.EXPERIMENT_ROOT))
    parser.add_argument("--tag", default="from_pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reasoning-max-batches", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    cfg = load_config(args.config)
    dirs = exp.ensure_dirs(output_dir)
    paths = exp.paths_for_run(dirs, cfg)
    stem = f"{cfg.run_id}_{args.tag}_test_reasoning_paths"
    paths["reasoning_jsonl_path"] = dirs["reasoning"] / f"{stem}.jsonl"
    paths["reasoning_csv_path"] = dirs["reasoning"] / f"{stem}.csv"

    device = torch.device(args.device)
    exp.set_seed(cfg.seed)
    cache = exp.MolecularCache(cfg.dataset)
    _, _, test_rows = exp.load_rows(cfg.dataset, cfg.fold, cfg.seed)
    test_data = exp.PairDataset(test_rows, cache, shuffle=False)
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.batch_size * 3,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
    )

    model = torch.load(str(Path(args.checkpoint_pkl).resolve()), map_location=device)
    model.to(device)
    exp.export_reasoning_outputs(model, test_loader, device, paths, cfg, ReasoningArgs(args.reasoning_max_batches))
    print(paths["reasoning_jsonl_path"])
    print(paths["reasoning_csv_path"])


if __name__ == "__main__":
    main()
