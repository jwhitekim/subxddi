"""
Precompute H_KG for all drug pairs and save as a cache dict.

Usage:
    python kg/precompute_hkg.py --max_pairs 100   # smoke test
    python kg/precompute_hkg.py                   # full run

Output:
    kg/hkg_cache.pt  →  {(drug_a, drug_b): tensor[384]}

NOTE: KGEncoder weights are random-init unless --kg_encoder_ckpt is provided.
      After a full training run, re-run this script with the trained checkpoint
      to produce meaningful KG features for subsequent training.
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _encode_batch(encoder, graphs, keys, cache):
    import dgl
    with torch.no_grad():
        batched = dgl.batch(graphs)
        result = encoder(batched)
        if result.dim() == 1:
            result = result.unsqueeze(0)
        for key, h_kg in zip(keys, result.cpu()):
            cache[key] = h_kg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/drugbank/fold0')
    parser.add_argument('--kg_graph', default='kg/kg_graph.gpickle')
    parser.add_argument('--emb_path', default='kg/pretrained/transe_entity_emb.npy')
    parser.add_argument('--e2id_path', default='kg/pretrained/transe_entity2id.json')
    parser.add_argument('--kg_code_dir', default='models/kg_encoder')
    parser.add_argument('--output', default='kg/hkg_cache.pt')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of valid subgraphs per DGL encoder call')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Limit pairs for smoke test (e.g. --max_pairs 100)')
    parser.add_argument('--kg_encoder_ckpt', type=str, default=None,
                        help='Path to KGEncoder state_dict (.pt). '
                             'If omitted, random init is used.')
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # ── Load code modules ──────────────────────────────────────────────────
    extract_mod = _load_module(
        'extract_subgraph',
        os.path.join(args.kg_code_dir, 'extract_subgraph.py'),
    )
    kg_enc_mod = _load_module(
        'kg_encoder_mod',
        os.path.join(args.kg_code_dir, 'kg_encoder.py'),
    )

    # ── Load KG data ───────────────────────────────────────────────────────
    print("KG graph 로드 중...")
    with open(args.kg_graph, 'rb') as f:
        G = pickle.load(f)
    print(f"  nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")

    emb_matrix, e2id = None, None
    if os.path.exists(args.emb_path) and os.path.exists(args.e2id_path):
        emb_matrix = np.load(args.emb_path)
        with open(args.e2id_path) as f:
            e2id = json.load(f)
        print(f"  TransE 임베딩: {emb_matrix.shape}")
    else:
        print("  TransE 임베딩 없음 → zero 벡터로 대체")

    # ── Collect unique drug pairs ─────────────────────────────────────────
    print("\nDrug pair 수집 중...")
    pairs = set()
    for csv_name in ['train.csv', 'test.csv']:
        path = os.path.join(args.data_dir, csv_name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            pairs.add((str(row['d1']), str(row['d2'])))
        print(f"  {csv_name}: {len(df):,} rows  (누계 unique pairs: {len(pairs):,})")

    pairs = sorted(pairs)
    total_pairs = len(pairs)
    print(f"총 unique pairs: {total_pairs:,}")

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"--max_pairs {args.max_pairs} 적용 → {len(pairs):,} pairs만 처리")

    # ── Build KGEncoder ────────────────────────────────────────────────────
    emb_dim = emb_matrix.shape[1] if emb_matrix is not None else 0
    encoder = kg_enc_mod.KGEncoder(emb_dim=emb_dim, k=1, hidden_dim=64, num_layers=2)
    if args.kg_encoder_ckpt and os.path.exists(args.kg_encoder_ckpt):
        encoder.load_state_dict(torch.load(args.kg_encoder_ckpt, map_location='cpu'))
        print(f"\nKGEncoder 체크포인트 로드: {args.kg_encoder_ckpt}")
    else:
        print("\nKGEncoder: random init (학습 후 --kg_encoder_ckpt로 재계산 권장)")
    encoder.eval()

    # ── Extract & encode ──────────────────────────────────────────────────
    cache = {}
    zero_tensor = torch.zeros(384)
    zero_count = 0
    valid_graphs, valid_keys = [], []

    t0 = time.time()
    print(f"\nSubgraph 추출 + 인코딩 (batch_size={args.batch_size})...")

    for drug_a, drug_b in tqdm(pairs):
        if drug_a not in G or drug_b not in G:
            cache[(drug_a, drug_b)] = zero_tensor
            zero_count += 1
            continue

        result = extract_mod.extract_enclosing_subgraph(
            G, drug_a, drug_b, k=1, emb_matrix=emb_matrix, e2id=e2id
        )
        if result is None:
            cache[(drug_a, drug_b)] = zero_tensor
            zero_count += 1
        else:
            g, _ = result
            valid_graphs.append(g)
            valid_keys.append((drug_a, drug_b))

        if len(valid_graphs) >= args.batch_size:
            _encode_batch(encoder, valid_graphs, valid_keys, cache)
            valid_graphs, valid_keys = [], []

    if valid_graphs:
        _encode_batch(encoder, valid_graphs, valid_keys, cache)

    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(cache, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024

    # ── Report ────────────────────────────────────────────────────────────
    n = len(pairs)
    valid_count = n - zero_count
    print(f"\n{'='*40}")
    print(f"전체 pair 수:           {n:>8,}")
    print(f"유효 (non-zero) pairs:  {valid_count:>8,}  ({valid_count/n*100:.1f}%)")
    print(f"zero tensor 처리:       {zero_count:>8,}  ({zero_count/n*100:.1f}%)")
    print(f"소요 시간:              {elapsed:>8.1f}s")
    print(f"처리 속도:              {n/elapsed:>8.1f} pairs/s")
    print(f"캐시 파일:              {args.output} ({size_mb:.1f} MB)")
    if args.max_pairs and total_pairs > args.max_pairs:
        est = elapsed / n * total_pairs
        print(f"전체 {total_pairs:,} pairs 예상 시간: {est/60:.1f}분")
    print('='*40)


if __name__ == '__main__':
    main()
