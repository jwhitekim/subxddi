import json
import os
import pickle
import random
from typing import Optional

import networkx as nx
import numpy as np
import torch
import dgl


_TRANSE_CACHE: dict = {}


def load_kg(path: str = "graph/kg_graph.gpickle") -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_transe_embeddings(
    emb_path: str = "pretrained/transe_entity_emb.npy",
    e2id_path: str = "pretrained/transe_entity2id.json",
) -> tuple:
    emb_matrix = np.load(emb_path)
    with open(e2id_path) as f:
        e2id = json.load(f)
    return emb_matrix, e2id


def _get_transe(
    emb_path: str = "pretrained/transe_entity_emb.npy",
    e2id_path: str = "pretrained/transe_entity2id.json",
) -> tuple:
    """Lazy-load TransE embeddings from disk; return (None, None) if files missing."""
    key = (emb_path, e2id_path)
    if key not in _TRANSE_CACHE:
        if os.path.exists(emb_path) and os.path.exists(e2id_path):
            _TRANSE_CACHE[key] = load_transe_embeddings(emb_path, e2id_path)
        else:
            _TRANSE_CACHE[key] = (None, None)
    return _TRANSE_CACHE[key]


def get_k_hop_neighbors(G: nx.DiGraph, node: str, k: int) -> set:
    """Return all nodes within k hops (undirected), including node itself."""
    visited = {node}
    frontier = {node}
    for _ in range(k):
        next_f = set()
        for n in frontier:
            next_f.update(G.successors(n))
            next_f.update(G.predecessors(n))
        frontier = next_f - visited
        visited.update(frontier)
    return visited


def extract_enclosing_subgraph(
    G: nx.DiGraph,
    drug_a: str,
    drug_b: str,
    k: int = 1,
    emb_matrix: Optional[np.ndarray] = None,
    e2id: Optional[dict] = None,
) -> Optional[tuple]:
    """Extract the enclosing subgraph for a drug pair (SumGNN §3.2-A).

    Nodes = [N_k(drug_a) ∩ N_k(drug_b)] ∪ {drug_a, drug_b}.
    Returns None if no shared nodes exist in the intersection.

    Returns:
        (dgl_graph, node_list) or None
    """
    if emb_matrix is None or e2id is None:
        emb_matrix, e2id = _get_transe()

    neighbors_a = get_k_hop_neighbors(G, drug_a, k)
    neighbors_b = get_k_hop_neighbors(G, drug_b, k)
    shared = (neighbors_a & neighbors_b) - {drug_a, drug_b}

    if not shared:
        return None

    # Drug→Gene 단방향 KG에서 k=1이면 drug_b ∉ N_k(drug_a)이므로
    # anchor drug를 교집합 외부에서 명시적으로 추가 (논문 정의와 불가피한 차이)
    nodes = {drug_a, drug_b} | shared
    sub_nx = G.subgraph(nodes).copy()
    return _nx_to_dgl(sub_nx, drug_a, drug_b, k, emb_matrix, e2id)


def _nx_to_dgl(
    sub_nx: nx.DiGraph,
    drug_a: str,
    drug_b: str,
    k: int,
    emb_matrix: Optional[np.ndarray],
    e2id: Optional[dict],
) -> tuple:
    """Convert nx.DiGraph subgraph to DGL, following SumGNN node feature layout.

    Node data:
        feat      : [N, emb_dim + 2*(k+1)]  — TransE emb || one-hot(d_u) || one-hot(d_v)
        id        : [N]  long               — 0=shared node, 1=drug_a, 2=drug_b
        node_type : [N]  long               — 0=drug, 1=gene
    Edge data:
        type      : [E]  long               — 0 for 'targets'
    """
    node_list = list(sub_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # Shortest-path distances from drug_a and drug_b (undirected, cutoff=k)
    G_undir = sub_nx.to_undirected()
    dist_u = nx.single_source_shortest_path_length(G_undir, drug_a, cutoff=k)
    dist_v = nx.single_source_shortest_path_length(G_undir, drug_b, cutoff=k)

    emb_dim = emb_matrix.shape[1] if emb_matrix is not None else 0

    def to_onehot(d: int) -> list:
        # Unreachable nodes (d > k) land in the last bucket (index k)
        vec = [0] * (k + 1)
        vec[min(d, k)] = 1
        return vec

    feat, id_labels, node_types = [], [], []
    for n in node_list:
        # TransE embedding; zero vector if entity not in vocabulary
        if emb_matrix is not None and e2id is not None and n in e2id:
            transe = emb_matrix[e2id[n]].tolist()
        else:
            transe = [0.0] * emb_dim

        pos_u = to_onehot(dist_u.get(n, k))
        pos_v = to_onehot(dist_v.get(n, k))
        feat.append(transe + pos_u + pos_v)

        id_labels.append(1 if n == drug_a else 2 if n == drug_b else 0)
        node_types.append(0 if sub_nx.nodes[n].get("type") == "drug" else 1)

    src = [node_to_idx[u] for u, _ in sub_nx.edges()]
    dst = [node_to_idx[v] for _, v in sub_nx.edges()]

    g = dgl.graph((src, dst), num_nodes=len(node_list))
    g.ndata["feat"]      = torch.tensor(feat, dtype=torch.float)
    g.ndata["id"]        = torch.tensor(id_labels, dtype=torch.long)
    g.ndata["node_type"] = torch.tensor(node_types, dtype=torch.long)
    g.edata["type"]      = torch.zeros(g.num_edges(), dtype=torch.long)

    return g, node_list


def main():
    G = load_kg()
    emb_matrix, e2id = _get_transe()

    if emb_matrix is not None:
        emb_dim = emb_matrix.shape[1]
        print(f"[TransE] loaded: {emb_matrix.shape[0]} entities, emb_dim={emb_dim}")
    else:
        emb_dim = 0
        print("[TransE] not found — using position vectors only (run train_transe.py first)")

    k = 1
    expected_feat_dim = emb_dim + 2 * (k + 1)

    # Try the specific test pair first, then fall back to random pairs
    drugs = [n for n, attr in G.nodes(data=True) if attr.get("type") == "drug"]
    random.seed(42)
    candidates = [("DB00357", "DB00773")] + [tuple(random.sample(drugs, 2)) for _ in range(20)]

    pairs_shown = 0
    for drug_a, drug_b in candidates:
        if drug_a not in G or drug_b not in G:
            continue
        result = extract_enclosing_subgraph(G, drug_a, drug_b, k=k,
                                            emb_matrix=emb_matrix, e2id=e2id)
        if result is None:
            continue
        g, node_list = result
        print(
            f"[pair] {drug_a} & {drug_b}\n"
            f"  nodes      : {g.num_nodes()}  edges: {g.num_edges()}\n"
            f"  feat shape : {tuple(g.ndata['feat'].shape)}  "
            f"(expected: (N, {expected_feat_dim}))\n"
            f"  id values  : {g.ndata['id'].tolist()}\n"
            f"  node_types : {g.ndata['node_type'].tolist()}\n"
        )
        pairs_shown += 1
        if pairs_shown >= 3:
            break

    if pairs_shown == 0:
        print("No valid pairs found in sample — check KG connectivity.")


if __name__ == "__main__":
    main()
