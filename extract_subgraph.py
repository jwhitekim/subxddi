import pickle
import random
from typing import Optional

import networkx as nx
import torch
import dgl


def load_kg(path: str = "kg_graph.gpickle") -> nx.DiGraph:
    """Load the pre-built KG from a pickle file and return the nx.DiGraph."""
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_enclosing_subgraph(
    G: nx.DiGraph, drug_a: str, drug_b: str
) -> Optional[tuple]:
    """Extract the enclosing subgraph for a drug pair.

    Shared-target enclosing subgraph: nodes = {drug_a, drug_b} + shared gene targets.
    Returns None if the two drugs share no target genes.

    Returns:
        (dgl_graph, node_ids) or None
        - dgl_graph: DGL graph with ndata['feat'], ndata['id'], edata['type']
        - node_ids: list of string node IDs in local index order (for explainability)
    """
    genes_a = set(G.successors(drug_a))
    genes_b = set(G.successors(drug_b))
    shared_genes = genes_a & genes_b

    if not shared_genes:
        return None

    nodes = {drug_a, drug_b} | shared_genes
    sub_nx = G.subgraph(nodes).copy()
    return _nx_to_dgl(sub_nx, drug_a, drug_b)


def _nx_to_dgl(sub_nx: nx.DiGraph, drug_a: str, drug_b: str) -> tuple:
    """Convert nx.DiGraph subgraph to DGL graph following SumGNN conventions.

    Node data:
        feat  : [N, 2] float  — (is_drug, is_gene)
        id    : [N]    float  — 0=gene, 1=drug_a, 2=drug_b
    Edge data:
        type  : [E]    long   — relation index (0 = 'targets')
    """
    node_ids = list(sub_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_ids)}

    feat, id_labels = [], []
    for n in node_ids:
        is_drug = 1 if sub_nx.nodes[n]["type"] == "drug" else 0
        feat.append([is_drug, 1 - is_drug])
        if n == drug_a:
            id_labels.append(1)
        elif n == drug_b:
            id_labels.append(2)
        else:
            id_labels.append(0)

    src = [node_to_idx[u] for u, _ in sub_nx.edges()]
    dst = [node_to_idx[v] for _, v in sub_nx.edges()]

    g = dgl.graph((src, dst))
    g.ndata["feat"] = torch.tensor(feat, dtype=torch.float)
    g.ndata["id"]   = torch.tensor(id_labels, dtype=torch.float)
    g.edata["type"] = torch.zeros(g.num_edges(), dtype=torch.long)

    return g, node_ids


def main():
    """Sanity check: sample 5 drug pairs and print DGL subgraph statistics."""
    G = load_kg()
    drugs = [n for n, attr in G.nodes(data=True) if attr.get("type") == "drug"]

    random.seed(42)
    pairs_found = 0
    while pairs_found < 5:
        drug_a, drug_b = random.sample(drugs, 2)
        result = extract_enclosing_subgraph(G, drug_a, drug_b)
        if result is None:
            continue

        g, node_ids = result
        shared_count = g.num_nodes() - 2
        print(
            f"[pair {pairs_found + 1}] {drug_a} & {drug_b}\n"
            f"  shared genes : {shared_count}\n"
            f"  nodes        : {g.num_nodes()}\n"
            f"  edges        : {g.num_edges()}\n"
            f"  feat shape   : {g.ndata['feat'].shape}\n"
            f"  id values    : {g.ndata['id'].tolist()}\n"
        )
        pairs_found += 1


if __name__ == "__main__":
    main()
