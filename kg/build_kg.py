import pandas as pd
import networkx as nx
import json
import pickle
import random


TSV_PATH = "dataset/kg_source/ChG-Miner_miner-chem-gene.tsv"


def parse(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", header=None, names=["Drug", "Gene"])
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"[parse] Loaded {len(df)} rows")
    return df


def build_triples(df: pd.DataFrame) -> pd.DataFrame:
    triples = pd.DataFrame({
        "head": df["Drug"].values,
        "relation": "targets",
        "tail": df["Gene"].values,
    })
    return triples


def print_stats(triples: pd.DataFrame) -> dict:
    num_triples = len(triples)
    num_drugs = triples["head"].nunique()
    num_genes = triples["tail"].nunique()
    avg_genes_per_drug = triples.groupby("head")["tail"].nunique().mean()
    avg_drugs_per_gene = triples.groupby("tail")["head"].nunique().mean()

    stats = {
        "num_triples": num_triples,
        "num_drugs": num_drugs,
        "num_genes": num_genes,
        "avg_genes_per_drug": round(avg_genes_per_drug, 4),
        "avg_drugs_per_gene": round(avg_drugs_per_gene, 4),
    }

    print("\n[stats]")
    print(f"  총 트리플 수          : {num_triples:,}")
    print(f"  유니크 Drug 노드 수   : {num_drugs:,}")
    print(f"  유니크 Gene 노드 수   : {num_genes:,}")
    print(f"  Drug당 평균 Gene 수   : {avg_genes_per_drug:.4f}")
    print(f"  Gene당 평균 Drug 수   : {avg_drugs_per_gene:.4f}")

    return stats


def build_graph(triples: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    drugs = triples["head"].unique()
    genes = triples["tail"].unique()

    G.add_nodes_from(drugs, type="drug")
    G.add_nodes_from(genes, type="gene")

    edges = [(row.head, row.tail, {"relation": "targets"}) for row in triples.itertuples()]
    G.add_edges_from(edges)

    print(f"\n[graph] Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return G


def extract_subgraph(G: nx.DiGraph, triples: pd.DataFrame, n_pairs: int = 3) -> None:
    drugs = list(triples["head"].unique())
    random.seed(42)
    sample_drugs = random.sample(drugs, min(n_pairs * 2, len(drugs)))

    print("\n[subgraph] Drug 쌍별 공유 Gene 추출")
    pairs_shown = 0
    for i in range(0, len(sample_drugs) - 1, 2):
        if pairs_shown >= n_pairs:
            break
        drug_a, drug_b = sample_drugs[i], sample_drugs[i + 1]
        genes_a = set(G.successors(drug_a))
        genes_b = set(G.successors(drug_b))
        shared = genes_a & genes_b
        for gene in sorted(shared):
            print(f"  Drug A ({drug_a}) -- {gene} -- Drug B ({drug_b})")
        print(f"  공유 Gene 수: {len(shared)}개\n")
        pairs_shown += 1


def save(triples: pd.DataFrame, G: nx.DiGraph, stats: dict) -> None:
    triples.to_csv("dataset/kg_source/kg_triples.tsv", sep="\t", index=False)
    print("[save] dataset/kg_source/kg_triples.tsv")

    with open("kg/kg_graph.gpickle", "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print("[save] kg/kg_graph.gpickle")

    with open("dataset/kg_source/kg_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("[save] dataset/kg_source/kg_stats.json")


def main():
    df = parse(TSV_PATH)
    triples = build_triples(df)
    stats = print_stats(triples)
    G = build_graph(triples)
    extract_subgraph(G, triples)
    save(triples, G, stats)
    print("\n[done] Knowledge Graph 구축 완료.")


if __name__ == "__main__":
    main()
