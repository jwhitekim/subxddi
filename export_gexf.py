import argparse
import pickle
import networkx as nx


def export(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        G = pickle.load(f)
    print(f"[load] Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    nx.write_gexf(G, output_path)
    print(f"[export] Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export KG graph pickle to GEXF format")
    parser.add_argument("--input", default="kg_graph.gpickle", help="Input pickle file")
    parser.add_argument("--output", default="kg_graph.gexf", help="Output GEXF file")
    args = parser.parse_args()

    export(args.input, args.output)
