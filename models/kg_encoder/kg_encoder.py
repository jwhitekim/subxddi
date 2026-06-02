import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class RGCNLayer(nn.Module):
    """Single R-GCN layer with basis decomposition removed (single relation).

    Follows SumGNN's repr accumulation: each layer appends its output to
    g.ndata['repr'] so the final repr has shape [N, num_layers, hidden_dim].
    Attention and TransE embeddings are removed per SubXDDI design.
    """

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        num_rels: int = 1,
        activation=None,
        dropout: float = 0.0,
        is_input_layer: bool = False,
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.activation = activation
        self.is_input_layer = is_input_layer

        self.weight = nn.Parameter(torch.Tensor(num_rels, inp_dim, out_dim))
        self.self_loop_weight = nn.Parameter(torch.Tensor(inp_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, g: dgl.DGLGraph):
        weight = self.weight
        input_key = "feat" if self.is_input_layer else "h"

        def msg_func(edges):
            w = weight[edges.data["type"]]
            x = edges.src[input_key]
            return {"msg": torch.bmm(x.unsqueeze(1), w).squeeze(1)}

        def reduce_func(nodes):
            return {"h": nodes.mailbox["msg"].sum(dim=1)}

        g.update_all(msg_func, reduce_func)

        # Self-loop applied outside reduce so nodes with no incoming edges
        # (e.g. drug anchors in a drug→gene KG) also receive W_self * h.
        # DGL zero-initialises 'h' for such nodes after update_all.
        h = g.ndata["h"] + g.ndata[input_key] @ self.self_loop_weight
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        g.ndata["h"] = h

        if self.is_input_layer:
            g.ndata["repr"] = h.unsqueeze(1)                                        # [N, 1, out]
        else:
            g.ndata["repr"] = torch.cat([g.ndata["repr"], h.unsqueeze(1)], dim=1)  # [N, l+1, out]


class KGEncoder(nn.Module):
    """DGL-based R-GCN encoder for KG enclosing subgraphs.

    inp_dim is derived automatically as emb_dim + 2*(k+1), matching the
    feat layout produced by extract_subgraph._nx_to_dgl().

    Output per graph:
        h_A    = repr[drug_a node].reshape(-1)                          [num_layers * hidden_dim]
        h_B    = repr[drug_b node].reshape(-1)                          [num_layers * hidden_dim]
        h_GSub = W_Sub(mean(repr[all nodes], dim=0)).reshape(-1)        [num_layers * hidden_dim]
        H_KG   = cat([h_A, h_B, h_GSub])                               [3 * num_layers * hidden_dim]

    Single graph  → shape [3 * num_layers * hidden_dim].
    Batched graph → shape [batch_size, 3 * num_layers * hidden_dim].

    # SumGNN layer-independent self-attention + threshold pruning 제거됨.
    # 대체: 상위 모듈에서 Cross-Attention(H_DSN, H_KG)으로 통합 (SubXDDI 설계).
    # SumGNN Channel2 패턴([h_u||h_v||h_GSub])을 채택. Channel1, Channel3 제거됨.
    """

    def __init__(
        self,
        emb_dim: int = 0,
        k: int = 1,
        hidden_dim: int = 64,
        num_rels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        inp_dim = emb_dim + 2 * (k + 1)

        self.layers = nn.ModuleList()
        self.layers.append(
            RGCNLayer(inp_dim, hidden_dim, num_rels, F.relu, dropout, is_input_layer=True)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_rels, F.relu, dropout)
            )
        self.w_sub = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Encode subgraph(s) into H_KG.

        Args:
            g: Single or dgl.batch()-ed graph with ndata['feat'] and ndata['id']
               (long: 1=drug_a, 2=drug_b, 0=other).

        Returns:
            H_KG: [3*num_layers*hidden_dim] for single graph,
                  [batch_size, 3*num_layers*hidden_dim] for batched.
        """
        g.ndata["h"] = g.ndata["feat"]
        for layer in self.layers:
            layer(g)

        repr_all = g.ndata["repr"]   # [total_N, num_layers, hidden_dim]
        id_all   = g.ndata["id"]     # [total_N]  long

        if g.batch_size == 1:
            return self._extract_repr(repr_all, id_all)

        offset = 0
        results = []
        for n_nodes in g.batch_num_nodes().tolist():
            r   = repr_all[offset : offset + n_nodes]
            ids = id_all[offset : offset + n_nodes]
            results.append(self._extract_repr(r, ids))
            offset += n_nodes

        return torch.stack(results)  # [batch_size, 3*num_layers*hidden_dim]

    def _extract_repr(
        self,
        repr: torch.Tensor,
        id_labels: torch.Tensor,
    ) -> torch.Tensor:
        head_ids = (id_labels == 1).nonzero(as_tuple=False).squeeze(1)
        tail_ids = (id_labels == 2).nonzero(as_tuple=False).squeeze(1)

        h_A    = repr[head_ids].reshape(-1)
        h_B    = repr[tail_ids].reshape(-1)
        # Eq.5: W_Sub applied to mean over all G_Sub nodes (drug + gene), per layer
        h_GSub = self.w_sub(repr.mean(dim=0)).reshape(-1)

        return torch.cat([h_A, h_B, h_GSub])


def main():
    """Smoke test: single graph and batched graph forward pass."""
    emb_dim, k, hidden_dim, num_layers = 64, 1, 64, 2
    out_dim = 3 * num_layers * hidden_dim  # 384
    inp_dim = emb_dim + 2 * (k + 1)       # 68

    model = KGEncoder(emb_dim=emb_dim, k=k, hidden_dim=hidden_dim, num_layers=num_layers)
    model.eval()

    def make_graph(n_genes: int = 2) -> dgl.DGLGraph:
        # Layout: nodes 0..n_genes-1 = genes, node n_genes = drug_a, node n_genes+1 = drug_b
        drug_a_idx = n_genes
        drug_b_idx = n_genes + 1
        n = n_genes + 2
        src = [drug_a_idx] * n_genes + [drug_b_idx] * n_genes
        dst = list(range(n_genes)) * 2
        g = dgl.graph((src, dst))
        g.ndata["feat"]      = torch.randn(n, inp_dim)
        g.ndata["id"]        = torch.tensor([0] * n_genes + [1, 2], dtype=torch.long)
        g.ndata["node_type"] = torch.tensor([1] * n_genes + [0, 0], dtype=torch.long)
        g.edata["type"]      = torch.zeros(g.num_edges(), dtype=torch.long)
        return g

    # Single graph
    g = make_graph(n_genes=2)
    with torch.no_grad():
        H_KG = model(g)
    print(f"Single graph  - H_KG shape: {tuple(H_KG.shape)}  (expected: ({out_dim},))")
    assert H_KG.shape == (out_dim,)

    # Batched graphs
    batch = dgl.batch([make_graph(2), make_graph(3)])
    with torch.no_grad():
        H_batch = model(batch)
    print(f"Batched (x2)  - H_KG shape: {tuple(H_batch.shape)}  (expected: (2, {out_dim}))")
    assert H_batch.shape == (2, out_dim)

    print("All assertions passed.")


if __name__ == "__main__":
    main()
