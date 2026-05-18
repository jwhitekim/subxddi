import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class RGCNLayer(nn.Module):
    """Single R-GCN layer with basis decomposition removed (single relation).

    Follows SumGNN's repr accumulation: each layer appends its output to
    g.ndata['repr'] so the final repr has shape [N, num_layers, hidden_dim].
    Attention and TransE embeddings are removed.
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
            w = weight[edges.data["type"]]          # [E, inp, out]
            x = edges.src[input_key]                # [E, inp]
            msg = torch.bmm(x.unsqueeze(1), w).squeeze(1)  # [E, out]
            return {"msg": msg}

        def reduce_func(nodes):
            agg = nodes.mailbox["msg"].sum(dim=1)                       # [N, out]
            self_loop = nodes.data[input_key] @ self.self_loop_weight   # [N, out]
            return {"h": agg + self_loop}

        g.update_all(msg_func, reduce_func)

        h = g.ndata["h"]
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        g.ndata["h"] = h

        # SumGNN-style per-layer repr accumulation
        if self.is_input_layer:
            g.ndata["repr"] = h.unsqueeze(1)                                        # [N, 1, out]
        else:
            g.ndata["repr"] = torch.cat([g.ndata["repr"], h.unsqueeze(1)], dim=1)  # [N, l+1, out]


class KGEncoder(nn.Module):
    """DGL-based R-GCN encoder for KG enclosing subgraphs.

    Follows SumGNN's graph_classifier pattern:
        repr accumulates one slice per layer  → [N, num_layers, hidden_dim]
        h_A    = repr[drug_a node].reshape(-1)   → [num_layers * hidden_dim]
        h_B    = repr[drug_b node].reshape(-1)   → [num_layers * hidden_dim]
        h_GSub = mean(repr, dim=0).reshape(-1)   → [num_layers * hidden_dim]
        H_KG   = cat([h_A, h_B, h_GSub])        → [3 * num_layers * hidden_dim]

    With default num_layers=2, hidden_dim=64: H_KG shape = [384].
    """

    def __init__(
        self,
        inp_dim: int = 2,
        hidden_dim: int = 64,
        num_rels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            RGCNLayer(inp_dim, hidden_dim, num_rels, F.relu, dropout, is_input_layer=True)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_rels, F.relu, dropout)
            )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Encode a subgraph into H_KG.

        Args:
            g: DGL graph with ndata['feat'] and ndata['id'] (1=drug_a, 2=drug_b).

        Returns:
            H_KG: tensor of shape [3 * num_layers * hidden_dim].
        """
        g.ndata["h"] = g.ndata["feat"]

        for layer in self.layers:
            layer(g)

        repr = g.ndata["repr"]  # [N, num_layers, hidden_dim]

        head_ids = (g.ndata["id"] == 1).nonzero(as_tuple=False).squeeze(1)
        tail_ids = (g.ndata["id"] == 2).nonzero(as_tuple=False).squeeze(1)

        h_A    = repr[head_ids].reshape(-1)      # [num_layers * hidden_dim]
        h_B    = repr[tail_ids].reshape(-1)      # [num_layers * hidden_dim]
        h_GSub = repr.mean(dim=0).reshape(-1)    # [num_layers * hidden_dim]

        return torch.cat([h_A, h_B, h_GSub])     # [3 * num_layers * hidden_dim]


def main():
    """Sanity check with a dummy 4-node subgraph (2 drugs sharing 1 gene + 1 extra gene)."""
    hidden_dim = 64

    # 2 drugs (idx 0,1) share gene idx 2; gene idx 3 is only reachable from drug 0
    src = torch.tensor([0, 1, 0])
    dst = torch.tensor([2, 2, 3])
    g = dgl.graph((src, dst))

    g.ndata["feat"] = torch.tensor([[1,0],[1,0],[0,1],[0,1]], dtype=torch.float)
    g.ndata["id"]   = torch.tensor([1, 2, 0, 0], dtype=torch.float)
    g.edata["type"] = torch.zeros(g.num_edges(), dtype=torch.long)

    model = KGEncoder(inp_dim=2, hidden_dim=hidden_dim)
    model.eval()

    with torch.no_grad():
        H_KG = model(g)

    print(f"H_KG shape: {H_KG.shape}")
    assert H_KG.shape == (3 * 2 * hidden_dim,), f"Expected ({3*2*hidden_dim},), got {H_KG.shape}"
    print("Assertion passed.")


if __name__ == "__main__":
    main()
