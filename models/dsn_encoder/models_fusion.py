import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                )
from torch_geometric.nn.conv import MessagePassing

from layers import (
                    CoAttentionLayer,
                    RESCAL,
                    IntraGraphAttention,
                    InterGraphAttention,
                    )
import time


FUSION_MODES = ("none", "concat", "mlp", "cross_attention")
DEFAULT_SUMGNN_DIR = os.environ.get(
    "SUMGNN_DIR",
    "/workspace/sum-gnn",
)


def _load_python_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

#sumgnn encder 추가 : extract_subgraph.py + kg_encoder.py --> H_KG 
## 05.31 - 모듈로 불러오는 방식으로 변경 수정
class SumGNNSubgraphEncoder(nn.Module):
    """Use sum-gnn/extract_subgraph.py and kg_encoder.py as the KG branch."""

    def __init__(
        self,
        sumgnn_dir=None,
        k=1,
        emb_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.sumgnn_dir = sumgnn_dir or DEFAULT_SUMGNN_DIR
        self.k = k
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = 3 * num_layers * hidden_dim
        self._loaded = False
        self._graph = None
        self._emb_matrix = None
        self._e2id = None
        self._extract_mod = None
        self._load_error = None
        self._encoder_state_dict = None

        self.encoder = self._build_encoder()

    def __getstate__(self):
        state = self.__dict__.copy()
        encoder = self._modules.get("encoder")
        if encoder is not None:
            state["_encoder_state_dict"] = {
                key: value.detach().cpu()
                for key, value in encoder.state_dict().items()
            }

        modules = state["_modules"].copy()
        modules.pop("encoder", None)
        state["_modules"] = modules
        state["_loaded"] = False
        state["_graph"] = None
        state["_emb_matrix"] = None
        state["_e2id"] = None
        state["_extract_mod"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        env_sumgnn_dir = os.environ.get("SUMGNN_DIR")
        if env_sumgnn_dir and not os.path.exists(os.path.join(self.sumgnn_dir, "kg_encoder.py")):
            self.sumgnn_dir = env_sumgnn_dir
        encoder_state = self.__dict__.pop("_encoder_state_dict", None)
        self.encoder = self._build_encoder()
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state)

    def _build_encoder(self):
        kg_encoder_mod = _load_python_module(
            "sumgnn_kg_encoder_runtime",
            os.path.join(self.sumgnn_dir, "kg_encoder.py"),
        )
        return kg_encoder_mod.KGEncoder(
            emb_dim=self.emb_dim,
            k=self.k,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def _load(self):
        if self._loaded:
            return
        self._loaded = True

        self._extract_mod = _load_python_module(
            "sumgnn_extract_subgraph_runtime",
            os.path.join(self.sumgnn_dir, "extract_subgraph.py"),
        )

        graph_candidates = [
            os.path.join(self.sumgnn_dir, "kg_graph_py37.gpickle"),
            os.path.join(self.sumgnn_dir, "kg_graph.gpickle"),
        ]
        for graph_path in graph_candidates:
            if not os.path.exists(graph_path):
                continue
            try:
                with open(graph_path, "rb") as f:
                    self._graph = pickle.load(f)
                break
            except Exception as exc:
                self._load_error = str(exc)

        emb_path = os.path.join(self.sumgnn_dir, "pretrained", "transe_entity_emb.npy")
        e2id_path = os.path.join(self.sumgnn_dir, "pretrained", "transe_entity2id.json")
        if os.path.exists(emb_path) and os.path.exists(e2id_path):
            try:
                self._emb_matrix = np.load(emb_path)
                with open(e2id_path) as f:
                    self._e2id = json.load(f)
            except Exception as exc:
                self._load_error = str(exc)
    # pair_ids는 (drug_a_id, drug_b_id) 형태의 리스트            
    def forward(self, pair_ids, batch_size, device, dtype):
        self._load()

        kg_repr = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        reasoning_paths = []
        valid_graphs = []
        valid_positions = []

        for idx, pair in enumerate(pair_ids or []):
            drug_a, drug_b = str(pair[0]), str(pair[1])
            if drug_a == drug_b:
                reasoning_paths.append(
                    self._reasoning_item(drug_a, drug_b, [], None, fallback_reason="self_pair")
                )
                continue

            result = self._extract_subgraph(drug_a, drug_b)
            if result is None:
                reasoning_paths.append(self._reasoning_item(drug_a, drug_b, [], None))
                continue

            g, node_list = result
            if not self._has_required_anchors(g):
                reasoning_paths.append(
                    self._reasoning_item(
                        drug_a, drug_b, node_list, g, fallback_reason="missing_anchor"
                    )
                )
                continue

            valid_graphs.append(g)
            valid_positions.append(idx)
            reasoning_paths.append(self._reasoning_item(drug_a, drug_b, node_list, g))

        if valid_graphs:
            import dgl

            # The installed DGL may be CPU-only even when PyTorch uses CUDA.
            # Keep the KG branch on CPU, then move H_KG to the DSN device.
            kg_device = torch.device("cpu")
            self.encoder.to(kg_device)
            batched_graph = dgl.batch(valid_graphs).to(kg_device)
            encoded = self.encoder(batched_graph).to(device=device, dtype=dtype)
            if encoded.dim() == 1:
                encoded = encoded.unsqueeze(0)
            kg_repr[torch.tensor(valid_positions, device=device)] = encoded

        return kg_repr, kg_repr.unsqueeze(1), reasoning_paths

    def _extract_subgraph(self, drug_a, drug_b):
        if self._graph is None or drug_a not in self._graph or drug_b not in self._graph:
            return None
        return self._extract_mod.extract_enclosing_subgraph(
            self._graph,
            drug_a,
            drug_b,
            k=self.k,
            emb_matrix=self._emb_matrix,
            e2id=self._e2id,
        )

    def _has_required_anchors(self, graph):
        id_values = graph.ndata["id"].tolist()
        return id_values.count(1) == 1 and id_values.count(2) == 1

    def _reasoning_item(self, drug_a, drug_b, node_list, graph, fallback_reason=None):
        gene_nodes = []
        if graph is not None:
            gene_nodes = [
                node for node in node_list
                if self._graph.nodes[node].get("type") == "gene"
            ]

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "num_paths": len(gene_nodes),
            "subgraph_nodes": len(node_list),
            "subgraph_edges": int(graph.num_edges()) if graph is not None else 0,
            "paths": [
                {
                    "drug_a": drug_a,
                    "relation_a": "targets",
                    "gene": gene,
                    "relation_b": "targets",
                    "drug_b": drug_b,
                }
                for gene in gene_nodes
            ],
            "source": "sum-gnn/extract_subgraph+KGEncoder",
            "fallback_reason": fallback_reason,
            "load_error": self._load_error,
        }


class DecoderFusion(nn.Module):
    def __init__(self, fusion_mode, dsn_dim, sumgnn_dim, cross_attention_heads=4):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError("fusion_mode must be one of {}".format(FUSION_MODES))

        self.fusion_mode = fusion_mode
        self.dsn_dim = dsn_dim
        self.sumgnn_dim = sumgnn_dim
        self.out_dim = dsn_dim

        if fusion_mode == "concat":
            self.out_dim = dsn_dim + sumgnn_dim
        elif fusion_mode == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(dsn_dim + sumgnn_dim, dsn_dim),
                nn.ReLU(),
                nn.Linear(dsn_dim, dsn_dim),
            )
        elif fusion_mode == "cross_attention":
            n_heads = max(1, min(cross_attention_heads, dsn_dim))
            while dsn_dim % n_heads != 0:
                n_heads -= 1
            self.sumgnn_proj = nn.Linear(sumgnn_dim, dsn_dim)
            self.cross_attention = nn.MultiheadAttention(dsn_dim, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(dsn_dim)

    def forward(self, dsn_repr, sumgnn_repr):
        if self.fusion_mode == "none" or sumgnn_repr is None:
            return dsn_repr

        sumgnn_repr = sumgnn_repr.to(device=dsn_repr.device, dtype=dsn_repr.dtype)
        if self.fusion_mode == "cross_attention":
            if sumgnn_repr.dim() == 2:
                sumgnn_repr = sumgnn_repr.unsqueeze(1)
            key_value = self.sumgnn_proj(sumgnn_repr)
            attended, _ = self.cross_attention(dsn_repr, key_value, key_value)
            return self.norm(dsn_repr + attended)

        if sumgnn_repr.dim() == 3:
            sumgnn_repr = sumgnn_repr.mean(dim=1)
        expanded = sumgnn_repr.unsqueeze(1).expand(-1, dsn_repr.size(1), -1)
        fused = torch.cat([dsn_repr, expanded], dim=-1)

        if self.fusion_mode == "concat":
            return fused
        return self.proj(fused)


def _patch_message_passing_compat(module):
    if not isinstance(module, MessagePassing):
        return
    if not hasattr(module, '_explain'):
        module._explain = False
    if not hasattr(module, '_edge_mask'):
        module._edge_mask = None
    if not hasattr(module, '_loop_mask'):
        module._loop_mask = None
    if not hasattr(module, '_apply_sigmoid'):
        module._apply_sigmoid = True
    if not hasattr(module, 'decomposed_layers'):
        module.decomposed_layers = 1


class MVN_DDI(nn.Module):
    def __init__(
        self,
        in_features,
        hidd_dim,
        kge_dim,
        rel_total,
        heads_out_feat_params,
        blocks_params,
        fusion_mode="none",
        sumgnn_dir=None,
        sumgnn_dim=384,
        sumgnn_max_paths=16,
        cross_attention_heads=4,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        self.fusion_mode = fusion_mode
        self.sumgnn_dir = sumgnn_dir or DEFAULT_SUMGNN_DIR
        self.sumgnn_max_paths = sumgnn_max_paths
        self.sumgnn_encoder = None
        if fusion_mode != "none":
            self.sumgnn_encoder = SumGNNSubgraphEncoder(sumgnn_dir=self.sumgnn_dir)
            sumgnn_dim = self.sumgnn_encoder.output_dim
        self.sumgnn_dim = sumgnn_dim

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.decoder_fusion = DecoderFusion(
            fusion_mode,
            self.kge_dim,
            self.sumgnn_dim,
            cross_attention_heads=cross_attention_heads,
        )
        decoder_dim = self.decoder_fusion.out_dim
        self.co_attention = CoAttentionLayer(decoder_dim)
        self.KGE = RESCAL(self.rel_total, decoder_dim)
        self._compat_patched = False

    def _ensure_pyg_compat(self):
        if not hasattr(self, '_compat_patched'):
            self._compat_patched = False
        if self._compat_patched:
            return
        for module in self.modules():
            _patch_message_passing_compat(module)
        self._compat_patched = True

    def _ensure_sumgnn_encoder(self):
        if self.sumgnn_encoder is None:
            self.sumgnn_encoder = SumGNNSubgraphEncoder(sumgnn_dir=self.sumgnn_dir)
            self.sumgnn_dim = self.sumgnn_encoder.output_dim
        return self.sumgnn_encoder

    def _get_sumgnn_features(self, pair_ids, batch_size, device, dtype):
        if pair_ids is None:
            zero = torch.zeros(batch_size, self.sumgnn_dim, device=device, dtype=dtype)
            tokens = zero.unsqueeze(1)
            return zero, tokens, []

        sumgnn_encoder = self._ensure_sumgnn_encoder()
        return sumgnn_encoder(pair_ids, batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, triples, return_reasoning=False):
        self._ensure_pyg_compat()
        if len(triples) == 5:
            h_data, t_data, rels, b_graph, pair_ids = triples
        else:
            h_data, t_data, rels, b_graph = triples
            pair_ids = None

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        # print(kge_heads.size(), kge_tails.size(), rels.size())
        reasoning_paths = []
        if self.fusion_mode != "none" or return_reasoning:
            sumgnn_pooled, sumgnn_tokens, reasoning_paths = self._get_sumgnn_features(
                pair_ids,
                batch_size=repr_h.size(0),
                device=repr_h.device,
                dtype=repr_h.dtype,
            )
            sumgnn_repr = sumgnn_tokens if self.fusion_mode == "cross_attention" else sumgnn_pooled
            kge_heads = self.decoder_fusion(kge_heads, sumgnn_repr)
            kge_tails = self.decoder_fusion(kge_tails, sumgnn_repr)

        # Fusion modes already combine DSN and Sum-GNN representations before RESCAL.
        # Skip the original co-attention there so the decoder sees the fused features directly.
        attentions = self.co_attention(kge_heads, kge_tails) if self.fusion_mode == "none" else None
        # attentions = None
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        if return_reasoning:
            return {
                "scores": scores,
                "reasoning_paths": reasoning_paths,
                "fusion_mode": self.fusion_mode,
            }
        return scores


#intra+inter
class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, h_data, t_data, b_graph):

        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep


        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x, t_data.edge_index, batch=t_data.batch)

        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)


        return h_data, t_data, h_global_graph_emb, t_global_graph_emb


CSS_DDI = MVN_DDI
