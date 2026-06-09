import importlib.util
import csv
import io
import json
import os
import pickle
import sys
import zipfile

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


class _HKGStorageRef:
    def __init__(self, storage_type, key, location, size):
        self.storage_type = storage_type
        self.key = str(key)
        self.location = location
        self.size = int(size)


class _HKGTensorSpec:
    def __init__(self, storage, storage_offset, size, stride):
        self.storage = storage
        self.storage_offset = int(storage_offset)
        self.size = tuple(int(value) for value in size)
        self.stride = tuple(int(value) for value in stride)


def _safe_rebuild_hkg_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return _HKGTensorSpec(storage, storage_offset, size, stride)


class _SafeHKGUnpickler(pickle.Unpickler):
    _ALLOWED_STORAGE_TYPES = {
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
        "HalfStorage": np.float16,
    }

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _safe_rebuild_hkg_tensor
        if module == "torch" and name in self._ALLOWED_STORAGE_TYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        raise pickle.UnpicklingError(f"Unsupported global in HKG cache: {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id in HKG cache: {pid!r}")
        _, storage_type, key, location, size = pid[:5]
        if storage_type not in self._ALLOWED_STORAGE_TYPES:
            raise pickle.UnpicklingError(f"Unsupported storage type in HKG cache: {storage_type}")
        return _HKGStorageRef(storage_type, key, location, size)


FUSION_MODES = (
    "none",
    "concat",
    "mlp",
    "cross_attention",
    "one_direct_cross_attention",
    "bidirect_cross_attention",
)
DECODER_MODES = (
    "original_decoder",
    "binary_no_coattn_no_rescal",
    "coattn_binary_no_rescal",
)


def _find_default_subxddi_kg_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = []
    cursor = here
    for _ in range(6):
        candidates.append(os.path.join(cursor, "subxddi-kg"))
        cursor = os.path.dirname(cursor)
    candidates.append("/workspace/subxddi-kg")

    for path in candidates:
        if os.path.exists(os.path.join(path, "kg_encoder.py")):
            return path
    return candidates[-1]


DEFAULT_SUBXDDI_KG_DIR = os.environ.get("SUBXDDI_KG_DIR", _find_default_subxddi_kg_dir())
# Backward-compatible alias for old scripts/configs.
DEFAULT_SUMGNN_DIR = DEFAULT_SUBXDDI_KG_DIR


def _load_python_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class SubXDDIKGSubgraphEncoder(nn.Module):
    """Use subxddi-kg/extract_subgraph.py and kg_encoder.py as the KG branch."""

    def __init__(
        self,
        subxddi_kg_dir=None,
        sumgnn_dir=None,
        k=1,
        emb_dim=None,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        max_paths=None,
        hkg_cache_dir=None,
        hkg_cache_path=None,
        prefer_hkg_cache=True,
    ):
        super().__init__()
        self.subxddi_kg_dir = subxddi_kg_dir or sumgnn_dir or DEFAULT_SUBXDDI_KG_DIR
        # Kept for checkpoints and older helper functions that still set sumgnn_dir.
        self.sumgnn_dir = self.subxddi_kg_dir
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_paths = max_paths
        self.hkg_cache_dir = hkg_cache_dir
        self.hkg_cache_path = hkg_cache_path
        self.prefer_hkg_cache = prefer_hkg_cache
        self.output_dim = 3 * num_layers * hidden_dim
        self._loaded = False
        self._graph = None
        self._emb_matrix = None
        self._e2id = None
        self._extract_mod = None
        self._hkg_cache = None
        self._hkg_cache_loaded = False
        self._hkg_cache_path = None
        self._load_error = None
        self._encoder_state_dict = None
        self.emb_dim = self._resolve_emb_dim(emb_dim)

        self.encoder = self._build_encoder()
        self.path_token_proj = self._build_path_token_proj()

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
        state["_hkg_cache"] = None
        state["_hkg_cache_loaded"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "subxddi_kg_dir"):
            self.subxddi_kg_dir = getattr(self, "sumgnn_dir", DEFAULT_SUBXDDI_KG_DIR)
        if not hasattr(self, "max_paths"):
            self.max_paths = None
        if not hasattr(self, "hkg_cache_dir"):
            self.hkg_cache_dir = None
        if not hasattr(self, "hkg_cache_path"):
            self.hkg_cache_path = None
        if not hasattr(self, "prefer_hkg_cache"):
            self.prefer_hkg_cache = True
        if not hasattr(self, "_hkg_cache"):
            self._hkg_cache = None
        if not hasattr(self, "_hkg_cache_loaded"):
            self._hkg_cache_loaded = False
        if not hasattr(self, "_hkg_cache_path"):
            self._hkg_cache_path = None
        self.sumgnn_dir = self.subxddi_kg_dir
        env_subxddi_kg_dir = os.environ.get("SUBXDDI_KG_DIR")
        if env_subxddi_kg_dir and not os.path.exists(os.path.join(self._kg_dir(), "kg_encoder.py")):
            self.subxddi_kg_dir = env_subxddi_kg_dir
            self.sumgnn_dir = env_subxddi_kg_dir
        encoder_state = self.__dict__.pop("_encoder_state_dict", None)
        self.encoder = self._build_encoder()
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state)
        if not hasattr(self, "path_token_proj"):
            self.path_token_proj = self._build_path_token_proj()

    def _kg_dir(self):
        subxddi_kg_dir = getattr(self, "subxddi_kg_dir", None)
        legacy_dir = getattr(self, "sumgnn_dir", None)
        if legacy_dir and legacy_dir != subxddi_kg_dir and subxddi_kg_dir in (None, DEFAULT_SUBXDDI_KG_DIR):
            subxddi_kg_dir = legacy_dir
        if not subxddi_kg_dir:
            subxddi_kg_dir = DEFAULT_SUBXDDI_KG_DIR
        self.subxddi_kg_dir = subxddi_kg_dir
        self.sumgnn_dir = subxddi_kg_dir
        return subxddi_kg_dir

    def _resolve_emb_dim(self, emb_dim):
        if emb_dim is not None:
            return emb_dim
        emb_path = os.path.join(self._kg_dir(), "pretrained", "transe_entity_emb.npy")
        if not os.path.exists(emb_path):
            return 0
        try:
            return int(np.load(emb_path, mmap_mode="r").shape[1])
        except Exception as exc:
            self._load_error = str(exc)
            return 0

    def _build_encoder(self):
        kg_dir = self._kg_dir()
        kg_encoder_mod = _load_python_module(
            "subxddi_kg_encoder_runtime",
            os.path.join(kg_dir, "kg_encoder.py"),
        )
        return kg_encoder_mod.KGEncoder(
            emb_dim=self.emb_dim,
            k=self.k,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def _build_path_token_proj(self):
        feat_dim = int(self.emb_dim) + 2 * (int(self.k) + 1)
        # Encoded path summary [h_A || h_gene || h_B] plus raw TransE/position
        # features [x_A || x_gene || x_B]. This keeps path tokens distinct even
        # when a pair-level H_KG cache is available.
        input_dim = self.output_dim + 3 * feat_dim
        return nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.LayerNorm(self.output_dim),
        )

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        kg_dir = self._kg_dir()

        self._extract_mod = _load_python_module(
            "subxddi_extract_subgraph_runtime",
            os.path.join(kg_dir, "extract_subgraph.py"),
        )

        graph_candidates = [
            os.path.join(kg_dir, "graph", "kg_graph.gpickle"),
            os.path.join(kg_dir, "kg_graph.gpickle"),
            os.path.join(kg_dir, "kg_graph_py37.gpickle"),
        ]
        for graph_path in graph_candidates:
            if not os.path.exists(graph_path):
                continue
            try:
                with open(graph_path, "rb") as f:
                    self._graph = pickle.load(f)
                self._load_error = None
                break
            except Exception as exc:
                self._load_error = str(exc)

        if self._graph is None:
            self._graph = self._build_graph_from_triples(kg_dir)
            if self._graph is not None and self._load_error:
                self._load_error = None

        emb_path = os.path.join(kg_dir, "pretrained", "transe_entity_emb.npy")
        e2id_path = os.path.join(kg_dir, "pretrained", "transe_entity2id.json")
        if os.path.exists(emb_path) and os.path.exists(e2id_path):
            try:
                self._emb_matrix = np.load(emb_path)
                with open(e2id_path) as f:
                    self._e2id = json.load(f)
            except Exception as exc:
                self._load_error = str(exc)
        self._load_hkg_cache()

    def _resolve_hkg_cache_path(self):
        if self.hkg_cache_path:
            return self.hkg_cache_path
        cache_dir = self.hkg_cache_dir or os.path.join(self._kg_dir(), "Hkg_encoding")
        if int(self.k) == 1:
            candidates = ["hkg_cache_k1.pt", "Hkg_1.pt", "hkg_cache.pt"]
        else:
            candidates = [
                f"hkg_cache_k{int(self.k)}.pt",
                f"Hkg_{int(self.k)}.pt",
                f"hkg_cache{int(self.k)}.pt",
            ]
        for filename in candidates:
            path = os.path.join(cache_dir, filename)
            if os.path.exists(path):
                return path
        return None

    def _load_hkg_cache(self):
        if self._hkg_cache_loaded:
            return
        self._hkg_cache_loaded = True
        cache_path = self._resolve_hkg_cache_path()
        self._hkg_cache_path = cache_path
        if not cache_path:
            return
        try:
            self._hkg_cache = self._safe_load_hkg_cache(cache_path)
            print(
                f"[HKG_CACHE] loaded k={self.k} entries={len(self._hkg_cache)} path={cache_path}",
                flush=True,
            )
        except Exception as exc:
            self._hkg_cache = None
            self._load_error = f"HKG cache load failed: {exc}"

    def _safe_load_hkg_cache(self, cache_path):
        with zipfile.ZipFile(cache_path) as zf:
            names = zf.namelist()
            data_pkl_name = next(name for name in names if name.endswith("/data.pkl"))
            prefix = data_pkl_name.rsplit("/", 1)[0]
            payload = zf.read(data_pkl_name)
            raw = _SafeHKGUnpickler(io.BytesIO(payload)).load()
            if not isinstance(raw, dict):
                raise ValueError("HKG cache root must be a dict")

            storage_cache = {}

            def tensor_from_spec(spec):
                if not isinstance(spec, _HKGTensorSpec):
                    return spec
                storage = spec.storage
                if storage.key not in storage_cache:
                    member = f"{prefix}/data/{storage.key}"
                    dtype = _SafeHKGUnpickler._ALLOWED_STORAGE_TYPES[storage.storage_type]
                    array = np.frombuffer(zf.read(member), dtype=dtype).copy()
                    storage_cache[storage.key] = torch.from_numpy(array)
                base = storage_cache[storage.key]
                return torch.as_strided(
                    base,
                    size=spec.size,
                    stride=spec.stride,
                    storage_offset=spec.storage_offset,
                ).clone()

            return {key: tensor_from_spec(value) for key, value in raw.items()}

    def _lookup_hkg_cache(self, drug_a, drug_b, device, dtype):
        if not isinstance(self._hkg_cache, dict):
            return None
        candidates = [
            (drug_a, drug_b),
            (drug_b, drug_a),
            f"{drug_a}|{drug_b}",
            f"{drug_b}|{drug_a}",
            f"{drug_a},{drug_b}",
            f"{drug_b},{drug_a}",
            f"{drug_a}_{drug_b}",
            f"{drug_b}_{drug_a}",
        ]
        value = None
        for key in candidates:
            if key in self._hkg_cache:
                value = self._hkg_cache[key]
                break
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ("hkg", "H_KG", "Hkg", "embedding", "repr", "tensor"):
                if key in value:
                    value = value[key]
                    break
        try:
            tensor = torch.as_tensor(value).to(device=device, dtype=dtype)
        except Exception:
            return None
        if tensor.numel() != self.output_dim:
            return None
        return tensor.reshape(self.output_dim)

    def _build_graph_from_triples(self, kg_dir):
        triples_candidates = [
            os.path.join(kg_dir, "data", "kg_triples.tsv"),
            os.path.join(kg_dir, "kg_triples.tsv"),
        ]
        for triples_path in triples_candidates:
            if not os.path.exists(triples_path):
                continue
            try:
                import networkx as nx

                graph = nx.DiGraph()
                with open(triples_path, newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        head = row.get("head") or row.get("Drug")
                        tail = row.get("tail") or row.get("Gene")
                        relation = row.get("relation") or "targets"
                        if not head or not tail:
                            continue
                        graph.add_node(head, type="drug")
                        graph.add_node(tail, type="gene")
                        graph.add_edge(head, tail, relation=relation)
                if graph.number_of_nodes() > 0:
                    return graph
            except Exception as exc:
                self._load_error = str(exc)
        return None
    # pair_ids는 (drug_a_id, drug_b_id) 형태의 리스트            
    def forward(self, pair_ids, batch_size, device, dtype):
        self._load()

        kg_repr = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        reasoning_paths = []
        valid_graphs = []
        valid_positions = []
        valid_path_node_indices = []
        path_token_rows = [[] for _ in range(batch_size)]

        for idx, pair in enumerate(pair_ids or []):
            drug_a, drug_b = str(pair[0]), str(pair[1])
            cached_hkg = self._lookup_hkg_cache(drug_a, drug_b, device, dtype)
            if drug_a == drug_b:
                item = self._reasoning_item(drug_a, drug_b, [], None, fallback_reason="self_pair")
                if cached_hkg is not None:
                    kg_repr[idx] = cached_hkg
                    item["hkg_cache_hit"] = True
                    item["hkg_cache_path"] = self._hkg_cache_path
                reasoning_paths.append(item)
                continue

            result = self._extract_subgraph(drug_a, drug_b)
            if result is None:
                item = self._reasoning_item(drug_a, drug_b, [], None)
                if cached_hkg is not None:
                    kg_repr[idx] = cached_hkg
                    item["hkg_cache_hit"] = True
                    item["hkg_cache_path"] = self._hkg_cache_path
                reasoning_paths.append(item)
                continue

            g, node_list = result
            if not self._has_required_anchors(g):
                item = self._reasoning_item(
                    drug_a, drug_b, node_list, g, fallback_reason="missing_anchor"
                )
                if cached_hkg is not None:
                    kg_repr[idx] = cached_hkg
                    item["hkg_cache_hit"] = True
                    item["hkg_cache_path"] = self._hkg_cache_path
                    path_token_rows[idx] = [cached_hkg.clone() for _ in item.get("paths", [])]
                reasoning_paths.append(item)
                continue

            item = self._reasoning_item(drug_a, drug_b, node_list, g)
            if cached_hkg is not None:
                kg_repr[idx] = cached_hkg
                item["hkg_cache_hit"] = True
                item["hkg_cache_path"] = self._hkg_cache_path
                if getattr(self, "prefer_hkg_cache", True):
                    item["hkg_cache_used_as_primary"] = True
                    self._fill_cached_path_tokens(
                        path_token_rows,
                        idx,
                        cached_hkg,
                        g,
                        node_list,
                        item,
                        device,
                        dtype,
                    )
                    reasoning_paths.append(item)
                    continue

            valid_graphs.append(g)
            valid_positions.append(idx)
            valid_path_node_indices.append(self._path_node_indices(node_list, item))
            reasoning_paths.append(item)

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
            self._fill_path_tokens(
                path_token_rows,
                valid_positions,
                valid_path_node_indices,
                batched_graph,
                device,
                dtype,
            )

        path_tokens, path_padding_mask = self._pad_path_tokens(kg_repr, path_token_rows)
        return kg_repr, path_tokens, path_padding_mask, reasoning_paths

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

    def _path_node_indices(self, node_list, reasoning_item):
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        return [
            node_to_idx[path["gene"]]
            for path in reasoning_item.get("paths", [])
            if path.get("gene") in node_to_idx
        ]

    def _fill_path_tokens(
        self,
        path_token_rows,
        valid_positions,
        valid_path_node_indices,
        batched_graph,
        device,
        dtype,
    ):
        repr_all = batched_graph.ndata["repr"]
        feat_all = batched_graph.ndata["feat"]
        id_all = batched_graph.ndata["id"]
        offset = 0
        for batch_pos, node_count, gene_indices in zip(
            valid_positions,
            batched_graph.batch_num_nodes().tolist(),
            valid_path_node_indices,
        ):
            graph_repr = repr_all[offset : offset + node_count]
            graph_feat = feat_all[offset : offset + node_count]
            graph_ids = id_all[offset : offset + node_count]
            offset += node_count

            head_ids = (graph_ids == 1).nonzero(as_tuple=False).squeeze(1)
            tail_ids = (graph_ids == 2).nonzero(as_tuple=False).squeeze(1)
            if head_ids.numel() != 1 or tail_ids.numel() != 1:
                continue

            h_a = graph_repr[head_ids[0]].reshape(-1)
            h_b = graph_repr[tail_ids[0]].reshape(-1)
            raw_a = graph_feat[head_ids[0]].reshape(-1)
            raw_b = graph_feat[tail_ids[0]].reshape(-1)
            proj_device = next(self.path_token_proj.parameters()).device
            tokens = []
            for gene_idx in gene_indices:
                if gene_idx >= node_count:
                    continue
                h_gene = self.encoder.w_sub(graph_repr[gene_idx]).reshape(-1)
                raw_gene = graph_feat[gene_idx].reshape(-1)
                path_input = torch.cat(
                    [h_a, h_gene, h_b, raw_a, raw_gene, raw_b],
                    dim=0,
                ).to(device=proj_device, dtype=dtype)
                token = self.path_token_proj(path_input).to(device=device, dtype=dtype)
                tokens.append(token)
            path_token_rows[batch_pos] = tokens

    def _fill_cached_path_tokens(
        self,
        path_token_rows,
        batch_pos,
        cached_hkg,
        graph,
        node_list,
        reasoning_item,
        device,
        dtype,
    ):
        gene_indices = self._path_node_indices(node_list, reasoning_item)
        if not gene_indices:
            return

        feat_all = graph.ndata["feat"]
        id_all = graph.ndata["id"]
        head_ids = (id_all == 1).nonzero(as_tuple=False).squeeze(1)
        tail_ids = (id_all == 2).nonzero(as_tuple=False).squeeze(1)
        if head_ids.numel() != 1 or tail_ids.numel() != 1:
            return

        block_dim = self.num_layers * self.hidden_dim
        cached = cached_hkg.detach().to(device=feat_all.device, dtype=feat_all.dtype)
        h_a = cached[:block_dim].reshape(-1)
        h_b = cached[block_dim : 2 * block_dim].reshape(-1)
        h_sub = cached[2 * block_dim : 3 * block_dim].reshape(-1)
        raw_a = feat_all[head_ids[0]].reshape(-1)
        raw_b = feat_all[tail_ids[0]].reshape(-1)
        proj_device = next(self.path_token_proj.parameters()).device
        tokens = []
        for gene_idx in gene_indices:
            if gene_idx >= graph.num_nodes():
                continue
            raw_gene = feat_all[gene_idx].reshape(-1)
            path_input = torch.cat(
                [h_a, h_sub, h_b, raw_a, raw_gene, raw_b],
                dim=0,
            ).to(device=proj_device, dtype=dtype)
            token = self.path_token_proj(path_input).to(device=device, dtype=dtype)
            tokens.append(token)
        path_token_rows[batch_pos] = tokens

    def _pad_path_tokens(self, kg_repr, path_token_rows):
        max_path_count = max([len(row) for row in path_token_rows] + [1])
        path_tokens = kg_repr.new_zeros((kg_repr.size(0), max_path_count, self.output_dim))
        path_padding_mask = torch.ones(
            (kg_repr.size(0), max_path_count),
            device=kg_repr.device,
            dtype=torch.bool,
        )

        for idx, row in enumerate(path_token_rows):
            if row:
                row_tensor = torch.stack(row).to(device=kg_repr.device, dtype=kg_repr.dtype)
                path_tokens[idx, : row_tensor.size(0)] = row_tensor
                path_padding_mask[idx, : row_tensor.size(0)] = False
            else:
                path_tokens[idx, 0] = kg_repr[idx]
                path_padding_mask[idx, 0] = False

        return path_tokens, path_padding_mask

    def _reasoning_item(self, drug_a, drug_b, node_list, graph, fallback_reason=None):
        gene_nodes = []
        if graph is not None:
            gene_nodes = [
                node for node in node_list
                if self._graph.nodes[node].get("type") == "gene"
            ]
        total_paths = len(gene_nodes)
        max_paths = getattr(self, "max_paths", None)
        if max_paths is not None and int(max_paths) > 0:
            gene_nodes = gene_nodes[: int(max_paths)]

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "num_paths": len(gene_nodes),
            "num_paths_total": total_paths,
            "paths_truncated": total_paths > len(gene_nodes),
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
            "source": "subxddi-kg/extract_subgraph+KGEncoder",
            "fallback_reason": fallback_reason,
            "load_error": self._load_error,
        }


SumGNNSubgraphEncoder = SubXDDIKGSubgraphEncoder


class DecoderFusion(nn.Module):
    def __init__(
        self,
        fusion_mode,
        dsn_dim,
        sumgnn_dim,
        cross_attention_heads=4,
        dropout=0.1,
        gated_bidirect=True,
    ):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError("fusion_mode must be one of {}".format(FUSION_MODES))

        self.fusion_mode = fusion_mode
        self.dsn_dim = dsn_dim
        self.sumgnn_dim = sumgnn_dim
        self.out_dim = dsn_dim
        self.dropout = nn.Dropout(dropout)
        self._warned_single_kg_token = False

        if fusion_mode == "concat":
            self.concat_proj = nn.Linear(dsn_dim + sumgnn_dim, dsn_dim)
        elif fusion_mode == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(dsn_dim + sumgnn_dim, dsn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dsn_dim, dsn_dim),
            )
            self.mlp_norm = nn.LayerNorm(dsn_dim)
        elif fusion_mode in ("cross_attention", "one_direct_cross_attention", "bidirect_cross_attention"):
            n_heads = max(1, min(cross_attention_heads, dsn_dim))
            while dsn_dim % n_heads != 0:
                n_heads -= 1
            self.sumgnn_proj = nn.Linear(sumgnn_dim, dsn_dim)
            self.cross_attention = nn.MultiheadAttention(
                dsn_dim, n_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(dsn_dim)
            if fusion_mode == "bidirect_cross_attention":
                self.kg_to_dsn_attention = nn.MultiheadAttention(
                    dsn_dim, n_heads, dropout=dropout, batch_first=True
                )
                self.kg_norm = nn.LayerNorm(dsn_dim)
                self.gated_bidirect = gated_bidirect
                self.gate = nn.Linear(dsn_dim * 2, dsn_dim)
                self.kg_pool_proj = nn.Linear(dsn_dim, dsn_dim)
                self.bidirect_norm = nn.LayerNorm(dsn_dim)

    def forward(
        self,
        dsn_repr,
        sumgnn_repr,
        return_attention=False,
        sumgnn_key_padding_mask=None,
    ):
        def maybe_with_attention(output, attention):
            if return_attention:
                return output, attention
            return output

        if self.fusion_mode == "none" or sumgnn_repr is None:
            return maybe_with_attention(dsn_repr, None)

        sumgnn_repr = sumgnn_repr.to(device=dsn_repr.device, dtype=dsn_repr.dtype)
        if sumgnn_key_padding_mask is not None:
            sumgnn_key_padding_mask = sumgnn_key_padding_mask.to(device=dsn_repr.device)
        if sumgnn_repr.dim() == 2:
            sumgnn_repr = sumgnn_repr.unsqueeze(1)

        if self.fusion_mode in ("cross_attention", "one_direct_cross_attention", "bidirect_cross_attention"):
            if sumgnn_repr.dim() == 2:
                sumgnn_repr = sumgnn_repr.unsqueeze(1)
            if sumgnn_repr.size(1) == 1 and not self._warned_single_kg_token:
                print(
                    "[WARN] DecoderFusion cross-attention received a single KG token; "
                    "attention source length is 1.",
                    flush=True,
                )
                self._warned_single_kg_token = True
            key_value = self.sumgnn_proj(sumgnn_repr)
            attended, dsn_to_kg_attention = self.cross_attention(
                dsn_repr,
                key_value,
                key_value,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=sumgnn_key_padding_mask,
            )
            dsn_out = self.norm(dsn_repr + self.dropout(attended))
            attention = {"dsn_to_kg": dsn_to_kg_attention}

            if self.fusion_mode != "bidirect_cross_attention":
                return maybe_with_attention(dsn_out, attention)

            kg_ctx, kg_to_dsn_attention = self.kg_to_dsn_attention(
                key_value,
                dsn_repr,
                dsn_repr,
                need_weights=True,
                average_attn_weights=False,
            )
            kg_out = self.kg_norm(key_value + self.dropout(kg_ctx))
            attention["kg_to_dsn"] = kg_to_dsn_attention
            if not self.gated_bidirect:
                return maybe_with_attention(dsn_out, attention)

            kg_pool = self._masked_token_mean(kg_out, sumgnn_key_padding_mask)
            kg_expand = kg_pool.unsqueeze(1).expand(-1, dsn_out.size(1), -1)
            gate = torch.sigmoid(self.gate(torch.cat([dsn_out, kg_expand], dim=-1)))
            kg_residual = self.kg_pool_proj(kg_expand)
            attention["gate"] = gate.detach()
            return maybe_with_attention(self.bidirect_norm(dsn_out + gate * kg_residual), attention)

        if sumgnn_repr.dim() == 3:
            sumgnn_repr = sumgnn_repr.mean(dim=1)
        expanded = sumgnn_repr.unsqueeze(1).expand(-1, dsn_repr.size(1), -1)
        fused = torch.cat([dsn_repr, expanded], dim=-1)

        if self.fusion_mode == "concat":
            return maybe_with_attention(self.concat_proj(fused), None)
        return maybe_with_attention(self.mlp_norm(dsn_repr + self.dropout(self.proj(fused))), None)

    def _masked_token_mean(self, tokens, padding_mask):
        if padding_mask is None:
            return tokens.mean(dim=1)
        keep = (~padding_mask).to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        denom = keep.sum(dim=1).clamp_min(1.0)
        return (tokens * keep).sum(dim=1) / denom


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
        subxddi_kg_dir=None,
        subxddi_kg_dim=None,
        subxddi_kg_max_paths=None,
        sumgnn_dir=None,
        sumgnn_dim=384,
        sumgnn_max_paths=16,
        cross_attention_heads=4,
        decoder="original_decoder",
        kg_k=1,
        fusion_dropout=0.1,
        hkg_cache_dir=None,
        hkg_cache_path=None,
        prefer_hkg_cache=True,
    ):
        super().__init__()
        if decoder not in DECODER_MODES:
            raise ValueError("decoder must be one of {}".format(DECODER_MODES))
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        self.fusion_mode = fusion_mode
        self.decoder = decoder
        self.kg_k = kg_k
        self.hkg_cache_dir = hkg_cache_dir
        self.hkg_cache_path = hkg_cache_path
        self.prefer_hkg_cache = prefer_hkg_cache
        self.subxddi_kg_dir = subxddi_kg_dir or sumgnn_dir or DEFAULT_SUBXDDI_KG_DIR
        self.sumgnn_dir = self.subxddi_kg_dir
        self.subxddi_kg_max_paths = (
            subxddi_kg_max_paths if subxddi_kg_max_paths is not None else sumgnn_max_paths
        )
        self.sumgnn_max_paths = self.subxddi_kg_max_paths
        self.sumgnn_encoder = None
        if subxddi_kg_dim is not None:
            sumgnn_dim = subxddi_kg_dim
        if fusion_mode != "none":
            self.sumgnn_encoder = SubXDDIKGSubgraphEncoder(
                subxddi_kg_dir=self.subxddi_kg_dir,
                k=self.kg_k,
                max_paths=self.subxddi_kg_max_paths,
                hkg_cache_dir=self.hkg_cache_dir,
                hkg_cache_path=self.hkg_cache_path,
                prefer_hkg_cache=self.prefer_hkg_cache,
            )
            sumgnn_dim = self.sumgnn_encoder.output_dim
        self.sumgnn_dim = sumgnn_dim
        self.subxddi_kg_dim = self.sumgnn_dim

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
            dropout=fusion_dropout,
        )
        print("decoder fusion out dim:", self.decoder_fusion.out_dim)
        decoder_dim = self.decoder_fusion.out_dim
        self.co_attention = CoAttentionLayer(decoder_dim)
        self.KGE = RESCAL(self.rel_total, decoder_dim)
        self.binary_relation = nn.Embedding(self.rel_total, decoder_dim)
        self.binary_classifier = nn.Sequential(
            nn.Linear(decoder_dim * 5, decoder_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(decoder_dim, 1),
        )
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
            self.subxddi_kg_dir = getattr(self, "subxddi_kg_dir", self.sumgnn_dir)
            self.sumgnn_encoder = SubXDDIKGSubgraphEncoder(
                subxddi_kg_dir=self.subxddi_kg_dir,
                k=getattr(self, "kg_k", 1),
                max_paths=getattr(self, "subxddi_kg_max_paths", None),
                hkg_cache_dir=getattr(self, "hkg_cache_dir", None),
                hkg_cache_path=getattr(self, "hkg_cache_path", None),
                prefer_hkg_cache=getattr(self, "prefer_hkg_cache", True),
            )
            self.sumgnn_dim = self.sumgnn_encoder.output_dim
            self.subxddi_kg_dim = self.sumgnn_dim
        return self.sumgnn_encoder

    def _get_sumgnn_features(self, pair_ids, batch_size, device, dtype):
        if pair_ids is None:
            zero = torch.zeros(batch_size, self.sumgnn_dim, device=device, dtype=dtype)
            tokens = zero.unsqueeze(1)
            padding_mask = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
            return zero, tokens, padding_mask, []

        sumgnn_encoder = self._ensure_sumgnn_encoder()
        return sumgnn_encoder(pair_ids, batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, triples, return_reasoning=False, return_attention=False):
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
        attention_info = {}
        if self.fusion_mode != "none" or return_reasoning:
            subxddi_kg_pooled, subxddi_kg_tokens, subxddi_kg_padding_mask, reasoning_paths = self._get_sumgnn_features(
                pair_ids,
                batch_size=repr_h.size(0),
                device=repr_h.device,
                dtype=repr_h.dtype,
            )
            attention_fusion = self.fusion_mode in (
                "cross_attention",
                "one_direct_cross_attention",
                "bidirect_cross_attention",
            )
            subxddi_kg_repr = (
                subxddi_kg_tokens if attention_fusion else subxddi_kg_pooled
            )
            subxddi_kg_mask = subxddi_kg_padding_mask if attention_fusion else None
            if return_attention:
                kge_heads, head_attention = self.decoder_fusion(
                    kge_heads,
                    subxddi_kg_repr,
                    return_attention=True,
                    sumgnn_key_padding_mask=subxddi_kg_mask,
                )
                kge_tails, tail_attention = self.decoder_fusion(
                    kge_tails,
                    subxddi_kg_repr,
                    return_attention=True,
                    sumgnn_key_padding_mask=subxddi_kg_mask,
                )
                attention_info["fusion_heads"] = head_attention
                attention_info["fusion_tails"] = tail_attention
                if subxddi_kg_mask is not None:
                    attention_info["kg_padding_mask"] = subxddi_kg_mask.detach()
            else:
                kge_heads = self.decoder_fusion(kge_heads, subxddi_kg_repr)
                kge_tails = self.decoder_fusion(kge_tails, subxddi_kg_repr)

        if self.decoder == "binary_no_coattn_no_rescal":
            scores = self._binary_scores(kge_heads, kge_tails, rels, use_coattn=False)
        elif self.decoder == "coattn_binary_no_rescal":
            scores = self._binary_scores(kge_heads, kge_tails, rels, use_coattn=True)
        else:
            attentions = self.co_attention(kge_heads, kge_tails)
            if return_attention:
                attention_info["co_attention"] = attentions.detach()
            scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        if return_reasoning or return_attention:
            if return_reasoning and return_attention:
                self._attach_path_attention(reasoning_paths, attention_info)
            result = {
                "scores": scores,
                "reasoning_paths": reasoning_paths,
                "fusion_mode": self.fusion_mode,
                "decoder": self.decoder,
            }
            if return_attention:
                result["attention"] = attention_info
            return result
        return scores

    def _attach_path_attention(self, reasoning_paths, attention_info):
        for label, key in (("fusion_head", "fusion_heads"), ("fusion_tail", "fusion_tails")):
            payload = attention_info.get(key) or {}
            dsn_to_kg = payload.get("dsn_to_kg")
            if dsn_to_kg is None:
                continue
            path_scores = dsn_to_kg.detach().mean(dim=(1, 2)).cpu()
            for item_idx, item in enumerate(reasoning_paths):
                paths = item.get("paths") or []
                if item_idx >= path_scores.size(0) or not paths:
                    continue
                values = [float(value) for value in path_scores[item_idx, : len(paths)].tolist()]
                item[f"{label}_path_attention"] = values
                for path, value in zip(paths, values):
                    path[f"{label}_attention"] = value

        for item in reasoning_paths:
            paths = item.get("paths") or []
            combined = []
            for path in paths:
                values = [
                    path[key]
                    for key in ("fusion_head_attention", "fusion_tail_attention")
                    if key in path
                ]
                if values:
                    value = float(sum(values) / len(values))
                    path["fusion_mean_attention"] = value
                    combined.append(value)
            if combined:
                item["fusion_path_attention"] = combined
                item["path_attention_source"] = "cross_attention_dsn_to_kg"

    def _binary_scores(self, heads, tails, rels, use_coattn=False):
        h_pool = heads.mean(dim=1)
        t_pool = tails.mean(dim=1)

        if use_coattn:
            alpha = self.co_attention(heads, tails)
            alpha = torch.softmax(alpha.reshape(alpha.size(0), -1), dim=-1).view_as(alpha)
            pair_grid = heads.unsqueeze(2) * tails.unsqueeze(1)
            pair_repr = (pair_grid * alpha.unsqueeze(-1)).sum(dim=(1, 2))
        else:
            pair_repr = h_pool * t_pool

        rel_ids = rels.reshape(-1)
        rel_repr = self.binary_relation(rel_ids)
        features = torch.cat(
            [h_pool, t_pool, torch.abs(h_pool - t_pool), pair_repr, rel_repr],
            dim=-1,
        )
        return self.binary_classifier(features).squeeze(-1)


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
