# test_pipeline.py
import json
import torch
from extract_subgraph import load_kg, extract_enclosing_subgraph
from kg_encoder import KGEncoder

# 1. KG 로드
G = load_kg()
print(f"KG nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# 2. 서브그래프 추출
result = extract_enclosing_subgraph(G, "DB00357", "DB00773", k=1)
if result is None:
    print("No shared genes — pair skipped")
    exit()

g, node_list = result
print(f"\n[Subgraph] nodes: {g.num_nodes()}, edges: {g.num_edges()}")
print(f"feat shape : {tuple(g.ndata['feat'].shape)}")
print(f"id values  : {g.ndata['id'].tolist()}")

# 3. emb_dim을 feat에서 역산
k = 1
actual_feat_dim = g.ndata["feat"].shape[1]   # = emb_dim + 2*(k+1)
emb_dim = actual_feat_dim - 2 * (k + 1)      # TransE 없으면 0
print(f"\nemb_dim (inferred): {emb_dim}")

# 4. KGEncoder 초기화 및 forward
model = KGEncoder(emb_dim=emb_dim, k=k, hidden_dim=64, num_layers=2)
model.eval()

with torch.no_grad():
    H_KG = model(g)

print(f"\n[H_KG] shape : {tuple(H_KG.shape)}")
print(f"[H_KG] first 10 values : {H_KG[:10].tolist()}")
print(f"[H_KG] min={H_KG.min():.4f}, max={H_KG.max():.4f}, mean={H_KG.mean():.4f}")

with open("embeddings/H_KG.json", "w") as f:
    json.dump(H_KG.tolist(), f, indent=2)
print("\n[H_KG] saved to embeddings/H_KG.json")