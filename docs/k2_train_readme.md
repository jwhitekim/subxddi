# DrugBank k=2 Training Code Drop

This folder contains only the code files needed to run the 0608 DrugBank k=2 experiment.
It intentionally excludes DrugBank data, KG assets, checkpoints, logs, and generated outputs.

## What to Copy

Copy these folders into the teammate's `Drug-Interaction-Research` repository:

```text
0608/scripts/
drugbank_test/
```

If their existing `subxddi-kg` code is not already updated, also copy:

```text
subxddi-kg/kg_encoder.py
subxddi-kg/extract_subgraph.py
```

The teammate should already have:

```text
Drug-Interaction-Research/dataset/drugbank/
subxddi-kg/kg_graph.gpickle
subxddi-kg/pretrained/transe_entity_emb.npy
subxddi-kg/pretrained/transe_entity2id.json
subxddi-kg/Hkg_encoding/hkg_cache_k2.pt
```

## Expected Layout

Recommended sibling-folder layout:

```text
<workspace>/
  Drug-Interaction-Research/
    0608/
      scripts/
    drugbank_test/
    dataset/
      drugbank/
  subxddi-kg/
    kg_graph.gpickle
    kg_encoder.py
    extract_subgraph.py
    pretrained/
    Hkg_encoding/
      hkg_cache_k2.pt
```

## Run k=2 Cross-Attention

From the teammate's machine:

```bash
cd <workspace>/Drug-Interaction-Research/0608

bash scripts/run_background.sh core \
  --ks 2 \
  --fusions cross_attention \
  --folds 0 \
  --path-counts 10 \
  --train-modes full \
  --batch-sizes 128 \
  --epochs 200 \
  --subxddi-kg-dir "$(pwd)/../../subxddi-kg" \
  --hkg-cache-dir "$(pwd)/../../subxddi-kg/Hkg_encoding"
```

Check progress:

```bash
cd <workspace>/Drug-Interaction-Research/0608
bash scripts/status_background.sh core
tail -f logs/background_core.log
```

Stop:

```bash
cd <workspace>/Drug-Interaction-Research/0608
bash scripts/stop_background.sh core
```

## Outputs

Main result files are written under `Drug-Interaction-Research/0608/`:

```text
all_results.csv
checkpoints/
history/
logs/
plots/
curves/
reasoning/
```

## Environment Used

```text
Python 3.7.17
torch 1.11.0+cu113
torch_geometric 2.0.4
dgl 0.9.1
rdkit 2022.09.5
scikit-learn 1.0.2
pandas 1.3.5
numpy 1.21.6
Pillow 9.5.0
networkx 2.6.3
```
