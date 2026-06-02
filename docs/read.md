# DSN-DDI + Sum-GNN Fusion

이 폴더는 기존 `Drug-Interaction-Research` 프로젝트 위에 얹어서 사용하는 **fusion patch 공유 폴더**입니다.

중요한 전제:

- `team_share`에는 fusion 관련 파일만 들어 있습니다.
- 기존 DSN-DDI original code와 dataset은 팀원 로컬의 `Drug-Interaction-Research` repo에 있어야 합니다.
- Sum-GNN KG encoder/subgraph extractor는 별도 `sum-gnn` 폴더가 있어야 합니다.
- 학습된 checkpoint는 `mlp`, `cross_attention`, `concat`별로 `.pkl`, `.pt`를 모두 포함했습니다.

## 1. 환경 구성 디렉터리 구조

권장 구조는 아래와 같습니다. `team_share`는 기존 `Drug-Interaction-Research` repo root 아래에 두고, Sum-GNN은 `/workspace/sum-gnn`에 둡니다.

```text
/workspace/
  Drug-Interaction-Research/
    dataset/
      drugbank/
        fold0/
          train.csv
          test.csv
        drug_data.pkl
        drug_graph.pkl

    drugbank_test/
      # 기존 DSN-DDI original files
      models.py
      layers.py
      custom_loss.py
      data_preprocessing.py
      transductive_train.py
      transductive_test.py
      inductive_train.py
      inductive_test.py

      # team_share/fusion_files 에서 복사할 fusion files
      models_fusion.py
      data_preprocessing_fusion.py
      transductive_train_fusion.py
      transductive_test_fusion.py
      inductive_train_fusion.py
      inductive_test_fusion.py
      run_fusion_experiments.py
      config.yaml
      config_decoder_only.yaml


  sum-gnn/
    extract_subgraph.py
    kg_encoder.py
    kg_graph_py37.gpickle
    pretrained/
      transe_entity2id.json
      transe_entity_emb.npy
```

기본 경로 기준:

```text
Drug-Interaction-Research repo: /workspace/Drug-Interaction-Research
Sum-GNN directory:              /workspace/sum-gnn
Shared checkpoints:             team_share/checkpoints/<fusion_mode>/
New training outputs:           outputs/checkpoints/<fusion_mode>/
Decoder-only training outputs:  outputs/checkpoints_decoder_only/<fusion_mode>/
```

## 2. Fusion 파일 설치

`Drug-Interaction-Research` repo root에서 fusion 파일만 기존 `drugbank_test/`로 복사합니다.

```bash
cd /workspace/Drug-Interaction-Research
cp team_share/fusion_files/* drugbank_test/
```

복사 후 확인:

```bash
ls drugbank_test/*fusion.py
```

## 3. Test 커맨드

가장 권장하는 방식은 `model_best_state_dict.pkl`을 사용하는 것입니다. 

### Cross-Attention 테스트

```bash
python drugbank_test/transductive_test_fusion.py \
  --fusion_mode cross_attention \
  --pkl_name team_share/checkpoints/cross_attention/model_best_state_dict.pkl \
  --sumgnn_dir /workspace/sum-gnn \
  --return_reasoning \
  --batch_size 16
```

### MLP 테스트

```bash
python drugbank_test/transductive_test_fusion.py \
  --fusion_mode mlp \
  --pkl_name team_share/checkpoints/mlp/model_best_state_dict.pkl \
  --sumgnn_dir /workspace/sum-gnn \
  --return_reasoning \
  --batch_size 16
```

### Concat 테스트

```bash
python drugbank_test/transductive_test_fusion.py \
  --fusion_mode concat \
  --pkl_name team_share/checkpoints/concat/model_best_state_dict.pkl \
  --sumgnn_dir /workspace/sum-gnn \
  --return_reasoning \
  --batch_size 16
```

### Full model pkl로 테스트하는 경우

아래처럼 `model_best.pkl`도 사용할 수 있습니다. 테스트 코드는 full model pkl을 로드하더라도 `--sumgnn_dir` 값을 런타임에 다시 덮어씁니다.

```bash
SUMGNN_DIR=/workspace/sum-gnn \
python drugbank_test/transductive_test_fusion.py \
  --fusion_mode cross_attention \
  --pkl_name team_share/checkpoints/cross_attention/model_best.pkl \
  --sumgnn_dir /workspace/sum-gnn \
  --return_reasoning \
  --batch_size 16
```

주의:

- `model_best.pkl`은 PyTorch pickle object라 코드 경로와 class 이름 영향이 있습니다.
- 다른 서버/다른 repo 구조에서는 `model_best_state_dict.pkl`을 권장합니다.
- full model pkl을 쓰면서 Sum-GNN 경로가 다르면 `SUMGNN_DIR` 환경변수와 `--sumgnn_dir`를 같은 값으로 지정하세요.
- `decoder_best.pt`는 전체 모델이 아니라 decoder/fusion weight만 들어 있으므로 최종 성능 테스트용으로는 권장하지 않습니다.

## 4. Train 커맨드

### 전체 모델 학습

아래는 `cross_attention`을 전체 모델 학습하는 예시입니다.

```bash
python drugbank_test/transductive_train_fusion.py \
  --fusion_mode cross_attention \
  --train_scope all \
  --batch_size 64 \
  --n_epochs 80 \
  --lr 0.001 \
  --weight_decay 0.0005 \
  --neg_samples 1 \
  --data_size_ratio 1 \
  --use_cuda 1 \
  --num_workers 2 \
  --sumgnn_dir /workspace/sum-gnn \
  --output_dir outputs/checkpoints/cross_attention \
  --pkl_name outputs/checkpoints/cross_attention/model_best.pkl \
  --target_train_loss 0.3 \
  --target_train_acc 0.8 \
  --early_stop_on_target 1
```

`mlp`, `concat`은 `--fusion_mode`와 output path만 바꾸면 됩니다.

```bash
--fusion_mode mlp
--fusion_mode concat
```

### Encoder 고정 + Decoder만 학습

가능합니다. `--train_scope decoder_only`를 사용하면 DSN-DDI encoder는 freeze되고, `decoder_fusion`과 `KGE/RESCAL`만 학습됩니다.

기존 학습 checkpoint의 encoder를 고정하고 decoder만 fine-tuning하려면 `--init_checkpoint`를 같이 사용합니다.

```bash
python drugbank_test/transductive_train_fusion.py \
  --fusion_mode cross_attention \
  --train_scope decoder_only \
  --init_checkpoint team_share/checkpoints/cross_attention/model_best_state_dict.pkl \
  --batch_size 64 \
  --n_epochs 5 \
  --lr 0.001 \
  --weight_decay 0.0005 \
  --neg_samples 1 \
  --data_size_ratio 1 \
  --use_cuda 1 \
  --num_workers 2 \
  --sumgnn_dir /workspace/sum-gnn \
  --output_dir outputs/checkpoints_decoder_only/cross_attention \
  --pkl_name outputs/checkpoints_decoder_only/cross_attention/model_best.pkl
```

주의:

- `--train_scope decoder_only`만 쓰면 새로 만든 모델의 encoder가 freeze됩니다.
- 기존 학습된 encoder를 고정하려면 반드시 `--init_checkpoint`를 같이 쓰는 것을 권장합니다.
- `decoder_best.pt`는 decoder weight만 있으므로 encoder 초기화용으로는 부족합니다.


## 5. 실행 전 체크

```bash
ls /workspace/sum-gnn/extract_subgraph.py
ls /workspace/sum-gnn/kg_encoder.py
ls team_share/checkpoints/cross_attention/model_best_state_dict.pkl
```

## 6. 실행 흐름 설명

전체 forward 흐름은 아래와 같습니다.

```text
Drug pair molecular graph
  -> DSN-DDI encoder
  -> DSN drug pair representation H_DSN

Drug pair id: (drug_a, drug_b)
  -> Sum-GNN extract_subgraph.py
  -> KG enclosing subgraph
  -> KGEncoder
  -> Sum-GNN/KG representation H_KG

H_DSN + H_KG
  -> fusion module
     - concat
     - mlp
     - cross_attention
  -> RESCAL decoder
  -> DDI prediction score
  -> optional reasoning path output
```

fusion mode별 차이:

```text
none:
  기존 DSN-DDI와 동일하게 동작합니다.

concat:
  DSN representation과 Sum-GNN representation을 concat한 뒤 decoder에 넣습니다.

mlp:
  concat 결과를 MLP projection으로 다시 decoder dimension에 맞춥니다.

cross_attention:
  DSN representation을 query로 사용하고, Sum-GNN representation을 key/value로 사용합니다.
```

fusion mode가 `concat`, `mlp`, `cross_attention`일 때는 기존 DSN-DDI의 CoAttention 연산을 생략하고 fused representation을 RESCAL decoder에 넣습니다.

## 7. 파일별 설명

### `models_fusion.py`

주요 모델 정의 파일입니다.

포함 내용:

- `SumGNNSubgraphEncoder`
- `DecoderFusion`
- fusion mode별 forward logic
- `return_reasoning=True`일 때 reasoning path 반환
- DGL CPU-only 환경 대응
- full model pickle 저장/로드 호환 처리

### `data_preprocessing_fusion.py`

기존 data loader 구조를 유지하면서 drug pair id를 추가합니다.

기존 batch 구조:

```text
(h_batch, t_batch, rels, b_graph)
```

fusion batch 구조:

```text
(h_batch, t_batch, rels, b_graph, pair_ids)
```

`pair_ids`는 Sum-GNN subgraph 추출에 사용됩니다.

### `transductive_train_fusion.py`

DrugBank transductive split 학습용 script입니다.

주요 기능:

- `--fusion_mode` 선택
- `--train_scope all/decoder_only` 선택
- epoch별 metric 저장
- loss/acc/roc plot 저장
- full model pkl 저장
- decoder pt 저장
- target train loss/acc 달성 시 early stop 가능

### `transductive_test_fusion.py`

DrugBank transductive split 테스트용 script입니다.

지원 checkpoint:

- full model pickle: `model_best.pkl`
- portable state_dict pickle: `model_best_state_dict.pkl`
- decoder-only checkpoint: `decoder_best.pt`

`--return_reasoning`을 켜면 prediction metric과 함께 reasoning sample을 출력합니다.

### `inductive_train_fusion.py`, `inductive_test_fusion.py`

inductive split용 fusion script입니다. 현재 주 실험은 transductive 기준입니다.

### `run_fusion_experiments.py`

config 파일을 읽어서 여러 fusion mode를 순차 실행합니다.

## 8. Input 설명

### Dataset input

기존 DSN-DDI와 동일한 DrugBank dataset을 사용합니다.

```text
dataset/drugbank/fold0/train.csv
dataset/drugbank/fold0/test.csv
dataset/drugbank/drug_data.pkl
dataset/drugbank/drug_graph.pkl
```

`train.csv`, `test.csv`는 최소한 아래 column을 사용합니다.

```text
d1, d2, type
```

### Sum-GNN input

Sum-GNN KG branch는 아래 파일을 사용합니다.

```text
sum-gnn/extract_subgraph.py
sum-gnn/kg_encoder.py
sum-gnn/kg_graph_py37.gpickle
sum-gnn/pretrained/transe_entity2id.json
sum-gnn/pretrained/transe_entity_emb.npy
```

각 drug pair `(d1, d2)`에 대해 KG enclosing subgraph를 추출하고, KGEncoder representation을 생성합니다.

## 9. Output 설명

### Model forward output

기본 output:

```python
scores
```

`--return_reasoning` 또는 `return_reasoning=True` 사용 시:

```python
{
    "scores": scores,
    "reasoning_paths": reasoning_paths,
}
```

reasoning path 예시:

```text
DB00678 --targets--> P08684 <--targets-- DB01259
DB00678 --targets--> P10632 <--targets-- DB01259
DB00678 --targets--> P33261 <--targets-- DB01259
DB00678 --targets--> P20815 <--targets-- DB01259
```

현재 reasoning path는 1-hop shared target gene 기반입니다. 따라서 drug가 KG에 없거나 공통 target gene이 없으면 `paths: []`가 나올 수 있습니다.

### Test metric output

테스트 실행 시 아래 metric을 출력합니다.

```text
test_acc
test_auc_roc
test_f1
test_precision
test_recall
test_int_ap
test_ap
```

## 10. 추가 옵션 사용 설명

### `--fusion_mode`

```text
none             기존 DSN-DDI와 동일
concat           DSN representation과 Sum-GNN representation concat
mlp              concat 후 MLP projection
cross_attention  DSN=query, Sum-GNN=key/value cross-attention
```

### `--train_scope`

```text
all           encoder + fusion + decoder 전체 학습
decoder_only  decoder_fusion + RESCAL KGE만 학습
```

현재 target 성능 실험은 `train_scope: all` 기준입니다.

### `--init_checkpoint`

decoder-only 학습 전에 기존 checkpoint를 먼저 로드할 때 사용합니다.

```bash
--init_checkpoint team_share/checkpoints/cross_attention/model_best_state_dict.pkl
```

권장 사용:

```text
--train_scope decoder_only + --init_checkpoint <full model state_dict>
```

이 조합이면 기존 학습된 encoder를 고정하고 decoder/fusion 쪽만 추가 학습할 수 있습니다.

### `--return_reasoning`

테스트 시 reasoning path sample을 출력합니다.

```bash
--return_reasoning
```

### `--sumgnn_dir`

Sum-GNN 폴더 위치를 지정합니다.

```bash
--sumgnn_dir /workspace/sum-gnn
```

### `--batch_size`

GPU memory가 부족하면 줄입니다.

```bash
--batch_size 16
```

### `--target_train_loss`, `--target_train_acc`, `--early_stop_on_target`

목표 성능에 도달하면 조기 종료합니다.

```bash
--target_train_loss 0.3 \
--target_train_acc 0.8 \
--early_stop_on_target 1
```

### `--cross_attention_heads`

cross-attention head 수를 지정합니다.

```bash
--cross_attention_heads 4
```

### `--data_size_ratio`

학습 데이터 일부만 사용할 때 사용합니다.

```bash
--data_size_ratio 0.1
```

## 11. 최종 출력물들

학습 script는 `--output_dir` 아래에 아래 파일을 저장합니다.

```text
model_best.pkl
  validation 기준 best full model입니다.

model_best_state_dict.pkl
  full model의 portable state_dict checkpoint입니다.

model_final_state_dict.pkl
  학습 종료 시점의 full model state_dict checkpoint입니다.

decoder_best.pt
  decoder_fusion + RESCAL KGE weight입니다.

decoder_final.pt
  학습 종료 시점의 decoder_fusion + RESCAL KGE weight입니다.

metrics.csv
  epoch별 train/validation metric입니다.

loss_plot.svg
  train/validation loss line plot입니다.

acc_roc_plot.svg
  train/validation acc, roc line plot입니다.

target_reached.txt
  target_train_loss/target_train_acc 조건을 달성했을 때 생성됩니다.

config.yaml
  해당 실험 설정입니다.

SHA256SUMS.txt
  checkpoint 무결성 확인용 checksum입니다.
```



##  `.pkl`과 `.pt` 사용 기준

### `model_best.pkl`

full model object입니다.

포함:

```text
DSN-DDI encoder
Sum-GNN fusion wrapper
DecoderFusion
RESCAL decoder
```


### `model_best_state_dict.pkl`

full model의 `state_dict`입니다.

사용 방식:

- `transductive_test_fusion.py`가 model을 먼저 만든 뒤 `load_state_dict`로 로드합니다.

### `decoder_best.pt`

decoder-only weight입니다.

포함:

```text
decoder_fusion
KGE / RESCAL relation embedding
co_attention 상태 또는 skipped marker
```

주의:

- DSN encoder weight는 포함하지 않습니다.
- 단독으로 최종 test 성능 평가용으로 쓰면 encoder가 새로 초기화될 수 있습니다.
- decoder만 재사용하거나 ablation/debug 목적으로 사용하세요.

## 추천 실행 순서



```bash
# 1. 기존 repo root로 이동
cd /workspace/Drug-Interaction-Research

# 2. fusion files 복사
cp team_share/fusion_files/* drugbank_test/

# 3. Sum-GNN 경로 확인
ls /workspace/sum-gnn/extract_subgraph.py
ls /workspace/sum-gnn/kg_encoder.py

# 4. cross_attention checkpoint 테스트
python drugbank_test/transductive_test_fusion.py \
  --fusion_mode cross_attention \
  --pkl_name team_share/checkpoints/cross_attention/model_best_state_dict.pkl \
  --sumgnn_dir /workspace/sum-gnn \
  --return_reasoning \
  --batch_size 16
```
