# SubXDDI: Subgraph-based eXplainable Drug-Drug Interaction Prediction

## 프로젝트 개요
DSN-DDI의 분자 인코더와 KG 기반 Subgraph Encoder를 Cross-Attention으로 융합하여
DDI(Drug-Drug Interaction) 예측과 약리학적 설명 가능성을 동시에 제공하는 모델.

## 아키텍처
```
Drug A/B 분자 그래프
        ↓
DSN-DDI Encoder (intra-view + inter-view)
→ repr_h, repr_t [batch, 4, ?]

KG Enclosing Subgraph (2-hop, Drug A–B 기준)
        ↓
KG-side GNN (DGL 기반)
→ H_KG [batch, 384]  (h_A || h_B || h_GSub)

Cross-Attention Fusion (H_DSN ↔ H_KG)
        ↓
CoAttention → RESCAL → DDI type 예측 (86 classes)
```

## 팀 역할
| 담당 | 내용 |
|------|------|
| 김준희 | KG 구축, Subgraph 추출, KG-side GNN, Cross-Attention Fusion |
| 김하은 | DSN-DDI 인코더 구현 (models_fusion.py 기반) |

## 폴더 구조
```
subxddi/
├── data/                        # 데이터셋
├── kg/                          # KG 파일 (kg_graph.gpickle 등)
├── models/
│   ├── dsn_encoder/             # DSN-DDI 인코더
│   ├── kg_encoder/              # KG-side GNN
│   └── fusion/                  # Cross-Attention Fusion 모듈
├── subgraph/                    # Subgraph 추출
├── utils/                       # 공통 유틸
├── transductive_train_fusion.py # 학습 진입점
└── transductive_test_fusion.py  # 평가 진입점
```

## 데이터
- DDI 레이블: DrugBank (86 DDI types, transductive setting)
- KG: BioSNAP ChG-Miner (Drug→Gene, targets 관계, 15,138 edges)
- Drug ID 통일: DrugBank Vocabulary (DB##### 형식)

## 실행
```bash
# 학습
python transductive_train_fusion.py --fusion_mode cross_attention

# 평가
python transductive_test_fusion.py --fusion_mode cross_attention
```

## 참고 논문
- [DSN-DDI](https://academic.oup.com/bib/article/24/1/bbac597/6966537) — Li et al., Briefings in Bioinformatics, 2023
- [SumGNN](https://academic.oup.com/bioinformatics/article/37/18/2988/6189090) — Yu et al., Bioinformatics, 2021
