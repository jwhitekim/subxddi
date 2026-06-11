"""
KG 공유 타겟 분석 스크립트 (설명가능성 케이스 스터디용)

목적: 두 약물에 대해 ① DDI ground truth 존재 여부와
      ② KG 공유 유전자 타겟을 함께 보여 준다.
      두 ground truth가 정합적이면 "예측이 KG 경로와 정합적"이라는
      케이스 스터디 한 줄 완성.

한계:
  - DDI 여부 / type 은 ddis.csv ground truth 기준이며, 모델 예측이 아니다.
  - 공유 유전자는 KG 사실이며, 모델 attention 결과가 아니다.
  - 배치 모드의 약물쌍은 무작위 추출이므로 "모델이 고른 쌍"이 아니다.
    이 표는 KG와 DDI 라벨의 정합성 경향을 보여 주는 용도이며,
    특정 모델 출력의 설명이 아니다.
  - KG에 없는 약물 ID는 빈 집합으로 처리된다.
  - dataset/ 내 DDI type → 설명 텍스트 매핑 파일 없음 (type 번호만 사용)

참고 (실제 데이터 기준):
  - 전체 DDI 쌍 중 양쪽 KG 존재 비율     : ~79%  (152,025 / 191,808)
  - 전체 DDI 쌍 중 공유 유전자 ≥ 1 비율  : ~44%  (85,089 / 191,808)
  - 양쪽 KG 존재 쌍 중 공유 유전자 ≥ 1  : ~56%  (85,089 / 152,025)
  ※ random 모드는 KG 존재 쌍만 후보로 사용하므로 기대 비율은 ~56%

사용법:
    # 단일 쌍 (기본)
    python utils/kg_shared_targets.py
    python utils/kg_shared_targets.py --drug_a DB00898 --drug_b DB01211

    # 배치 모드 — random: 실제 비율 반영 (기본값)
    python utils/kg_shared_targets.py --n 10
    python utils/kg_shared_targets.py --n 10 --mode random --seed 42 --out outputs/case_study.csv

    # 배치 모드 — shared: 공유 유전자 있는 쌍 우선 (정합 사례 강조)
    python utils/kg_shared_targets.py --n 10 --mode shared --seed 42

    # 경로 명시
    python utils/kg_shared_targets.py \\
        --triples dataset/kg_source/kg_triples.tsv \\
        --ddis    dataset/drugbank/ddis.csv \\
        --drug_a DB00945 --drug_b DB01050
"""

import argparse
import random
from collections import defaultdict

import pandas as pd

DEFAULT_TRIPLES = "dataset/kg_source/kg_triples.tsv"
DEFAULT_DDIS    = "dataset/drugbank/ddis.csv"
DEFAULT_DRUG_A  = "DB00898"
DEFAULT_DRUG_B  = "DB01211"
SHARED_PREVIEW  = 5   # 배치 표에서 공유 유전자를 몇 개까지 미리 보여줄지


# ── 데이터 로드 ────────────────────────────────────────────────────────────

def load_drug_targets(triples_path: str) -> dict[str, set[str]]:
    df = pd.read_csv(triples_path, sep="\t")
    # 컬럼: head(DrugID), relation, tail(GeneID) — "targets" 관계만 사용
    targets_df = df[df["relation"] == "targets"]
    drug_targets: dict[str, set[str]] = defaultdict(set)
    for _, row in targets_df.iterrows():
        drug_targets[str(row["head"])].add(str(row["tail"]))
    return dict(drug_targets)


def load_ddis(ddis_path: str) -> pd.DataFrame:
    df = pd.read_csv(ddis_path, usecols=["d1", "d2", "type"])
    # type NaN/결측 행 제거
    df = df.dropna(subset=["type"])
    df["d1"]   = df["d1"].astype(str)
    df["d2"]   = df["d2"].astype(str)
    df["type"] = df["type"].astype(int)
    # self-pair 제거 (d1 == d2)
    df = df[df["d1"] != df["d2"]].reset_index(drop=True)
    return df


# ── 단일 쌍 분석 ───────────────────────────────────────────────────────────

def lookup_ddi(ddis: pd.DataFrame, drug_a: str, drug_b: str) -> dict:
    """(A,B) / (B,A) 순서 무관 검색. ground truth, 모델 예측 아님."""
    match = ddis[
        ((ddis["d1"] == drug_a) & (ddis["d2"] == drug_b)) |
        ((ddis["d1"] == drug_b) & (ddis["d2"] == drug_a))
    ]
    if match.empty:
        return {"exists": False, "ddi_type": None}
    return {"exists": True, "ddi_type": int(match.iloc[0]["type"])}


def analyze_kg(drug_targets: dict, drug_a: str, drug_b: str) -> dict:
    targets_a = drug_targets.get(drug_a, set())
    targets_b = drug_targets.get(drug_b, set())
    shared = targets_a & targets_b
    return {
        "targets_a":   len(targets_a),
        "targets_b":   len(targets_b),
        "shared_count": len(shared),
        "shared_genes": sorted(shared),
    }


def print_single(drug_a: str, drug_b: str, ddi: dict, kg: dict) -> None:
    W = 54
    print(f"\n{'='*W}")
    print(f"  DDI × KG 정합성 케이스 스터디")
    print(f"{'='*W}")
    print(f"  약물 A : {drug_a}")
    print(f"  약물 B : {drug_b}")

    print(f"{'-'*W}")
    print(f"  [DDI ground truth — ddis.csv 기준, 모델 예측 아님]")
    if ddi["exists"]:
        print(f"  DDI 여부 : 있음")
        print(f"  DDI type : {ddi['ddi_type']}")
    else:
        print(f"  DDI 여부 : 없음")

    print(f"{'-'*W}")
    print(f"  [KG 공유 타겟 — KG 사실, 모델 attention 아님]")
    print(f"  약물 A 타겟 : {kg['targets_a']}개")
    print(f"  약물 B 타겟 : {kg['targets_b']}개")
    print(f"  공유 유전자 : {kg['shared_count']}개")
    if kg["shared_count"] == 0:
        print(f"  → 공유 타겟 없음 (KG 기준)")
    else:
        for gene in kg["shared_genes"]:
            print(f"     · {gene}")

    print(f"{'-'*W}")
    if ddi["exists"] and kg["shared_count"] > 0:
        print(f"  ✓ DDI 있음 + 공유 타겟 있음 → KG 경로와 정합적")
    elif ddi["exists"] and kg["shared_count"] == 0:
        print(f"  △ DDI 있음 + 공유 타겟 없음 → KG 경로로 설명 불가")
    elif not ddi["exists"] and kg["shared_count"] > 0:
        print(f"  △ DDI 없음 + 공유 타겟 있음 → KG만으로는 과예측 가능성")
    else:
        print(f"  — DDI 없음 + 공유 타겟 없음")
    print(f"{'='*W}\n")


# ── 배치 모드 ──────────────────────────────────────────────────────────────

def sample_batch(ddis: pd.DataFrame, drug_targets: dict,
                 n: int, seed: int, mode: str) -> list[dict]:
    """
    ddis.csv 의 실제 DDI 쌍에서 n개를 추출한다.
    self-pair / type NaN은 load_ddis 단계에서 이미 제거됨.

    mode='random' : 순수 무작위 — 실제 공유 유전자 비율(~44%)을 그대로 반영
    mode='shared' : 공유 유전자 있는 쌍 우선, 부족하면 0개 쌍으로 채움
                    (정합 사례 강조용 케이스 스터디에 사용)

    추출은 무작위(seed 고정)이므로 "모델이 고른 쌍"이 아님.
    """
    rng = random.Random(seed)

    # KG 양쪽 존재 쌍만 대상
    candidates = []
    for _, row in ddis.iterrows():
        a, b = row["d1"], row["d2"]
        if a not in drug_targets or b not in drug_targets:
            continue
        shared = drug_targets[a] & drug_targets[b]
        candidates.append({
            "drug_a":   a,
            "drug_b":   b,
            "ddi_type": int(row["type"]),
            "shared":   sorted(shared),
        })

    if mode == "shared":
        with_shared    = [r for r in candidates if r["shared"]]
        without_shared = [r for r in candidates if not r["shared"]]
        rng.shuffle(with_shared)
        rng.shuffle(without_shared)
        pool = (with_shared + without_shared)[:n]
    else:  # random
        rng.shuffle(candidates)
        pool = candidates[:n]

    return pool


def build_batch_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        preview = " | ".join(r["shared"][:SHARED_PREVIEW])
        if len(r["shared"]) > SHARED_PREVIEW:
            preview += f" … (+{len(r['shared']) - SHARED_PREVIEW})"
        rows.append({
            "drug_a":       r["drug_a"],
            "drug_b":       r["drug_b"],
            "ddi_type":     r["ddi_type"],
            "shared_count": len(r["shared"]),
            "shared_genes": preview if r["shared"] else "(없음)",
        })
    return pd.DataFrame(rows)


# ── 진입점 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DDI ground truth + KG 공유 타겟 정합성 분석"
    )
    parser.add_argument("--triples", default=DEFAULT_TRIPLES, help="kg_triples.tsv 경로")
    parser.add_argument("--ddis",    default=DEFAULT_DDIS,    help="ddis.csv 경로")
    parser.add_argument("--drug_a",  default=DEFAULT_DRUG_A,  help="단일 모드: 약물 A ID")
    parser.add_argument("--drug_b",  default=DEFAULT_DRUG_B,  help="단일 모드: 약물 B ID")
    parser.add_argument("--n",       type=int, default=None,
                        help="배치 모드: 출력할 약물쌍 수 (지정 시 배치 모드)")
    parser.add_argument("--mode",    default="random", choices=["random", "shared"],
                        help="random: 실제 비율 반영 무작위 / shared: 공유 유전자 있는 쌍 우선")
    parser.add_argument("--seed",    type=int, default=42, help="배치 모드: 랜덤 시드")
    parser.add_argument("--out",     default=None,
                        help="배치 결과 CSV 저장 경로 (예: outputs/case_study.csv)")
    args = parser.parse_args()

    drug_targets = load_drug_targets(args.triples)
    ddis         = load_ddis(args.ddis)

    if args.n is not None:
        # ── 배치 모드 ──────────────────────────────────────────────────────
        records = sample_batch(ddis, drug_targets, args.n, args.seed, args.mode)
        df = build_batch_df(records)

        n_shared = sum(1 for r in records if r["shared"])
        print(f"\n[배치 케이스 스터디] n={args.n} mode={args.mode} seed={args.seed}")
        print(f"  공유 유전자 있는 쌍: {n_shared}/{len(records)}")
        print(f"  DDI type: ddis.csv ground truth / 공유 유전자: KG 사실")
        print(f"  무작위 추출 — 모델이 선택한 쌍이 아님\n")
        print(df.to_string(index=False))

        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\n저장: {args.out}")
    else:
        # ── 단일 쌍 모드 ──────────────────────────────────────────────────
        ddi = lookup_ddi(ddis, args.drug_a, args.drug_b)
        kg  = analyze_kg(drug_targets, args.drug_a, args.drug_b)
        print_single(args.drug_a, args.drug_b, ddi, kg)


if __name__ == "__main__":
    main()
