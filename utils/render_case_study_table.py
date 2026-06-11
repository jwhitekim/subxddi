"""
case_study.csv → 발표용 표 이미지 렌더링
출력: outputs/case_study_table.png
"""

from pathlib import Path
import os

MPLCONFIGDIR = Path("outputs/.matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

INPUT_CSV  = "outputs/case_study.csv"
OUTPUT_PNG = "outputs/case_study_table.png"
MAX_ROWS   = 10
DPI        = 220
GENE_FONT_SIZE = 8.2

# ── 색상 팔레트 ────────────────────────────────────────────────────────────
COLOR_HEADER     = "#1f3a5f"   # 헤더 배경 (진파랑)
COLOR_HEADER_FG  = "white"
COLOR_ROW_ODD    = "#f0f4fa"   # zebra, odd rows
COLOR_ROW_EVEN   = "white"     # zebra, even rows
COLOR_BORDER     = "#c5cfe0"
FONT_FAMILY      = "DejaVu Sans"


def load_data(path: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.head(n).reset_index(drop=True)


def format_genes(text: str) -> str:
    if pd.isna(text) or text == "(없음)":
        return "-"
    parts = [p.strip() for p in str(text).split("|")]
    return " | ".join(parts)


def render(df: pd.DataFrame, output_path: str) -> tuple[int, int]:
    col_labels = ["Drug A", "Drug B", "DDI Type", "Shared", "Shared Genes"]
    col_keys   = ["drug_a", "drug_b", "ddi_type", "shared_count", "shared_genes"]

    # shared_genes는 발표 표에서 행 높이가 커지지 않도록 한 줄로 정리한다.
    df = df.copy()
    df["shared_genes"] = df["shared_genes"].apply(format_genes)

    # 각 행의 라인 수 계산 → 행 높이 결정
    def line_count(row):
        return max(str(row[k]).count("\n") + 1 for k in col_keys)
    row_lines = [line_count(df.iloc[i]) for i in range(len(df))]

    BASE_H   = 0.38   # 한 라인당 인치
    HEADER_H = 0.55
    row_heights = [BASE_H * n for n in row_lines]
    total_h = HEADER_H + sum(row_heights) + 0.3   # 여백

    # 컬럼 너비 비율 (합 = 1)
    col_widths = [0.16, 0.16, 0.13, 0.12, 0.43]
    fig_w = 11.0

    fig, ax = plt.subplots(figsize=(fig_w, total_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    def col_x_edges():
        edges = [0.0]
        for w in col_widths:
            edges.append(edges[-1] + w)
        return edges

    edges = col_x_edges()
    PAD_X = 0.008

    def draw_cell(x0, x1, y0, y1, text, bg, fg="black",
                  fontsize=9.5, bold=False, ha="center"):
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="square,pad=0",
            linewidth=0.5,
            edgecolor=COLOR_BORDER,
            facecolor=bg,
            transform=ax.transData,
            clip_on=False,
        )
        ax.add_patch(rect)
        text_x = x0 + PAD_X if ha == "left" else (x0 + x1) / 2
        ax.text(
            text_x, (y0 + y1) / 2,
            text,
            color=fg,
            fontsize=fontsize,
            fontfamily=FONT_FAMILY,
            fontweight="bold" if bold else "normal",
            ha=ha,
            va="center",
            multialignment="center",
            transform=ax.transData,
            clip_on=False,
        )

    # ── 헤더 ──────────────────────────────────────────────────────────────
    y_top = total_h
    y_bot = y_top - HEADER_H
    for ci, label in enumerate(col_labels):
        draw_cell(
            edges[ci], edges[ci + 1], y_bot, y_top,
            label, bg=COLOR_HEADER, fg=COLOR_HEADER_FG,
            fontsize=10, bold=True,
        )

    # ── 데이터 행 ──────────────────────────────────────────────────────────
    y_cursor = y_bot
    for ri in range(len(df)):
        h = row_heights[ri]
        y_top_r = y_cursor
        y_bot_r = y_cursor - h
        bg = COLOR_ROW_ODD if ri % 2 == 0 else COLOR_ROW_EVEN

        for ci, key in enumerate(col_keys):
            val = str(df.iloc[ri][key])
            fontsize = GENE_FONT_SIZE if key == "shared_genes" else 9
            draw_cell(
                edges[ci], edges[ci + 1], y_bot_r, y_top_r,
                val, bg=bg, fontsize=fontsize, ha="center",
            )
        y_cursor = y_bot_r

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    from PIL import Image
    with Image.open(output_path) as img:
        return img.size   # (width, height) px


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    df = load_data(INPUT_CSV, MAX_ROWS)
    w, h = render(df, OUTPUT_PNG)
    print(f"저장 완료 : {OUTPUT_PNG}")
    print(f"이미지 크기: {w} × {h} px  (dpi={DPI})")
