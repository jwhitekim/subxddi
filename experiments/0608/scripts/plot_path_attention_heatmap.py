#!/usr/bin/env python3
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


ATTENTION_COLUMNS = ("head", "tail", "mean")


def iter_rows(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_row(path, row_index=None, drug_a=None, drug_b=None, pair_kind=None):
    for row in iter_rows(path):
        if pair_kind and row.get("pair_kind") != pair_kind:
            continue
        if row_index is not None and int(row.get("row_index", -1)) == row_index:
            return row
        if drug_a and drug_b and row.get("drug_a") == drug_a and row.get("drug_b") == drug_b:
            return row
    raise SystemExit("No matching reasoning row found.")


def coerce_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def path_rows(row, sort_by="mean", top_n=0):
    paths = row.get("paths") or []
    head_values = row.get("fusion_head_path_attention") or []
    tail_values = row.get("fusion_tail_path_attention") or []
    mean_values = row.get("fusion_path_attention") or []
    rows = []
    for idx, item in enumerate(paths):
        head = coerce_float(item.get("fusion_head_attention"), None)
        tail = coerce_float(item.get("fusion_tail_attention"), None)
        mean = coerce_float(item.get("fusion_mean_attention"), None)
        if head is None and idx < len(head_values):
            head = coerce_float(head_values[idx])
        if tail is None and idx < len(tail_values):
            tail = coerce_float(tail_values[idx])
        if mean is None and idx < len(mean_values):
            mean = coerce_float(mean_values[idx])
        if mean is None and head is not None and tail is not None:
            mean = (head + tail) / 2.0
        rows.append({
            "rank_source_index": idx + 1,
            "drug_a": row.get("drug_a"),
            "drug_b": row.get("drug_b"),
            "pair_kind": row.get("pair_kind"),
            "relation": row.get("relation"),
            "path_label": "{} --{}--> {} --{}--> {}".format(
                item.get("drug_a") or row.get("drug_a"),
                item.get("relation_a") or "",
                item.get("gene") or "path_{}".format(idx + 1),
                item.get("relation_b") or "",
                item.get("drug_b") or row.get("drug_b"),
            ),
            "gene": item.get("gene") or "path_{}".format(idx + 1),
            "relation_a": item.get("relation_a") or "",
            "relation_b": item.get("relation_b") or "",
            "head": coerce_float(head),
            "tail": coerce_float(tail),
            "mean": coerce_float(mean),
        })
    rows.sort(key=lambda item: item.get(sort_by, 0.0), reverse=True)
    if top_n and top_n > 0:
        rows = rows[:top_n]
    return rows


def aggregate_rows(reasoning_jsonl, sort_by="mean", top_n=50):
    grouped = defaultdict(lambda: {
        "count": 0,
        "head_sum": 0.0,
        "tail_sum": 0.0,
        "mean_sum": 0.0,
        "max": 0.0,
        "examples": [],
    })
    for row in iter_rows(reasoning_jsonl):
        for item in path_rows(row):
            key = (item["relation_a"], item["gene"], item["relation_b"])
            bucket = grouped[key]
            bucket["count"] += 1
            bucket["head_sum"] += item["head"]
            bucket["tail_sum"] += item["tail"]
            bucket["mean_sum"] += item["mean"]
            bucket["max"] = max(bucket["max"], item["mean"])
            if len(bucket["examples"]) < 3:
                bucket["examples"].append("{}->{}".format(item["drug_a"], item["drug_b"]))

    rows = []
    for (relation_a, gene, relation_b), bucket in grouped.items():
        count = max(1, bucket["count"])
        rows.append({
            "path_label": "* --{}--> {} --{}--> *".format(relation_a, gene, relation_b),
            "gene": gene,
            "relation_a": relation_a,
            "relation_b": relation_b,
            "count": bucket["count"],
            "head": bucket["head_sum"] / count,
            "tail": bucket["tail_sum"] / count,
            "mean": bucket["mean_sum"] / count,
            "max": bucket["max"],
            "examples": ", ".join(bucket["examples"]),
        })
    rows.sort(key=lambda item: item.get(sort_by, 0.0), reverse=True)
    return rows[:top_n] if top_n and top_n > 0 else rows


def value_range(rows, columns):
    values = [coerce_float(row.get(column)) for row in rows for column in columns]
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    if low == high:
        high = low + 1e-12
    return low, high


def heat_color(value, low, high):
    ratio = (coerce_float(value) - low) / max(high - low, 1e-12)
    ratio = max(0.0, min(1.0, ratio))
    red = 255
    green = int(245 - 170 * ratio)
    blue = int(238 - 205 * ratio)
    return "rgb({}, {}, {})".format(red, green, blue)


def heat_cell(value, low, high):
    return '<td class="num" style="background:{}">{:.6f}</td>'.format(
        heat_color(value, low, high),
        coerce_float(value),
    )


def metric_cards(row):
    items = [
        ("pair_kind", row.get("pair_kind")),
        ("drug_a", row.get("drug_a")),
        ("drug_b", row.get("drug_b")),
        ("relation", row.get("relation")),
        ("probability", row.get("probability")),
        ("num_paths", row.get("num_paths")),
        ("num_paths_total", row.get("num_paths_total")),
        ("hkg_cache_hit", row.get("hkg_cache_hit")),
        ("hkg_cache_used_as_primary", row.get("hkg_cache_used_as_primary")),
    ]
    cards = []
    for label, value in items:
        if value is None:
            continue
        text = "{:.6f}".format(value) if isinstance(value, float) else str(value)
        cards.append(
            '<div class="metric"><span>{}</span><strong>{}</strong></div>'.format(
                html.escape(label), html.escape(text)
            )
        )
    return "\n".join(cards)


def heatmap_table(rows, aggregate=False):
    if not rows:
        return '<p class="empty">No path-level attention rows were found.</p>'
    metric_columns = ("head", "tail", "mean", "max") if aggregate else ATTENTION_COLUMNS
    low, high = value_range(rows, metric_columns)
    headers = ["Rank", "Path", "Gene", "Drug A attn", "Drug B attn", "Mean attn"]
    if aggregate:
        headers.extend(["Max mean", "Count", "Example pairs"])
    body = []
    for rank, item in enumerate(rows, start=1):
        cells = [
            "<td>{}</td>".format(rank),
            "<td>{}</td>".format(html.escape(item["path_label"])),
            "<td><strong>{}</strong></td>".format(html.escape(str(item["gene"]))),
            heat_cell(item["head"], low, high),
            heat_cell(item["tail"], low, high),
            heat_cell(item["mean"], low, high),
        ]
        if aggregate:
            cells.append(heat_cell(item["max"], low, high))
            cells.append('<td class="num">{}</td>'.format(item["count"]))
            cells.append("<td>{}</td>".format(html.escape(item.get("examples", ""))))
        body.append("<tr>{}</tr>".format("".join(cells)))
    return (
        '<table class="heatmap"><thead><tr>{}</tr></thead><tbody>{}</tbody></table>'.format(
            "".join("<th>{}</th>".format(html.escape(header)) for header in headers),
            "\n".join(body),
        )
    )


def matrix_table(row, field, title):
    matrix = row.get(field)
    if not matrix:
        return ""
    paths = row.get("paths") or []
    genes = [item.get("gene") or "path_{}".format(idx + 1) for idx, item in enumerate(paths)]
    col_count = max(len(inner) for inner in matrix if isinstance(inner, list))
    values = [
        coerce_float(value)
        for inner in matrix
        if isinstance(inner, list)
        for value in inner
    ]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    if low == high:
        high = low + 1e-12
    header = ["<th>DSN token</th>"]
    for idx in range(col_count):
        label = genes[idx] if idx < len(genes) else "pad_{}".format(idx + 1)
        header.append("<th>{}</th>".format(html.escape(label)))
    rows = []
    for row_idx, values in enumerate(matrix):
        cells = ["<td>q{}</td>".format(row_idx + 1)]
        for col_idx in range(col_count):
            value = coerce_float(values[col_idx] if col_idx < len(values) else 0.0)
            cells.append(heat_cell(value, low, high))
        rows.append("<tr>{}</tr>".format("".join(cells)))
    return (
        '<section class="panel"><h2>{}</h2><table class="heatmap matrix"><thead><tr>{}</tr></thead><tbody>{}</tbody></table></section>'.format(
            html.escape(title),
            "".join(header),
            "\n".join(rows),
        )
    )


def dashboard_html(title, table_html, cards="", matrices=""):
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f7f8fb; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 0 0 12px; }}
  .subtitle {{ color: #64748b; margin: 0 0 18px; }}
  .panel {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 8px; padding: 18px; margin-bottom: 18px; overflow-x: auto; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
  .metric {{ border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; background: #fbfdff; }}
  .metric span {{ display: block; color: #64748b; font-size: 12px; margin-bottom: 4px; }}
  .metric strong {{ font-size: 15px; word-break: break-word; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 9px 10px; text-align: left; vertical-align: middle; font-size: 13px; }}
  th {{ background: #f1f5f9; color: #334155; font-weight: 700; white-space: nowrap; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .matrix th, .matrix td {{ text-align: center; }}
  .empty {{ color: #64748b; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="subtitle">Darker red cells indicate higher path-level attention. Mean attention is the average of Drug A-side and Drug B-side path attention.</p>
  {cards_section}
  <section class="panel"><h2>Path-Level Attention Heatmap</h2>{table_html}</section>
  {matrices}
</main>
</body>
</html>
""".format(
        title=html.escape(title),
        cards_section='<section class="panel metrics">{}</section>'.format(cards) if cards else "",
        table_html=table_html,
        matrices=matrices,
    )


def write_csv(rows, output_csv, aggregate=False):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "path_label", "gene", "relation_a", "relation_b", "head", "tail", "mean"]
    if aggregate:
        fieldnames.extend(["max", "count", "examples"])
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            payload = dict(row)
            payload["rank"] = rank
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Create path-level attention heatmaps from reasoning JSONL.")
    parser.add_argument("reasoning_jsonl", type=Path)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--drug-a")
    parser.add_argument("--drug-b")
    parser.add_argument("--pair-kind")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate path attention across all reasoning rows.")
    parser.add_argument("--sort-by", choices=["head", "tail", "mean", "max", "count"], default="mean")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = args.csv_output or args.output.with_suffix(".csv")

    if args.aggregate:
        rows = aggregate_rows(args.reasoning_jsonl, sort_by=args.sort_by, top_n=args.top_n)
        title = "Global Path-Level Attention Heatmap"
        html_text = dashboard_html(title, heatmap_table(rows, aggregate=True))
        write_csv(rows, csv_output, aggregate=True)
    else:
        if args.row_index is None and not (args.drug_a and args.drug_b):
            raise SystemExit("Use --row-index or both --drug-a and --drug-b, or use --aggregate.")
        row = load_row(args.reasoning_jsonl, args.row_index, args.drug_a, args.drug_b, args.pair_kind)
        rows = path_rows(row, sort_by=args.sort_by, top_n=args.top_n)
        if not rows:
            raise SystemExit("Selected reasoning row has no path-level attention rows.")
        title = "{} {} -> {} rel={} prob={:.4f}".format(
            row.get("pair_kind", ""),
            row.get("drug_a"),
            row.get("drug_b"),
            row.get("relation"),
            coerce_float(row.get("probability")),
        )
        matrices = "\n".join([
            matrix_table(row, "fusion_head_dsn_to_kg_attention", "Drug A DSN-to-Path Attention"),
            matrix_table(row, "fusion_tail_dsn_to_kg_attention", "Drug B DSN-to-Path Attention"),
        ])
        html_text = dashboard_html(title, heatmap_table(rows), cards=metric_cards(row), matrices=matrices)
        write_csv(rows, csv_output)

    args.output.write_text(html_text)
    print(args.output)
    print(csv_output)


if __name__ == "__main__":
    main()
