#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

import networkx as nx


def load_row(path, row_index=None, drug_a=None, drug_b=None):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
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


def attention_value(path_item):
    return float(
        path_item.get("fusion_mean_attention")
        or path_item.get("fusion_head_attention")
        or path_item.get("fusion_tail_attention")
        or 0.0
    )


def build_graph(row, sort_by="mean", top_n=0):
    graph = nx.MultiDiGraph()
    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    graph.add_node(drug_a, kind="drug", mean=0.0)
    graph.add_node(drug_b, kind="drug", mean=0.0)

    for item in path_rows(row, sort_by=sort_by, top_n=top_n):
        gene = item["gene"]
        graph.add_node(gene, kind="gene", mean=item["mean"], head=item["head"], tail=item["tail"])
        graph.add_edge(
            drug_a,
            gene,
            relation=item["relation_a"],
            attention=item["head"],
            attention_name="head",
        )
        graph.add_edge(
            gene,
            drug_b,
            relation=item["relation_b"],
            attention=item["tail"],
            attention_name="tail",
        )
    return graph


def graph_layout(graph, row):
    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    genes = [n for n, data in graph.nodes(data=True) if data.get("kind") == "gene"]
    y_step = 1.0 / max(len(genes), 1)
    pos = {drug_a: (0.0, 0.5), drug_b: (2.0, 0.5)}
    for i, gene in enumerate(genes):
        pos[gene] = (1.0, 1.0 - (i + 0.5) * y_step)
    return pos, genes


def draw_png(row, graph, output):
    import matplotlib.pyplot as plt

    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    pos, genes = graph_layout(graph, row)
    node_colors = [
        "#4C78A8" if data.get("kind") == "drug" else "#F58518"
        for _, data in graph.nodes(data=True)
    ]
    attentions = [data.get("attention", 0.0) for _, _, data in graph.edges(data=True)]
    max_attention = max(attentions) if attentions else 1.0
    widths = [1.5 + 8.0 * (value / max_attention if max_attention else 0.0) for value in attentions]

    plt.figure(figsize=(10, max(4, 1.1 * len(genes))))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1600, edgecolors="#222222")
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        edge_color=attentions,
        edge_cmap=plt.cm.viridis,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.05",
    )
    edge_labels = {
        (u, v, k): f"{data.get('relation', '')}\n{data.get('attention_name', 'attn')}={data.get('attention', 0.0):.4f}"
        for u, v, k, data in graph.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title(graph_title(row))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=220)


def graph_title(row):
    return (
        f"{row.get('pair_kind', '')} {row.get('drug_a')} -> {row.get('drug_b')} "
        f"rel={row.get('relation')} prob={row.get('probability', 0.0):.4f}"
    )


def draw_svg(row, graph):
    pos, _ = graph_layout(graph, row)
    width = 980
    height = max(360, 120 * max(1, graph.number_of_nodes() - 2))
    margin_x = 120
    margin_y = 70

    def xy(node):
        x, y = pos[node]
        return margin_x + x * ((width - 2 * margin_x) / 2.0), margin_y + (1.0 - y) * (height - 2 * margin_y)

    attentions = [data.get("attention", 0.0) for _, _, data in graph.edges(data=True)]
    means = [data.get("mean", 0.0) for _, data in graph.nodes(data=True) if data.get("kind") == "gene"]
    max_attention = max(attentions + means) if attentions or means else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#444"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(graph_title(row))}</text>',
    ]

    for u, v, data in graph.edges(data=True):
        x1, y1 = xy(u)
        x2, y2 = xy(v)
        attention = data.get("attention", 0.0)
        stroke_width = 1.5 + 8.0 * (attention / max_attention if max_attention else 0.0)
        label = f"{data.get('relation', '')} {data.get('attention_name', 'attn')}={attention:.4f}"
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#4a5568" stroke-width="{stroke_width:.2f}" marker-end="url(#arrow)" opacity="0.78"/>'
        )
        parts.append(
            f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 8:.1f}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#111827">{html.escape(label)}</text>'
        )

    for node, data in graph.nodes(data=True):
        x, y = xy(node)
        is_drug = data.get("kind") == "drug"
        fill = "#4C78A8" if is_drug else "#F58518"
        radius = 34 if is_drug else 34 + 14 * (data.get("mean", 0.0) / max_attention if max_attention else 0.0)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="#202020" stroke-width="1.5"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + (0 if not is_drug else 4):.1f}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" font-weight="700" fill="#ffffff">{html.escape(str(node))}</text>'
        )
        if not is_drug:
            parts.append(
                f'<text x="{x:.1f}" y="{y + 15:.1f}" text-anchor="middle" '
                f'font-family="Arial" font-size="10" fill="#fff7ed">mean={data.get("mean", 0.0):.4f}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def bar_html(value, max_value):
    pct = 100.0 * (value / max_value if max_value else 0.0)
    return (
        '<div class="bar"><span style="width:{pct:.2f}%"></span></div>'
        '<span class="value">{value:.6f}</span>'
    ).format(pct=pct, value=value)


def metric_cards(row):
    items = [
        ("score", row.get("score")),
        ("probability", row.get("probability")),
        ("num_paths", row.get("num_paths")),
        ("num_paths_total", row.get("num_paths_total")),
        ("subgraph_nodes", row.get("subgraph_nodes")),
        ("subgraph_edges", row.get("subgraph_edges")),
        ("co_attn_raw_mean", row.get("co_attention_raw_mean")),
        ("co_attn_raw_max", row.get("co_attention_raw_max")),
        ("co_attn_softmax_max", row.get("co_attention_softmax_max")),
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


def gene_table(row, sort_by="mean", top_n=0):
    rows = path_rows(row, sort_by=sort_by, top_n=top_n)
    if not rows:
        return '<p class="empty">No reasoning paths for this row.</p>'
    max_value = max([max(item["head"], item["tail"], item["mean"]) for item in rows] + [1e-12])
    body = []
    for rank, item in enumerate(rows, start=1):
        body.append(
            "<tr>"
            "<td>{rank}</td>"
            '<td><strong>{gene}</strong><div class="muted">source path #{src}</div></td>'
            "<td>{rel_a}</td>"
            "<td>{rel_b}</td>"
            "<td>{head}</td>"
            "<td>{tail}</td>"
            "<td>{mean}</td>"
            "</tr>".format(
                rank=rank,
                gene=html.escape(str(item["gene"])),
                src=item["rank_source_index"],
                rel_a=html.escape(item["relation_a"]),
                rel_b=html.escape(item["relation_b"]),
                head=bar_html(item["head"], max_value),
                tail=bar_html(item["tail"], max_value),
                mean=bar_html(item["mean"], max_value),
            )
        )
    return (
        '<table class="gene-table">'
        "<thead><tr><th>Rank</th><th>Gene</th><th>Drug A relation</th><th>Drug B relation</th><th>Head attn</th><th>Tail attn</th><th>Mean attn</th></tr></thead>"
        "<tbody>{}</tbody></table>"
    ).format("\n".join(body))


def matrix_table(row, field, title):
    matrix = row.get(field)
    if not matrix:
        return ""
    paths = row.get("paths") or []
    col_count = max(len(inner) for inner in matrix if isinstance(inner, list))
    genes = [item.get("gene") or "path_{}".format(idx + 1) for idx, item in enumerate(paths)]
    max_value = max(
        [coerce_float(value) for inner in matrix if isinstance(inner, list) for value in inner] + [1e-12]
    )
    header = ["<th>DSN token</th>"]
    for idx in range(col_count):
        label = genes[idx] if idx < len(genes) else "pad_{}".format(idx + 1)
        header.append("<th>{}</th>".format(html.escape(label)))
    rows = []
    for row_idx, values in enumerate(matrix):
        cells = ["<td>q{}</td>".format(row_idx + 1)]
        for col_idx in range(col_count):
            value = coerce_float(values[col_idx] if col_idx < len(values) else 0.0)
            alpha = 0.12 + 0.78 * (value / max_value if max_value else 0.0)
            cells.append(
                '<td style="background:rgba(14, 165, 233, {alpha:.3f})">{value:.4f}</td>'.format(
                    alpha=alpha, value=value
                )
            )
        rows.append("<tr>{}</tr>".format("".join(cells)))
    return (
        '<section><h2>{}</h2><table class="matrix"><thead><tr>{}</tr></thead><tbody>{}</tbody></table></section>'.format(
            html.escape(title), "".join(header), "".join(rows)
        )
    )


def dashboard_html(row, graph, sort_by="mean", top_n=0):
    matrix_html = "\n".join([
        matrix_table(row, "fusion_head_dsn_to_kg_attention", "Head DSN-to-KG attention matrix"),
        matrix_table(row, "fusion_tail_dsn_to_kg_attention", "Tail DSN-to-KG attention matrix"),
    ])
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f7f8fb; }}
  main {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; margin: 28px 0 12px; }}
  .subtitle {{ color: #64748b; margin: 0 0 18px; }}
  .panel {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
  .metric {{ border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; background: #fbfdff; }}
  .metric span {{ display: block; color: #64748b; font-size: 12px; margin-bottom: 4px; }}
  .metric strong {{ font-size: 15px; word-break: break-word; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 9px 10px; text-align: left; vertical-align: middle; font-size: 13px; }}
  th {{ background: #f1f5f9; color: #334155; font-weight: 700; }}
  .gene-table td:nth-child(5), .gene-table td:nth-child(6), .gene-table td:nth-child(7) {{ min-width: 150px; }}
  .muted {{ color: #64748b; font-size: 12px; margin-top: 3px; }}
  .bar {{ display: inline-block; width: 82px; height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; margin-right: 8px; vertical-align: middle; }}
  .bar span {{ display: block; height: 100%; background: #0ea5e9; }}
  .value {{ font-variant-numeric: tabular-nums; }}
  .matrix th, .matrix td {{ text-align: center; font-variant-numeric: tabular-nums; }}
  .empty {{ color: #64748b; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="subtitle">Sorted by {sort_by} attention. Left graph edge is head attention; right graph edge is tail attention.</p>
  <section class="panel metrics">{cards}</section>
  <section class="panel">{svg}</section>
  <section class="panel">
    <h2>Gene-level path attention</h2>
    {table}
  </section>
  <section class="panel">
    {matrix_html}
  </section>
</main>
</body>
</html>
""".format(
        title=html.escape(graph_title(row)),
        sort_by=html.escape(sort_by),
        cards=metric_cards(row),
        svg=draw_svg(row, graph),
        table=gene_table(row, sort_by=sort_by, top_n=top_n),
        matrix_html=matrix_html or '<p class="empty">No attention matrix fields were found.</p>',
    )


def draw_graph(row, output, sort_by="mean", top_n=0):
    graph = build_graph(row, sort_by=sort_by, top_n=top_n)
    if graph.number_of_edges() == 0:
        raise SystemExit("Selected row has no reasoning paths to draw.")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".png":
        draw_png(row, graph, output)
    elif output.suffix.lower() == ".html":
        output.write_text(dashboard_html(row, graph, sort_by=sort_by, top_n=top_n))
    else:
        svg = draw_svg(row, graph)
        output.write_text(svg)
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Plot reasoning paths with path attention scores.")
    parser.add_argument("reasoning_jsonl", type=Path)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--drug-a")
    parser.add_argument("--drug-b")
    parser.add_argument("--sort-by", choices=["head", "tail", "mean"], default="mean")
    parser.add_argument("--top-n", type=int, default=0, help="Limit graph/table to the top N genes after sorting. 0 means all.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.row_index is None and not (args.drug_a and args.drug_b):
        raise SystemExit("Use --row-index or both --drug-a and --drug-b.")

    row = load_row(args.reasoning_jsonl, args.row_index, args.drug_a, args.drug_b)
    draw_graph(row, args.output, sort_by=args.sort_by, top_n=args.top_n)


if __name__ == "__main__":
    main()
