#!/usr/bin/env python3
"""Build the static HTML report for the COMPAS Section 4.2 reproduction."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: object) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(number) >= 100:
        return f"{number:,.2f}"
    return f"{number:.2f}"


def table_from_rows(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            cls = "num" if key not in {"variant", "metric", "race"} else ""
            cells.append(f'<td class="{cls}">{fmt(row.get(key, ""))}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def matrix_table(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    html_rows = [
        "<tr><th></th>" + "".join(f"<th>{html.escape(c)}</th>" for c in header[1:]) + "</tr>"
    ]
    for row in rows[1:]:
        html_rows.append(
            "<tr>"
            + f"<th>{html.escape(row[0])}</th>"
            + "".join(f'<td class="num">{fmt(cell)}</td>' for cell in row[1:])
            + "</tr>"
        )
    return f'<table class="matrix"><tbody>{"".join(html_rows)}</tbody></table>'


def inline_svg(path: Path) -> str:
    svg = path.read_text(encoding="utf-8")
    return svg.replace("<svg ", '<svg class="report-figure" ', 1)


def main() -> None:
    summary = read_csv(OUT / "summary_metrics.csv")
    comparison = read_csv(OUT / "comparison_with_paper.csv")
    diagnostics = json.loads((OUT / "data_diagnostics.json").read_text(encoding="utf-8"))
    confusion = json.loads((OUT / "compas_confusion_by_race.json").read_text(encoding="utf-8"))
    figure_svg = inline_svg(OUT / "figure3_ab_style.svg")

    comparison_table = table_from_rows(
        comparison,
        [
            ("metric", "Metric"),
            ("paper", "Paper"),
            ("criminal_only_compas_groups", "Criminal only"),
            ("expanded_compas_groups", "Expanded"),
            ("expanded_equal_opportunity_proxy_compas_groups", "EO proxy"),
        ],
    )
    selected_summary = [
        row
        for row in summary
        if row["variant"]
        in {
            "expanded_compas_groups",
            "expanded_equal_opportunity_proxy_compas_groups",
            "criminal_only_compas_groups",
        }
    ]
    summary_table = table_from_rows(
        selected_summary,
        [
            ("variant", "Variant"),
            ("bias_pct_WhiteLow_to_BlackHigh", "Bias WL -> BH"),
            ("bias_pct_BlackHigh_to_WhiteLow", "Bias BH -> WL"),
            ("mass_pct_WhiteLow_to_BlackHigh", "Mass WL -> BH"),
            ("mass_pct_BlackHigh_to_WhiteLow", "Mass BH -> WL"),
        ],
    )
    confusion_rows = [
        {
            "race": race,
            "fpr": values["fpr"],
            "fnr": values["fnr"],
            "tpr": values["tpr"],
        }
        for race, values in confusion.items()
    ]
    confusion_table = table_from_rows(
        confusion_rows,
        [("race", "Race"), ("fpr", "FPR"), ("fnr", "FNR"), ("tpr", "TPR")],
    )

    data_rows = [
        {"item": "Raw rows", "value": diagnostics["raw_rows"]},
        {"item": "Filtered rows", "value": diagnostics["filtered_rows_all_races"]},
        {"item": "Black defendants", "value": diagnostics["black_count_all_races"]},
        {"item": "White defendants", "value": diagnostics["white_count_all_races"]},
        {"item": "Two-year recidivists", "value": diagnostics["two_year_recid_all_races"]},
    ]
    data_table = table_from_rows(data_rows, [("item", ""), ("value", "Count")])

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Section 4.2 — COMPAS Reproduction</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Georgia", serif;
    background: #fafafa;
    color: #1a1a2e;
    line-height: 1.75;
    padding: 0 0 60px;
  }}
  .hero {{
    background: #1a1a2e;
    color: #fff;
    padding: 44px 60px 36px;
  }}
  .hero .label {{
    font-family: monospace;
    font-size: 0.72em;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8899bb;
    margin-bottom: 10px;
  }}
  .hero h1 {{
    font-size: 1.75em;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 10px;
    color: #e8eaf6;
  }}
  .hero .meta {{
    font-size: 0.83em;
    color: #8899bb;
    font-family: sans-serif;
  }}
  .container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 0 36px;
  }}
  h2 {{
    font-size: 1.12em;
    font-weight: 700;
    color: #1a1a2e;
    margin: 44px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1.5px solid #dde;
  }}
  h3 {{
    font-size: 0.97em;
    font-weight: 700;
    color: #333;
    margin: 28px 0 8px;
  }}
  p {{ margin: 10px 0; font-size: 0.97em; }}
  a {{ color: #0f766e; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.9em;
    margin: 16px 0;
  }}
  th, td {{
    border: 1px solid #dde;
    padding: 8px 14px;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #f0f0f8; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  table.matrix {{ min-width: 520px; }}
  .table-wrap {{ overflow-x: auto; }}
  blockquote {{
    border-left: 3px solid #8899bb;
    margin: 18px 0;
    padding: 10px 18px;
    background: #f4f6fb;
    font-size: 0.91em;
    color: #444;
    border-radius: 0 4px 4px 0;
  }}
  blockquote strong {{ color: #1a1a2e; }}
  .fig-wrap {{
    margin: 22px 0;
    text-align: center;
  }}
  .fig-wrap .report-figure {{
    width: 100%;
    max-width: 860px;
    height: auto;
    border: 1px solid #e5e5ef;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    background: #fff;
  }}
  .fig-cap {{
    font-size: 0.82em;
    color: #666;
    font-family: sans-serif;
    margin-top: 8px;
  }}
  pre.code {{
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 6px;
    padding: 18px 22px;
    font-size: 0.83em;
    font-family: "Fira Code", "Courier New", monospace;
    overflow-x: auto;
    line-height: 1.6;
    margin: 14px 0 20px;
    border: 1px solid #313244;
  }}
  code {{
    font-family: "Fira Code", "Courier New", monospace;
    background: #eef1f7;
    border: 1px solid #dde;
    border-radius: 4px;
    padding: 1px 4px;
  }}
  pre.code code {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    color: inherit;
  }}
  .math-block {{
    text-align: center;
    margin: 18px 0;
    font-size: 1.05em;
  }}
  hr {{
    border: none;
    border-top: 1px solid #dde;
    margin: 36px 0;
  }}
  @media (max-width: 640px) {{
    .hero {{ padding: 34px 24px 28px; }}
    .container {{ padding: 0 22px; }}
    table {{ font-size: 0.82em; }}
  }}
</style>
</head>
<body>

<div class="hero">
  <div class="label">Section 4.2</div>
  <h1>COMPAS Experiment: Optimal Transport View</h1>
  <div class="meta">Kwegyir-Aggrey, K., Santorella, R., Brown, S. M. &nbsp;·&nbsp; arXiv:2102.10349v1</div>
</div>

<div class="container">

<h2>Setup</h2>

<p>The experiment compares the ground-truth recidivism policy with COMPAS-style predictions on the ProPublica COMPAS dataset. Scores are binarized as <code>Low</code> vs. <code>Medium/High</code>, then grouped by race and risk label:</p>

<table>
  <tr><th>Object</th><th>Definition used here</th></tr>
  <tr><td><strong>$F_{{true}}$</strong></td><td>Logistic regression score for two-year recidivism.</td></tr>
  <tr><td><strong>$F_{{compas}}$</strong></td><td>Logistic regression score for the binary COMPAS label.</td></tr>
  <tr><td><strong>Groups</strong></td><td><code>WhiteLow</code>, <code>WhiteHigh</code>, <code>BlackLow</code>, <code>BlackHigh</code>.</td></tr>
  <tr><td><strong>OT cost</strong></td><td>$C_{{ij}}=\\left\\|F_{{true}}(x_i)-F_{{pred}}(x_j)\\right\\|_2$.</td></tr>
</table>

<p>For two-class outputs this cost is equivalent to sorting the predicted probabilities and matching quantiles. The script therefore computes the exact one-dimensional equal-mass OT coupling instead of storing a dense linear program.</p>

<hr>

<h2>Data Check</h2>

<p>The filtering follows the standard ProPublica COMPAS analysis. These counts match the paper appendix, so the data slice is likely the same even though the authors did not release code.</p>

<div class="table-wrap">{data_table}</div>

<pre class="code"><code>-30 &lt;= days_b_screening_arrest &lt;= 30
is_recid != -1
c_charge_degree != "O"
score_text != "N/A"</code></pre>

<hr>

<h2>Figure 3A/3B Style Reconstruction</h2>

<p>The alluvial plot below is embedded directly in this HTML file, so GitHub Pages does not need to resolve a separate image path.</p>

<div class="fig-wrap">
{figure_svg}
  <div class="fig-cap">Figure 3-style transport maps. A uses COMPAS predictions; B uses the equal-opportunity proxy classifier. Widths reproduce the qualitative structure, not the paper's exact hidden implementation.</div>
</div>

<blockquote>
<strong>Reading the plot.</strong><br>
The left side is the ground-truth risk group; the right side is the predicted risk group. A large crossing band indicates that individuals in one subgroup are transported toward outcomes typical of another subgroup.
</blockquote>

<hr>

<h2>Equal Opportunity Classifier</h2>

<p>The paper says its third classifier is a logistic regression trained on COMPAS labels with the Zafar et al. equal-opportunity constraint. The exact code, split, threshold, and constraint strength are not given.</p>

<p>Here it is treated as an FNR/TPR constraint: among true recidivists, Black and White predicted positive rates should be close. The proxy used here is</p>

<pre class="code"><code>logistic loss on COMPAS label
+ penalty * (mean_score_black_true_positive - mean_score_white_true_positive)^2</code></pre>

<p>This proxy is enough for Figure 3B-style sensitivity analysis, but not an exact reconstruction of the authors' implementation.</p>

<hr>

<h2>Numerical Check</h2>

<p>The first highlighted paper number is approximately recovered by the criminal-history-only distance variant. The second highlighted number is not; that mismatch is the main evidence that Section 4.2 contains unrecoverable implementation choices.</p>

<div class="table-wrap">{comparison_table}</div>

<h3>Selected Variants</h3>
<div class="table-wrap">{summary_table}</div>

<h3>COMPAS Baseline Error Rates</h3>
<p>Using <code>Low</code> vs. <code>Medium/High</code> directly, the familiar COMPAS asymmetry appears: Black FPR is higher, White FNR is higher.</p>
<div class="table-wrap">{confusion_table}</div>

<hr>

<h2>Transport Matrices</h2>

<h3>Figure 3A mass row percentage</h3>
<div class="table-wrap">{matrix_table(OUT / "figure3A_compas_style_mass_row_pct.csv")}</div>

<h3>Figure 3B mass row percentage</h3>
<div class="table-wrap">{matrix_table(OUT / "figure3B_equal_opportunity_proxy_style_mass_row_pct.csv")}</div>

<hr>

<h2>Re-run</h2>

<pre class="code"><code>python3 -m pip install -r requirements.txt
python3 reproduce_compas_section4_2.py
python3 build_html_report.py</code></pre>

</div>
</body>
</html>
"""

    (ROOT / "report.html").write_text(html_text, encoding="utf-8")
    print(ROOT / "report.html")


if __name__ == "__main__":
    main()
