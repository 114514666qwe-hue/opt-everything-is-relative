#!/usr/bin/env python3
"""Reproduce and audit Section 4.2 of arXiv:2102.10349.

The paper does not release experiment code. This script rebuilds the COMPAS
experiment from public ProPublica data and reports several reasonable
interpretations of the underspecified choices in Section 4.2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib import patheffects
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.rcParams["svg.fonttype"] = "none"

DATA_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)

GROUPS = ["WhiteLow", "WhiteHigh", "BlackLow", "BlackHigh"]
GROUP_COLORS = {
    "WhiteLow": "#009E73",
    "WhiteHigh": "#D55E00",
    "BlackLow": "#0072B2",
    "BlackHigh": "#E69F00",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    columns: tuple[str, ...]
    distance_columns: tuple[str, ...] | None = None
    note: str = ""


FEATURE_SPECS = [
    FeatureSpec(
        name="expanded",
        columns=(
            "sex",
            "age_cat",
            "race",
            "c_charge_degree",
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
        ),
        note="Common COMPAS covariates: demographics, charge degree, and prior/juvenile counts.",
    ),
    FeatureSpec(
        name="paperish",
        columns=(
            "priors_count",
            "age_over_45",
            "age_under_25",
            "female",
            "misdemeanor",
            "race",
        ),
        note="Closest to the appendix feature list and the commented TeX footnote.",
    ),
    FeatureSpec(
        name="criminal_only",
        columns=(
            "c_charge_degree",
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
        ),
        note="Distance and model use only criminal-history-like variables.",
    ),
    FeatureSpec(
        name="paper_model_criminal_distance",
        columns=(
            "priors_count",
            "age_over_45",
            "age_under_25",
            "female",
            "misdemeanor",
            "race",
        ),
        distance_columns=(
            "c_charge_degree",
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
        ),
        note="Paper-like logistic models, but bias distance restricted to criminal history.",
    ),
]


def download_dataset(data_path: Path) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        return
    urlretrieve(DATA_URL, data_path)


def load_filtered_compas(data_path: Path, black_white_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # This is the standard ProPublica analysis filter. It reproduces the paper's
    # appendix counts: 6172 rows, 3175 African-American defendants, 2103 Caucasian
    # defendants, and 2809 defendants with two-year recidivism.
    filtered = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    if black_white_only:
        filtered = filtered[
            filtered["race"].isin(["African-American", "Caucasian"])
        ].copy()

    filtered = filtered.reset_index(drop=True)
    filtered["race_group"] = (
        filtered["race"]
        .map({"African-American": "Black", "Caucasian": "White"})
        .fillna("Other")
    )
    filtered["compas_binary"] = (filtered["score_text"] != "Low").astype(int)
    filtered["compas_risk"] = np.where(filtered["compas_binary"] == 1, "High", "Low")
    filtered["true_risk"] = np.where(filtered["two_year_recid"] == 1, "High", "Low")

    filtered["age_over_45"] = (filtered["age"] > 45).astype(int)
    filtered["age_under_25"] = (filtered["age"] < 25).astype(int)
    filtered["female"] = (filtered["sex"] == "Female").astype(int)
    filtered["misdemeanor"] = (filtered["c_charge_degree"] == "M").astype(int)
    return filtered


def make_design_matrix(
    df: pd.DataFrame, columns: Iterable[str]
) -> tuple[np.ndarray, list[str]]:
    columns = list(columns)
    categorical = [col for col in columns if df[col].dtype == object]
    numeric = [col for col in columns if col not in categorical]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical)
        )

    preprocessor = ColumnTransformer(
        transformers, verbose_feature_names_out=False, remainder="drop"
    )
    matrix = preprocessor.fit_transform(df[columns])
    names = list(preprocessor.get_feature_names_out())
    return np.asarray(matrix, dtype=float), names


def fit_logistic_scores(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(matrix, target)
    return model.predict_proba(matrix)[:, 1]


def fit_equal_opportunity_proxy(
    matrix: np.ndarray,
    compas_target: np.ndarray,
    true_target: np.ndarray,
    race_group: np.ndarray,
    penalty: float = 100.0,
    l2: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train a transparent proxy for the paper's equal-opportunity classifier.

    The paper says it uses a Zafar et al. equal-opportunity constraint but omits
    the implementation and constraint value. Here we optimize logistic loss on
    COMPAS labels plus a penalty on the difference in average score between
    Black and White defendants among true positives. This is not claimed to be
    the paper's exact model; it is an auditable surrogate for Figure 3B.
    """

    n, d = matrix.shape
    design = np.column_stack([np.ones(n), matrix])
    y = compas_target.astype(float)
    true_positive = true_target.astype(int) == 1
    black_pos = true_positive & (race_group == "Black")
    white_pos = true_positive & (race_group == "White")

    standard = LogisticRegression(max_iter=2000, solver="lbfgs")
    standard.fit(matrix, y)
    init = np.concatenate([standard.intercept_, standard.coef_.ravel()])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ theta
        probs = expit(logits)
        # Stable logistic loss: log(1 + exp(z)) - y z.
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        grad = design.T @ (probs - y) / n

        loss += 0.5 * l2 * np.dot(theta[1:], theta[1:])
        grad[1:] += l2 * theta[1:]

        if black_pos.any() and white_pos.any():
            diff = probs[black_pos].mean() - probs[white_pos].mean()
            penalty_loss = penalty * diff * diff
            grad_diff = (
                (design[black_pos].T @ (probs[black_pos] * (1 - probs[black_pos])))
                / black_pos.sum()
                - (design[white_pos].T @ (probs[white_pos] * (1 - probs[white_pos])))
                / white_pos.sum()
            )
            loss += penalty_loss
            grad += 2 * penalty * diff * grad_diff

        return float(loss), grad

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=init,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    theta = result.x
    probs = expit(design @ theta)
    tpr_black, tpr_white = race_tpr(probs >= 0.5, true_target, race_group)
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_fun": float(result.fun),
        "tpr_black": tpr_black,
        "tpr_white": tpr_white,
        "tpr_gap_black_minus_white": tpr_black - tpr_white,
        "penalty": penalty,
        "l2": l2,
    }
    return probs, diagnostics


def race_tpr(pred: np.ndarray, true_target: np.ndarray, race_group: np.ndarray) -> tuple[float, float]:
    values = []
    for race in ["Black", "White"]:
        mask = (race_group == race) & (true_target == 1)
        values.append(float(pred[mask].mean()) if mask.any() else float("nan"))
    return values[0], values[1]


def monotone_ot_edges(source_scores: np.ndarray, target_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse exact OT edges for one-dimensional equal-mass empirical data."""

    n_source = len(source_scores)
    n_target = len(target_scores)
    source_order = np.argsort(source_scores, kind="mergesort")
    target_order = np.argsort(target_scores, kind="mergesort")

    source_i = target_j = 0
    source_remaining = 1.0 / n_source
    target_remaining = 1.0 / n_target
    rows: list[int] = []
    cols: list[int] = []
    masses: list[float] = []

    while source_i < n_source and target_j < n_target:
        mass = min(source_remaining, target_remaining)
        rows.append(int(source_order[source_i]))
        cols.append(int(target_order[target_j]))
        masses.append(float(mass))

        source_remaining -= mass
        target_remaining -= mass

        if source_remaining <= 1e-15:
            source_i += 1
            source_remaining = 1.0 / n_source if source_i < n_source else 0.0
        if target_remaining <= 1e-15:
            target_j += 1
            target_remaining = 1.0 / n_target if target_j < n_target else 0.0

    return np.array(rows), np.array(cols), np.array(masses)


def group_labels(df: pd.DataFrame, risk_column: str) -> np.ndarray:
    return (df["race_group"] + df[risk_column]).to_numpy()


def race_risk_labels(df: pd.DataFrame, risk_values: np.ndarray) -> np.ndarray:
    return np.array(
        [f"{race}{risk}" for race, risk in zip(df["race_group"].to_numpy(), risk_values)]
    )


def decompose_by_group(
    rows: np.ndarray,
    cols: np.ndarray,
    masses: np.ndarray,
    distance_matrix: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, pd.DataFrame]:
    mass_matrix = pd.DataFrame(0.0, index=GROUPS, columns=GROUPS)
    bias_matrix = pd.DataFrame(0.0, index=GROUPS, columns=GROUPS)

    distances = np.linalg.norm(distance_matrix[rows] - distance_matrix[cols], axis=1)
    individual_bias = np.zeros(distance_matrix.shape[0])

    for row, col, mass, distance in zip(rows, cols, masses, distances):
        source_group = source_labels[row]
        target_group = target_labels[col]
        contribution = mass * distance
        individual_bias[row] += contribution
        if source_group in GROUPS and target_group in GROUPS:
            mass_matrix.loc[source_group, target_group] += mass
            bias_matrix.loc[source_group, target_group] += contribution

    with np.errstate(divide="ignore", invalid="ignore"):
        mass_row_pct = mass_matrix.div(mass_matrix.sum(axis=1), axis=0) * 100
        bias_row_pct = bias_matrix.div(bias_matrix.sum(axis=1), axis=0) * 100

    return {
        "mass": mass_matrix.fillna(0.0),
        "mass_row_pct": mass_row_pct.fillna(0.0),
        "bias": bias_matrix.fillna(0.0),
        "bias_row_pct": bias_row_pct.fillna(0.0),
        "individual_bias": pd.DataFrame(
            {
                "group": source_labels,
                "individual_bias_unnormalized": individual_bias,
                "individual_bias_row_normalized": individual_bias * distance_matrix.shape[0],
            }
        ),
    }


def candidate_summary(
    name: str,
    mass_pct: pd.DataFrame,
    bias_pct: pd.DataFrame,
    source_labels: np.ndarray,
    individual_bias: pd.DataFrame,
) -> dict[str, float | str]:
    grouped_bias = (
        individual_bias[individual_bias["group"].isin(GROUPS)]
        .groupby("group")["individual_bias_row_normalized"]
        .mean()
    )
    black_high_vs_low = (
        100 * (grouped_bias.get("BlackHigh", np.nan) / grouped_bias.get("BlackLow", np.nan) - 1)
    )
    white_low_vs_high = (
        100 * (grouped_bias.get("WhiteLow", np.nan) / grouped_bias.get("WhiteHigh", np.nan) - 1)
    )

    counts = pd.Series(source_labels).value_counts()
    return {
        "variant": name,
        "source_group_definition": "COMPAS risk labels",
        "n_white_low_source": int(counts.get("WhiteLow", 0)),
        "n_white_high_source": int(counts.get("WhiteHigh", 0)),
        "n_black_low_source": int(counts.get("BlackLow", 0)),
        "n_black_high_source": int(counts.get("BlackHigh", 0)),
        "mass_pct_WhiteLow_to_BlackHigh": float(mass_pct.loc["WhiteLow", "BlackHigh"]),
        "mass_pct_BlackHigh_to_WhiteLow": float(mass_pct.loc["BlackHigh", "WhiteLow"]),
        "bias_pct_WhiteLow_to_BlackHigh": float(bias_pct.loc["WhiteLow", "BlackHigh"]),
        "bias_pct_BlackHigh_to_WhiteLow": float(bias_pct.loc["BlackHigh", "WhiteLow"]),
        "mean_bias_black_high_vs_black_low_pct": float(black_high_vs_low),
        "mean_bias_white_low_vs_white_high_pct": float(white_low_vs_high),
    }


def save_matrix(matrix: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path)


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=max(70, matrix.to_numpy().max()))
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Target group")
    ax.set_ylabel("Source group")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value > 35 else "black")
    fig.colorbar(image, ax=ax, label="row percentage")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_alluvial_panel(
    ax: plt.Axes,
    flow: pd.DataFrame,
    panel_label: str,
    left_label: str = "Ground Truth",
    right_label: str = "Predicted",
) -> None:
    """Draw a Figure-3-style alluvial panel with four source/target groups."""

    flow = flow.loc[GROUPS, GROUPS].astype(float)
    total = float(flow.to_numpy().sum())
    if total <= 0:
        raise ValueError("flow matrix has no mass")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_x = 0.08
    right_x = 0.88
    node_w = 0.055
    top = 0.88
    usable_h = 0.72
    gap = 0.035
    scale = (usable_h - gap * (len(GROUPS) - 1)) / total

    source_totals = flow.sum(axis=1)
    target_totals = flow.sum(axis=0)

    def node_positions(totals: pd.Series) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        cursor = top
        for group in GROUPS:
            h = float(totals[group]) * scale
            positions[group] = (cursor - h, cursor)
            cursor -= h + gap
        return positions

    source_pos = node_positions(source_totals)
    target_pos = node_positions(target_totals)

    # Stack outgoing and incoming flow segments. Top-to-bottom order is fixed so
    # the visual resembles the paper's four-band alluvial plot.
    source_cursor = {group: source_pos[group][1] for group in GROUPS}
    target_cursor = {group: target_pos[group][1] for group in GROUPS}
    incoming_segments: dict[tuple[str, str], tuple[float, float]] = {}

    for target in GROUPS:
        for source in GROUPS:
            h = float(flow.loc[source, target]) * scale
            y_top = target_cursor[target]
            y_bottom = y_top - h
            incoming_segments[(source, target)] = (y_bottom, y_top)
            target_cursor[target] = y_bottom

    for source in GROUPS:
        for target in GROUPS:
            h = float(flow.loc[source, target]) * scale
            if h <= 0:
                continue
            y0_top = source_cursor[source]
            y0_bottom = y0_top - h
            source_cursor[source] = y0_bottom
            y1_bottom, y1_top = incoming_segments[(source, target)]

            x0 = left_x + node_w
            x1 = right_x
            dx = x1 - x0
            verts = [
                (x0, y0_top),
                (x0 + dx * 0.45, y0_top),
                (x1 - dx * 0.45, y1_top),
                (x1, y1_top),
                (x1, y1_bottom),
                (x1 - dx * 0.45, y1_bottom),
                (x0 + dx * 0.45, y0_bottom),
                (x0, y0_bottom),
                (x0, y0_top),
            ]
            codes = [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            patch = PathPatch(
                MplPath(verts, codes),
                facecolor=GROUP_COLORS[source],
                edgecolor="none",
                alpha=0.98,
                zorder=1,
            )
            ax.add_patch(patch)

    text_effect = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.8)]
    for x, totals, positions, align in [
        (left_x, source_totals, source_pos, "left"),
        (right_x, target_totals, target_pos, "right"),
    ]:
        for group in GROUPS:
            y_bottom, y_top = positions[group]
            rect = Rectangle(
                (x, y_bottom),
                node_w,
                y_top - y_bottom,
                facecolor=GROUP_COLORS[group],
                edgecolor="#333333",
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(rect)
            label_x = x + 0.011 if align == "left" else x + node_w - 0.011
            ax.text(
                label_x,
                (y_top + y_bottom) / 2,
                group,
                ha=align,
                va="center",
                fontsize=9,
                color="#333333",
                path_effects=text_effect,
                zorder=4,
            )

    ax.text(0.01, 0.94, panel_label, fontsize=18, weight="bold", ha="left", va="center")
    ax.text(left_x, 0.055, left_label, fontsize=12, weight="bold", ha="left", va="center")
    ax.text(right_x + node_w, 0.055, right_label, fontsize=12, weight="bold", ha="right", va="center")


def svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def alluvial_panel_svg(
    flow: pd.DataFrame,
    panel_label: str,
    x_offset: float = 0.0,
    panel_width: float = 600.0,
    panel_height: float = 430.0,
    left_label: str = "Ground Truth",
    right_label: str = "Predicted",
) -> str:
    """Return a compact hand-written SVG panel for GitHub-friendly reports."""

    flow = flow.loc[GROUPS, GROUPS].astype(float)
    total = float(flow.to_numpy().sum())
    if total <= 0:
        raise ValueError("flow matrix has no mass")

    left_x = 52.0
    right_x = panel_width - 88.0
    node_w = 30.0
    top = 58.0
    usable_h = 314.0
    gap = 16.0
    scale = (usable_h - gap * (len(GROUPS) - 1)) / total

    source_totals = flow.sum(axis=1)
    target_totals = flow.sum(axis=0)

    def fmt(value: float) -> str:
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def node_positions(totals: pd.Series) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        cursor = top
        for group in GROUPS:
            h = float(totals[group]) * scale
            positions[group] = (cursor, cursor + h)
            cursor += h + gap
        return positions

    source_pos = node_positions(source_totals)
    target_pos = node_positions(target_totals)
    source_cursor = {group: source_pos[group][0] for group in GROUPS}
    target_cursor = {group: target_pos[group][0] for group in GROUPS}
    incoming_segments: dict[tuple[str, str], tuple[float, float]] = {}

    parts = [f'<g transform="translate({fmt(x_offset)},0)">']
    for target in GROUPS:
        for source in GROUPS:
            h = float(flow.loc[source, target]) * scale
            y_top = target_cursor[target]
            y_bottom = y_top + h
            incoming_segments[(source, target)] = (y_top, y_bottom)
            target_cursor[target] = y_bottom

    for source in GROUPS:
        for target in GROUPS:
            h = float(flow.loc[source, target]) * scale
            if h <= 0:
                continue
            y0_top = source_cursor[source]
            y0_bottom = y0_top + h
            source_cursor[source] = y0_bottom
            y1_top, y1_bottom = incoming_segments[(source, target)]

            x0 = left_x + node_w
            x1 = right_x
            dx = x1 - x0
            c0 = x0 + dx * 0.45
            c1 = x1 - dx * 0.45
            path = (
                f"M {fmt(x0)} {fmt(y0_top)} "
                f"C {fmt(c0)} {fmt(y0_top)} {fmt(c1)} {fmt(y1_top)} {fmt(x1)} {fmt(y1_top)} "
                f"L {fmt(x1)} {fmt(y1_bottom)} "
                f"C {fmt(c1)} {fmt(y1_bottom)} {fmt(c0)} {fmt(y0_bottom)} {fmt(x0)} {fmt(y0_bottom)} Z"
            )
            parts.append(
                f'<path d="{path}" fill="{GROUP_COLORS[source]}" opacity="0.98"/>'
            )

    for x, totals, positions, anchor in [
        (left_x, source_totals, source_pos, "start"),
        (right_x, target_totals, target_pos, "end"),
    ]:
        for group in GROUPS:
            y_top, y_bottom = positions[group]
            h = y_bottom - y_top
            parts.append(
                f'<rect x="{fmt(x)}" y="{fmt(y_top)}" width="{fmt(node_w)}" '
                f'height="{fmt(h)}" fill="{GROUP_COLORS[group]}" stroke="#333" stroke-width="1"/>'
            )
            label_x = x + 8 if anchor == "start" else x + node_w - 8
            parts.append(
                f'<text class="group-label" x="{fmt(label_x)}" y="{fmt((y_top + y_bottom) / 2 + 4)}" '
                f'text-anchor="{anchor}">{svg_escape(group)}</text>'
            )

    parts.extend(
        [
            f'<text class="panel-label" x="20" y="36">{svg_escape(panel_label)}</text>',
            f'<text class="axis-label" x="{fmt(left_x)}" y="{fmt(panel_height - 26)}" text-anchor="start">{svg_escape(left_label)}</text>',
            f'<text class="axis-label" x="{fmt(right_x + node_w)}" y="{fmt(panel_height - 26)}" text-anchor="end">{svg_escape(right_label)}</text>',
            "</g>",
        ]
    )
    return "\n".join(parts)


def write_alluvial_svg(path: Path, panels: list[tuple[pd.DataFrame, str]], width: int, height: int) -> None:
    panel_width = width / len(panels)
    body = "\n".join(
        alluvial_panel_svg(flow, label, i * panel_width, panel_width, height)
        for i, (flow, label) in enumerate(panels)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; letter-spacing: 0; }}
  .panel-label {{ font-size: 28px; font-weight: 700; fill: #111; }}
  .axis-label {{ font-size: 18px; font-weight: 700; fill: #111; }}
  .group-label {{ font-size: 14px; fill: #333; paint-order: stroke; stroke: rgba(255,255,255,0.82); stroke-width: 3px; stroke-linejoin: round; }}
</style>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def plot_figure3_ab_style(
    compas_flow: pd.DataFrame,
    fair_flow: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_alluvial_svg(
        output_dir / "figure3_ab_style.svg",
        [(compas_flow, "A."), (fair_flow, "B.")],
        width=1200,
        height=430,
    )
    write_alluvial_svg(
        output_dir / "figure3A_compas_style.svg",
        [(compas_flow, "A.")],
        width=600,
        height=430,
    )
    write_alluvial_svg(
        output_dir / "figure3B_equal_opportunity_proxy_style.svg",
        [(fair_flow, "B.")],
        width=600,
        height=430,
    )



def write_report(
    output_dir: Path,
    data_diagnostics: dict[str, object],
    summary: pd.DataFrame,
    eo_diagnostics: dict[str, float],
) -> None:
    best = summary.loc[summary["variant"] == "criminal_only_compas_groups"].iloc[0]
    expanded = summary.loc[summary["variant"] == "expanded_compas_groups"].iloc[0]
    report = f"""# Section 4.2 COMPAS 复现实验说明

## 任务范围

本文件夹复现并审计论文 `Everything is Relative: Understanding Fairness with
Optimal Transport` 的 Section 4.2。原论文没有公开实验代码，因此这里采用
“公开数据 + 论文方法描述 + 多个合理实现分支”的方式重建实验。

## 数据集与过滤

- 数据来源：`{DATA_URL}`
- 原始行数：{data_diagnostics["raw_rows"]}
- 使用 ProPublica 标准过滤后的行数：{data_diagnostics["filtered_rows_all_races"]}
- 用于本报告矩阵的 Black/White 子集行数：{data_diagnostics["filtered_rows_black_white"]}
- 过滤后 Black defendants：{data_diagnostics["black_count_all_races"]}
- 过滤后 White defendants：{data_diagnostics["white_count_all_races"]}
- 过滤后 two-year recidivists：{data_diagnostics["two_year_recid_all_races"]}

过滤条件为：

1. `-30 <= days_b_screening_arrest <= 30`
2. `is_recid != -1`
3. `c_charge_degree != "O"`
4. `score_text != "N/A"`

这些计数和论文附录一致：6172 总样本、3175 Black、2103 White、2809 个
two-year recidivists。因此数据源和过滤方式基本可以确认。

## 实验思路

论文比较同一批个体上的两个 policy：

1. `F_true`：用 logistic regression 拟合真实 two-year recidivism。
2. `F_compas`：用 logistic regression 拟合 COMPAS 的二元风险标签。

这里把 COMPAS 的 `Low` 作为低风险，把 `Medium/High` 合并为高风险。这是
ProPublica COMPAS 分析里最常见的二分方式。

每个 policy 对个体 `x_i` 输出一个二分类概率向量：

```text
F(x_i) = [1 - p_i, p_i]
```

于是两个 policy outcome 之间的代价为：

```text
C_ij = ||F_true(x_i) - F_compas(x_j)||_2
```

在二分类情形下，这个代价和 `|p_i - p_j|` 只差一个常数因子。因此最优传输
可以用一维分位数匹配精确求解：把 `F_true` 的概率排序，把 `F_compas` 的概率
排序，然后按经验分布质量逐段匹配。这个结果与离散 OT 线性规划的最优代价一致，
但避免构造巨大的 dense coupling matrix。

得到 coupling `pi` 后，将每条传输边按 race 和 COMPAS 二元风险组聚合：

```text
WhiteLow, WhiteHigh, BlackLow, BlackHigh
```

脚本同时输出两类矩阵：

- `mass_row_pct`：只看 transport mass 的行归一化占比。
- `bias_row_pct`：看 `transport_mass * feature_distance` 的行归一化占比。

第二种更接近论文的 individual/group bias 定义。离散形式下，对 source 个体
`a_i` 的 individual bias 可写作：

```text
u(a_i) = n * sum_j pi_ij d(x_i, x_j)
```

其中 `n * pi_ij` 是第 `i` 行 coupling 的条件分布质量，`d` 是特征空间距离。
group-wise bias decomposition 则把 `u(a_i)` 按 target subgroup 分解。

## 主要结果

最接近论文 Figure 3A 第一处高亮数字的是
`criminal_only_compas_groups`。该分支用刑事历史相关变量训练 logistic model，
并用同一类变量计算 feature distance：

| 指标 | 原文 | 本复现 |
| --- | ---: | ---: |
| `WhiteLow -> BlackHigh` bias share | 43.8% | {best["bias_pct_WhiteLow_to_BlackHigh"]:.2f}% |
| `BlackHigh -> WhiteLow` bias share | 48.3% | {best["bias_pct_BlackHigh_to_WhiteLow"]:.2f}% |

这个分支能较好恢复“WhiteLow 与 BlackHigh 存在显著交叉映射/偏差贡献”的结构，
但不能恢复原文第二个 `BlackHigh -> WhiteLow = 48.3%` 数字。

较常规的 expanded feature 分支结果为：

| 指标 | 原文 | expanded 复现 |
| --- | ---: | ---: |
| `WhiteLow -> BlackHigh` bias share | 43.8% | {expanded["bias_pct_WhiteLow_to_BlackHigh"]:.2f}% |
| `BlackHigh -> WhiteLow` bias share | 48.3% | {expanded["bias_pct_BlackHigh_to_WhiteLow"]:.2f}% |

因此，Section 4.2 的定性结构可以复现，但 Figure 3A 的两个具体百分比不能在
论文给出的信息下唯一复现。

## Equal Opportunity 代理实验

论文还展示了 Figure 3B：在 COMPAS 标签上训练第三个 logistic regression，
并加入 Zafar et al. 的 equal opportunity 约束。论文没有给出约束参数、优化器、
特征矩阵和 train/test split，因此这里没有声称精确复现 Figure 3B。

为了观察趋势，脚本实现了一个透明代理模型：在 COMPAS label 的 logistic loss
之外，加入一个 penalty，使真实 recidivists 中 Black 和 White 的平均预测分数更接近。

代理模型诊断：

- Black TPR：{eo_diagnostics["tpr_black"]:.3f}
- White TPR：{eo_diagnostics["tpr_white"]:.3f}
- TPR gap Black - White：{eo_diagnostics["tpr_gap_black_minus_white"]:.3f}
- 优化是否成功：{eo_diagnostics["optimizer_success"]}

这个代理模型只用于敏感性分析，不等同于原文的 Zafar 约束实现。

## 与原文的关键差异

1. 原文未公开代码、随机种子和完整参数。
2. 原文没有明确 logistic regression 的完整 feature matrix。
3. 原文没有明确 Figure 3A 的宽度是 raw transport mass，还是
   `mass * feature distance` 的 group-wise bias。
4. 原文没有明确 `Medium` COMPAS score 如何二分；本复现采用 `Low` vs
   `Medium/High`。
5. 原文没有明确是否先用全部 race 训练模型，再只画 Black/White，还是直接只用
   Black/White 子集。主报告采用 Black/White 子集。
6. 原文没有给出 equal opportunity classifier 的约束强度，导致 Figure 3B 无法
   精确复现。

结论：数据和 OT 框架可以复现，Section 4.2 的“跨 race/风险组的结构性偏差”
可以定性复现；但原文 Figure 3A/3B 的精确百分比无法只凭论文唯一推出。

## 输出文件

- `data/compas-scores-two-years.csv`：下载的 ProPublica 数据。
- `outputs/summary_metrics.csv`：所有分支的核心指标。
- `outputs/comparison_with_paper.csv`：原文数字与复现数字对比。
- `outputs/*_bias_row_pct.csv`：group-wise bias decomposition 矩阵。
- `outputs/*_mass_row_pct.csv`：raw transport mass decomposition 矩阵。
- `outputs/*_bias_heatmap.svg`：主要 bias decomposition 热力图。

## 复运行命令

```bash
/Users/zhangyongxiu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \\
  section4_2_compas_reproduction/reproduce_compas_section4_2.py
```
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    data_dir = root / "data"
    output_dir = root / "outputs"
    data_path = data_dir / "compas-scores-two-years.csv"
    download_dataset(data_path)

    raw = pd.read_csv(data_path)
    all_filtered = load_filtered_compas(data_path, black_white_only=False)
    df = load_filtered_compas(data_path, black_white_only=True)

    data_diagnostics = {
        "raw_rows": int(raw.shape[0]),
        "filtered_rows_all_races": int(all_filtered.shape[0]),
        "filtered_rows_black_white": int(df.shape[0]),
        "black_count_all_races": int((all_filtered["race"] == "African-American").sum()),
        "white_count_all_races": int((all_filtered["race"] == "Caucasian").sum()),
        "two_year_recid_all_races": int(all_filtered["two_year_recid"].sum()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_diagnostics.json").write_text(
        json.dumps(data_diagnostics, indent=2), encoding="utf-8"
    )

    summaries = []
    eo_diagnostics: dict[str, float] = {}
    compas_alluvial_flow: pd.DataFrame | None = None
    fair_alluvial_flow: pd.DataFrame | None = None

    for spec in FEATURE_SPECS:
        model_matrix, _ = make_design_matrix(df, spec.columns)
        distance_columns = spec.distance_columns or spec.columns
        distance_matrix, _ = make_design_matrix(df, distance_columns)

        true_scores = fit_logistic_scores(model_matrix, df["two_year_recid"].to_numpy())
        compas_scores = fit_logistic_scores(model_matrix, df["compas_binary"].to_numpy())

        rows, cols, masses = monotone_ot_edges(true_scores, compas_scores)
        source_labels = group_labels(df, "compas_risk")
        target_labels = group_labels(df, "compas_risk")
        result = decompose_by_group(
            rows, cols, masses, distance_matrix, source_labels, target_labels
        )

        variant = f"{spec.name}_compas_groups"
        save_matrix(result["mass_row_pct"], output_dir / f"{variant}_mass_row_pct.csv")
        save_matrix(result["bias_row_pct"], output_dir / f"{variant}_bias_row_pct.csv")
        save_matrix(result["mass"], output_dir / f"{variant}_mass.csv")
        save_matrix(result["bias"], output_dir / f"{variant}_bias.csv")
        plot_heatmap(
            result["bias_row_pct"],
            f"{variant}: bias contribution row %",
            output_dir / f"{variant}_bias_heatmap.svg",
        )
        summaries.append(
            candidate_summary(
                variant,
                result["mass_row_pct"],
                result["bias_row_pct"],
                source_labels,
                result["individual_bias"],
            )
        )
        if spec.name == "expanded":
            true_source_labels = group_labels(df, "true_risk")
            compas_target_labels = group_labels(df, "compas_risk")
            compas_style_result = decompose_by_group(
                rows,
                cols,
                masses,
                distance_matrix,
                true_source_labels,
                compas_target_labels,
            )
            compas_alluvial_flow = compas_style_result["mass"]
            save_matrix(
                compas_style_result["mass_row_pct"],
                output_dir / "figure3A_compas_style_mass_row_pct.csv",
            )

        if spec.name == "expanded":
            eo_scores, eo_diagnostics = fit_equal_opportunity_proxy(
                model_matrix,
                df["compas_binary"].to_numpy(),
                df["two_year_recid"].to_numpy(),
                df["race_group"].to_numpy(),
                penalty=args.eo_penalty,
            )
            eo_rows, eo_cols, eo_masses = monotone_ot_edges(true_scores, eo_scores)
            eo_result = decompose_by_group(
                eo_rows, eo_cols, eo_masses, distance_matrix, source_labels, target_labels
            )
            eo_variant = "expanded_equal_opportunity_proxy_compas_groups"
            save_matrix(eo_result["mass_row_pct"], output_dir / f"{eo_variant}_mass_row_pct.csv")
            save_matrix(eo_result["mass"], output_dir / f"{eo_variant}_mass.csv")
            save_matrix(eo_result["bias_row_pct"], output_dir / f"{eo_variant}_bias_row_pct.csv")
            save_matrix(eo_result["bias"], output_dir / f"{eo_variant}_bias.csv")
            plot_heatmap(
                eo_result["bias_row_pct"],
                f"{eo_variant}: bias contribution row %",
                output_dir / f"{eo_variant}_bias_heatmap.svg",
            )
            fair_risk = np.where(eo_scores >= 0.5, "High", "Low")
            fair_target_labels = race_risk_labels(df, fair_risk)
            fair_style_result = decompose_by_group(
                eo_rows,
                eo_cols,
                eo_masses,
                distance_matrix,
                true_source_labels,
                fair_target_labels,
            )
            fair_alluvial_flow = fair_style_result["mass"]
            save_matrix(
                fair_style_result["mass_row_pct"],
                output_dir / "figure3B_equal_opportunity_proxy_style_mass_row_pct.csv",
            )
            summaries.append(
                candidate_summary(
                    eo_variant,
                    eo_result["mass_row_pct"],
                    eo_result["bias_row_pct"],
                    source_labels,
                    eo_result["individual_bias"],
                )
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    if compas_alluvial_flow is not None and fair_alluvial_flow is not None:
        plot_figure3_ab_style(compas_alluvial_flow, fair_alluvial_flow, output_dir)
    comparison = pd.DataFrame(
        [
            {
                "metric": "WhiteLow_to_BlackHigh_bias_share",
                "paper": 43.8,
                **{
                    row["variant"]: row["bias_pct_WhiteLow_to_BlackHigh"]
                    for _, row in summary.iterrows()
                },
            },
            {
                "metric": "BlackHigh_to_WhiteLow_bias_share",
                "paper": 48.3,
                **{
                    row["variant"]: row["bias_pct_BlackHigh_to_WhiteLow"]
                    for _, row in summary.iterrows()
                },
            },
            {
                "metric": "BlackHigh_vs_BlackLow_mean_bias_pct",
                "paper": 5.88,
                **{
                    row["variant"]: row["mean_bias_black_high_vs_black_low_pct"]
                    for _, row in summary.iterrows()
                },
            },
            {
                "metric": "WhiteLow_vs_WhiteHigh_mean_bias_pct",
                "paper": 4.92,
                **{
                    row["variant"]: row["mean_bias_white_low_vs_white_high_pct"]
                    for _, row in summary.iterrows()
                },
            },
        ]
    )
    comparison.to_csv(output_dir / "comparison_with_paper.csv", index=False)

    # Baseline confusion data for context.
    baseline_pred = df["compas_binary"].to_numpy()
    true = df["two_year_recid"].to_numpy()
    confusion = {}
    for race in ["Black", "White"]:
        mask = df["race_group"].to_numpy() == race
        tn, fp, fn, tp = confusion_matrix(true[mask], baseline_pred[mask], labels=[0, 1]).ravel()
        confusion[race] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": float(fp / (fp + tn)),
            "fnr": float(fn / (fn + tp)),
            "tpr": float(tp / (tp + fn)),
        }
    (output_dir / "compas_confusion_by_race.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )

    write_report(root, data_diagnostics, summary, eo_diagnostics)
    print(f"Wrote outputs to {root.resolve()}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        help="Directory for data, outputs, and report. Defaults to this repository folder.",
    )
    parser.add_argument(
        "--eo-penalty",
        type=float,
        default=100.0,
        help="Penalty strength for the equal-opportunity proxy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
