#!/usr/bin/env python3
"""Build processed benchmark artifacts (plots/tables/summary) from raw outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPEED_METHODS = [
    ("Lott", "Lott triangulation", "Lott"),
    (
        "Lott cert+fallback",
        "Lott certified+fallback triangulation",
        "Lott certified+fallback",
    ),
    ("Hartley-Sturm", "Hartley-Sturm triangulation", "HS"),
    ("Lindstrom niter1", "Lindstrom niter1 triangulation", "Lindstrom niter1"),
    ("Lindstrom niter2", "Lindstrom niter2 triangulation", "Lindstrom niter2"),
    ("Kanatani", "Kanatani triangulation", "Kanatani"),
]

FIG_DPI = 220
FIG_FONT = "DejaVu Serif"

COLOR = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "gray": "#4d4d4d",
    "light_gray": "#d9d9d9",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FIG_FONT,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8, length=4, color=COLOR["gray"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def parse_number(text: str):
    token = text.strip()
    if token == "":
        return token
    if token.lower() in {"nan", "+nan", "-nan"}:
        return float("nan")
    int_match = re.fullmatch(r"[+-]?\d+", token)
    if int_match:
        try:
            return int(token)
        except ValueError:
            pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_equals_tokens(line: str) -> Dict[str, object]:
    out = {}
    for tok in line.split():
        if "=" not in tok:
            continue
        key, value = tok.split("=", 1)
        out[key.strip()] = parse_number(value)
    return out


def parse_speed_summary(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        colon = re.match(r"^([^:]+):\s*(.+)$", line)
        if colon:
            data[colon.group(1).strip()] = parse_number(colon.group(2).strip())
            continue
        data.update(parse_equals_tokens(line))
    return data


def parse_correctness_summary(path: Path) -> List[Dict[str, object]]:
    suites: List[Dict[str, object]] = []
    current: Dict[str, object] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            if current:
                suites.append(current)
                current = {}
            continue
        if line.startswith("suite="):
            if current:
                suites.append(current)
                current = {}
            current["suite"] = line.split("=", 1)[1]
            continue
        current.update(parse_equals_tokens(line))
    if current:
        suites.append(current)
    return suites


def parse_approximation_summary(path: Path) -> List[Dict[str, object]]:
    methods: List[Dict[str, object]] = []
    current: Dict[str, object] = {}
    in_method_block = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            if in_method_block and current.get("method"):
                methods.append(current)
                current = {}
                in_method_block = False
            continue
        if line.startswith("method="):
            if in_method_block and current.get("method"):
                methods.append(current)
            current = {"method": line.split("=", 1)[1]}
            in_method_block = True
            continue
        if "=" in line and in_method_block:
            current.update(parse_equals_tokens(line))
    if in_method_block and current.get("method"):
        methods.append(current)
    return methods


def parse_scaling_summary(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    fit: Dict[str, object] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = parse_equals_tokens(line)
        if not tokens:
            continue
        if "npts" in tokens and "total_ns_mean" in tokens and "ns_per_pt_mean" in tokens:
            rows.append(tokens)
            continue
        if any(str(k).startswith("fit_") for k in tokens.keys()) or any(
            str(k).startswith("ns_per_pt_") for k in tokens.keys()
        ):
            fit.update(tokens)
    rows.sort(key=lambda r: as_float(r.get("npts")))
    return rows, fit


def parse_point_csv(path: Path):
    suites: Dict[str, Dict[str, np.ndarray]] = {}
    total_rows = 0
    valid_rows = 0
    accumulator: Dict[str, Dict[str, List[float]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            suite = row["suite"]
            lott_finite = row.get("lott_finite", "0") == "1"
            hs_finite = row.get("hs_finite", "0") == "1"
            if not (lott_finite and hs_finite):
                continue
            valid_rows += 1
            if suite not in accumulator:
                accumulator[suite] = {"deltaE": [], "E_ours": [], "E_hs": []}
            accumulator[suite]["deltaE"].append(float(row["deltaE"]))
            accumulator[suite]["E_ours"].append(float(row["E_ours"]))
            accumulator[suite]["E_hs"].append(float(row["E_hs"]))
    for suite, values in accumulator.items():
        suites[suite] = {k: np.asarray(v) for k, v in values.items()}
    return suites, total_rows, valid_rows


def as_float(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt_scientific(value) -> str:
    v = as_float(value)
    if math.isnan(v):
        return "--"
    return f"{v:.3e}"


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def latest_stamp(raw_dir: Path) -> str:
    matches = sorted(raw_dir.glob("*_bench_correctness.txt"))
    if not matches:
        raise FileNotFoundError(f"No *_bench_correctness.txt files found in {raw_dir}")
    latest = matches[-1].name
    return latest[: -len("_bench_correctness.txt")]


def speed_rows_from_summary(speed_summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method_label, speed_prefix, residual_suffix in SPEED_METHODS:
        mean = as_float(speed_summary.get(f"{speed_prefix} ns/pt mean"))
        if math.isnan(mean):
            continue
        rows.append(
            {
                "method": method_label,
                "ns_per_pt_mean": mean,
                "ns_per_pt_std": as_float(speed_summary.get(f"{speed_prefix} ns/pt std")),
                "ns_per_pt_ci95": as_float(speed_summary.get(f"{speed_prefix} ns/pt ci95")),
                "mean_abs_epipolar": as_float(
                    speed_summary.get(f"Mean |x'Fx| {residual_suffix}")
                ),
                "finite_ratio": as_float(speed_summary.get(f"Finite ratio {residual_suffix}")),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_runtime(speed_rows: List[Dict[str, object]], out_path: Path) -> None:
    labels = [row["method"] for row in speed_rows]
    means = [as_float(row["ns_per_pt_mean"]) for row in speed_rows]
    ci95 = [as_float(row["ns_per_pt_ci95"]) for row in speed_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    palette = [
        COLOR["blue"],
        COLOR["orange"],
        COLOR["green"],
        COLOR["purple"],
        COLOR["red"],
        COLOR["brown"],
    ]
    ax.bar(x, means, yerr=ci95, capsize=5, color=palette[: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("ns / point")
    ax.set_title("Triangulation Runtime (Mean +/- 95% CI)")
    style_axes(ax, grid_axis="y")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_delta_hist(point_data: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    if not point_data:
        return
    delta = np.concatenate([d["deltaE"] for d in point_data.values()])
    log_abs = np.log10(np.abs(delta) + 1e-30)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(log_abs, bins=120, color=COLOR["blue"], alpha=0.9)
    ax.set_xlabel(r"log10(|DeltaE| + 1e-30)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Objective Gap Magnitude")
    style_axes(ax, grid_axis="y")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_cost_scatter(point_data: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    if not point_data:
        return
    e_ours = np.concatenate([d["E_ours"] for d in point_data.values()])
    e_hs = np.concatenate([d["E_hs"] for d in point_data.values()])
    eps = 1e-30
    x = np.maximum(e_hs, eps)
    y = np.maximum(e_ours, eps)
    if x.size > 30000:
        rng = np.random.default_rng(12345)
        idx = rng.choice(x.size, size=30000, replace=False)
        x = x[idx]
        y = y[idx]
    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.loglog(x, y, ".", markersize=1.5, alpha=0.15, color=COLOR["blue"])
    ax.loglog([low, high], [low, high], "--", color=COLOR["gray"], linewidth=1.0)
    ax.set_xlabel("E_HS")
    ax.set_ylabel("E_ours")
    ax.set_title("Objective Agreement Scatter")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    style_axes(ax, grid_axis="both")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_suite_boxplot(point_data: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    if not point_data:
        return
    suites = sorted(point_data.keys())
    values = [np.log10(np.abs(point_data[s]["deltaE"]) + 1e-30) for s in suites]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bp = ax.boxplot(values, labels=suites, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e6eef7")
        patch.set_edgecolor(COLOR["blue"])
    for median in bp["medians"]:
        median.set_color(COLOR["red"])
        median.set_linewidth(1.3)
    ax.set_ylabel(r"log10(|DeltaE| + 1e-30)")
    ax.set_title("Objective Gap by Suite")
    style_axes(ax, grid_axis="y")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_approximation_pareto(
    approx_rows: List[Dict[str, object]], out_path: Path
) -> None:
    if not approx_rows:
        return
    x = np.asarray([as_float(r.get("speed_ns_per_pt_mean")) for r in approx_rows], dtype=float)
    y = np.asarray([as_float(r.get("deltaE_abs_p95")) for r in approx_rows], dtype=float)
    labels = [str(r.get("method", "unknown")) for r in approx_rows]

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y >= 0.0)
    if not np.any(valid):
        return

    xv = x[valid]
    yv = np.maximum(y[valid], 1e-30)
    labels_v = [labels[i] for i in range(len(labels)) if valid[i]]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.loglog(xv, yv, "o", color=COLOR["blue"], markersize=6)
    for idx, (xi, yi, lab) in enumerate(zip(xv, yv, labels_v)):
        yoff = 4 if idx % 2 == 0 else -12
        ax.annotate(
            lab, (xi, yi), textcoords="offset points", xytext=(5, yoff), fontsize=8
        )
    ax.set_xlabel("Runtime (ns/point)")
    ax.set_ylabel(r"|$\Delta E$| p95 vs Lott Full")
    ax.set_title("Approximation Ladder Pareto")
    style_axes(ax, grid_axis="both")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_scaling_total_ns(
    scaling_rows: List[Dict[str, object]], scaling_fit: Dict[str, object], out_path: Path
) -> None:
    if not scaling_rows:
        return
    x = np.asarray([as_float(r.get("npts")) for r in scaling_rows], dtype=float)
    y = np.asarray([as_float(r.get("total_ns_mean")) for r in scaling_rows], dtype=float)
    yerr = np.asarray([as_float(r.get("total_ns_ci95")) for r in scaling_rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (x > 0.0) & (y > 0.0)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = y[valid]
    yerrv = yerr[valid]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.errorbar(xv, yv, yerr=yerrv, fmt="o", capsize=4, color=COLOR["blue"])

    slope = as_float(scaling_fit.get("fit_slope_ns_per_pt"))
    intercept = as_float(scaling_fit.get("fit_intercept_ns"))
    if np.isfinite(slope) and np.isfinite(intercept):
        xfit = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
        yfit = slope * xfit + intercept
        ax.plot(xfit, yfit, "-", color=COLOR["red"], linewidth=1.6)

    ax.set_xlabel("Correspondences per batch")
    ax.set_ylabel("Total runtime (ns)")
    ax.set_title("Lott Scaling: Total Runtime vs Batch Size")
    style_axes(ax, grid_axis="both")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_scaling_ns_per_pt(scaling_rows: List[Dict[str, object]], out_path: Path) -> None:
    if not scaling_rows:
        return
    x = np.asarray([as_float(r.get("npts")) for r in scaling_rows], dtype=float)
    y = np.asarray([as_float(r.get("ns_per_pt_mean")) for r in scaling_rows], dtype=float)
    yerr = np.asarray([as_float(r.get("ns_per_pt_ci95")) for r in scaling_rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (x > 0.0) & (y > 0.0)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = y[valid]
    yerrv = yerr[valid]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.errorbar(xv, yv, yerr=yerrv, fmt="o-", capsize=4, color=COLOR["green"])
    ax.set_xlabel("Correspondences per batch")
    ax.set_ylabel("Runtime (ns/point)")
    ax.set_title("Lott Scaling: Per-Point Runtime Stability")
    style_axes(ax, grid_axis="both")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def write_speed_table_tex(path: Path, speed_rows: List[Dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by scripts/build_figures.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Mean ns/pt & 95\\% CI & Mean $|x'^T F x|$ & Finite ratio \\\\",
        "\\midrule",
    ]
    for row in speed_rows:
        lines.append(
            f"{latex_escape(row['method'])} & {fmt_scientific(row['ns_per_pt_mean'])} & "
            f"{fmt_scientific(row['ns_per_pt_ci95'])} & "
            f"{fmt_scientific(row['mean_abs_epipolar'])} & "
            f"{as_float(row['finite_ratio']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_correctness_table_tex(path: Path, suites: List[Dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by scripts/build_figures.py",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Suite & Mean $\\Delta E$ & Worst $\\Delta E$ & $\\Delta E_{95}$ & NaN count & Max-step roots & Cert fallback rate \\\\",
        "\\midrule",
    ]
    for s in suites:
        lines.append(
            f"{latex_escape(s.get('suite','unknown'))} & "
            f"{fmt_scientific(s.get('mean_cost_gap_Eours_minus_Ehs'))} & "
            f"{fmt_scientific(s.get('worst_cost_gap_Eours_minus_Ehs'))} & "
            f"{fmt_scientific(s.get('gap_p95'))} & "
            f"{int(as_float(s.get('nan_count', 0)))} & "
            f"{int(as_float(s.get('solver_roots_max_steps', 0)))} & "
            f"{fmt_scientific(s.get('solver_cert_nonunique_or_failure_rate'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_approximation_table_tex(path: Path, approx_rows: List[Dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by scripts/build_figures.py",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Mean ns/pt & 95\\% CI & $|\\Delta E|_{95}$ & $|\\Delta E|_{99}$ & Mean $|x'^T F x|$ \\\\",
        "\\midrule",
    ]
    for r in approx_rows:
        lines.append(
            f"{latex_escape(r.get('method','unknown'))} & "
            f"{fmt_scientific(r.get('speed_ns_per_pt_mean'))} & "
            f"{fmt_scientific(r.get('speed_ns_per_pt_ci95'))} & "
            f"{fmt_scientific(r.get('deltaE_abs_p95'))} & "
            f"{fmt_scientific(r.get('deltaE_abs_p99'))} & "
            f"{fmt_scientific(r.get('speed_mean_abs_epi'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_scaling_table_tex(path: Path, scaling_rows: List[Dict[str, object]], scaling_fit: Dict[str, object]) -> None:
    lines = [
        "% Auto-generated by scripts/build_figures.py",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "Batch size & Mean total ns & Mean ns/pt & 95\\% CI (ns/pt) \\\\",
        "\\midrule",
    ]
    for r in scaling_rows:
        lines.append(
            f"{int(as_float(r.get('npts')))} & "
            f"{fmt_scientific(r.get('total_ns_mean'))} & "
            f"{fmt_scientific(r.get('ns_per_pt_mean'))} & "
            f"{fmt_scientific(r.get('ns_per_pt_ci95'))} \\\\"
        )

    fit_r2 = as_float(scaling_fit.get("fit_r2"))
    fit_slope = as_float(scaling_fit.get("fit_slope_ns_per_pt"))
    if np.isfinite(fit_r2) and np.isfinite(fit_slope):
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{4}}{{l}}{{Linear fit total-ns slope={fit_slope:.3f} ns/pt, $R^2$={fit_r2:.5f}}} \\\\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_markdown_summary(
    path: Path,
    stamp: str,
    speed_path: Path,
    scaling_path: Path | None,
    corr_path: Path,
    approx_path: Path | None,
    points_path: Path | None,
    speed_rows: List[Dict[str, object]],
    scaling_rows: List[Dict[str, object]],
    scaling_fit: Dict[str, object],
    suites: List[Dict[str, object]],
    approx_rows: List[Dict[str, object]],
    total_rows: int,
    valid_rows: int,
) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"Stamp: `{stamp}`",
        f"Speed source: `{speed_path}`",
    ]
    if scaling_path is not None and scaling_path.exists():
        lines.append(f"Scaling source: `{scaling_path}`")
    lines.extend(
        [
        f"Correctness source: `{corr_path}`",
        ]
    )
    if approx_path is not None and approx_path.exists():
        lines.append(f"Approximation source: `{approx_path}`")
    if points_path is not None and points_path.exists():
        lines.append(f"Per-point source: `{points_path}`")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Method | mean ns/pt | ci95 ns/pt | mean |x'Fx| | finite ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in speed_rows:
        lines.append(
            f"| {row['method']} | {fmt_scientific(row['ns_per_pt_mean'])} | "
            f"{fmt_scientific(row['ns_per_pt_ci95'])} | "
            f"{fmt_scientific(row['mean_abs_epipolar'])} | "
            f"{as_float(row['finite_ratio']):.4f} |"
        )

    if scaling_rows:
        lines.extend(
            [
                "",
                "## Complexity Scaling",
                "",
                "| Batch size | mean total ns | mean ns/pt | ci95 ns/pt |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for r in scaling_rows:
            lines.append(
                f"| {int(as_float(r.get('npts')))} | "
                f"{fmt_scientific(r.get('total_ns_mean'))} | "
                f"{fmt_scientific(r.get('ns_per_pt_mean'))} | "
                f"{fmt_scientific(r.get('ns_per_pt_ci95'))} |"
            )

        fit_slope = as_float(scaling_fit.get("fit_slope_ns_per_pt"))
        fit_r2 = as_float(scaling_fit.get("fit_r2"))
        ns_ratio = as_float(scaling_fit.get("ns_per_pt_ratio_max_over_min"))
        lines.extend(
            [
                "",
                f"- Linear fit slope (total ns vs npts): `{fit_slope:.6g}` ns/pt",
                f"- Linear fit $R^2$: `{fit_r2:.6g}`",
                f"- Per-point max/min ratio across sizes: `{ns_ratio:.6g}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "| Suite | mean DeltaE | worst DeltaE | gap p95 | nan_count | valid_points | max_steps | unbracketed | epipole r95 | epipoles inside | c ratio p95 | c hits | c misses | solver c-near-zero | solver c-near-zero non-x | cert points | cert one-root | cert non-1/failure | cert non-1 rate | cert failures | cert ivt conflicts | cert left-endpoint roots | cert longdouble rescues |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in suites:
        lines.append(
            f"| {s.get('suite','unknown')} | "
            f"{fmt_scientific(s.get('mean_cost_gap_Eours_minus_Ehs'))} | "
            f"{fmt_scientific(s.get('worst_cost_gap_Eours_minus_Ehs'))} | "
            f"{fmt_scientific(s.get('gap_p95'))} | "
            f"{int(as_float(s.get('nan_count', 0)))} | "
            f"{int(as_float(s.get('valid_points', 0)))} | "
            f"{int(as_float(s.get('solver_roots_max_steps', 0)))} | "
            f"{int(as_float(s.get('solver_roots_unbracketed', 0)))} | "
            f"{fmt_scientific(s.get('epipole_max_radius_p95'))} | "
            f"{int(as_float(s.get('epipoles_inside_unit_focal', 0)))} | "
            f"{fmt_scientific(s.get('c_ratio_p95'))} | "
            f"{int(as_float(s.get('c_target_hits', 0)))} | "
            f"{int(as_float(s.get('c_target_misses', 0)))} | "
            f"{int(as_float(s.get('solver_c_near_zero_points', 0)))} | "
            f"{int(as_float(s.get('solver_c_near_zero_non_x_points', 0)))} | "
            f"{int(as_float(s.get('solver_cert_points', 0)))} | "
            f"{int(as_float(s.get('solver_cert_rootcount_eq1', 0)))} | "
            f"{int(as_float(s.get('solver_cert_nonunique_or_failure', 0)))} | "
            f"{fmt_scientific(s.get('solver_cert_nonunique_or_failure_rate'))} | "
            f"{int(as_float(s.get('solver_cert_failures', 0)))} | "
            f"{int(as_float(s.get('solver_cert_ivt_conflict', 0)))} | "
            f"{int(as_float(s.get('solver_cert_endpoint_root_left', 0)))} | "
            f"{int(as_float(s.get('solver_cert_longdouble_rescues', 0)))} |"
        )

    if approx_rows:
        lines.extend(
            [
                "",
                "## Approximation Ladder",
                "",
                "| Method | mean ns/pt | ci95 ns/pt | |DeltaE| p95 vs full | |DeltaE| p99 vs full | mean |x'Fx| |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in approx_rows:
            lines.append(
                f"| {r.get('method','unknown')} | "
                f"{fmt_scientific(r.get('speed_ns_per_pt_mean'))} | "
                f"{fmt_scientific(r.get('speed_ns_per_pt_ci95'))} | "
                f"{fmt_scientific(r.get('deltaE_abs_p95'))} | "
                f"{fmt_scientific(r.get('deltaE_abs_p99'))} | "
                f"{fmt_scientific(r.get('speed_mean_abs_epi'))} |"
            )

    if total_rows > 0:
        lines.extend(
            [
                "",
                "## Per-point Data",
                "",
                f"- Total rows: `{total_rows}`",
                f"- Valid rows: `{valid_rows}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- `figures/{stamp}_runtime_ci95.png`",
            f"- `figures/{stamp}_deltaE_logabs_hist.png`",
            f"- `figures/{stamp}_Eours_vs_Ehs_scatter.png`",
            f"- `figures/{stamp}_deltaE_by_suite_boxplot.png`",
        ]
    )
    if approx_rows:
        lines.append(f"- `figures/{stamp}_approximation_pareto.png`")
    if scaling_rows:
        lines.append(f"- `figures/{stamp}_scaling_total_ns.png`")
        lines.append(f"- `figures/{stamp}_scaling_ns_per_pt.png`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--stamp", type=str, default="")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    stamp = args.stamp or latest_stamp(raw_dir)
    configure_plot_style()

    speed_path = raw_dir / f"{stamp}_bench_speed.txt"
    scaling_path = raw_dir / f"{stamp}_bench_scaling.txt"
    corr_path = raw_dir / f"{stamp}_bench_correctness.txt"
    approx_path = raw_dir / f"{stamp}_bench_approximation.txt"
    points_path = raw_dir / f"{stamp}_bench_correctness_points.csv"
    if not speed_path.exists() or not corr_path.exists():
        raise FileNotFoundError(
            f"Missing benchmark files for stamp {stamp}: {speed_path} {corr_path}"
        )

    ensure_dir(processed_dir)
    figure_dir = processed_dir / "figures"
    table_dir = processed_dir / "tables"
    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    speed_summary = parse_speed_summary(speed_path)
    speed_rows = speed_rows_from_summary(speed_summary)
    scaling_rows: List[Dict[str, object]] = []
    scaling_fit: Dict[str, object] = {}
    if scaling_path.exists():
        scaling_rows, scaling_fit = parse_scaling_summary(scaling_path)
    suites = parse_correctness_summary(corr_path)
    approx_rows: List[Dict[str, object]] = []
    if approx_path.exists():
        approx_rows = parse_approximation_summary(approx_path)

    total_rows = 0
    valid_rows = 0
    point_data = {}
    if points_path.exists():
        point_data, total_rows, valid_rows = parse_point_csv(points_path)

    speed_csv = processed_dir / f"{stamp}_speed_summary.csv"
    scaling_csv = processed_dir / f"{stamp}_scaling_summary.csv"
    corr_csv = processed_dir / f"{stamp}_correctness_suite_summary.csv"
    approx_csv = processed_dir / f"{stamp}_approximation_summary.csv"
    write_csv(
        speed_csv,
        ["method", "ns_per_pt_mean", "ns_per_pt_std", "ns_per_pt_ci95", "mean_abs_epipolar", "finite_ratio"],
        speed_rows,
    )
    corr_fields = sorted({key for row in suites for key in row.keys()})
    write_csv(corr_csv, corr_fields, suites)
    if scaling_rows:
        scaling_fields = sorted(
            {key for row in scaling_rows for key in row.keys()} | set(scaling_fit.keys())
        )
        scaling_rows_with_fit = list(scaling_rows)
        if scaling_fit:
            scaling_rows_with_fit.append(scaling_fit)
        write_csv(scaling_csv, scaling_fields, scaling_rows_with_fit)
    if approx_rows:
        approx_fields = sorted({key for row in approx_rows for key in row.keys()})
        write_csv(approx_csv, approx_fields, approx_rows)

    runtime_png = figure_dir / f"{stamp}_runtime_ci95.png"
    delta_hist_png = figure_dir / f"{stamp}_deltaE_logabs_hist.png"
    scatter_png = figure_dir / f"{stamp}_Eours_vs_Ehs_scatter.png"
    suite_box_png = figure_dir / f"{stamp}_deltaE_by_suite_boxplot.png"
    approx_pareto_png = figure_dir / f"{stamp}_approximation_pareto.png"
    scaling_total_png = figure_dir / f"{stamp}_scaling_total_ns.png"
    scaling_nspt_png = figure_dir / f"{stamp}_scaling_ns_per_pt.png"

    plot_runtime(speed_rows, runtime_png)
    if point_data:
        plot_delta_hist(point_data, delta_hist_png)
        plot_cost_scatter(point_data, scatter_png)
        plot_suite_boxplot(point_data, suite_box_png)
    if approx_rows:
        plot_approximation_pareto(approx_rows, approx_pareto_png)
    if scaling_rows:
        plot_scaling_total_ns(scaling_rows, scaling_fit, scaling_total_png)
        plot_scaling_ns_per_pt(scaling_rows, scaling_nspt_png)

    speed_tex = table_dir / f"{stamp}_speed_table.tex"
    scaling_tex = table_dir / f"{stamp}_scaling_table.tex"
    corr_tex = table_dir / f"{stamp}_correctness_table.tex"
    approx_tex = table_dir / f"{stamp}_approximation_table.tex"
    write_speed_table_tex(speed_tex, speed_rows)
    if scaling_rows:
        write_scaling_table_tex(scaling_tex, scaling_rows, scaling_fit)
    write_correctness_table_tex(corr_tex, suites)
    if approx_rows:
        write_approximation_table_tex(approx_tex, approx_rows)

    summary_md = processed_dir / f"{stamp}_benchmark_summary.md"
    write_markdown_summary(
        summary_md,
        stamp,
        speed_path,
        scaling_path if scaling_path.exists() else None,
        corr_path,
        approx_path if approx_path.exists() else None,
        points_path if points_path.exists() else None,
        speed_rows,
        scaling_rows,
        scaling_fit,
        suites,
        approx_rows,
        total_rows,
        valid_rows,
    )

    # Stable "latest" aliases.
    (processed_dir / "latest_stamp.txt").write_text(stamp + "\n")
    (processed_dir / "benchmark_summary.md").write_text(summary_md.read_text())
    (processed_dir / "speed_summary.csv").write_text(speed_csv.read_text())
    if scaling_rows:
        (processed_dir / "scaling_summary.csv").write_text(scaling_csv.read_text())
    (processed_dir / "correctness_suite_summary.csv").write_text(corr_csv.read_text())
    if approx_rows:
        (processed_dir / "approximation_summary.csv").write_text(approx_csv.read_text())
    (table_dir / "speed_table.tex").write_text(speed_tex.read_text())
    if scaling_rows:
        (table_dir / "scaling_table.tex").write_text(scaling_tex.read_text())
    (table_dir / "correctness_table.tex").write_text(corr_tex.read_text())
    if approx_rows:
        (table_dir / "approximation_table.tex").write_text(approx_tex.read_text())
    (figure_dir / "runtime_ci95.png").write_bytes(runtime_png.read_bytes())
    if point_data:
        (figure_dir / "deltaE_logabs_hist.png").write_bytes(delta_hist_png.read_bytes())
        (figure_dir / "Eours_vs_Ehs_scatter.png").write_bytes(scatter_png.read_bytes())
        (figure_dir / "deltaE_by_suite_boxplot.png").write_bytes(
            suite_box_png.read_bytes()
        )
    if approx_rows:
        (figure_dir / "approximation_pareto.png").write_bytes(
            approx_pareto_png.read_bytes()
        )
    if scaling_rows:
        (figure_dir / "scaling_total_ns.png").write_bytes(scaling_total_png.read_bytes())
        (figure_dir / "scaling_ns_per_pt.png").write_bytes(scaling_nspt_png.read_bytes())

    print(f"Processed stamp: {stamp}")
    print(f"Wrote summary: {summary_md}")
    if scaling_rows:
        print(f"Wrote scaling table: {scaling_tex}")
    if approx_rows:
        print(f"Wrote approximation table: {approx_tex}")
    print(f"Wrote tables: {speed_tex}, {corr_tex}")
    print(f"Wrote figures directory: {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
