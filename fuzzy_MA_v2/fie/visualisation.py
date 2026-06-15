"""
fie/visualisation.py  ─  Plotting utilities for the FIE
=========================================================
All matplotlib code lives here, keeping engine.py dependency-free
from visualisation libraries.

Usage
-----
    from fie.visualisation import plot_membership_functions, plot_evaluation
    plot_membership_functions()              # all 5 variables
    plot_evaluation(cs=70, ir=0.6, fh=40)   # single-case diagram
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .engine import _MF, _UNI, _mu, evaluate, fuzzify, FIEResult

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (consistent with architecture diagram)
# ---------------------------------------------------------------------------
_COLOURS = {
    "baja":        "#4CAF50",
    "media":       "#FFC107",
    "alta":        "#F44336",
    "minima":      "#4CAF50",
    "moderada":    "#FFC107",
    "elevada":     "#F44336",
    "deficiente":  "#F44336",
    "aceptable":   "#FFC107",
    "excelente":   "#4CAF50",
    "bajo":        "#F44336",
    "medio":       "#FFC107",
    "alto":        "#4CAF50",
    "innecesario": "#4CAF50",
    "condicional": "#FFC107",
    "critico":     "#F44336",
}

_TITLES = {
    "cs":  "CS — Semantic Complexity  [0, 100]",
    "ir":  "IR — Response Uncertainty  [0, 1]",
    "fh":  "FH — Historical Reliability  [0, 100]",
    "nc":  "NC — Trust Level  [0, 1]",
    "ien": "IEN — Escalation Index  [0, 100]",
}

# English labels for the linguistic terms (internal keys are Spanish).
_LABEL_EN = {
    "baja": "Low", "media": "Medium", "alta": "High",
    "minima": "Minimal", "moderada": "Moderate", "elevada": "High",
    "deficiente": "Poor", "aceptable": "Acceptable", "excelente": "Excellent",
    "bajo": "Low", "medio": "Medium", "alto": "High",
    "innecesario": "Unnecessary", "condicional": "Conditional", "critico": "Critical",
}


def _plot_single_var(ax: plt.Axes, var: str, crisp_value: Optional[float] = None) -> None:
    """Plot all MF curves for one variable on the given Axes."""
    x = np.linspace(*_UNI[var])
    for label, params in _MF[var].items():
        y = _mu(var, label, x)
        ax.plot(x, y, label=_LABEL_EN.get(label, label.capitalize()),
                color=_COLOURS.get(label, "grey"), lw=2)

    if crisp_value is not None:
        ax.axvline(crisp_value, color="black", ls="--", lw=1.5,
                   label=f"value={crisp_value:.2f}")

    ax.set_title(_TITLES[var], fontsize=9, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("μ(x)", fontsize=8)


def plot_membership_functions(
    cs: Optional[float] = None,
    ir: Optional[float] = None,
    fh: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render all 5 variable membership functions in a 3×2 grid.
    Optionally overlay the crisp input values as dashed vertical lines.

    Parameters
    ----------
    cs, ir, fh : optional crisp input values to annotate
    save_path  : if provided, save to this path instead of showing
    """
    fig, axs = plt.subplots(3, 2, figsize=(13, 9))
    fig.suptitle("Fuzzy Inference Engine — Membership Functions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(pad=3.5, rect=[0, 0, 1, 0.96])

    _plot_single_var(axs[0, 0], "cs",  crisp_value=cs)
    _plot_single_var(axs[0, 1], "ir",  crisp_value=ir)
    _plot_single_var(axs[1, 0], "fh",  crisp_value=fh)
    _plot_single_var(axs[1, 1], "nc")
    _plot_single_var(axs[2, 0], "ien")
    axs[2, 1].axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_evaluation(
    cs: float,
    ir: float,
    fh: float,
    save_path: Optional[str] = None,
) -> FIEResult:
    """
    Full single-evaluation visualisation:
      - Row 1: Input MFs with crisp value markers
      - Row 2: Aggregated output MFs with centroid marker

    Returns the FIEResult for programmatic use.
    """
    result = evaluate(cs, ir, fh)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        f"FIE Evaluation — CS={cs}  IR={ir}  FH={fh}\n"
        f"NC={result.nc:.3f}  IEN={result.ien:.2f}  "
        f"({'ESCALATE' if result.should_escalate else 'LOCAL'})  "
        f"[{result.eval_ms:.2f} ms]",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: Inputs ──
    for col, (var, val) in enumerate(zip(["cs", "ir", "fh"], [cs, ir, fh])):
        ax = fig.add_subplot(gs[0, col])
        _plot_single_var(ax, var, crisp_value=val)

    # ── Row 2: Outputs (aggregated) ──
    for col, (var, max_alphas, crisp_val) in enumerate([
        ("nc",  result.nc_activations,  result.nc),
        ("ien", result.ien_activations, result.ien),
    ]):
        ax = fig.add_subplot(gs[1, col])
        x = np.linspace(*_UNI[var])
        agg = np.zeros_like(x)
        for label, alpha in max_alphas.items():
            if alpha > 0.0:
                clipped = np.minimum(alpha, _mu(var, label, x))
                ax.fill_between(x, clipped, alpha=0.25, color=_COLOURS.get(label, "grey"))
                ax.plot(x, _mu(var, label, x), "--", lw=1,
                        color=_COLOURS.get(label, "grey"),
                        label=_LABEL_EN.get(label, label))
                agg = np.maximum(agg, clipped)
        ax.fill_between(x, agg, alpha=0.35, color="steelblue", label="Aggregated")
        ax.plot(x, agg, color="steelblue", lw=2)
        ax.axvline(crisp_val, color="black", ls="--", lw=2,
                   label=f"centroid={crisp_val:.3f}")
        ax.set_title(_TITLES[var], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    # Empty cell
    fig.add_subplot(gs[1, 2]).axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return result


def plot_pareto_from_results(
    results_path: str = "outputs/benchmark_results.json",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render the Pareto frontier from REAL benchmark output
    (``python benchmark.py`` → outputs/benchmark_results.json).
    """
    import json

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    style = {
        "pure-local":   ("#4CAF50", "o"),
        "pure-cloud":   ("#F44336", "s"),
        "fuzzy-hybrid": ("#2196F3", "D"),
    }
    configs = {}
    for m in data["table2_metrics"]:
        cost = m["costo_usd_por_1k_tareas"]
        acc = m.get("precision_global_pct") or 0.0
        runtime_min = (m.get("tiempo_total_config_s") or 0.0) / 60.0
        colour, marker = style.get(m["config"], ("grey", "o"))
        # Label carries accuracy, cost, and the wall-clock runtime of the run
        label = f"{m['config']}\n{acc:.1f}% / ${cost:.2f} / {runtime_min:.0f} min"
        configs[label] = (cost, acc, colour, marker)

    return _render_pareto(configs, save_path,
                          subtitle=f"{data['n_tasks']}-task benchmark — real data "
                                   f"(label: accuracy / cost / wall-clock)")


def plot_pareto(save_path: Optional[str] = None) -> plt.Figure:
    """
    Render the Pareto frontier figure referenced as Fig. 1 in the paper
    (Section 6.2).  Uses the exact benchmark numbers from Table 2.
    """
    configs = {
        "Pure-Local\n(68.4% / $0.00)":  (0.00,  68.4, "#4CAF50", "o"),
        "Pure-Cloud\n(96.2% / $24.50)": (24.50, 96.2, "#F44336", "s"),
        "Fuzzy-Hybrid\n(94.1% / $14.21)": (14.21, 94.1, "#2196F3", "D"),
    }

    return _render_pareto(configs, save_path,
                          subtitle="150-task benchmark — MICAI 2026")


def _render_pareto(
    configs: dict,
    save_path: Optional[str],
    subtitle: str,
) -> plt.Figure:
    """Shared Pareto renderer: configs = {label: (cost, accuracy, colour, marker)}."""
    fig, ax = plt.subplots(figsize=(7, 5))

    points = [(x, y) for x, y, _, _ in configs.values()]

    # Non-dominated points (higher accuracy and/or lower cost) form the frontier
    frontier = sorted(
        p for p in points
        if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in points)
    )
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                "k--", lw=1.5, label="Pareto frontier", zorder=1)

    for label, (x, y, colour, marker) in configs.items():
        ax.scatter(x, y, color=colour, marker=marker, s=160, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(8, -14), fontsize=8)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.set_xlabel("Operational Cost per 1 000 Tasks (USD)", fontsize=10)
    ax.set_ylabel("Task Resolution Accuracy (%)", fontsize=10)
    ax.set_title(f"Pareto Frontier: Accuracy vs. Cost\n({subtitle})",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(min(xs) - 2, max(xs) + 4)
    ax.set_ylim(max(0, min(ys) - 10), 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig
