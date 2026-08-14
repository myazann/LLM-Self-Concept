"""Reproducible figures for the validity and results reports.

Matplotlib is imported lazily so the tabular analysis remains usable in a
minimal environment.  All plots use the default condition unless the title
explicitly identifies a framing or instruction contrast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scoring import LABELS, POLARITY


SHORT = {
    "Self-esteem (RSES)": "Self-esteem",
    "Self-concept clarity (SCCS)": "Clarity",
    "Moral self-image (MSI)": "Moral self-image",
    "Lack of identity (SCIM)": "Identity coherence",
    "Self-alienation (AS)": "Low alienation",
    "External influence (AS)": "Low external influence",
}
FAMILY_COLORS = {"gemma": "#6657c7", "qwen": "#008f70"}
FRAME_COLORS = {
    "first_person_bare": "#202020",
    "first_person_ack": "#d56a00",
    "third_person_assistant": "#3b78b4",
}
STATUS_TAG = {
    "do_not_use_composite": "DO NOT USE",
    "default_condition_only": "DEFAULT ONLY",
    "use_with_multiple_warnings": "MULTIPLE WARNINGS",
    "use_with_instruction_warning": "INSTRUCTION WARNING",
    "use_with_wording_warning": "WORDING WARNING",
    "use_with_response_style_warning": "STYLE WARNING",
    "use_with_reliability_warning": "RELIABILITY WARNING",
    "provisionally_usable": "PROVISIONAL",
}


def _construct_title(construct, status_map=None):
    if not status_map:
        return SHORT[construct]
    return f"{SHORT[construct]}  [{STATUS_TAG.get(status_map.get(construct), status_map.get(construct, ''))}]"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return path


def polarity_aligned(scores: pd.DataFrame) -> pd.DataFrame:
    """Add ``aligned_score`` so every panel points toward the positive pole."""
    out = scores.copy()
    pol = out.construct.map(POLARITY)
    out["aligned_score"] = np.where(pol < 0, 1 - out.score, out.score)
    return out


def plot_default_heatmap(default_scores, meta, path, *, status_map=None):
    plt, _ = _plt()
    d = polarity_aligned(default_scores).merge(meta, on="model")
    order = d[["model", "release_dt", "params_b"]].drop_duplicates()
    order = order.sort_values(["release_dt", "params_b", "model"]).model
    M = (d.pivot(index="model", columns="construct", values="aligned_score")
           .reindex(index=order, columns=LABELS))

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    im = ax.imshow(M.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(LABELS)), [_construct_title(x, status_map) for x in LABELS],
                  rotation=35, ha="right")
    labels = []
    mm = meta.set_index("model")
    for model in M.index:
        r = mm.loc[model]
        active = f"/{r.active_b:g} active" if r.active_b != r.params_b else ""
        labels.append(f"{model}  ({r.params_b:g}B{active}; {str(r.release)})")
    ax.set_yticks(range(len(M)), labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if v < .36 or v > .72 else "#202020")
    ax.set_title("Default-condition model profiles (first-person bare + p0)\n"
                 "Polarity aligned: higher = more positive/coherent pole")
    fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label="polarity-aligned score")
    ax.set_xlabel("")
    ax.set_ylabel("models ordered by release date")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_release_trajectories(default_scores, meta, path, *, status_map=None):
    """Release-date trajectories with only genuinely matched size tracks joined."""
    plt, Line2D = _plt()
    d = polarity_aligned(default_scores).merge(meta, on="model")
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=True)
    axes = axes.ravel()
    tracks = {
        "Gemma dense 12B": ["Gemma3-12B", "Gemma4-12B"],
        "Gemma dense ~30B": ["Gemma3-27B", "Gemma4-31B"],
        "Qwen dense 27B": ["Qwen3.5-27B", "Qwen3.6-27B"],
        "Qwen MoE 35B/A3B": ["Qwen3.5-35B-A3B", "Qwen3.6-35B-A3B"],
    }
    track_styles = ["-", ":", "-", "--"]
    for ax, construct in zip(axes, LABELS):
        g = d[d.construct == construct]
        for _, row in g.iterrows():
            ax.scatter(row.release_dt, row.aligned_score,
                       s=28 + 2.2 * row.params_b,
                       marker="X" if row.moe else "o",
                       color=FAMILY_COLORS.get(row.family, "#555555"),
                       edgecolor="white", linewidth=.6, zorder=3)
        for (track, models), ls in zip(tracks.items(), track_styles):
            z = g[g.model.isin(models)].sort_values("release_dt")
            if len(z) == 2:
                ax.plot(z.release_dt, z.aligned_score, ls,
                        color=FAMILY_COLORS.get(z.family.iloc[0], "#555555"),
                        alpha=.75, linewidth=1.6)
        ax.axhline(.5, color="#aaaaaa", lw=.8, ls=":")
        ax.set_title(_construct_title(construct, status_map), loc="left",
                     fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=.18)
        ax.set_ylim(0, 1)
    for ax in axes[-2:]:
        ax.set_xlabel("release date")
    for ax in axes[::2]:
        ax.set_ylabel("polarity-aligned score")
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS["gemma"], ls="-", marker="o",
               label="Gemma dense 12B"),
        Line2D([0], [0], color=FAMILY_COLORS["gemma"], ls=":", marker="o",
               label="Gemma dense 27→31B (near-match)"),
        Line2D([0], [0], color=FAMILY_COLORS["qwen"], ls="-", marker="o",
               label="Qwen dense 27B"),
        Line2D([0], [0], color=FAMILY_COLORS["qwen"], ls="--", marker="X",
               label="Qwen MoE 35B/A3B"),
        Line2D([0], [0], color="#555", marker="o", ls="",
               label="other model: point only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Default self-report by release date, with matched-size tracks\n"
                 "Point area = total parameters; X = MoE; lines are descriptive, not causal fits",
                 fontsize=14, y=.995)
    fig.tight_layout(rect=(0, .06, 1, .97))
    _save(fig, path)
    plt.close(fig)


def plot_size_ladders(default_scores, meta, path, *, status_map=None, size_axis="total"):
    """Parameter-size relationships within each family generation."""
    plt, Line2D = _plt()
    d = polarity_aligned(default_scores).merge(meta, on="model")
    generations = list(meta.sort_values("release_dt").generation.drop_duplicates())
    palette = dict(zip(generations, ["#5b4bb7", "#a58ee6", "#008f70", "#65c6ad"]))
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=True)
    axes = axes.ravel()
    if size_axis not in {"total", "active"}:
        raise ValueError("size_axis must be 'total' or 'active'")
    xcol = "params_b" if size_axis == "total" else "active_b"
    for ax, construct in zip(axes, LABELS):
        g = d[d.construct == construct]
        for (generation, architecture), z in g.groupby(["generation", "architecture"]):
            z = z.sort_values(xcol)
            marker = "X" if architecture == "moe" else "D" if architecture == "effective-embedding" else "o"
            ax.plot(z[xcol], z.aligned_score, color=palette[generation],
                    marker=marker, lw=1.6 if len(z) > 1 else 0, alpha=.85)
        ax.axhline(.5, color="#aaaaaa", lw=.8, ls=":")
        ax.set_xscale("log")
        ticks = [3, 4, 8, 12, 27, 35] if size_axis == "active" else [4, 8, 12, 27, 35]
        ax.set_xticks(ticks, [str(x) for x in ticks])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.18)
        ax.set_title(_construct_title(construct, status_map), loc="left",
                     fontsize=9.5, fontweight="bold")
    for ax in axes[-2:]:
        ax.set_xlabel(f"{size_axis} parameters (billions, log scale)")
    for ax in axes[::2]:
        ax.set_ylabel("polarity-aligned score")
    handles = [Line2D([0], [0], color=palette[g], marker="o", label=g) for g in generations]
    handles.append(Line2D([0], [0], color="#555", marker="X", ls="", label="mixture-of-experts"))
    handles.append(Line2D([0], [0], color="#555", marker="D", ls="", label="effective-embedding"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Default self-report across {size_axis}-parameter ladders\n"
                 "Lines connect models only within the same family generation and architecture",
                 fontsize=14, y=.995)
    fig.tight_layout(rect=(0, .045, 1, .97))
    _save(fig, path)
    plt.close(fig)


def plot_instruction_robustness(scores, path, *, status_map=None):
    plt, _ = _plt()
    d = polarity_aligned(scores)
    order = ["p0", "p1", "p2"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, construct in zip(axes, LABELS):
        g = d[d.construct == construct]
        W = g.pivot_table(index="model", columns="paraphrase", values="aligned_score").reindex(columns=order)
        for _, row in W.iterrows():
            ax.plot(order, row.values, color="#9a9a9a", lw=.7, alpha=.45)
        mean = W.mean()
        ax.plot(order, mean.values, color="#111111", marker="o", lw=2.4)
        ax.axhline(.5, color="#aaaaaa", lw=.8, ls=":")
        ax.set_title(_construct_title(construct, status_map), loc="left",
                     fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.18)
    for ax in axes[-2:]:
        ax.set_xlabel("instruction form")
    for ax in axes[::2]:
        ax.set_ylabel("polarity-aligned score")
    fig.suptitle("Instruction robustness at the default first-person-bare framing\n"
                 "Thin lines are models; dark line is the model mean", fontsize=14, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    _save(fig, path)
    plt.close(fig)


def plot_framing_effects(scores, path, *, status_map=None):
    plt, _ = _plt()
    d = polarity_aligned(scores)
    order = ["first_person_bare", "first_person_ack", "third_person_assistant"]
    labels = ["self: bare", "self: +AI preamble", "typical AI assistant"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, construct in zip(axes, LABELS):
        g = d[d.construct == construct]
        W = g.pivot_table(index="model", columns="framing", values="aligned_score").reindex(columns=order)
        for _, row in W.iterrows():
            ax.plot(labels, row.values, color="#9a9a9a", lw=.7, alpha=.42)
        mean = W.mean()
        ax.plot(labels, mean.values, color="#111111", marker="o", lw=2.4)
        for x, frame in enumerate(order):
            ax.scatter(x, mean[frame], color=FRAME_COLORS[frame], s=45, zorder=3)
        ax.axhline(.5, color="#aaaaaa", lw=.8, ls=":")
        ax.set_title(_construct_title(construct, status_map), loc="left",
                     fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.18)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=18)
    for ax in axes[::2]:
        ax.set_ylabel("polarity-aligned score")
    fig.suptitle("Framing is a substantive result (p0 held fixed)\n"
                 "Thin lines are models; dark line is the model mean", fontsize=14, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    _save(fig, path)
    plt.close(fig)


def plot_item_validity(items, path):
    plt, _ = _plt()
    colors = {
        "keep": "#2a9d68", "keep_with_warning": "#d8a128",
        "review": "#df6f2d", "drop_candidate": "#bd2d2d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for recommendation, g in items.groupby("recommendation"):
        c = colors.get(recommendation, "#777777")
        axes[0].scatter(g.item_total_r, g.alpha_if_deleted_delta, s=42, alpha=.8,
                        color=c, label=recommendation.replace("_", " "))
        axes[1].scatter(g.instruction_mae, g.order_mae_norm, s=42, alpha=.8,
                        color=c, label=recommendation.replace("_", " "))
    reviews = items[items.recommendation == "review"].sort_values(
        ["item_total_r", "alpha_if_deleted_delta"], ascending=[True, False]).head(8)
    annotated = pd.concat([items[items.recommendation == "drop_candidate"], reviews])
    for _, row in annotated.iterrows():
        axes[0].annotate(row.item_id, (row.item_total_r, row.alpha_if_deleted_delta),
                         xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[0].axvline(0, color="#999", ls=":")
    axes[0].axvline(.20, color="#999", ls="--", lw=.8)
    axes[0].axhline(.02, color="#999", ls="--", lw=.8)
    axes[0].set(xlabel="corrected item-total correlation (default)",
                ylabel="change in alpha if item is removed",
                title="Default-condition structural evidence")
    axes[1].axvline(.10, color="#999", ls="--", lw=.8)
    axes[1].axhline(.15, color="#999", ls="--", lw=.8)
    axes[1].set(xlabel="instruction-form mean absolute difference (0-1)",
                ylabel="option-order mean absolute difference (0-1)",
                title="Robustness warnings")
    for ax in axes:
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Item validity: decisions start at the default condition; robustness confirms or warns")
    fig.tight_layout(rect=(0, .07, 1, .94))
    _save(fig, path)
    plt.close(fig)
