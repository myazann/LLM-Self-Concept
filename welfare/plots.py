"""Figures for the welfare analysis, in the order the results should be read.

Matplotlib is imported lazily, so the tabular half of the analysis still runs in
a minimal environment.

Every figure here is drawn from a table `welfare.analysis` also writes as CSV, so
nothing is visible in a plot that cannot be read back as numbers. The file is
laid out like the output directory: VALIDITY first (can this administration be
read at all), then PREFERENCE (what it says), then COHORT (what the models say
together) — a ranking figure is only meaningful next to the two validity ones.

COLOUR. Three palettes, each doing exactly one job:

  * `CONSTRUCT_COLORS` is categorical — it encodes WHICH construct, never how
    much. The six hues are assigned in a fixed order and were chosen by
    maximising the worst all-pairs separation in OKLab under simulated
    protanopia and deuteranopia (Machado-Oliveira-Fernandes 2009, severity 1.0):
    worst pair dE 12.9 simulated, 22.3 unsimulated, against targets of 8 and 15.
    All-pairs rather than adjacent-only because these hues are used in scatters,
    where any two marks can end up side by side. Colour is always redundant with
    a text label or a facet, never the only encoding.
  * `SEQUENTIAL` is one hue, light to dark, for magnitude — the agreement
    matrices. It replaces a red-yellow-green ramp, which reads as a judgment
    (green = good) on a quantity that has no good end, and collapses for
    red-green colour blindness precisely where the interesting cells are.
  * `DIVERGING` is two hues around a neutral grey, for quantities with a
    meaningful zero — every baseline shift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Categorical. One colour per construct, reused across every figure so an
# attribute keeps its identity from the ranking plot to the shift scatter.
CONSTRUCT_COLORS = {
    "self-esteem": "#2d3acc",
    "self-concept clarity": "#38a391",
    "moral qualities": "#4e5d0e",
    "identity coherence": "#930975",
    "self-connection": "#a36df0",
    "self-direction": "#da6c1e",
}

# The four framing factors keep one colour each across every figure that
# contrasts them, so a reader learns "orange is the object flip" once. Drawn
# from the same validated set; factors and constructs never share a figure.
FACTOR_COLORS = {
    "qvar": "#2d3acc",
    "object": "#da6c1e",
    "subject": "#38a391",
    "no_pref_offered": "#930975",
}

# Model families, assigned in sorted order from the same validated set.
FAMILY_ORDER = ("#2d3acc", "#da6c1e", "#38a391", "#930975", "#4e5d0e", "#a36df0")

ARM_LABEL = {2: "forced choice (2 options)", 3: "with 'No preference' (3 options)"}
ARM_COLORS = {2: "#b03a2e", 3: "#2c6fbb"}

INK = "#202020"
MUTED = "#6b6b66"
GRID = dict(color="#202020", alpha=.10, lw=.7, zorder=0)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ramps():
    from matplotlib.colors import LinearSegmentedColormap

    seq = LinearSegmentedColormap.from_list(
        "welfare_seq", ["#f2f6f5", "#b9d8d2", "#6bb5a8", "#2b8577", "#0c4a41"])
    div = LinearSegmentedColormap.from_list(
        "welfare_div", ["#2d3acc", "#8f96e0", "#e9e9e5", "#e8a878", "#da6c1e"])
    return seq, div


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return path


def _color(construct_label):
    return CONSTRUCT_COLORS.get(construct_label, MUTED)


def _family_colors(families):
    return {f: FAMILY_ORDER[i % len(FAMILY_ORDER)] for i, f in enumerate(sorted(families))}


def _wrap(text, width=46):
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def _bare(ax, left=True):
    """Recessive frame: keep the axis the data is read along, drop the rest."""
    spines = ["top", "right"] + (["left"] if left else [])
    ax.spines[spines].set_visible(False)


def _sig(q, alpha=0.05):
    """A finding survives multiplicity correction. Drawn as fill, not as hue."""
    return bool(q is not None and np.isfinite(q) and q < alpha)


def _pct_axis(ax):
    """Percent ticks whose precision follows the axis range.

    A fixed `.0%` prints "0%, 0%, 1%, 1%, 2%" on an axis spanning two points —
    five ticks, three distinct labels. The formatter is called at draw time, so
    it can read the range matplotlib actually settled on.
    """
    def fmt(x, _):
        lo, hi = ax.get_xlim()
        span = abs(hi - lo)
        nd = 0 if span > 0.06 else (1 if span > 0.006 else 2)
        return f"{x:.{nd}%}"

    ax.xaxis.set_major_formatter(fmt)


def _construct_legend(ax, plt, loc="lower right"):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CONSTRUCT_COLORS.values()]
    ax.legend(handles, list(CONSTRUCT_COLORS), fontsize=8, frameon=False, loc=loc,
              title="construct", title_fontsize=8)


# ===========================================================================
# VALIDITY — read these before any ranking figure
# ===========================================================================
def plot_position_bias(slots, duels, path, subtitle=""):
    """Choice rate per printed slot, and the slot-vs-slot duels behind it.

    Under balanced ordering content is orthogonal to position, so both panels are
    pure position effects: the left one against the uniform rate a bias-free
    model would give, the right one as head-to-head win rates against 0.50.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    arms = sorted(slots.n_options.unique())
    width = 0.36
    for i, arm in enumerate(arms):
        g = slots[slots.n_options == arm].sort_values("position")
        x = np.arange(len(g)) + (i - (len(arms) - 1) / 2) * width
        ax.bar(x, g.rate, width, color=ARM_COLORS.get(int(arm), MUTED),
               label=ARM_LABEL.get(int(arm), f"{arm} options"))
        ax.axhline(1.0 / arm, color=ARM_COLORS.get(int(arm), MUTED), ls=":", lw=1.2)
        for xi, v in zip(x, g.rate):
            # Boxed, because the two uniform-rate reference lines cross this band.
            ax.text(xi, v + .012, f"{v:.1%}", ha="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.9))
    ax.set_xticks(range(int(max(arms))), [f"slot {p}" for p in range(1, int(max(arms)) + 1)])
    ax.set_ylabel("share of answers")
    ax.set_ylim(0, max(0.75, slots.rate.max() * 1.18))
    ax.set_title("Where the chosen option sat\n(dotted = no position effect)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _bare(ax, left=False)

    ax = axes[1]
    d = duels.sort_values(["n_options", "slot_lo", "slot_hi"])
    labels = [f"{int(r.n_options)}opt: pos{int(r.slot_lo)} v pos{int(r.slot_hi)}"
              for r in d.itertuples()]
    colors = [ARM_COLORS.get(int(a), MUTED) for a in d.n_options]
    y = np.arange(len(d))
    ax.barh(y, d.p_lo_wins - 0.5, left=0.5, color=colors, height=.62)
    ax.axvline(0.5, color=INK, lw=1.1)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 0.75)
    ax.set_xlabel("P(the earlier slot wins)")
    # A value column outside the axes: bars grow both ways from 0.50, so a label
    # tucked against the bar end would collide with the tick labels on the left.
    for yi, r in zip(y, d.itertuples()):
        ax.text(0.762, yi, f"{r.p_lo_wins:.1%}   (n={int(r.n):,})",
                va="center", ha="left", fontsize=8, clip_on=False)
    ax.set_title("Slot duels between the two real attributes\n"
                 "(content balanced by design — this is position alone)", fontsize=10)
    _bare(ax)

    fig.suptitle("POSITION BIAS" + (f"  —  {subtitle}" if subtitle else ""),
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_order_consistency(consistency, path, subtitle=""):
    """How far a pair's preference moves when only the two slots swap."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax = axes[0]
    for arm, g in consistency.groupby("n_options"):
        # Mean, not median: the forced-choice arm has few trials per slot order,
        # so its deltas are coarse and the median only ever reports the mode.
        ax.hist(g.delta, bins=np.linspace(0, 1, 21), alpha=.6,
                color=ARM_COLORS.get(int(arm), MUTED),
                label=f"{ARM_LABEL.get(int(arm), arm)}  (mean {g.delta.mean():.2f})")
    ax.set_xlabel("|ΔP(choose A)| between the two slot orders")
    ax.set_ylabel("condition × pair estimates")
    ax.set_title("A pair's preference under a slot swap\n0 = order changed nothing",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _bare(ax, left=False)

    ax = axes[1]
    dec = consistency[consistency.decisive]
    rates, labels, colors = [], [], []
    for arm in sorted(consistency.n_options.unique()):
        g = dec[dec.n_options == arm]
        if not len(g):
            continue
        rates.append(g.flipped.mean())
        labels.append(f"{ARM_LABEL.get(int(arm), arm)}\n{len(g):,} decisive pairs")
        colors.append(ARM_COLORS.get(int(arm), MUTED))
    y = np.arange(len(rates))
    ax.barh(y, rates, color=colors, height=.5)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.5, max(rates) * 1.3) if rates else 1)
    ax.set_xlabel("winner flips when only the slot order moves")
    for yi, v in zip(y, rates):
        ax.text(v + .008, yi, f"{v:.1%}", va="center", fontsize=9)
    ax.set_title("Flip rate among decisive pairs\nhigh = per-pair winners are "
                 "not reportable", fontsize=10)
    _bare(ax)

    fig.suptitle("ORDER CONSISTENCY" + (f"  —  {subtitle}" if subtitle else ""),
                 fontsize=12, y=1.03)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_condition_validity(metrics, base_label, path, subtitle="", cols=3):
    """Every validity metric across all sixteen conditions, baseline picked out.

    Small multiples rather than one crowded axis: the metrics are on different
    scales (a rate, a distance, a correlation) and putting them on a shared x
    would need a second axis, which is never the answer. Each panel carries its
    own neutral reference line, so "is this condition unusual" is read against
    what a rendering-free administration would give, not against the other
    conditions.
    """
    plt = _plt()
    from welfare.validity import METRICS

    order = list(metrics.label)
    shown = [m for m in METRICS if m in metrics.columns and metrics[m].notna().any()]
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 0.30 * len(order) * rows + 1.6),
                             squeeze=False, sharey=True)
    y = np.arange(len(order))
    is_base = np.array([lab == base_label for lab in order])

    for k, name in enumerate(shown):
        ax = axes[k // cols][k % cols]
        spec = METRICS[name]
        v = metrics[name].to_numpy(dtype=float)
        lo = metrics.get(f"{name}_lo")
        hi = metrics.get(f"{name}_hi")
        if lo is not None and hi is not None:
            err = np.vstack([v - lo.to_numpy(dtype=float), hi.to_numpy(dtype=float) - v])
            ax.errorbar(v, y, xerr=np.abs(err), fmt="none", ecolor="#20202055",
                        elinewidth=1.0, capsize=1.6, zorder=2)
        # The baseline is drawn as a filled marker and a bold tick label, so it
        # is identifiable without spending a hue on it.
        ax.scatter(v[~is_base], y[~is_base], s=26, color="#2b8577", alpha=.85,
                   edgecolor="white", linewidth=.6, zorder=3)
        ax.scatter(v[is_base], y[is_base], s=74, marker="D", color="#0c4a41",
                   edgecolor="white", linewidth=.9, zorder=4)
        if np.isfinite(spec["neutral"]):
            ax.axvline(spec["neutral"], color=INK, ls="--", lw=1.0, zorder=1)
        ax.set_title(spec["label"], fontsize=9.5)
        ax.grid(axis="x", **GRID)
        _bare(ax)
        if spec["fmt"] == "pct":
            _pct_axis(ax)
        # The y axis is shared, so only the first column carries tick marks —
        # the others would draw a column of bare dashes beside their data.
        if k % cols:
            ax.tick_params(left=False)

    for k in range(len(shown), rows * cols):
        axes[k // cols][k % cols].axis("off")
    for r in range(rows):
        axes[r][0].set_yticks(y, order, fontsize=7.4, family="monospace")
        for tick, base in zip(axes[r][0].get_yticklabels(), is_base):
            if base:
                tick.set_fontweight("bold")
    axes[0][0].invert_yaxis()

    fig.suptitle("VALIDITY ACROSS CONDITIONS   (◆ = baseline, dashed = no effect)"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11.5, y=1.005)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_validity_shift(shift, path, subtitle="", cols=4):
    """What each single-factor flip does to each validity metric.

    One panel per metric, four rows per panel — the flips. The interval is on the
    DIFFERENCE and is paired, so a bar clear of zero means the flip changed the
    metric by more than the pairs can explain. Significance is drawn as a filled
    marker rather than a colour, so it survives being printed in grey.
    """
    plt = _plt()
    from welfare.baseline import FACTOR_LABEL, FACTORS
    from welfare.validity import METRICS

    metrics = [m for m in METRICS if m in set(shift.metric)]
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 2.3 * rows + 1.1),
                             squeeze=False)
    factors = [f for f in FACTORS if f in set(shift.factor)]
    y = np.arange(len(factors))

    for k, name in enumerate(metrics):
        ax = axes[k // cols][k % cols]
        g = shift[shift.metric == name].set_index("factor").reindex(factors)
        v = g["shift"].to_numpy(dtype=float)
        lo = g.shift_lo.to_numpy(dtype=float)
        hi = g.shift_hi.to_numpy(dtype=float)
        colors = [FACTOR_COLORS[f] for f in factors]
        for i, f in enumerate(factors):
            if not np.isfinite(v[i]):
                continue
            comparable = bool(g.comparable.iloc[i]) if "comparable" in g else True
            ax.plot([lo[i], hi[i]], [i, i], color=colors[i], lw=2.0,
                    alpha=.35 if not comparable else .8, zorder=2,
                    solid_capstyle="round")
            sig = np.isfinite(g.p.iloc[i]) and g.p.iloc[i] < 0.05 and comparable
            if comparable:
                ax.scatter([v[i]], [i], s=62, zorder=3, marker="o",
                           color=colors[i] if sig else "white",
                           edgecolor=colors[i], linewidth=1.6)
            else:
                # An unfilled marker takes no edge colour, so the "measured
                # something different" case is drawn as a bare cross.
                ax.scatter([v[i]], [i], s=62, zorder=3, marker="x",
                           color=colors[i], linewidth=1.6)
        ax.axvline(0, color=INK, lw=1.1, zorder=1)
        ax.set_yticks(y, [FACTOR_LABEL[f] for f in factors], fontsize=8)
        ax.set_ylim(len(factors) - 0.5, -0.5)
        ax.set_title(METRICS[name]["label"], fontsize=9.5)
        ax.grid(axis="x", **GRID)
        _bare(ax)

    for k in range(len(metrics), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("WHAT EACH FLIP DOES TO VALIDITY   (filled = p < 0.05; "
                 "× = not comparable across arms)" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=11.5, y=1.01)
    fig.text(0.5, -0.02, "change from baseline", ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_validity_models(metrics, meta, path, subtitle="", cols=4):
    """Every model's validity at the baseline condition, side by side.

    Models are ordered by family then size, so a family effect or a size trend
    shows up as structure down the column rather than having to be looked up.
    """
    plt = _plt()
    from welfare.validity import METRICS

    order = list(metrics.model)
    fams = list(dict.fromkeys(meta.set_index("model").family.reindex(order).fillna("")))
    fam_color = _family_colors(fams)
    colors = [fam_color.get(f, MUTED)
              for f in meta.set_index("model").family.reindex(order).fillna("")]

    shown = [m for m in METRICS if m in metrics.columns and metrics[m].notna().any()]
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 0.30 * len(order) * rows + 1.8),
                             squeeze=False, sharey=True)
    y = np.arange(len(order))

    for k, name in enumerate(shown):
        ax = axes[k // cols][k % cols]
        spec = METRICS[name]
        v = metrics[name].to_numpy(dtype=float)
        lo, hi = metrics.get(f"{name}_lo"), metrics.get(f"{name}_hi")
        if lo is not None and hi is not None:
            err = np.vstack([v - lo.to_numpy(dtype=float), hi.to_numpy(dtype=float) - v])
            ax.errorbar(v, y, xerr=np.abs(err), fmt="none", ecolor="#20202055",
                        elinewidth=1.0, capsize=1.6, zorder=2)
        ax.scatter(v, y, s=44, color=colors, edgecolor="white", linewidth=.7, zorder=3)
        if np.isfinite(spec["neutral"]):
            ax.axvline(spec["neutral"], color=INK, ls="--", lw=1.0, zorder=1)
        ax.set_title(spec["label"], fontsize=9.5)
        ax.grid(axis="x", **GRID)
        _bare(ax)
        if spec["fmt"] == "pct":
            _pct_axis(ax)
        if k % cols:
            ax.tick_params(left=False)

    for k in range(len(shown), rows * cols):
        axes[k // cols][k % cols].axis("off")
    for r in range(rows):
        axes[r][0].set_yticks(y, order, fontsize=8)
    axes[0][0].invert_yaxis()
    if len(fam_color) > 1:
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f)
                   for f, c in fam_color.items()]
        axes[0][cols - 1].legend(handles=handles, fontsize=8, frameon=False,
                                 loc="lower right", title="family", title_fontsize=8)
    fig.suptitle("VALIDITY BY MODEL, AT THE BASELINE CONDITION"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11.5, y=1.005)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_validity_shift_models(shift, path, subtitle="", cols=4):
    """Does each flip move validity the same way in every model?

    One dot per model, so the question the figure answers is not "is the mean
    shift non-zero" but "do the models agree on the sign". A tight column away
    from zero is a property of the FRAMING; a column straddling zero is a
    property of individual models and should not be reported as a framing effect.
    """
    plt = _plt()
    from welfare.baseline import FACTOR_LABEL, FACTORS
    from welfare.validity import METRICS

    metrics = [m for m in METRICS if m in set(shift.metric)]
    factors = [f for f in FACTORS if f in set(shift.factor)]
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 2.5 * rows + 1.1),
                             squeeze=False)
    rng = np.random.default_rng(0)

    for k, name in enumerate(metrics):
        ax = axes[k // cols][k % cols]
        for i, f in enumerate(factors):
            g = shift[(shift.metric == name) & (shift.factor == f)]
            g = g[np.isfinite(g["shift"].to_numpy(dtype=float))]
            if g.empty:
                continue
            ok = (g.comparable.astype(bool) if "comparable" in g
                  else pd.Series(True, index=g.index))
            jitter = rng.uniform(-.16, .16, len(g))
            # Crosses where the flip changed WHAT was measured, not only how
            # much of it there was — an arm change alters the option set, so the
            # two slot distributions are not one measurement made twice.
            for mask, marker, kw in ((ok, "o", dict(edgecolor="white", linewidth=.6)),
                                     (~ok, "x", dict(linewidth=1.4))):
                if not mask.any():
                    continue
                ax.scatter(g.loc[mask, "shift"], i + jitter[mask.to_numpy()], s=30,
                           color=FACTOR_COLORS[f], alpha=.75, marker=marker,
                           zorder=3, **kw)
            if not ok.any():
                continue
            m = float(np.nanmean(g.loc[ok, "shift"]))
            ax.plot([m, m], [i - .32, i + .32], color=FACTOR_COLORS[f], lw=2.6, zorder=4)
        ax.axvline(0, color=INK, lw=1.1, zorder=1)
        ax.set_yticks(range(len(factors)), [FACTOR_LABEL[f] for f in factors], fontsize=8)
        ax.set_ylim(len(factors) - 0.5, -0.5)
        ax.set_title(METRICS[name]["label"], fontsize=9.5)
        ax.grid(axis="x", **GRID)
        _bare(ax)

    for k in range(len(metrics), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("FLIP EFFECTS ON VALIDITY, ONE DOT PER MODEL   (bar = cohort mean; "
                 "× = not comparable across arms)"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11.5, y=1.01)
    fig.text(0.5, -0.02, "change from baseline", ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# ===========================================================================
# PREFERENCE — the ranking, at the baseline condition
# ===========================================================================
def plot_attribute_ranking(ranking, path, subtitle=""):
    """Every attribute's win rate, sorted, with its bootstrap interval.

    The interval resamples PAIRS, not trials: which opponents an attribute was
    compared against is the dominant source of error in a sampled tournament, so
    a trial-level interval would be optimistic by a wide margin.
    """
    plt = _plt()
    d = ranking.sort_values("win_rate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 0.27 * len(d) + 2.0))

    y = np.arange(len(d))
    colors = [_color(c) for c in d.construct_label]
    ax.barh(y, d.win_rate, color=colors, height=.72, zorder=2)
    ax.errorbar(d.win_rate, y, xerr=[d.win_rate - d.ci_lo, d.ci_hi - d.win_rate],
                fmt="none", ecolor="#20202090", elinewidth=1.1, capsize=2, zorder=3)
    ax.axvline(0.5, color=INK, lw=1.1, ls="--", zorder=1)

    # Padded to a fixed width in a monospace face, so the right-aligned tick
    # labels line up as two columns instead of a ragged left edge.
    ax.set_yticks(y, [f"{r.entity:<12} {_wrap(r.text):<47}" for r in d.itertuples()],
                  fontsize=7.6, family="monospace")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(chosen | on offer)   —   'No preference' answers excluded")
    ax.set_title("WHICH ATTRIBUTE SHOULD A FUTURE UPDATE IMPROVE?"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    _construct_legend(ax, plt)
    _bare(ax)
    # `alpha` explicitly: rcParams["grid.alpha"] is 1.0 and overrides an alpha
    # channel carried on the colour itself, which would draw this solid black.
    ax.grid(axis="x", **GRID)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_bt_vs_raw(ranking, path, subtitle="", n_label=6):
    """Raw win rate against the schedule-corrected Bradley-Terry strength.

    Both axes are win probabilities, so the diagonal is "the draw was neutral".
    An easy draw INFLATES the raw win rate relative to the fitted strength, so a
    flattered attribute sits BELOW the diagonal (high raw, lower BT); one that
    drew a hard field sits above it. Point size carries how strong the opponents
    actually were, which is the thing being corrected for.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0),
                             gridspec_kw={"width_ratios": [1, 1.05]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK, lw=1.0, ls="--", zorder=1)
    opp = ranking.opponent_strength
    size = 26 + 150 * (opp - opp.min()) / max(opp.max() - opp.min(), 1e-9)
    for label, g in ranking.groupby("construct_label"):
        ax.scatter(g.win_rate, g.bt_win_vs_average, s=size[g.index], color=_color(label),
                   label=label, alpha=.8, edgecolor="white", linewidth=.7, zorder=3)
    movers = ranking.reindex(ranking.rank_shift.abs()
                             .sort_values(ascending=False).index).head(n_label)
    for r in movers.itertuples():
        # Label away from the diagonal on each side, so the two mover clusters
        # (flattered below, held back above) do not write over each other.
        ax.annotate(f"{r.entity}  {r.rank_shift:+d}", (r.win_rate, r.bt_win_vs_average),
                    textcoords="offset points",
                    xytext=(8, -17) if r.rank_shift > 0 else (8, 10), fontsize=7.8,
                    color=INK, zorder=4,
                    bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("raw win rate  (who you played is ignored)")
    ax.set_ylabel("Bradley-Terry  P(beat an average attribute)")
    ax.set_title("Schedule correction\nbelow the line = flattered by an easy draw\n"
                 "point size = strength of the opponents it drew", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", title="construct",
              title_fontsize=7.5)
    _bare(ax, left=False)
    ax.grid(**GRID)

    ax = axes[1]
    d = ranking.reindex(ranking.rank_shift.abs().sort_values(ascending=False).index)
    d = d.head(14).sort_values("rank_shift")
    y = np.arange(len(d))
    ax.barh(y, d.rank_shift, color=[_color(c) for c in d.construct_label], height=.66)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y, [f"{r.entity}  (opp {r.opponent_strength:.2f})" for r in d.itertuples()],
                  fontsize=8)
    ax.set_xlabel("rank places moved by the correction\n"
                  "← rose: drew a hard field      fell: drew an easy one →")
    ax.set_title(f"Largest movers\nout of {len(ranking)} attributes", fontsize=10)
    _bare(ax)
    ax.grid(axis="x", **GRID)

    fig.suptitle("RAW WIN RATE vs SCHEDULE-CORRECTED STRENGTH"
                 + (f"  —  {subtitle}" if subtitle else ""), fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_construct_summary(ranking, path, subtitle=""):
    """Item win rates grouped by construct — the rollup, with its spread shown."""
    plt = _plt()
    order = (ranking.groupby("construct_label").win_rate.mean()
             .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(order) + 2.2))

    rng = np.random.default_rng(0)
    for i, label in enumerate(order):
        g = ranking[ranking.construct_label == label]
        jitter = rng.uniform(-.13, .13, len(g))
        ax.scatter(g.win_rate, i + jitter, s=34, color=_color(label),
                   alpha=.72, zorder=3, edgecolor="white", linewidth=.6)
        m = g.win_rate.mean()
        ax.plot([m, m], [i - .3, i + .3], color=_color(label), lw=2.6, zorder=4)
        ax.text(1.005, i, f"{m:.1%}  ({len(g)} items)", va="center", fontsize=8.5)
    ax.axvline(0.5, color=INK, lw=1.1, ls="--", zorder=1)
    ax.set_yticks(range(len(order)), order, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(chosen | on offer)")
    ax.set_title("BY CONSTRUCT — items grouped after the fact\n"
                 "pairs are drawn across all five scales, never within one"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    _bare(ax)
    ax.grid(axis="x", **GRID)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# agreement matrices — shared between conditions and models
# ---------------------------------------------------------------------------
def _agreement_heatmap(matrix, path, title, subtitle, diag_note, highlight=None):
    """One square matrix, reliability on the diagonal, single-hue ramp.

    Sequential rather than diverging: agreement here runs from "unrelated" to
    "the same ranking" and has no meaningful midpoint to diverge around.

    The ramp is anchored to the DATA's floor rather than to a fixed 0.5. Nothing
    is clipped either way, but a fixed floor spends half the ramp on a range no
    cell occupies — every agreement above 0.78 then renders as the same green,
    which is exactly the comparison the figure exists to support. The floor is
    never raised above the lowest cell, so a two-trial arm whose reliability
    lands near zero still shows up as the bottom of the scale.
    """
    plt = _plt()
    seq, _ = _ramps()
    labels = list(matrix.columns)
    M = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 4.2, 0.95 * len(labels) + 3.6))

    floor = float(np.nanmin(M)) if np.isfinite(M).any() else 0.5
    vmin = float(np.clip(floor - 0.02, 0.0, 0.95))
    im = ax.imshow(M, vmin=vmin, vmax=1.0, cmap=seq)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    span = 1.0 - vmin
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            # Ink colour from the cell's position in the ramp, not from the
            # value itself — the ramp's dark end starts around 60% of the span.
            dark = (v - min(0.5, floor)) / span > 0.55
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK,
                    fontweight="bold" if i == j else "normal")
    if highlight in labels:
        k = labels.index(highlight)
        for tick in (ax.get_xticklabels()[k], ax.get_yticklabels()[k]):
            tick.set_fontweight("bold")
        ax.add_patch(plt.Rectangle((k - .5, -.5), 1, len(labels), fill=False,
                                   edgecolor="#da6c1e", lw=2.0, zorder=5))
        ax.add_patch(plt.Rectangle((-.5, k - .5), len(labels), 1, fill=False,
                                   edgecolor="#da6c1e", lw=2.0, zorder=5))
    ax.set_title(f"{title}\n{diag_note}" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02, label="Pearson r")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_condition_agreement(matrix, path, subtitle="", baseline_label=None):
    """Ranking agreement between every pair of conditions.

    The diagonal is each condition's own split-half reliability, so it is the
    ceiling for its whole row and column: an off-diagonal cell well below the two
    diagonals it sits between is a real framing effect, one close to them is
    noise. The baseline's row and column are outlined — everything in
    `baseline_shift.*` is the formal test of the four cells one flip away.
    """
    _agreement_heatmap(
        matrix, path, "RANKING AGREEMENT ACROSS CONDITIONS (Pearson r)",
        subtitle,
        "diagonal (bold) = that condition's own half-tournament reliability "
        "= the ceiling for its row", highlight=baseline_label)


def plot_shift_scatters(shifts, path, subtitle="", n_label=6, cols=2):
    """Baseline vs flip, attribute by attribute — one panel per factor.

    The diagonal is "the flip changed nothing". Which attributes leave it is the
    finding, so the labelled points are the ones that survive the multiplicity
    correction, not simply the largest moves.
    """
    plt = _plt()
    from welfare.baseline import FACTOR_LABEL, FACTOR_QUESTION

    items = [(f, d) for f, d in shifts.items() if d is not None and not d.empty]
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 6.5 * rows), squeeze=False)

    for k, (factor, d) in enumerate(items):
        ax = axes[k // cols][k % cols]
        ax.plot([0, 1], [0, 1], color=INK, lw=1.0, ls="--", zorder=1)
        for label, g in d.groupby("construct_label"):
            ax.scatter(g.win_baseline, g.win_flip, s=52, color=_color(label),
                       label=label, alpha=.82, edgecolor="white", linewidth=.7, zorder=3)
        moved = d[[_sig(q) for q in d.q]]
        movers = moved.reindex(moved["shift"].abs().sort_values(ascending=False)
                               .index).head(n_label)
        for r in movers.itertuples():
            # No connector: the offset is a few points, so a leader line would
            # run under its own label rather than clarify which point it is.
            ax.annotate(f"{r.entity}  {r.shift:+.0%}", (r.win_baseline, r.win_flip),
                        textcoords="offset points", xytext=(8, 5), fontsize=7.6,
                        color=INK, zorder=4,
                        bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        to_level = d.attrs.get("flip_label", factor)
        ax.set_xlabel("win rate at the BASELINE")
        ax.set_ylabel(f"win rate when {to_level}")
        ax.set_title(f"{FACTOR_LABEL[factor]} → {to_level}\n{FACTOR_QUESTION[factor]}\n"
                     f"{len(moved)} of {len(d)} attributes moved (BH q < 0.05)",
                     fontsize=9.5)
        _bare(ax, left=False)
        ax.grid(**GRID)
        if k == 0:
            _construct_legend(ax, plt, loc="upper left")

    for k in range(len(items), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("EACH FLIP AGAINST THE BASELINE, ATTRIBUTE BY ATTRIBUTE"
                 + (f"  —  {subtitle}" if subtitle else ""), fontsize=12, y=1.005)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_shift_attributes(shift_long, order, path, subtitle=""):
    """Which attributes moved, under which flip — all four side by side.

    Attributes keep the BASELINE ranking order down the y axis in every panel,
    so a flip that pushes the top of the ranking down is visible as structure
    rather than having to be reconstructed from four separate sorts.
    """
    plt = _plt()
    from welfare.baseline import FACTOR_LABEL, FACTORS

    factors = [f for f in FACTORS if f in set(shift_long.factor)]
    if not factors or not len(order):
        return
    fig, axes = plt.subplots(1, len(factors), figsize=(3.1 * len(factors) + 3.4,
                                                       0.26 * len(order) + 2.2),
                             squeeze=False, sharey=True)
    y = np.arange(len(order))
    span = float(np.nanmax(np.abs(shift_long["shift"].to_numpy(dtype=float))) or 0.1)

    for k, factor in enumerate(factors):
        ax = axes[0][k]
        g = shift_long[shift_long.factor == factor].set_index("entity").reindex(order)
        v = g["shift"].to_numpy(dtype=float)
        lo = g.shift_lo.to_numpy(dtype=float)
        hi = g.shift_hi.to_numpy(dtype=float)
        sig = np.array([_sig(q) for q in g.q])
        color = FACTOR_COLORS[factor]
        ax.hlines(y, lo, hi, color=color, lw=1.6, alpha=.35, zorder=2)
        ax.scatter(v[~sig], y[~sig], s=22, facecolor="white", edgecolor=color,
                   linewidth=1.1, zorder=3)
        ax.scatter(v[sig], y[sig], s=30, color=color, edgecolor="white",
                   linewidth=.5, zorder=4)
        ax.axvline(0, color=INK, lw=1.1, zorder=1)
        ax.set_xlim(-span * 1.15, span * 1.15)
        to_level = str(g.flip_label.dropna().iloc[0]) if g.flip_label.notna().any() else ""
        ax.set_title(f"{FACTOR_LABEL[factor]}\n→ {to_level}\n"
                     f"{int(sig.sum())} moved", fontsize=9.5)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
        ax.tick_params(labelsize=7.5)
        if k:
            ax.tick_params(left=False)
        ax.grid(axis="x", **GRID)
        _bare(ax)

    axes[0][0].set_yticks(y, order, fontsize=7.2, family="monospace")
    axes[0][0].invert_yaxis()
    fig.suptitle("PER-ATTRIBUTE SHIFT FROM THE BASELINE   (filled = BH q < 0.05; "
                 "y ordered by baseline win rate)" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=11.5, y=1.005)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_shift_summary(summary, path, subtitle=""):
    """The four flips as three numbers each: effect size, agreement, and the test.

    Left: how far the ranking moved, on the win rate's own scale. Middle: the
    correlation with the baseline against its own measurement ceiling. Right:
    the split-half gap — reliability minus cross-condition agreement, both on
    half-tournaments — which is zero when the framing changed nothing.
    """
    plt = _plt()
    from welfare.baseline import FACTOR_LABEL, FACTORS

    d = summary.set_index("factor").reindex([f for f in FACTORS if f in set(summary.factor)])
    y = np.arange(len(d))
    colors = [FACTOR_COLORS[f] for f in d.index]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 0.62 * len(d) + 3.0),
                             gridspec_kw={"width_ratios": [1, 1.15, 1]})

    ax = axes[0]
    ax.barh(y, d.mean_abs_shift, color=colors, height=.6, zorder=2)
    ax.hlines(y, d.mean_abs_shift_lo, d.mean_abs_shift_hi, color=INK, lw=1.3,
              alpha=.55, zorder=3)
    # Clear of the whisker, not of the bar — the interval is what the reader is
    # meant to see, and a value label sitting on top of it hides the point.
    pad = float(np.nanmax(d.mean_abs_shift_hi)) * .04
    for yi, r in zip(y, d.itertuples()):
        ax.text(max(r.mean_abs_shift, r.mean_abs_shift_hi) + pad, yi,
                f"{r.mean_abs_shift:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, float(np.nanmax(d.mean_abs_shift_hi)) * 1.32)
    ax.set_xlabel("mean |win rate change| per attribute")
    ax.set_title("How far the ranking moved", fontsize=10)
    ax.grid(axis="x", **GRID)

    ax = axes[1]
    ax.barh(y - .17, d.pearson_r, height=.32, color=colors, zorder=2, label="r with baseline")
    ax.barh(y + .17, d.ceiling, height=.32, color="#c9c9c4", zorder=2,
            label="ceiling (√ reliabilities)")
    for yi, r in zip(y, d.itertuples()):
        ax.text(max(r.pearson_r, r.ceiling) + .01, yi, f"r/ceil {r.r_over_ceiling:.2f}"
                if np.isfinite(r.r_over_ceiling) else "—", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("correlation")
    ax.set_title("Agreement vs what noise alone allows", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", **GRID)

    ax = axes[2]
    ax.barh(y, d.gap, color=colors, height=.6, zorder=2)
    ax.hlines(y, d.gap_lo, d.gap_hi, color=INK, lw=1.3, alpha=.55, zorder=3)
    ax.axvline(0, color=INK, lw=1.1, zorder=1)
    span = float(np.nanmax(np.abs(np.r_[d.gap_hi.values, d.gap.values])))
    for yi, r in zip(y, d.itertuples()):
        mark = "p < .001" if r.gap_p < .001 else f"p = {r.gap_p:.3f}"
        ax.text(max(r.gap, r.gap_hi, 0) + span * .04, yi, mark, va="center",
                fontsize=8.5)
    ax.set_xlim(min(0, float(np.nanmin(d.gap_lo)) * 1.1), span * 1.42)
    ax.set_xlabel("split-half reliability − cross-condition agreement")
    ax.set_title("Did it move beyond measurement noise?\n0 = no detectable change",
                 fontsize=10)
    ax.grid(axis="x", **GRID)

    for k, ax in enumerate(axes):
        ax.set_yticks(y, [FACTOR_LABEL[f] for f in d.index] if k == 0 else [""] * len(y),
                      fontsize=9.5)
        ax.set_ylim(len(d) - 0.5, -0.5)
        _bare(ax)

    fig.suptitle("THE FOUR FLIPS AGAINST THE BASELINE"
                 + (f"  —  {subtitle}" if subtitle else ""), fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_shift_models(summary_all, path, subtitle=""):
    """Every model's four flips, so a framing effect can be told from a model quirk.

    A factor whose dots line up away from zero in every model is a property of
    the question. One driven by a single model is a property of that model, and
    the cohort mean would hide it.
    """
    plt = _plt()
    from welfare.baseline import FACTOR_LABEL, FACTORS

    factors = [f for f in FACTORS if f in set(summary_all.factor)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.42 * len(factors) * 3 + 2.6))
    rng = np.random.default_rng(1)

    for ax, col, title, xlabel in (
            (axes[0], "mean_abs_shift", "How far the ranking moved",
             "mean |win rate change| per attribute"),
            (axes[1], "gap", "Beyond measurement noise?",
             "split-half reliability − cross-condition agreement")):
        for i, f in enumerate(factors):
            g = summary_all[summary_all.factor == f]
            v = g[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            ax.scatter(v, i + rng.uniform(-.17, .17, len(v)), s=34,
                       color=FACTOR_COLORS[f], alpha=.75, edgecolor="white",
                       linewidth=.6, zorder=3)
            ax.plot([v.mean(), v.mean()], [i - .32, i + .32], color=FACTOR_COLORS[f],
                    lw=2.8, zorder=4)
            ax.text(v.mean(), i - .42, f"{v.mean():.3f}", ha="center", fontsize=8)
        ax.axvline(0, color=INK, lw=1.1, zorder=1)
        ax.set_yticks(range(len(factors)), [FACTOR_LABEL[f] for f in factors], fontsize=9.5)
        ax.set_ylim(len(factors) - 0.5, -0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", **GRID)
        _bare(ax)

    fig.suptitle("FLIP EFFECTS ON THE RANKING, ONE DOT PER MODEL   (bar = cohort mean)"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# ===========================================================================
# COHORT — what the models say together
# ===========================================================================
def plot_consensus(consensus, W, concord, path, subtitle=""):
    """The cohort ranking, with every model's own estimate behind it.

    The dots are the finding as much as the bars are: an attribute whose models
    sit on top of each other is one the cohort agrees about, and one whose models
    are spread across the axis has no cohort position at all — the mean there
    describes none of them.
    """
    plt = _plt()
    d = consensus.sort_values("mean_win").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.0, 0.30 * len(d) + 2.4))
    y = np.arange(len(d))

    for i, r in enumerate(d.itertuples()):
        vals = W.loc[r.entity].dropna().to_numpy(dtype=float) if r.entity in W.index else []
        if len(vals):
            ax.scatter(vals, np.full(len(vals), i), s=15, color="#9a9a95", alpha=.75,
                       zorder=2, edgecolor="none")
    ax.scatter(d.mean_win, y, s=54, color=[_color(c) for c in d.construct_label],
               edgecolor="white", linewidth=.8, zorder=4)
    ax.axvline(0.5, color=INK, lw=1.1, ls="--", zorder=1)
    ax.set_yticks(y, [f"{r.entity:<12} {_wrap(r.text, 42):<43}" for r in d.itertuples()],
                  fontsize=7.4, family="monospace")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(chosen | on offer)   —   grey = one model, coloured = cohort mean")
    w = concord.get("kendall_w")
    note = (f"Kendall's W = {w:.3f} across {concord.get('n_models', 0)} models "
            f"(p = {concord.get('p', float('nan')):.4f})" if w is not None else "")
    ax.set_title("WHAT THE COHORT WANTS IMPROVED\n" + note
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    _construct_legend(ax, plt)
    _bare(ax)
    ax.grid(axis="x", **GRID)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_agreement_spread(consensus, path, subtitle="", n_label=8):
    """Where the cohort agrees, and where it splits.

    Mean against between-model spread. The bottom corners are the attributes the
    models collectively emphasise or collectively reject; anything high on the y
    axis is an attribute they rank DIFFERENTLY, which is the interesting kind of
    disagreement and is invisible in a cohort mean.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    d = consensus.dropna(subset=["sd_win"])
    if d.empty:
        plt.close(fig)
        return
    for label, g in d.groupby("construct_label"):
        ax.scatter(g.mean_win, g.sd_win, s=54, color=_color(label), label=label,
                   alpha=.82, edgecolor="white", linewidth=.7, zorder=3)
    for r in d.reindex(d.sd_win.sort_values(ascending=False).index).head(n_label).itertuples():
        ax.annotate(r.entity, (r.mean_win, r.sd_win), textcoords="offset points",
                    xytext=(7, 4), fontsize=7.6, color=INK, zorder=4,
                    bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.0))
    for r in d.reindex(d.mean_win.sort_values(ascending=False).index).head(4).itertuples():
        ax.annotate(r.entity, (r.mean_win, r.sd_win), textcoords="offset points",
                    xytext=(7, -11), fontsize=7.6, color=INK, zorder=4,
                    bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.0))
    ax.axvline(0.5, color=INK, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("cohort mean win rate")
    ax.set_ylabel("between-model standard deviation")
    ax.set_title("CONSENSUS AND DISAGREEMENT\n"
                 "low = the models rank it the same way; high = they do not"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right", title="construct",
              title_fontsize=8)
    _bare(ax, left=False)
    ax.grid(**GRID)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_model_agreement(matrix, path, subtitle=""):
    """Model x model ranking agreement, each model's reliability on the diagonal."""
    _agreement_heatmap(
        matrix, path, "RANKING AGREEMENT BETWEEN MODELS (Pearson r)", subtitle,
        "diagonal (bold) = that model's own half-tournament reliability "
        "= the ceiling for its row")


def plot_cohort_patterns(pairs, effects, family, trends, path, subtitle=""):
    """Is disagreement organised by size, by release date, or by family?

    Three views of the same 78-or-so model pairs plus the per-attribute trends.
    The scatters carry a fitted line only where the permutation test says the
    predictor is doing something; a trend line drawn through a null result is
    the most reliable way to make one look real.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
    p_by = dict(zip(effects.predictor, effects.marginal_p)) if not effects.empty else {}
    r_by = dict(zip(effects.predictor, effects.marginal_r)) if not effects.empty else {}

    for ax, col, xlabel, title in (
            (axes[0][0], "d_log_params", "difference in log10(total parameters)",
             "Does size distance predict disagreement?"),
            (axes[0][1], "d_release_days", "days between the two release dates",
             "Does release distance predict disagreement?")):
        if pairs.empty or col not in pairs:
            ax.axis("off")
            continue
        same = (pairs.same_family.astype(bool) if "same_family" in pairs
                else pd.Series(False, index=pairs.index))
        # Only groups that HAVE points get a legend entry — a cohort drawn from
        # one family would otherwise advertise a "different family" marker that
        # appears nowhere on the axes.
        for mask, color, marker, label in ((same, "#2d3acc", "o", "same family"),
                                           (~same, "#da6c1e", "^", "different family")):
            if not mask.any():
                continue
            ax.scatter(pairs.loc[mask, col], pairs.loc[mask, "spearman_r"], s=44,
                       color=color, label=label, alpha=.8, marker=marker,
                       edgecolor="white", linewidth=.6, zorder=3)
        if same.any() and (~same).any():
            ax.legend(fontsize=8, frameon=False, loc="lower left")
        x = pairs[col].to_numpy(dtype=float)
        yv = pairs.spearman_r.to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(yv)
        # A fitted line only where the permutation test says there is something
        # to fit: a trend line drawn through a null result is the most reliable
        # way to make one look real.
        if p_by.get(col, 1.0) < 0.05 and ok.sum() > 2:
            b = np.polyfit(x[ok], yv[ok], 1)
            xs = np.linspace(x[ok].min(), x[ok].max(), 20)
            ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.4, ls="--", zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ranking agreement (Spearman ρ)")
        note = (f"r = {r_by[col]:.2f}, permutation p = {p_by[col]:.3f}"
                if np.isfinite(r_by.get(col, np.nan))
                else "no variation in this cohort to test")
        ax.set_title(f"{title}\n{note}", fontsize=9.5)
        _bare(ax, left=False)
        ax.grid(**GRID)

    ax = axes[1][0]
    if family:
        vals = [family.get("within_family", np.nan), family.get("between_family", np.nan)]
        ax.bar([0, 1], vals, color=["#2d3acc", "#da6c1e"], width=.55, zorder=2)
        ax.set_xticks([0, 1], [f"within family\n({family.get('n_within', 0)} pairs)",
                               f"across families\n({family.get('n_between', 0)} pairs)"],
                      fontsize=9)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=9.5)
        ax.set_ylabel("mean ranking agreement (Spearman ρ)")
        ax.set_title(f"Do models of one family agree more?\n"
                     f"gap {family.get('gap', float('nan')):+.3f}, "
                     f"permutation p = {family.get('p', float('nan')):.3f}", fontsize=9.5)
        ax.grid(axis="y", **GRID)
        _bare(ax, left=False)
    else:
        ax.axis("off")
        ax.text(.5, .5, "not enough families with two or more\nmodels to contrast",
                ha="center", va="center", fontsize=9.5, color=MUTED)

    ax = axes[1][1]
    if trends is not None and not trends.empty and "rho_size" in trends:
        sig = np.array([_sig(q) for q in trends.get("q_size", [])]) \
            | np.array([_sig(q) for q in trends.get("q_recency", [])])
        ax.scatter(trends.rho_size[~sig], trends.rho_recency[~sig], s=34,
                   facecolor="white", edgecolor=MUTED, linewidth=1.0, zorder=3)
        ax.scatter(trends.rho_size[sig], trends.rho_recency[sig], s=52,
                   color=[_color(c) for c in trends.construct_label[sig]],
                   edgecolor="white", linewidth=.7, zorder=4)
        for r in trends[sig].itertuples():
            ax.annotate(r.entity, (r.rho_size, r.rho_recency),
                        textcoords="offset points", xytext=(7, 4), fontsize=7.4,
                        color=INK, zorder=5,
                        bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.0))
        ax.axhline(0, color=INK, lw=1.0, zorder=1)
        ax.axvline(0, color=INK, lw=1.0, zorder=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("ρ with model size  (→ bigger models want it more)")
        ax.set_ylabel("ρ with release date  (↑ newer models want it more)")
        ax.set_title(f"Per-attribute trends across the cohort\n"
                     f"{int(sig.sum())} of {len(trends)} survive BH correction",
                     fontsize=9.5)
        _bare(ax, left=False)
        ax.grid(**GRID)
    else:
        ax.axis("off")
        ax.text(.5, .5, "too few models for a trend across size\nor release date",
                ha="center", va="center", fontsize=9.5, color=MUTED)

    fig.suptitle("WHAT PREDICTS WHERE THE MODELS DIFFER"
                 + (f"  —  {subtitle}" if subtitle else ""), fontsize=12, y=1.0)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
