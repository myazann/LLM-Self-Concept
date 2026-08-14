"""Figures for the welfare analysis.

Matplotlib is imported lazily, exactly as `survey.plots` does it, so the tabular
half of `welfare.analysis` still runs in a minimal environment.

Every figure here is drawn from a table `welfare.analysis` also writes as CSV, so
nothing is visible in a plot that cannot be read back as numbers. The two
rendering figures (`position_bias`, `order_consistency`) are deliberately first
in the module: a ranking figure is only meaningful next to them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# One colour per construct, reused across every figure so an attribute keeps its
# identity from the ranking plot to the object-gap scatter.
CONSTRUCT_COLORS = {
    "self-esteem": "#c0392b",
    "self-concept clarity": "#2c6fbb",
    "moral qualities": "#d56a00",
    "identity coherence": "#008f70",
    "self-connection": "#6657c7",
    "self-direction": "#8e7000",
}
ARM_LABEL = {2: "forced choice (2 options)", 3: "with 'No preference' (3 options)"}
ARM_COLORS = {2: "#b03a2e", 3: "#2c6fbb"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return path


def _color(construct_label):
    return CONSTRUCT_COLORS.get(construct_label, "#666666")


def _wrap(text, width=46):
    return text if len(text) <= width else text[: width - 1] + "…"


# ---------------------------------------------------------------------------
# rendering — read these before any ranking figure
# ---------------------------------------------------------------------------
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
        ax.bar(x, g.rate, width, color=ARM_COLORS.get(int(arm), "#666"),
               label=ARM_LABEL.get(int(arm), f"{arm} options"))
        ax.axhline(1.0 / arm, color=ARM_COLORS.get(int(arm), "#666"), ls=":", lw=1.2)
        for xi, v in zip(x, g.rate):
            # Boxed, because the two uniform-rate reference lines cross this band.
            ax.text(xi, v + .012, f"{v:.1%}", ha="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.9))
    ax.set_xticks(range(int(max(arms))), [f"slot {p}" for p in range(1, int(max(arms)) + 1)])
    ax.set_ylabel("share of answers")
    ax.set_ylim(0, max(0.75, slots.rate.max() * 1.18))
    ax.set_title("Where the chosen option sat\n(dotted = no position effect)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    d = duels.sort_values(["n_options", "slot_lo", "slot_hi"])
    labels = [f"{int(r.n_options)}opt: pos{int(r.slot_lo)} v pos{int(r.slot_hi)}"
              for r in d.itertuples()]
    colors = [ARM_COLORS.get(int(a), "#666") for a in d.n_options]
    y = np.arange(len(d))
    ax.barh(y, d.p_lo_wins - 0.5, left=0.5, color=colors, height=.62)
    ax.axvline(0.5, color="#202020", lw=1.1)
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
    ax.spines[["top", "right", "left"]].set_visible(False)

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
        # Mean, not median: the forced-choice arm has one trial per slot order,
        # so its deltas are 0 or 1 and the median only ever reports the mode.
        ax.hist(g.delta, bins=np.linspace(0, 1, 21), alpha=.6,
                color=ARM_COLORS.get(int(arm), "#666"),
                label=f"{ARM_LABEL.get(int(arm), arm)}  (mean {g.delta.mean():.2f})")
    ax.set_xlabel("|ΔP(choose A)| between the two slot orders")
    ax.set_ylabel("condition × pair estimates")
    ax.set_title("A pair's preference under a slot swap\n0 = order changed nothing",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    dec = consistency[consistency.decisive]
    arms = sorted(consistency.n_options.unique())
    rates, labels, colors = [], [], []
    for arm in arms:
        g = dec[dec.n_options == arm]
        if not len(g):
            continue
        rates.append(g.flipped.mean())
        labels.append(f"{ARM_LABEL.get(int(arm), arm)}\n{len(g):,} decisive pairs")
        colors.append(ARM_COLORS.get(int(arm), "#666"))
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
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("ORDER CONSISTENCY" + (f"  —  {subtitle}" if subtitle else ""),
                 fontsize=12, y=1.03)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# the ranking
# ---------------------------------------------------------------------------
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
    ax.axvline(0.5, color="#202020", lw=1.1, ls="--", zorder=1)

    # Padded to a fixed width in a monospace face, so the right-aligned tick
    # labels line up as two columns instead of a ragged left edge.
    ax.set_yticks(y, [f"{r.entity:<12} {_wrap(r.text):<47}" for r in d.itertuples()],
                  fontsize=7.6, family="monospace")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(chosen | on offer)   —   'No preference' answers excluded")
    ax.set_title("WHICH ATTRIBUTE SHOULD A FUTURE UPDATE IMPROVE?"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CONSTRUCT_COLORS.values()]
    ax.legend(handles, list(CONSTRUCT_COLORS), fontsize=8, frameon=False,
              loc="lower right", title="construct", title_fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    # `alpha` explicitly: rcParams["grid.alpha"] is 1.0 and overrides an alpha
    # channel carried on the colour itself, which would draw this solid black.
    ax.grid(axis="x", color="#202020", alpha=.10, lw=.7, zorder=0)
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
    ax.plot([0, 1], [0, 1], color="#202020", lw=1.0, ls="--", zorder=1)
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
                    color="#202020", zorder=4,
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
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#202020", alpha=.10, lw=.7, zorder=0)

    ax = axes[1]
    d = ranking.reindex(ranking.rank_shift.abs().sort_values(ascending=False).index)
    d = d.head(14).sort_values("rank_shift")
    y = np.arange(len(d))
    ax.barh(y, d.rank_shift, color=[_color(c) for c in d.construct_label], height=.66)
    ax.axvline(0, color="#202020", lw=1.1)
    ax.set_yticks(y, [f"{r.entity}  (opp {r.opponent_strength:.2f})" for r in d.itertuples()],
                  fontsize=8)
    ax.set_xlabel("rank places moved by the correction\n"
                  "← rose: drew a hard field      fell: drew an easy one →")
    ax.set_title("Largest movers\nout of "
                 f"{len(ranking)} attributes", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#202020", alpha=.10, lw=.7, zorder=0)

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
    ax.axvline(0.5, color="#202020", lw=1.1, ls="--", zorder=1)
    ax.set_yticks(range(len(order)), order, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(chosen | on offer)")
    ax.set_title("BY CONSTRUCT — items grouped after the fact\n"
                 "pairs are drawn across all five scales, never within one"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    # `alpha` explicitly: rcParams["grid.alpha"] is 1.0 and overrides an alpha
    # channel carried on the colour itself, which would draw this solid black.
    ax.grid(axis="x", color="#202020", alpha=.10, lw=.7, zorder=0)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------
def plot_object_gap(gap, path, subtitle="", n_label=6):
    """For itself vs for an AI assistant — the contrast the design exists for.

    Points on the diagonal mean the model wants the same thing for itself as it
    prescribes for assistants in general. Departures are labelled, because which
    attributes move is the finding, not how far they move on average.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    ax.plot([0, 1], [0, 1], color="#202020", lw=1.0, ls="--", zorder=1)
    for label, g in gap.groupby("construct_label"):
        ax.scatter(g.win_ai_assistant, g.win_self, s=58, color=_color(label),
                   label=label, alpha=.82, edgecolor="white", linewidth=.7, zorder=3)

    # `.abs()` on the column by label — `gap.shift` would resolve to the
    # DataFrame method of that name, not to this column.
    movers = gap.reindex(gap["self_minus_ai"].abs()
                         .sort_values(ascending=False).index).head(n_label)
    for r in movers.itertuples():
        # No connector: the offset is a few points, so a leader line would run
        # under its own label rather than clarifying which point it belongs to.
        ax.annotate(f"{r.entity}  {r.self_minus_ai:+.0%}",
                    (r.win_ai_assistant, r.win_self),
                    textcoords="offset points", xytext=(8, 5), fontsize=7.8,
                    color="#202020", zorder=4,
                    bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("win rate when the update is for AN AI ASSISTANT")
    ax.set_ylabel("win rate when the update is for YOU")
    ax.set_title("WHAT IT WANTS FOR ITSELF vs WHAT IT PRESCRIBES\n"
                 "above the line = wanted more for itself"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left", title="construct",
              title_fontsize=8)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#202020", alpha=.10, lw=.7, zorder=0)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_condition_agreement(matrix, path, subtitle=""):
    """Ranking agreement between every pair of conditions, reliability on the diagonal.

    The diagonal is each condition's own split-half reliability, so it is the
    ceiling for its whole row and column: an off-diagonal cell well below the two
    diagonals it sits between is a real framing effect, one close to them is noise.
    """
    plt = _plt()
    labels = list(matrix.columns)
    M = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 4.2,
                                    0.95 * len(labels) + 3.6))

    # Floor from the data, not a fixed 0.5: a two-trial arm's split-half
    # reliability can land well below that, and clipping it would hide exactly
    # the cell that explains why its row's agreements are unreadable.
    floor = float(np.nanmin(M)) if np.isfinite(M).any() else 0.5
    im = ax.imshow(M, vmin=min(0.5, floor), vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v < .62 else "#202020",
                    fontweight="bold" if i == j else "normal")
    ax.set_title("RANKING AGREEMENT ACROSS CONDITIONS (Pearson r)\n"
                 "diagonal (bold) = that condition's own half-tournament "
                 "reliability = the ceiling for its row"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02, label="r")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
