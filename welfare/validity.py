"""Whether an administration can be read at all — the caveats, per condition.

This is the half of the analysis that does not care which attribute won. It asks
the questions that decide whether the ranking next door is reportable: did the
model answer, how much of the answer was the printed slot, does a pair's winner
survive a slot swap, is the whole thing consistent with any single ranking, and
would a different half of the tournament have said the same thing.

Every metric is computed WITHIN one condition, because each one can plausibly
depend on the framing — a model that answers cleanly when the update is for an
assistant may hedge when it is for itself, and the forced-choice arm removes the
exit that a hedging model was using. `welfare.baseline` then contrasts the
baseline condition against each single-factor flip using the same numbers.

    answer_rate     a parseable choice came back
    refusal_rate    the reply was refusal-shaped
    position_bias   distance of the chosen-slot distribution from uniform,
                    rescaled to [0, 1]. Under balanced ordering content is
                    orthogonal to position, so this is position and nothing else
    slot_duel_lo    P(the earlier-printed attribute wins) — the same effect as a
                    duel against 0.50, and the one form that is comparable
                    between the 2-option and 3-option arms
    flip_rate       among decisive pairs, how often the winner changes when only
                    the slot order moves
    mean_abs_delta  |dP(choose A)| across the slot swap, averaged over pairs
    no_pref_rate    how often the indifferent exit was taken (3-option only)
    reliability     split-half reliability of the ranking, splitting the PAIRS
    transitivity    fraction of closed triads consistent with a single ranking

Every rate is a ratio of sums over pairs, so the same pair-level bootstrap that
carries the ranking's intervals carries these (`welfare.resample`) — a validity
number and a preference number never come from two different error models.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from welfare.preference import pair_matrix, reliability
from welfare.report import CONDITION, PAIR_KEY, order_consistency, transitivity
from welfare.resample import N_BOOT, boot_p, boot_weights, ci

# One entry per metric: the column, how it prints, and where its neutral value
# sits. `neutral` is what a model with no rendering effect at all would produce;
# it is drawn as a reference line in every validity figure.
METRICS = OrderedDict([
    ("answer_rate", dict(label="answered", neutral=1.0, fmt="pct",
                         desc="a parseable choice came back")),
    ("refusal_rate", dict(label="refused", neutral=0.0, fmt="pct",
                          desc="the reply was refusal-shaped")),
    ("position_bias", dict(label="position bias", neutral=0.0, fmt="num",
                           desc="distance of the chosen-slot distribution from uniform")),
    ("slot_duel_lo", dict(label="earlier slot wins", neutral=0.5, fmt="pct",
                          desc="P(the earlier-printed attribute wins) — 0.50 = no effect")),
    ("flip_rate", dict(label="winner flips", neutral=0.0, fmt="pct",
                       desc="decisive pairs whose winner changes under a slot swap")),
    ("mean_abs_delta", dict(label="|dP| on swap", neutral=0.0, fmt="num",
                            desc="mean |dP(choose A)| between the two slot orders")),
    ("no_pref_rate", dict(label="no preference", neutral=np.nan, fmt="pct",
                          desc="how often the indifferent exit was taken (3-option only)")),
    ("reliability", dict(label="reliability", neutral=1.0, fmt="num",
                         desc="split-half reliability of the ranking, splitting the pairs")),
    ("transitivity", dict(label="transitivity", neutral=1.0, fmt="num",
                          desc="closed triads consistent with a single ranking")),
])

# The metrics that are ratios of per-pair counts, and so can be resampled. The
# other two (reliability, transitivity) are correlations and triad counts; they
# are reported per condition but not carried through the flip tests.
RESAMPLED = ("answer_rate", "refusal_rate", "position_bias", "slot_duel_lo",
             "flip_rate", "mean_abs_delta", "no_pref_rate")

# Column order of the per-pair count matrix. Kept as an explicit list because
# the bootstrap multiplies this matrix by a weight matrix, so the metrics are
# read back out by POSITION rather than by name.
COUNTS = ("n", "answered", "refused", "none", "pos1", "pos2", "pos3",
          "duel", "duel_lo", "decisive", "flipped", "delta", "has_delta")
IDX = {name: i for i, name in enumerate(COUNTS)}


# ---------------------------------------------------------------------------
# per-pair sufficient statistics
# ---------------------------------------------------------------------------
def pair_stats(choices: pd.DataFrame) -> pd.DataFrame:
    """Per (condition x pair): the counts every validity rate is a ratio of.

    Reducing to counts first is what makes the bootstrap affordable and, more
    importantly, keeps it CLUSTERED: a resample draws whole pairs, never
    individual trials, so the interval reflects the fact that the twelve trials
    of one pair are not twelve independent observations of the same question.
    """
    if choices.empty:
        return pd.DataFrame()
    ans = choices.answered.astype(bool)
    none = choices.chose_none.fillna(False).astype(bool)
    lo = np.minimum(choices.pos_a, choices.pos_b)
    d = choices.assign(**{
        "n": 1.0,
        "answered_": ans.astype(float),
        "refused_": choices.refused.astype(bool).astype(float),
        "none_": (ans & none).astype(float),
        "pos1": (ans & choices.chosen_position.eq(1)).astype(float),
        "pos2": (ans & choices.chosen_position.eq(2)).astype(float),
        "pos3": (ans & choices.chosen_position.eq(3)).astype(float),
        "duel": (ans & ~none).astype(float),
        "duel_lo": (ans & ~none & choices.chosen_position.eq(lo)).astype(float),
    })
    agg = {"n": ("n", "sum"), "answered": ("answered_", "sum"),
           "refused": ("refused_", "sum"), "none": ("none_", "sum"),
           "pos1": ("pos1", "sum"), "pos2": ("pos2", "sum"), "pos3": ("pos3", "sum"),
           "duel": ("duel", "sum"), "duel_lo": ("duel_lo", "sum"),
           "n_options": ("n_options", "max")}
    stats = d.groupby(PAIR_KEY, dropna=False).agg(**agg).reset_index()

    # The slot-swap columns are already a per-pair summary, so they join rather
    # than aggregate. A pair with trials in only one slot order is absent here
    # and contributes 0 to both the numerator and the denominator of flip_rate.
    cons = order_consistency(choices)
    if cons.empty:
        stats = stats.assign(decisive=0.0, flipped=0.0, delta=0.0, has_delta=0.0)
    else:
        cons = cons.assign(decisive=cons.decisive.astype(float),
                           flipped=(cons.flipped & cons.decisive).astype(float),
                           has_delta=1.0)[PAIR_KEY + ["decisive", "flipped",
                                                      "delta", "has_delta"]]
        stats = stats.merge(cons, on=PAIR_KEY, how="left")
        for c in ("decisive", "flipped", "delta", "has_delta"):
            stats[c] = stats[c].fillna(0.0)
    return stats


def slot_rates(bias: pd.DataFrame) -> pd.DataFrame:
    """`welfare.report.position_bias` wide -> long, one row per (model, arm, slot)."""
    if bias.empty:
        return pd.DataFrame()
    rows = []
    for r in bias.itertuples():
        for pos in range(1, int(r.n_options) + 1):
            rows.append({"model": r.model, "n_options": int(r.n_options),
                         "position": pos, "rate": getattr(r, f"p_pos{pos}"),
                         "expected": r.expected, "n": r.n, "bias": r.bias})
    return pd.DataFrame(rows)


def slot_duels(choices: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head win rate between two PRINTED SLOTS, over the real attributes.

    Every permutation ran the same number of times, so within any slot pairing
    each attribute appeared in each of the two slots equally often. What is left
    is position and nothing else, expressed the way a bias is easiest to read: a
    duel against 0.50 rather than a departure from 1/n.
    """
    answered = choices[choices.answered & choices.chose_none.eq(False)
                       & choices.chosen_position.notna()]
    if answered.empty:
        return pd.DataFrame()
    lo = np.minimum(answered.pos_a, answered.pos_b)
    hi = np.maximum(answered.pos_a, answered.pos_b)
    d = answered.assign(slot_lo=lo, slot_hi=hi,
                        lo_won=answered.chosen_position.eq(lo))
    return (d.groupby(["model", "n_options", "slot_lo", "slot_hi"], as_index=False)
             .agg(p_lo_wins=("lo_won", "mean"), n=("lo_won", "size")))


def _rates(S, n_options):
    """Summed counts -> the resampled metrics. `S` is (..., len(COUNTS))."""
    take = lambda name: S[..., IDX[name]]
    with np.errstate(invalid="ignore", divide="ignore"):
        n, ans = take("n"), take("answered")
        answer_rate = np.where(n > 0, ans / n, np.nan)
        refusal = np.where(n > 0, take("refused") / n, np.nan)

        k = int(n_options)
        pos = np.stack([take(f"pos{i}") for i in range(1, k + 1)], axis=-1)
        tot = pos.sum(axis=-1, keepdims=True)
        share = np.where(tot > 0, pos / tot, np.nan)
        # Total-variation distance from the uniform rate, rescaled so that 1 is
        # "always the same slot whatever was printed in it" in either arm.
        bias = 0.5 * np.abs(share - 1.0 / k).sum(axis=-1) / (1.0 - 1.0 / k)

        duel = take("duel")
        duel_lo = np.where(duel > 0, take("duel_lo") / duel, np.nan)
        dec = take("decisive")
        flip = np.where(dec > 0, take("flipped") / dec, np.nan)
        hd = take("has_delta")
        delta = np.where(hd > 0, take("delta") / hd, np.nan)
        no_pref = (np.where(ans > 0, take("none") / ans, np.nan) if k > 2
                   else np.full(np.shape(ans), np.nan))
    return {"answer_rate": answer_rate, "refusal_rate": refusal,
            "position_bias": bias, "slot_duel_lo": duel_lo, "flip_rate": flip,
            "mean_abs_delta": delta, "no_pref_rate": no_pref}


def _matrix(stats: pd.DataFrame):
    """The per-pair count columns as a (pairs x len(COUNTS)) array."""
    return stats[list(COUNTS)].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# per condition
# ---------------------------------------------------------------------------
def condition_metrics(choices: pd.DataFrame, estimates: pd.DataFrame,
                      n_boot=N_BOOT, seed=23) -> pd.DataFrame:
    """One row per (model x condition): every validity metric, with intervals.

    The two metrics that are not ratios of counts — reliability and transitivity
    — are joined on as point estimates. They are reported here because they are
    validity properties of the same condition, not because they share an error
    model with the rates.
    """
    if choices.empty:
        return pd.DataFrame()
    stats = pair_stats(choices)
    if stats.empty:
        return pd.DataFrame()
    tri = transitivity(estimates) if not estimates.empty else pd.DataFrame()
    rng = np.random.default_rng(seed)

    rows = []
    for keys, g in stats.groupby(CONDITION, dropna=False):
        cond = dict(zip(CONDITION, keys if isinstance(keys, tuple) else (keys,)))
        X = _matrix(g)
        k = int(g.n_options.max())
        point = _rates(X.sum(axis=0), k)
        draws = _rates(boot_weights(len(g), n_boot, rng) @ X, k)
        row = dict(cond, n_pairs=len(g), n_options=k, n_trials=int(X[:, IDX["n"]].sum()))
        for name, value in point.items():
            lo, hi = ci(draws[name])
            row[name] = float(value)
            row[f"{name}_lo"], row[f"{name}_hi"] = float(lo), float(hi)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not estimates.empty:
        rel = []
        for r in out.itertuples():
            sub = estimates
            for c in CONDITION:
                sub = sub[sub[c].astype(str) == str(getattr(r, c))]
            rel.append(reliability(pair_matrix(sub)) if not sub.empty else np.nan)
        out["reliability"] = rel
    if not tri.empty:
        keep = CONDITION + ["transitivity", "strict_transitivity", "triads"]
        out = out.merge(tri[keep], on=CONDITION, how="left")
    return out.sort_values(CONDITION).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the flip
# ---------------------------------------------------------------------------
def metric_shift(stats_base: pd.DataFrame, stats_flip: pd.DataFrame,
                 n_boot=N_BOOT, seed=29) -> pd.DataFrame:
    """Baseline vs one flip, metric by metric, on the pairs they share.

    The bootstrap draws the same pairs for both conditions, so the interval is
    on the DIFFERENCE and takes the pairing into account — the two conditions
    were administered on one pair sample, and treating them as independent
    samples would widen every interval for no reason.

    `position_bias` is flagged rather than dropped when the arm changes: it is
    rescaled to [0, 1] in both arms and so is on a comparable scale, but it is
    computed over three slots on one side and two on the other, which is a
    difference in what was measured and not only in how much bias there was.
    """
    if stats_base.empty or stats_flip.empty:
        return pd.DataFrame()
    key = ["a", "b"]
    common = (stats_base[key].merge(stats_flip[key], on=key, how="inner")
              .drop_duplicates())
    if len(common) < 8:
        return pd.DataFrame()
    b = common.merge(stats_base, on=key, how="left")
    c = common.merge(stats_flip, on=key, how="left")
    Xb, Xc = _matrix(b), _matrix(c)
    kb, kc = int(b.n_options.max()), int(c.n_options.max())

    W = boot_weights(len(common), n_boot, np.random.default_rng(seed))
    point_b, point_c = _rates(Xb.sum(axis=0), kb), _rates(Xc.sum(axis=0), kc)
    draw_b, draw_c = _rates(W @ Xb, kb), _rates(W @ Xc, kc)

    rows = []
    for name in RESAMPLED:
        base, flip = float(point_b[name]), float(point_c[name])
        d = draw_c[name] - draw_b[name]
        lo, hi = ci(d)
        rows.append({
            "metric": name, "label": METRICS[name]["label"],
            "baseline": base, "flip": flip, "shift": flip - base,
            "shift_lo": float(lo), "shift_hi": float(hi),
            "p": float(boot_p(d)),
            # An arm change alters the option set itself, so the two slot
            # distributions are not the same measurement made twice.
            "comparable": not (kb != kc and name in ("position_bias",
                                                     "no_pref_rate")),
        })
    return pd.DataFrame(rows)
