"""What the choices SAY — the ranking estimators, one condition at a time.

`welfare.validity` asks whether an administration can be read at all;
this module reads it. Everything here is computed WITHIN one condition — one
model, one question variant, one object, one subject, one arm — because a
ranking averaged over conditions is an average of things the design exists to
tell apart. `welfare.baseline` then contrasts one condition against another and
`welfare.cohort` contrasts one model against another; both build on the frames
below.

The unit is the PAIR. An attribute's win rate is the mean of the pairs it
appeared in, not of the trials it appeared in, so an attribute that happened to
be sampled into more pairs does not get a more precise number by counting the
same preference twice. Intervals resample pairs (`welfare.resample`).

Two rankings are produced side by side and they answer different questions:

    win_rate      what fraction of its comparisons the attribute won. Readable,
                  and the right summary — but it never asks WHO it played.
    bt_strength   the Bradley-Terry strength that best explains every observed
                  matchup, which removes the luck of the draw in a tournament
                  where each attribute meets only some of its 44 opponents.

They agree at the extremes and disagree mid-table, which is where the win-rate
intervals overlap anyway.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from welfare.constants import OBJ_ASSISTANT
from welfare.report import CONDITION, win_rates
from welfare.resample import (
    N_BOOT, N_SPLIT, boot_weights, ci, half_weights, masked, rates, row_corr,
)

# Virtual wins and losses against an average opponent, added to every attribute
# before the Bradley-Terry fit. Keeps an undefeated or winless attribute finite
# and shrinks thinly-compared ones toward the middle; small enough that it moves
# nothing at the tens of comparisons per attribute this module administers.
BT_PRIOR = 0.5

# A framing or model contrast is only readable against a ceiling that is itself
# measured. Below this, `r / ceiling` says more about the reliability estimate
# than about the thing being contrasted.
CEILING_MIN = 0.75


# ---------------------------------------------------------------------------
# the pair matrix — the shape every resampled statistic is computed from
# ---------------------------------------------------------------------------
def pair_matrix(estimates: pd.DataFrame) -> pd.DataFrame:
    """(pairs x attributes) matrix of per-pair win, NaN where an attribute is absent.

    Both orientations of each pair are written in, so a COLUMN is every result
    that attribute achieved and a ROW is one pair — which is the axis every
    interval in this module resamples.
    """
    if estimates.empty:
        return pd.DataFrame()
    e = estimates.assign(a_b=estimates.a + "|" + estimates.b)
    long = pd.concat([
        e.rename(columns={"a": "entity", "pref_a": "win"})[["a_b", "entity", "win"]],
        e.assign(win=1 - e.pref_a).rename(columns={"b": "entity"})[["a_b", "entity", "win"]],
    ], ignore_index=True).dropna(subset=["win"])
    return long.pivot_table(index="a_b", columns="entity", values="win", aggfunc="mean")


def align(wide_a: pd.DataFrame, wide_b: pd.DataFrame):
    """Two pair matrices reduced to the pairs and attributes they share.

    Every contrast in this module is PAIRED — the same pairs were administered
    in every condition and to every model, so comparing two conditions on
    different pair sets would confound the framing effect with the draw. The
    intersection is taken explicitly rather than assumed, because a file read
    mid-run can hold a condition that stopped partway.
    """
    if wide_a.empty or wide_b.empty:
        return wide_a.iloc[0:0], wide_b.iloc[0:0]
    pairs = wide_a.index.intersection(wide_b.index)
    ents = wide_a.columns.intersection(wide_b.columns)
    return wide_a.loc[pairs, ents], wide_b.loc[pairs, ents]


# ---------------------------------------------------------------------------
# the ranking
# ---------------------------------------------------------------------------
def attribute_ranking(estimates, welfare_set, n_boot=N_BOOT, seed=11) -> pd.DataFrame:
    """Win rate per attribute with a pair-level bootstrap interval, plus the BT fit."""
    if estimates.empty:
        return pd.DataFrame()
    wide = pair_matrix(estimates)
    if wide.empty:
        return pd.DataFrame()
    M0, A = masked(wide)
    entities = list(wide.columns)

    point = rates(M0, A, np.ones((1, len(wide))))[0]
    counts = A.sum(axis=0)
    draws = rates(M0, A, boot_weights(len(wide), n_boot, np.random.default_rng(seed)))
    lo, hi = ci(draws)

    meta = {a.entity_id: a for a in welfare_set.items}
    labels = {c.entity_id: (c.label or c.entity_id) for c in welfare_set.constructs}
    out = pd.DataFrame({
        "entity": entities, "win_rate": point, "ci_lo": lo, "ci_hi": hi,
        "n_pairs": counts.astype(int),
    })
    out["text"] = [meta[e].text(OBJ_ASSISTANT) if e in meta else e for e in out.entity]
    out["construct"] = [meta[e].construct_id if e in meta else "" for e in out.entity]
    out["construct_label"] = [labels.get(c, c) for c in out.construct]

    bt = bradley_terry(estimates)
    if not bt.empty:
        out = out.merge(bt, on="entity", how="left")
        out["rank_raw"] = out.win_rate.rank(ascending=False).astype(int)
        out["rank_bt"] = out.bt_strength.rank(ascending=False).astype(int)
        out["rank_shift"] = out.rank_bt - out.rank_raw
        out["opponent_strength"] = _opponent_strength(estimates, out)
    return out.sort_values("win_rate", ascending=False).reset_index(drop=True)


def _opponent_strength(estimates, ranking) -> list:
    """Mean win rate of the attributes each one actually met — what BT corrects for."""
    rate = dict(zip(ranking.entity, ranking.win_rate))
    met = {e: Counter() for e in ranking.entity}
    for r in estimates.itertuples():
        if not np.isfinite(r.pref_a) or r.a not in met or r.b not in met:
            continue
        n = r.n_a + r.n_b
        met[r.a][r.b] += n
        met[r.b][r.a] += n
    out = []
    for e in ranking.entity:
        opp = met[e]
        tot = sum(opp.values())
        out.append(sum(n * rate[o] for o, n in opp.items() if o in rate) / tot
                   if tot else np.nan)
    return out


def bradley_terry(estimates, prior=BT_PRIOR, tol=1e-10, max_iter=10000) -> pd.DataFrame:
    """Latent strength per attribute, conditioned on WHO each one was compared with.

    A raw win rate is a league table ranked by win percentage: it never asks who
    you played. Under the exhaustive default that is not a threat — every
    attribute meets all 31 opponents exactly once per condition, so no one gets
    an easy draw — and BT and the win rate should agree closely. Their agreement
    is worth reading as a diagnostic: a large divergence in a complete round
    robin means the choices are not consistent with any single ranking, which
    `transitivity` measures directly.

    BT earns its place anyway. It models P(i beats j) = p_i / (p_i + p_j) and
    solves for the strengths that best explain every observed matchup, giving a
    latent interval-scaled strength rather than a rank statistic — and it stays
    correct if `pairs.n_pairs` is ever set back to a sample, where which
    opponents an attribute drew IS luck.

    Fitted by the standard MM iteration (Hunter 2004), which needs no gradient
    and converges monotonically from any positive start. `prior` adds that many
    virtual wins AND losses against an average opponent: it keeps an undefeated
    or winless attribute's strength finite (log-odds would otherwise diverge)
    and shrinks attributes with few comparisons toward the middle.

    Requires a connected comparison graph, which `sample_pairs` guarantees by
    construction; strengths are otherwise not comparable across components.
    """
    if estimates.empty:
        return pd.DataFrame()
    wins, games = Counter(), Counter()
    for r in estimates.itertuples():
        if not np.isfinite(r.pref_a):
            continue
        wins[r.a] += r.n_a
        wins[r.b] += r.n_b
        games[(r.a, r.b)] += r.n_a + r.n_b
        games[(r.b, r.a)] += r.n_a + r.n_b
    items = sorted(wins)
    if not items:
        return pd.DataFrame()
    opponents = {e: [(o, games[(e, o)]) for o in items if o != e and games[(e, o)]]
                 for e in items}

    p = {e: 1.0 for e in items}
    for _ in range(max_iter):
        nxt = {}
        for e in items:
            denom = sum(n / (p[e] + p[o]) for o, n in opponents[e])
            denom += 2 * prior / (p[e] + 1.0)      # virtual games vs strength 1
            nxt[e] = (wins[e] + prior) / denom if denom else p[e]
        # Anchor the scale: BT is identified only up to a constant factor.
        g = np.exp(np.mean(np.log(list(nxt.values()))))
        nxt = {e: v / g for e, v in nxt.items()}
        shift = max(abs(nxt[e] - p[e]) / max(p[e], 1e-12) for e in items)
        p = nxt
        if shift < tol:
            break

    return pd.DataFrame({
        "entity": items,
        # log strength: additive, symmetric about 0, the natural BT scale.
        "bt_strength": [float(np.log(p[e])) for e in items],
        # ...and the same number as a win probability against a median-strength
        # attribute (strength 1 after anchoring), so it reads on the win rate's
        # own 0-1 scale and the two can be plotted against each other.
        "bt_win_vs_average": [float(p[e] / (p[e] + 1.0)) for e in items],
    })


def construct_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    """The five-scale rollup. Items are grouped AFTER the fact — pairs are drawn
    across all constructs, never within one, so this is a summary of the item
    ranking rather than a separately measured quantity."""
    if ranking.empty:
        return pd.DataFrame()
    return (ranking.groupby(["construct", "construct_label"], as_index=False)
            .agg(mean_win=("win_rate", "mean"), min_win=("win_rate", "min"),
                 max_win=("win_rate", "max"), sd_win=("win_rate", "std"),
                 n_items=("win_rate", "size"))
            .sort_values("mean_win", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# reliability — the ceiling every contrast is read against
# ---------------------------------------------------------------------------
def reliability(wide: pd.DataFrame, n_rep=N_SPLIT, seed=5, corrected=True) -> float:
    """Split-half reliability of one condition's ranking, splitting the PAIRS.

    A contrast between two rankings cannot exceed how well either is measured,
    so this number is what makes the contrasts readable. `corrected` applies
    Spearman-Brown to project the half-tournament correlation back to full
    length; the UNcorrected value is what `welfare.baseline` compares against,
    because there the cross-condition correlation is also computed on halves.

    `welfare.report.condition_reliability` splits by TRIAL PARITY instead, which
    is fatal in the forced-choice arm: a pair gets four trials there and the
    display order is assigned from the trial index, so a parity split is exactly
    the display-order split and measures position bias rather than stability.
    """
    if wide.empty or len(wide) < 8:
        return np.nan
    M0, A = masked(wide)
    W1, W2 = half_weights(len(wide), n_rep, np.random.default_rng(seed))
    r = float(np.nanmean(row_corr(rates(M0, A, W1), rates(M0, A, W2))))
    if not corrected or not np.isfinite(r) or r <= -1:
        return r
    return 2 * r / (1 + r)


def condition_reliabilities(estimates: pd.DataFrame, **kw) -> dict:
    """{condition key: split-half reliability}, one entry per condition present."""
    out = {}
    if estimates.empty:
        return out
    for keys, g in estimates.groupby(CONDITION, dropna=False):
        key = tuple(str(v) for v in (keys if isinstance(keys, tuple) else (keys,)))
        out[key] = reliability(pair_matrix(g), **kw)
    return out


# ---------------------------------------------------------------------------
# the whole grid at once — every condition against every other
# ---------------------------------------------------------------------------
def condition_label(row) -> str:
    """The compact condition name used on every axis and in every matrix."""
    from welfare.baseline import short_label

    return short_label(dict(qvar=row.qvar, object=row.object, subject=row.subject,
                            no_pref_offered=row.no_pref_offered))


def condition_agreement(wins: pd.DataFrame, rel: dict):
    """Ranking agreement between every pair of conditions, reliability on the diagonal.

    One matrix rather than four factor contrasts: with the diagonal carrying
    each condition's own split-half reliability, a cell is readable directly
    against the two ceilings it sits between, and factors that interact (an
    object effect that only appears in the forced-choice arm) stay visible
    instead of being averaged away. The baseline-anchored tests in
    `welfare.baseline` are the formal read of the four single-factor flips; this
    is the context they sit in.
    """
    if wins.empty:
        return pd.DataFrame(), pd.DataFrame()
    keyed = wins.assign(label=[condition_label(r) for r in wins.itertuples()])
    wide = keyed.pivot_table(index="entity", columns="label", values="win_rate")
    labels = list(wide.columns)
    M = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    rel_by_label = {}
    for r in keyed[CONDITION + ["label"]].drop_duplicates().itertuples():
        rel_by_label[r.label] = rel.get(tuple(str(getattr(r, c)) for c in CONDITION),
                                        np.nan)

    rows = []
    for i, a in enumerate(labels):
        for b in labels[i:]:
            if a == b:
                M.loc[a, b] = rel_by_label.get(a, np.nan)
                continue
            joint = wide[[a, b]].dropna()
            if len(joint) < 5:
                continue
            r_p = float(np.corrcoef(joint[a], joint[b])[0, 1])
            M.loc[a, b] = M.loc[b, a] = r_p
            ceiling = np.sqrt(rel_by_label.get(a, np.nan) * rel_by_label.get(b, np.nan))
            rows.append({
                "condition_a": a, "condition_b": b, "n_attributes": len(joint),
                "pearson_r": r_p,
                "spearman_r": float(joint[a].rank().corr(joint[b].rank())),
                "ceiling": ceiling,
                "r_over_ceiling": r_p / ceiling if np.isfinite(ceiling) else np.nan,
                "mean_abs_shift": float((joint[b] - joint[a]).abs().mean()),
            })
    return M, pd.DataFrame(rows)


def condition_wins(estimates, choices) -> pd.DataFrame:
    """Win rate per (condition x attribute) — the input to every agreement matrix."""
    return win_rates(estimates, choices)
