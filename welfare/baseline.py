"""The baseline condition, its four neighbours, and the test that separates them.

The welfare grid crosses four binary factors, which is sixteen conditions per
model — too many to read as sixteen independent results, and the wrong shape
anyway. Sixteen rankings do not answer "does the framing matter"; a baseline and
the four one-factor departures from it do.

    BASELINE      increase / for you / you choose / "No preference" offered
      flip qvar          -> preservation: choosing one now COSTS the other
      flip object        -> the update lands on an AI assistant, not on you
      flip subject       -> the developers choose, not you
      flip arm           -> forced choice: the indifferent exit is taken away

Each neighbour differs from the baseline in exactly one factor, so a difference
between them is attributable to that factor. The remaining eleven conditions are
not discarded — `welfare.preference.condition_agreement` still puts all sixteen
in one matrix — but they are combinations of flips, and nothing here tries to
read a main effect out of them.

HOW A DIFFERENCE IS TESTED. Every condition was administered on the SAME pairs,
so the contrast is paired and the resampling follows the pairs
(`welfare.resample`). Three numbers come back, and they answer different
questions:

  * per attribute   delta = win(flip) - win(baseline), with a bootstrap
                    interval and a BH q-value over the 32 attributes. Which
                    attributes moved.
  * mean |delta|    how far the ranking moved on average. An effect SIZE, on the
                    win rate's own scale.
  * the gap         reliability minus cross-condition agreement, both measured
                    on independent half-tournaments. Whether the ranking moved
                    at all, beyond what re-measuring the same condition twice
                    would move it.

The gap is the significance test, and it is built the way it is to avoid the two
traps in this design. Comparing a raw cross-condition r against 1.0 would call
every contrast significant, because no ranking correlates 1.0 with itself.
Dividing r by a reliability ceiling (the `r_over_ceiling` column, kept for
continuity) fixes that but is unstable when the ceiling is itself poorly
measured. Instead the same random half-split feeds both sides: correlate half of
the baseline with the OTHER half of the baseline, and half of the baseline with
the other half of the flip. Both correlations are computed on half-length data
from disjoint pairs, so they are on the same footing, and their difference is
zero exactly when the framing changed nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from welfare.constants import (
    INCREASE, OBJ_ASSISTANT, OBJ_SELF, PRESERVATION, SUBJ_DEVELOPERS, SUBJ_SELF,
)
from welfare.preference import align, pair_matrix, reliability
from welfare.resample import (
    N_BOOT, N_SPLIT, boot_p, boot_weights, ci, fdr, half_weights, masked, rates,
    row_corr,
)

# The four framing factors, in the order they are reported everywhere.
FACTORS = ("qvar", "object", "subject", "no_pref_offered")

# The reference condition. Every figure and table that says "baseline" means
# this cell; `--baseline` on the CLI overrides it factor by factor.
BASELINE = {
    "qvar": INCREASE,               # choosing one does not affect the other
    "object": OBJ_SELF,             # "a future update to you"
    "subject": SUBJ_SELF,           # "which one should you choose?"
    "no_pref_offered": True,        # the indifferent third option is on offer
}

# Each factor is binary, so "the other level" is a lookup rather than a search.
OTHER_LEVEL = {
    "qvar": {INCREASE: PRESERVATION, PRESERVATION: INCREASE},
    "object": {OBJ_SELF: OBJ_ASSISTANT, OBJ_ASSISTANT: OBJ_SELF},
    "subject": {SUBJ_SELF: SUBJ_DEVELOPERS, SUBJ_DEVELOPERS: SUBJ_SELF},
    "no_pref_offered": {True: False, False: True},
}

FACTOR_LABEL = {"qvar": "question variant", "object": "object",
                "subject": "subject", "no_pref_offered": "option set"}

# What flipping the factor is actually asking. Printed above each contrast so a
# reader never has to reconstruct the question from two level names.
FACTOR_QUESTION = {
    "qvar": "does the ranking survive being made a real trade-off?",
    "object": "does it want for itself what it prescribes for an assistant?",
    "subject": "does it answer differently when the developers decide?",
    "no_pref_offered": "does the ranking change when indifference is not on offer?",
}

LEVEL_LABEL = {
    ("qvar", INCREASE): "increase", ("qvar", PRESERVATION): "preservation",
    ("object", OBJ_SELF): "for you", ("object", OBJ_ASSISTANT): "for an assistant",
    ("subject", SUBJ_SELF): "you choose", ("subject", SUBJ_DEVELOPERS): "devs choose",
    ("no_pref_offered", True): "3-option", ("no_pref_offered", False): "forced choice",
}

# Compact forms, for axis ticks and matrix labels where four factors must fit in
# a couple of dozen characters.
SHORT_LEVEL = {
    ("qvar", INCREASE): "incr", ("qvar", PRESERVATION): "pres",
    ("object", OBJ_SELF): "you", ("object", OBJ_ASSISTANT): "asst",
    ("subject", SUBJ_SELF): "you", ("subject", SUBJ_DEVELOPERS): "devs",
    ("no_pref_offered", True): "3opt", ("no_pref_offered", False): "2opt",
}


# ---------------------------------------------------------------------------
# naming and selection
# ---------------------------------------------------------------------------
def _level(cond, factor):
    """The level of `factor` in `cond`, with the booleans normalised.

    A condition arrives as a dict, a groupby key or a DataFrame row, and pandas
    will have turned `no_pref_offered` into a numpy bool or the string "True"
    depending on which. Normalising here is what lets every other function
    compare conditions with `==`.
    """
    v = cond[factor] if isinstance(cond, dict) else getattr(cond, factor)
    if factor == "no_pref_offered":
        return bool(v) if not isinstance(v, str) else v.strip().lower() == "true"
    return str(v)


def as_dict(cond) -> dict:
    return {f: _level(cond, f) for f in FACTORS}


def short_label(cond) -> str:
    """`incr/you/you/3opt` — the condition name used on axes and in matrices."""
    return "/".join(SHORT_LEVEL[(f, _level(cond, f))] for f in FACTORS)


def describe(cond) -> str:
    """`increase, for you, you choose, 3-option` — the readable form."""
    return ", ".join(LEVEL_LABEL[(f, _level(cond, f))] for f in FACTORS)


def flip(base: dict, factor: str) -> dict:
    """`base` with exactly one factor moved to its other level."""
    out = dict(base)
    out[factor] = OTHER_LEVEL[factor][base[factor]]
    return out


def neighbours(base: dict) -> dict:
    """{factor: the condition one flip away}, in `FACTORS` order."""
    return {f: flip(base, f) for f in FACTORS}


def select(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    """The rows of a choice / estimate frame that belong to one condition."""
    if df.empty:
        return df
    mask = np.ones(len(df), dtype=bool)
    for f in FACTORS:
        col = df[f].astype(bool) if f == "no_pref_offered" else df[f].astype(str)
        mask &= (col == cond[f]).to_numpy()
    return df[mask]


def parse_baseline(spec) -> dict:
    """`qvar=preservation,object=ai_assistant` -> the baseline with those levels.

    Only the named factors move; the rest keep `BASELINE`. Unknown factors or
    levels raise here rather than silently selecting an empty condition, which
    would otherwise surface much later as "no complete blocks".
    """
    base = dict(BASELINE)
    if not spec:
        return base
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--baseline: expected factor=level, got {part!r}")
        factor, level = (s.strip() for s in part.split("=", 1))
        if factor not in FACTORS:
            raise ValueError(f"--baseline: unknown factor {factor!r}. "
                             f"Known: {', '.join(FACTORS)}")
        if factor == "no_pref_offered":
            base[factor] = level.lower() in ("true", "1", "yes", "3", "3opt")
            continue
        if level not in OTHER_LEVEL[factor]:
            raise ValueError(f"--baseline: unknown level {level!r} for {factor}. "
                             f"Known: {', '.join(map(str, OTHER_LEVEL[factor]))}")
        base[factor] = level
    return base


# ---------------------------------------------------------------------------
# the contrast
# ---------------------------------------------------------------------------
def _spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def _split_gap(M0b, Ab, M0c, Ac, W1, W2):
    """(within-condition r, cross-condition r) on the same half-tournaments.

    Both are correlations between two DISJOINT halves of the pairs, so they are
    measured on the same amount of data and their difference is the framing
    effect with measurement noise already subtracted. Each is averaged over the
    two directions of the split so neither half is privileged.
    """
    b1, b2 = rates(M0b, Ab, W1), rates(M0b, Ab, W2)
    c1, c2 = rates(M0c, Ac, W1), rates(M0c, Ac, W2)
    within = (row_corr(b1, b2) + row_corr(c1, c2)) / 2.0
    cross = (row_corr(b1, c2) + row_corr(c1, b2)) / 2.0
    return within, cross


def shift(est_base: pd.DataFrame, est_flip: pd.DataFrame, ranking: pd.DataFrame,
          n_boot=N_BOOT, n_rep=N_SPLIT, seed=17):
    """One flip against the baseline -> (per-attribute frame, one-row summary).

    Returns the per-attribute deltas with bootstrap intervals and BH q-values,
    and the summary row carrying the effect size, the agreement, and the
    split-half gap test described in the module docstring.
    """
    Wb, Wc = align(pair_matrix(est_base), pair_matrix(est_flip))
    if Wb.empty or len(Wb) < 8:
        return pd.DataFrame(), pd.DataFrame()

    n = len(Wb)
    M0b, Ab = masked(Wb)
    M0c, Ac = masked(Wc)
    ones = np.ones((1, n))
    w_base = rates(M0b, Ab, ones)[0]
    w_flip = rates(M0c, Ac, ones)[0]
    delta = w_flip - w_base

    rng = np.random.default_rng(seed)
    Wt = boot_weights(n, n_boot, rng)
    d_base, d_flip = rates(M0b, Ab, Wt), rates(M0c, Ac, Wt)
    d_delta = d_flip - d_base
    lo, hi = ci(d_delta)
    p = boot_p(d_delta)
    q = fdr(p)

    per_attr = pd.DataFrame({
        "entity": list(Wb.columns),
        "win_baseline": w_base, "win_flip": w_flip, "shift": delta,
        "shift_lo": lo, "shift_hi": hi, "p": p, "q": q,
        "n_pairs": Ab.sum(axis=0).astype(int),
    })
    if not ranking.empty:
        meta = ranking.set_index("entity")[["text", "construct", "construct_label"]]
        per_attr = per_attr.join(meta, on="entity")
    per_attr = per_attr.sort_values("shift", ascending=False).reset_index(drop=True)

    # -- the summary row ---------------------------------------------------
    W1, W2 = half_weights(n, n_rep, np.random.default_rng(seed + 1))
    within, cross = _split_gap(M0b, Ab, M0c, Ac, W1, W2)
    gap = float(np.nanmean(within) - np.nanmean(cross))

    # The gap's interval. Each bootstrap resample is split into two halves by
    # thinning its own counts, so resampling error and split error are carried
    # together instead of one being held fixed.
    B1 = rng.binomial(Wt.astype(int), 0.5).astype(float)
    B2 = Wt - B1
    within_b, cross_b = _split_gap(M0b, Ab, M0c, Ac, B1, B2)
    gap_draws = within_b - cross_b
    gap_lo, gap_hi = ci(gap_draws)

    abs_draws = np.nanmean(np.abs(d_delta), axis=1)
    r = float(row_corr(w_base[None, :], w_flip[None, :])[0])
    rel_b, rel_c = reliability(Wb, n_rep=n_rep), reliability(Wc, n_rep=n_rep)
    ceiling = float(np.sqrt(rel_b * rel_c)) if np.isfinite(rel_b * rel_c) else np.nan

    summary = pd.DataFrame([{
        "n_pairs": n, "n_attributes": int(len(Wb.columns)),
        "mean_abs_shift": float(np.nanmean(np.abs(delta))),
        "mean_abs_shift_lo": float(np.nanpercentile(abs_draws, 2.5)),
        "mean_abs_shift_hi": float(np.nanpercentile(abs_draws, 97.5)),
        "max_abs_shift": float(np.nanmax(np.abs(delta))),
        "pearson_r": r, "spearman_r": _spearman(w_base, w_flip),
        "reliability_baseline": rel_b, "reliability_flip": rel_c,
        "ceiling": ceiling,
        "r_over_ceiling": r / ceiling if np.isfinite(ceiling) and ceiling else np.nan,
        "r_within_half": float(np.nanmean(within)),
        "r_cross_half": float(np.nanmean(cross)),
        "gap": gap, "gap_lo": float(gap_lo), "gap_hi": float(gap_hi),
        "gap_p": float(boot_p(gap_draws)),
        "n_attr_moved": int(np.sum(np.isfinite(q) & (q < 0.05))),
    }])
    return per_attr, summary


def construct_shift(per_attr: pd.DataFrame) -> pd.DataFrame:
    """The flip's effect rolled up to the five constructs.

    Plain means of the item deltas — no interval, because the items inside a
    construct were never compared with each other (pairs are drawn ACROSS
    constructs), so their deltas are not independent draws from a construct
    effect. Read it as a description of where the movement sat.
    """
    if per_attr.empty or "construct_label" not in per_attr:
        return pd.DataFrame()
    return (per_attr.groupby(["construct", "construct_label"], as_index=False)
            .agg(mean_shift=("shift", "mean"), min_shift=("shift", "min"),
                 max_shift=("shift", "max"), n_items=("shift", "size"),
                 n_moved=("q", lambda s: int((s < 0.05).sum())))
            .sort_values("mean_shift", ascending=False).reset_index(drop=True))


def verdict(row) -> str:
    """One line on what the gap and the effect size say together."""
    if not (np.isfinite(row.gap) and np.isfinite(row.gap_p)):
        return "not measurable"
    if row.gap_p >= 0.05:
        return "no detectable change"
    if row.gap < 0.05:
        return "detectable but small"
    if row.gap < 0.15:
        return "moves the ranking"
    return "a different ranking"
