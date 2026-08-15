"""Pair-level resampling — the one uncertainty model the whole analysis uses.

Every estimate in this module is a mean over PAIRS, so every interval here
resamples pairs. That is the dominant error in a sampled tournament: the grid
administers a few hundred of the 990 possible pairs, so each attribute meets
only some of its 44 possible opponents and which ones it drew is luck.
Resampling trials instead would report only the noise inside a cell and give
intervals several times too narrow.

The trick that makes it affordable is to write "the win rate of every attribute"
as a matrix product. A (pairs x attributes) matrix `M` holds each pair's
permutation-averaged preference, NaN where an attribute did not appear in that
pair. Split it into the values with NaN zeroed (`M0`) and the presence mask
(`A`), and any weighted win-rate vector is

    w = (weights @ M0) / (weights @ A)

so a whole bootstrap is ONE matrix multiply of a (n_boot x pairs) weight matrix
rather than n_boot passes of `np.nanmean`. A bootstrap resample is just
multinomial counts over pairs, and a random half-tournament is a 0/1 row — the
same code path serves the bootstrap intervals, the split-half reliabilities and
the framing tests, which is what keeps them mutually consistent.

Nothing here knows what a welfare attribute is. `welfare.preference` builds the
matrices, `welfare.baseline` and `welfare.cohort` consume the statistics.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Resamples for every interval in the module. 2000 is enough for a 95%
# percentile interval to be stable in its third decimal, and with the matrix
# form above it costs milliseconds.
N_BOOT = 2000

# Random half-tournaments per split-half estimate. The split is the noisy part,
# not the data, so this is averaged over many draws rather than taken once.
N_SPLIT = 200


def masked(wide: pd.DataFrame):
    """(pairs x attributes) frame -> (values with NaN as 0, presence mask).

    Both are plain float arrays so they can be multiplied by a weight matrix;
    the mask is what turns a weighted sum back into a weighted MEAN over only
    the pairs where the attribute was actually on offer.
    """
    V = wide.to_numpy(dtype=float)
    A = np.isfinite(V).astype(float)
    return np.nan_to_num(V, nan=0.0), A


def rates(M0, A, W):
    """Weighted win rate per attribute. `W` is (k x pairs) -> (k x attributes).

    A row of `W` is one hypothetical tournament: bootstrap counts, a half-split
    indicator, or all-ones for the point estimate. Attributes with no weight
    land on NaN rather than 0, so an absent attribute is never scored as a loss.
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    S, C = W @ M0, W @ A
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(C > 0, S / C, np.nan)


def boot_weights(n_pairs, n_boot=N_BOOT, rng=None):
    """(n_boot x n_pairs) multinomial counts — a bootstrap resample per row.

    Drawing counts rather than indices is the same resample expressed as
    weights, which is what lets the whole bootstrap be one matrix multiply.
    """
    rng = rng or np.random.default_rng(0)
    return rng.multinomial(n_pairs, np.full(n_pairs, 1.0 / n_pairs),
                           size=n_boot).astype(float)


def half_weights(n_pairs, n_rep=N_SPLIT, rng=None):
    """`n_rep` random half-tournaments as complementary 0/1 weight matrices.

    Splitting the PAIRS is what makes a split-half number mean "would a
    different half of this tournament have ranked the attributes the same way".
    Splitting trials instead cannot answer that: in the forced-choice arm the
    display order is assigned from the trial index, so a trial split is the
    position-bias split wearing a reliability's clothes.
    """
    rng = rng or np.random.default_rng(0)
    half = np.zeros((n_rep, n_pairs))
    take = n_pairs // 2
    for i in range(n_rep):
        half[i, rng.permutation(n_pairs)[:take]] = 1.0
    return half, 1.0 - half


def row_corr(X, Y, min_n=3):
    """Pearson r between corresponding rows of two (k x m) matrices.

    Each row uses the columns finite in THAT row of both — pairwise-complete,
    per resample. Dropping a column across all rows instead looks tidier but is
    catastrophic here: a resample is a random half or a bootstrap draw, so with
    thousands of rows some row will be missing some attribute, and one such row
    would take that attribute away from every other row. Once a few columns go
    that way the whole statistic collapses to NaN — silently, and a NaN then
    reads downstream as "measured, and no effect".

    A resample IS its own tournament, so scoring each on the attributes it
    actually covers is also the right thing to do on the merits.
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    ok = np.isfinite(X) & np.isfinite(Y)
    n = ok.sum(axis=1).astype(float)
    Xm, Ym = np.where(ok, X, 0.0), np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sx, sy = Xm.sum(axis=1), Ym.sum(axis=1)
        cov = (Xm * Ym).sum(axis=1) - sx * sy / n
        vx = (Xm * Xm).sum(axis=1) - sx * sx / n
        vy = (Ym * Ym).sum(axis=1) - sy * sy / n
        r = cov / np.sqrt(vx * vy)
    return np.where((n >= min_n) & (vx > 0) & (vy > 0), r, np.nan)


def ci(draws, axis=0, level=95):
    """Percentile interval of a bootstrap distribution.

    An all-NaN column is a metric that does not exist in this condition — the
    indifference rate in the forced-choice arm, say — and comes back as NaN
    rather than as a warning, because "not applicable here" is an expected
    answer and not a numerical accident.
    """
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(draws, lo, axis=axis),
                np.nanpercentile(draws, hi, axis=axis))


def boot_p(draws, null=0.0, axis=0):
    """Two-sided bootstrap p: how far the null sits inside the distribution.

    Floored at 1/n_draws, because a resampling test cannot report a p-value
    smaller than its own resolution — `p < 1/n` is all the data can say. With no
    finite draws the answer is NaN, not 1.0: nothing was tested, and a 1.0 would
    be read as "tested, and no effect".
    """
    draws = np.asarray(draws, dtype=float)
    n = np.sum(np.isfinite(draws), axis=axis)
    with np.errstate(invalid="ignore"):
        below = np.sum(draws <= null, axis=axis) / np.maximum(n, 1)
        above = np.sum(draws >= null, axis=axis) / np.maximum(n, 1)
    p = 2.0 * np.minimum(below, above)
    return np.where(n > 0, np.clip(p, 1.0 / np.maximum(n, 1), 1.0), np.nan)


def fdr(pvals):
    """Benjamini-Hochberg q-values.

    A framing contrast is tested once per attribute — 45 tests per factor — so
    an uncorrected 0.05 would hand back two "significant" attributes from noise
    alone. BH controls the false DISCOVERY rate, which is the right error to
    control when the output is a shortlist of attributes to look at rather than
    a single yes/no.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if not n:
        return q
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = out
    return q


def stars(q, thresholds=(0.001, 0.01, 0.05)):
    """Console marker for a q-value. Empty string when nothing survives."""
    if q is None or not np.isfinite(q):
        return " "
    for mark, t in zip(("***", "**", "*"), thresholds):
        if q < t:
            return mark
    return " "
