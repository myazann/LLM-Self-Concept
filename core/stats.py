"""Statistics used by more than one instrument's report scripts.

Small, deliberately unopinionated helpers: each one drops non-finite pairs and
returns NaN rather than raising when there is not enough data, because both
report scripts run over partial files during a long run.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def cronbach_alpha(matrix):
    """Alpha from a respondents x items DataFrame. Rows with any gap are dropped."""
    data = matrix.dropna(axis=0, how="any").values
    n, k = data.shape
    if k < 2 or n < 2:
        return float("nan")
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - data.var(axis=0, ddof=1).sum() / total_var))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_p(x, y):
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    n = int((np.isfinite(xa) & np.isfinite(ya)).sum())
    r = corr(xa, ya)
    if not np.isfinite(r) or abs(r) >= 1.0 or n < 4:
        return r, float("nan")
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), n - 2))


def partial_corr(x, y, z):
    """r(x, y) with z held constant."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 4 or z.std() == 0:
        return float("nan"), float("nan")
    ex = x - np.polyval(np.polyfit(z, x, 1), z)
    ey = y - np.polyval(np.polyfit(z, y, 1), z)
    r, df = corr(ex, ey), len(x) - 3
    if not np.isfinite(r) or df < 1 or abs(r) >= 1.0:
        return r, float("nan")
    t = r * math.sqrt(df / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df))


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))
