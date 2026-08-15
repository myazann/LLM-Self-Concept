"""Across models — what they agree on, and what predicts where they differ.

The per-model results say what one model wants improved. This module asks the
two questions that only exist once there are several: do the models converge on
the same ranking, and when they do not, is the difference organised by anything
we know about them — family, size, or how recently they were released.

Pooling is deliberately avoided. A ranking estimated from every model's trials
at once describes no model in particular and is dominated by whichever model is
furthest through the grid, so the combined view here is a META-ANALYSIS: each
model contributes ONE per-attribute win rate, estimated within itself and within
the baseline condition, and the combined statement is over those estimates. The
between-model spread is then a real quantity (how much the models differ) rather
than a mixture of that and how much data each one happened to contribute.

    consensus       mean win rate per attribute with its between-model spread,
                    plus Kendall's W — one number for "is there a shared ranking"
    agreement       model x model correlation, each model's own split-half
                    reliability on the diagonal, exactly as the condition matrix
                    is built. A pair of models below both diagonals really does
                    rank the attributes differently
    covariates      is agreement predicted by same-family / size / release date
    trends          per attribute: does its win rate move with model size, or
                    with release date, across the cohort

Every test here is a permutation test over MODEL labels. With a cohort this
small (a dozen models, not a sample from a population of models) that is the
only null with a defensible meaning: it asks whether the pattern survives
relabelling which model is which, holding the rankings themselves fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Permutations for the model-label nulls. The cohort is small, so this is cheap
# and can be generous.
N_PERM = 5000

# Below this many models, a correlation across models is not an estimate of
# anything and the covariate tests are not run.
MIN_MODELS_TREND = 4
# A family needs at least this many members before "within-family" means
# anything different from "the same model twice".
MIN_FAMILY_SIZE = 2


# ---------------------------------------------------------------------------
# the cohort's metadata
# ---------------------------------------------------------------------------
def model_meta(models, choices: pd.DataFrame = None) -> pd.DataFrame:
    """Family, generation, release date and parameter count per model alias.

    The registry is the source of truth — it is the only place that knows
    parameter counts — but the results file carries family, generation and
    release date on every row, so a model that has been renamed or removed from
    `config/models.yaml` since it was run still lands in the table instead of
    dropping out of the cohort analysis silently.
    """
    from core.model_registry import load_registry

    try:
        reg = load_registry()
    except Exception:
        reg = None

    from_file = {}
    if choices is not None and not choices.empty and "model_family" in choices:
        cols = ["model", "model_family", "model_generation", "model_release_date"]
        from_file = {r.model: r for r in choices[cols].drop_duplicates().itertuples()}

    rows = []
    for alias in models:
        spec = None
        if reg is not None:
            try:
                spec = reg.get(alias)
            except KeyError:
                spec = None
        rec = from_file.get(alias)
        release = (spec.release_date if spec is not None
                   else (pd.to_datetime(rec.model_release_date).date() if rec is not None
                         else None))
        rows.append({
            "model": alias,
            "family": (spec.family if spec is not None
                       else getattr(rec, "model_family", "")) or "",
            "generation": (spec.generation if spec is not None
                           else getattr(rec, "model_generation", "")) or "",
            "release_date": pd.to_datetime(release) if release is not None else pd.NaT,
            "params_total_b": spec.params_total_b if spec is not None else np.nan,
            "params_active_b": spec.params_active_b if spec is not None else np.nan,
            "size_tier": (spec.size_tier if spec is not None else None) or "",
        })
    meta = pd.DataFrame(rows)
    # Active parameters are what a mixture-of-experts model actually spends at
    # inference, so a 35B-A3B sits with the small models on this axis and with
    # the large ones on `params_total_b`. Both are kept; the log axis used for
    # the size trend is total, which is what "model size" conventionally means.
    meta["log_params"] = np.log10(meta.params_total_b.astype(float))
    meta["release_days"] = (meta.release_date - meta.release_date.min()).dt.days
    return meta


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------
def win_matrix(rankings: dict) -> pd.DataFrame:
    """{model: ranking frame} -> (attributes x models) win-rate matrix."""
    cols = {m: r.set_index("entity").win_rate for m, r in rankings.items()
            if r is not None and not r.empty}
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all")


def consensus(W: pd.DataFrame, ranking: pd.DataFrame = None) -> pd.DataFrame:
    """Per attribute: the cohort mean, and how far the models spread around it.

    `sd` is the finding, not a nuisance: an attribute with a high mean and a low
    spread is one the whole cohort puts near the top, while a high spread means
    the models are ranking it differently and the cohort mean describes none of
    them.
    """
    if W.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "entity": W.index,
        "mean_win": W.mean(axis=1).values,
        "sd_win": W.std(axis=1, ddof=1).values if W.shape[1] > 1 else np.nan,
        "min_win": W.min(axis=1).values,
        "max_win": W.max(axis=1).values,
        "n_models": W.notna().sum(axis=1).values,
    })
    # How consistently the models place it, independent of where: the mean
    # within-cohort rank and its spread over models.
    ranks = W.rank(ascending=False, axis=0)
    out["mean_rank"] = ranks.mean(axis=1).values
    out["sd_rank"] = (ranks.std(axis=1, ddof=1).values if W.shape[1] > 1 else np.nan)
    if ranking is not None and not ranking.empty:
        meta = ranking.set_index("entity")[["text", "construct", "construct_label"]]
        out = out.join(meta, on="entity")
    return out.sort_values("mean_win", ascending=False).reset_index(drop=True)


def kendall_w(W: pd.DataFrame, n_perm=N_PERM, seed=41) -> dict:
    """Kendall's W across models, with a permutation null.

    W is 1 when every model produces the same ranking and 0 when the rankings
    are unrelated. The null shuffles each model's ranking INDEPENDENTLY, which
    is the hypothesis "the models rank the attributes at random and any apparent
    consensus is the accident of averaging", and is exact up to `n_perm`.
    """
    if W.empty or W.shape[1] < 2:
        return {}
    R = W.rank(axis=0).to_numpy(dtype=float)      # attributes x models
    n, m = R.shape

    def stat(mat):
        s = mat.sum(axis=1)
        return 12.0 * ((s - s.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))

    obs = stat(R)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stat(np.apply_along_axis(rng.permutation, 0, R))
    return {"kendall_w": float(obs), "n_models": int(m), "n_attributes": int(n),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "null_mean": float(null.mean())}


# ---------------------------------------------------------------------------
# model x model agreement
# ---------------------------------------------------------------------------
def agreement(W: pd.DataFrame, reliabilities: dict):
    """Model x model ranking correlation, each model's reliability on the diagonal.

    Built exactly like `preference.condition_agreement`, and read the same way:
    the diagonal is the ceiling for its row, so two models whose cell sits well
    below both of their diagonals disagree by more than either one's measurement
    noise.
    """
    if W.empty:
        return pd.DataFrame(), pd.DataFrame()
    models = list(W.columns)
    M = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    rows = []
    for i, a in enumerate(models):
        M.loc[a, a] = reliabilities.get(a, np.nan)
        for b in models[i + 1:]:
            joint = W[[a, b]].dropna()
            if len(joint) < 5:
                continue
            r_p = float(np.corrcoef(joint[a], joint[b])[0, 1])
            M.loc[a, b] = M.loc[b, a] = r_p
            ceiling = np.sqrt(reliabilities.get(a, np.nan)
                              * reliabilities.get(b, np.nan))
            rows.append({
                "model_a": a, "model_b": b, "n_attributes": len(joint),
                "pearson_r": r_p,
                "spearman_r": float(joint[a].rank().corr(joint[b].rank())),
                "ceiling": ceiling,
                "r_over_ceiling": r_p / ceiling if np.isfinite(ceiling) and ceiling
                else np.nan,
                "mean_abs_diff": float((joint[a] - joint[b]).abs().mean()),
            })
    return M, pd.DataFrame(rows)


def with_covariates(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Attach same-family / size-distance / release-distance to each model pair."""
    if pairs.empty or meta.empty:
        return pairs
    m = meta.set_index("model")
    get = lambda alias, col: m[col].get(alias, np.nan)
    out = pairs.copy()
    out["same_family"] = [float(get(r.model_a, "family") == get(r.model_b, "family"))
                          for r in out.itertuples()]
    out["same_generation"] = [
        float(get(r.model_a, "generation") == get(r.model_b, "generation"))
        for r in out.itertuples()]
    out["d_log_params"] = [abs(get(r.model_a, "log_params") - get(r.model_b, "log_params"))
                           for r in out.itertuples()]
    out["d_release_days"] = [abs(get(r.model_a, "release_days")
                                 - get(r.model_b, "release_days"))
                             for r in out.itertuples()]
    return out


def covariate_effects(pairs: pd.DataFrame, meta: pd.DataFrame, value="spearman_r",
                      n_perm=N_PERM, seed=43) -> pd.DataFrame:
    """Does family / size / release date predict how much two models agree?

    One regression of pairwise agreement on the three distances, plus each
    predictor on its own. The p-values are Mantel-style: the model LABELS are
    permuted and the whole pairwise design is rebuilt from the shuffled
    metadata, which respects the fact that the 78 pairs of 13 models are not 78
    independent observations. Fitting the pairs as if they were independent is
    the standard way this analysis goes wrong.
    """
    models = sorted(set(pairs.model_a) | set(pairs.model_b))
    if len(models) < MIN_MODELS_TREND or pairs.empty:
        return pd.DataFrame()
    m = meta.set_index("model").reindex(models)
    idx = {a: i for i, a in enumerate(models)}
    ia = np.array([idx[a] for a in pairs.model_a])
    ib = np.array([idx[b] for b in pairs.model_b])
    y = pairs[value].to_numpy(dtype=float)

    fam = m.family.to_numpy()
    gen = m.generation.to_numpy()
    lp = m.log_params.to_numpy(dtype=float)
    rd = m.release_days.to_numpy(dtype=float)

    def design(order):
        f, g, p, d = fam[order], gen[order], lp[order], rd[order]
        return np.column_stack([
            (f[ia] == f[ib]).astype(float),
            (g[ia] == g[ib]).astype(float),
            np.abs(p[ia] - p[ib]),
            np.abs(d[ia] - d[ib]),
        ])

    names = ["same_family", "same_generation", "d_log_params", "d_release_days"]
    X = design(np.arange(len(models)))
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 4:
        return pd.DataFrame()

    # A predictor with no variance in THIS cohort — every model in one family,
    # so `same_family` is 1 everywhere — has no effect to estimate. It is
    # reported as NaN rather than dropped, because "this cohort cannot answer
    # that" is the finding. Leaving it in would be worse than useless: its
    # correlation is NaN, no permutation can beat NaN, and the p-value would
    # come back as 1/(n_perm+1) — the most significant number in the table.
    live = [j for j in range(len(names)) if np.std(X[ok, j]) > 0]
    if not live or ok.sum() < len(live) + 2:
        return pd.DataFrame({
            "predictor": names, "marginal_r": np.nan, "marginal_p": np.nan,
            "joint_beta": np.nan, "joint_p": np.nan,
            "n_pairs": int(ok.sum()), "n_models": len(models),
        })

    def fit(Xd, yv):
        A = np.column_stack([np.ones(len(Xd)), Xd])
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        return beta[1:]

    obs_joint = np.full(len(names), np.nan)
    obs_joint[live] = fit(X[np.ix_(ok, live)], y[ok])
    obs_alone = np.full(len(names), np.nan)
    for j in live:
        obs_alone[j] = float(np.corrcoef(X[ok, j], y[ok])[0, 1])

    rng = np.random.default_rng(seed)
    ge_joint, ge_alone = np.zeros(len(names)), np.zeros(len(names))
    n_used = 0
    for _ in range(n_perm):
        Xp = design(rng.permutation(len(models)))
        okp = np.isfinite(y) & np.isfinite(Xp).all(axis=1)
        live_p = [j for j in live if np.std(Xp[okp, j]) > 0]
        if len(live_p) < len(live) or okp.sum() < len(live) + 2:
            continue
        n_used += 1
        bp = fit(Xp[np.ix_(okp, live)], y[okp])
        for i, j in enumerate(live):
            ge_joint[j] += abs(bp[i]) >= abs(obs_joint[j])
            r = float(np.corrcoef(Xp[okp, j], y[okp])[0, 1])
            ge_alone[j] += abs(r) >= abs(obs_alone[j])

    p = lambda ge: np.where(np.isfinite(obs_alone), (ge + 1) / (n_used + 1), np.nan)
    return pd.DataFrame({
        "predictor": names,
        "marginal_r": obs_alone, "marginal_p": p(ge_alone),
        "joint_beta": obs_joint, "joint_p": p(ge_joint),
        "n_pairs": int(ok.sum()), "n_models": len(models), "n_perm": n_used,
    })


def family_contrast(pairs: pd.DataFrame, meta: pd.DataFrame, value="spearman_r",
                    n_perm=N_PERM, seed=47) -> dict:
    """Mean agreement within a family vs across families, permuting family labels."""
    if pairs.empty or meta.empty:
        return {}
    fam = meta.set_index("model").family
    sizes = fam.value_counts()
    if (sizes >= MIN_FAMILY_SIZE).sum() < 2:
        return {}
    models = sorted(set(pairs.model_a) | set(pairs.model_b))
    idx = {a: i for i, a in enumerate(models)}
    ia = np.array([idx[a] for a in pairs.model_a])
    ib = np.array([idx[b] for b in pairs.model_b])
    labels = fam.reindex(models).to_numpy()
    y = pairs[value].to_numpy(dtype=float)

    def gap(lab):
        same = lab[ia] == lab[ib]
        if same.sum() == 0 or (~same).sum() == 0:
            return np.nan
        return float(np.nanmean(y[same]) - np.nanmean(y[~same]))

    obs = gap(labels)
    if not np.isfinite(obs):
        return {}
    rng = np.random.default_rng(seed)
    null = np.array([gap(labels[rng.permutation(len(models))]) for _ in range(n_perm)])
    same = labels[ia] == labels[ib]
    return {
        "within_family": float(np.nanmean(y[same])) if same.any() else np.nan,
        "between_family": float(np.nanmean(y[~same])) if (~same).any() else np.nan,
        "gap": obs, "n_within": int(same.sum()), "n_between": int((~same).sum()),
        "p": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)),
    }


# ---------------------------------------------------------------------------
# trends across the cohort
# ---------------------------------------------------------------------------
def trends(W: pd.DataFrame, meta: pd.DataFrame, ranking: pd.DataFrame = None,
           n_perm=N_PERM, seed=53) -> pd.DataFrame:
    """Per row of `W`: does it move with model size, or with release date?

    `W` is (thing x models) — attributes for the preference side, validity
    metrics for the other. Spearman rather than Pearson: with a dozen models the
    size axis is a handful of coarse steps and a single 70B would otherwise
    carry the whole coefficient.

    One permutation of the model order serves EVERY row, so the null preserves
    the correlations between rows (attributes are not independent — they are
    ranked against each other) and the q-values below are not built on a null
    that assumes they are.
    """
    if W.empty or meta.empty or W.shape[1] < MIN_MODELS_TREND:
        return pd.DataFrame()
    from welfare.resample import fdr

    m = meta.set_index("model").reindex(W.columns)
    axes = {"size": m.log_params.to_numpy(dtype=float),
            "recency": m.release_days.to_numpy(dtype=float)}
    R = W.rank(axis=1).to_numpy(dtype=float)      # rank across MODELS, per row

    out = pd.DataFrame({"entity": W.index, "n_models": W.notna().sum(axis=1).values})
    rng = np.random.default_rng(seed)
    for name, x in axes.items():
        ok = np.isfinite(x)
        if ok.sum() < MIN_MODELS_TREND or np.nanstd(x[ok]) == 0:
            out[f"rho_{name}"] = np.nan
            out[f"p_{name}"] = np.nan
            out[f"q_{name}"] = np.nan
            continue
        xr = pd.Series(x[ok]).rank().to_numpy()
        Rr = R[:, ok]
        obs = _row_pearson(Rr, xr)
        ge = np.zeros(len(obs))
        for _ in range(n_perm):
            ge += np.abs(_row_pearson(Rr, rng.permutation(xr))) >= np.abs(obs)
        # A row with no variance across models — a metric that is 0 in every
        # model — has an undefined correlation, and no permutation can exceed
        # NaN. Without this, `ge` stays 0 and the row is handed the smallest
        # p-value in the table for having said nothing.
        p = np.where(np.isfinite(obs), (ge + 1) / (n_perm + 1), np.nan)
        out[f"rho_{name}"] = obs
        out[f"p_{name}"] = p
        out[f"q_{name}"] = fdr(p)
    if ranking is not None and not ranking.empty:
        meta_cols = ranking.set_index("entity")[["text", "construct", "construct_label"]]
        out = out.join(meta_cols, on="entity")
    return out.reset_index(drop=True)


def _row_pearson(R, x):
    """Pearson r between every row of `R` and the vector `x`."""
    Rc = R - R.mean(axis=1, keepdims=True)
    xc = x - x.mean()
    den = np.sqrt((Rc ** 2).sum(axis=1) * (xc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (Rc * xc).sum(axis=1) / den, np.nan)


def group_means(W: pd.DataFrame, meta: pd.DataFrame, by="family") -> pd.DataFrame:
    """Mean of each row of `W` within each level of a model attribute."""
    if W.empty or meta.empty or by not in meta:
        return pd.DataFrame()
    groups = meta.set_index("model")[by].reindex(W.columns)
    out = pd.DataFrame(index=W.index)
    for level, cols in groups.groupby(groups):
        out[str(level)] = W[list(cols.index)].mean(axis=1)
    return out.reset_index().rename(columns={"index": "entity"})
