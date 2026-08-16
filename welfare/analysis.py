"""Substantive results for the welfare module — tables and figures.

`welfare.report` audits an administration: did the model answer, how much of the
answer was the rendering, does the winner survive a slot swap. This script takes
the administration as given and asks what it SAYS — which attributes a model
wants improved, whether that ranking survives being asked a different way, and
whether the models agree with each other.

    python -m welfare.analysis                     # local + API results, when present
    python -m welfare.analysis --include-partial   # also use half-collected blocks
    python -m welfare.analysis --baseline object=ai_assistant
    python -m welfare.analysis run.jsonl --out dir/ # explicitly use one result file
    python -m welfare.analysis --no-save           # console only, no files

TWO CATEGORIES, TWO DIRECTORIES. The output is split by what a reader is
allowed to conclude from it:

    validity/     can this administration be read at all — position bias, slot
                  swaps, answer rates, reliability, transitivity
    preference/   what it says once it can — the ranking, the framing tests, and
                  what the cohort of models has in common

Each holds `models/<alias>/` (one model at a time, never pooled) and
`combined/` (every model at once, as a meta-analysis over per-model estimates
rather than a pool of their trials).

ONE CONDITION IS THE BASELINE. The grid crosses four binary factors, so a model
has sixteen conditions and sixteen rankings. The reference is

    increase / the update is for YOU / YOU choose / "No preference" offered

and everything else is reported as a departure from it: each of the four factors
is flipped on its own, and `baseline_shift.*` says which attributes moved, how
far, and whether that is more than re-measuring the same condition twice would
have moved them. `--baseline` moves the reference cell.

PARTIAL RUNS ARE THE NORMAL CASE. The current full grid is 31,744 cells per
model, and the runner fills it condition by condition, so a file read mid-run
holds some complete blocks and one that stopped halfway. A half-filled block is
NOT a smaller sample of the same thing: pairs are administered in the order
`sample_pairs` emits them, so it covers a biased subset of the tournament and
gives some attributes almost no comparisons. Incomplete blocks are therefore
dropped by default and listed in COVERAGE, with `--include-partial` to override.

Where the numbers come from: `welfare.validity` (the caveats),
`welfare.preference` (the ranking), `welfare.baseline` (the framing tests),
`welfare.cohort` (across models), all resampling pairs through
`welfare.resample`. This file is the console and the filing cabinet.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from core.report import Saver, fmt, section
from core.schema import load_records
from welfare import baseline as B
from welfare import cohort as C
from welfare import plots as P
from welfare import preference as PR
from welfare import validity as V
from welfare.attributes import load_welfare
from welfare.config import load_config
from welfare.constants import MODULE
from welfare.grid import n_pairs_for, trials_per_cell
from welfare.report import (
    CONDITION, choice_frame, order_consistency, pair_estimates, position_bias,
)
from welfare.resample import N_BOOT, stars


# Local runs and API batch collection deliberately use separate stores. Analysis
# treats both as one cohort by default; an explicit positional path remains an
# escape hatch for analysing a single run or an archived result set.
DEFAULT_RESULT_PATHS = ("welfare.jsonl", "welfare_api.jsonl")


def _result_paths(paths=None) -> list[str]:
    """Normalize one/many explicit paths, or select every canonical store."""
    if paths is None:
        return list(DEFAULT_RESULT_PATHS)
    if isinstance(paths, (str, os.PathLike)):
        return [os.fspath(paths)]
    normalized = [os.fspath(path) for path in paths]
    # argparse supplies [] (rather than its declared None default) for an absent
    # nargs="*" positional on supported Python versions.
    return normalized or list(DEFAULT_RESULT_PATHS)


def load_analysis_records(paths=None):
    """Load and merge welfare rows from the selected result stores.

    Missing canonical stores are normal (a study may have only local or only
    API models), but an explicitly requested missing path is an error. Cell-key
    deduplication prevents a copied/collected result from receiving extra weight;
    when a retried cell has both an infrastructure-error row and a real result,
    the real result wins.
    """
    requested = _result_paths(paths)
    explicit = paths is not None and bool(paths)
    missing = [path for path in requested if not os.path.exists(path)]
    if explicit and missing:
        raise FileNotFoundError(", ".join(missing))
    available = [path for path in requested if os.path.exists(path)]
    if not available:
        raise FileNotFoundError(", ".join(requested))

    by_cell = {}
    for path in available:
        for record in load_records(path, MODULE):
            key = record.cell_key()
            previous = by_cell.get(key)
            previous_error = (previous is not None
                              and str(previous.notes).startswith("error:"))
            current_error = str(record.notes).startswith("error:")
            if previous is None or (previous_error and not current_error):
                by_cell[key] = record
    return list(by_cell.values()), available


def _slug(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(text))


def provenance(records) -> pd.DataFrame:
    """Family, generation and release date per model, as the RECORDS state them.

    The registry is the richer source (it alone knows parameter counts), but a
    model that has since been renamed in `config/models.yaml` would drop out of
    the cohort analysis silently. What the run itself wrote is the fallback.
    """
    seen = {}
    for r in records:
        if r.model_id not in seen:
            seen[r.model_id] = {"model": r.model_id,
                                "model_family": r.model_family,
                                "model_generation": r.model_generation,
                                "model_release_date": r.model_release_date}
    return pd.DataFrame(list(seen.values()))


# ---------------------------------------------------------------------------
# coverage — which blocks are safe to analyse
# ---------------------------------------------------------------------------
def expected_pair_count(cfg, welfare_set=None):
    """Pairs one complete block should hold, or None if it cannot be determined.

    Falls back to `cfg.n_pairs` only when no attribute set is supplied, since an
    exhaustive design's pair count is a property of the item bank rather than of
    the config.
    """
    if welfare_set is not None:
        return n_pairs_for(welfare_set, cfg)
    return cfg.n_pairs


def coverage(choices: pd.DataFrame, cfg, welfare_set=None) -> pd.DataFrame:
    """Per (model x condition): how much of its design the file actually holds.

    `expected_trials` comes from the config that would produce this file, so a
    block is complete when it has every configured pair at full trial depth. When
    the file was produced under a different config the fraction is still the
    honest one to sort by, it just may not reach 1.0 — which is why the console
    prints the expectation it used.

    The expectation is `n_pairs_for`, not `cfg.n_pairs`: under the exhaustive
    default `cfg.n_pairs` is null, and reading it literally would score a
    complete run as 0% collected.
    """
    if choices.empty:
        return pd.DataFrame()
    expected_pairs = expected_pair_count(cfg, welfare_set)
    rows = []
    for keys, g in choices.groupby(CONDITION, dropna=False):
        no_pref = dict(zip(CONDITION, keys))["no_pref_offered"]
        expected = (expected_pairs * trials_per_cell(bool(no_pref), cfg)
                    if expected_pairs else np.nan)
        rows.append({
            **dict(zip(CONDITION, keys)),
            "n_pairs": g.groupby(["a", "b"]).ngroups,
            "expected_pairs": expected_pairs,
            "n_trials": len(g),
            "expected_trials": expected,
            "fraction": len(g) / expected if expected else np.nan,
        })
    tab = pd.DataFrame(rows)
    tab["complete"] = tab.fraction >= 0.999
    return tab.sort_values(CONDITION).reset_index(drop=True)


def print_coverage(cov, cfg, base, saver, welfare_set=None):
    section("COVERAGE — which blocks are complete enough to analyse?")
    if cov.empty:
        print("  no choice rows in this file")
        return cov
    n_pairs = expected_pair_count(cfg, welfare_set)
    how = "exhaustive" if cfg.n_pairs is None else "sampled"
    print(f"  design in config/welfare.yaml: {n_pairs} pairs ({how}) x "
          f"{trials_per_cell(True, cfg)} trials (3-option) / "
          f"{trials_per_cell(False, cfg)} trials (forced choice)")
    print(f"  baseline condition: {B.describe(base)}\n")
    n_ok = int(cov.complete.sum())
    per_model = cov.groupby("model").agg(blocks=("complete", "size"),
                                         done=("complete", "sum"),
                                         trials=("n_trials", "sum"))
    print(f"    {'model':<18} {'blocks complete':>16} {'trials':>12} {'baseline':>10}")
    for model, r in per_model.iterrows():
        bl = cov[(cov.model == model)]
        bl = bl[np.ones(len(bl), dtype=bool)
                & (bl.qvar == base["qvar"]) & (bl.object == base["object"])
                & (bl.subject == base["subject"])
                & (bl.no_pref_offered.astype(bool) == base["no_pref_offered"])]
        mark = ("—" if bl.empty else ("yes" if bool(bl.complete.iloc[0])
                                      else f"{bl.fraction.iloc[0]:.0%}"))
        print(f"    {model:<18} {int(r.done):>9,} / {int(r.blocks):<4} "
              f"{int(r.trials):>12,} {mark:>10}")
    print(f"\n  {n_ok} of {len(cov)} blocks complete "
          f"({cov.loc[cov.complete, 'n_trials'].sum():,} of {cov.n_trials.sum():,} rows).")
    if n_ok < len(cov):
        print("  Partial blocks cover the FIRST pairs of the tournament, not a random")
        print("  subset, so their rankings are biased. Dropped unless --include-partial.")
    saver.csv(cov, "coverage.csv", "design completeness per model x condition")
    return cov


def select_complete(choices, cov, include_partial):
    """Restrict to the blocks whose design is fully collected."""
    if include_partial or cov.empty or cov.complete.all():
        return choices
    ok = cov[cov.complete][CONDITION]
    if ok.empty:
        return choices.iloc[0:0]
    return choices.merge(ok, on=CONDITION, how="inner")


# ---------------------------------------------------------------------------
# one model
# ---------------------------------------------------------------------------
def analyse_validity(model, choices, estimates, base, n_boot, saver, save):
    """The validity half for ONE model: all conditions, and the four flips."""
    metrics = V.condition_metrics(choices, estimates, n_boot=n_boot)
    if metrics.empty:
        return {}
    metrics.insert(0, "label", [B.short_label(r) for r in metrics.itertuples()])
    metrics["is_baseline"] = metrics.label == B.short_label(base)

    stats = V.pair_stats(choices)
    base_stats = B.select(stats, base)
    shifts = []
    for factor, cond in B.neighbours(base).items():
        s = V.metric_shift(base_stats, B.select(stats, cond), n_boot=n_boot)
        if not s.empty:
            s.insert(0, "factor", factor)
            s.insert(1, "flip_level", str(cond[factor]))
            shifts.append(s)
    shift = pd.concat(shifts, ignore_index=True) if shifts else pd.DataFrame()

    row = metrics[metrics.is_baseline]
    if not row.empty:
        r = row.iloc[0]
        print(f"    validity at baseline:  answered {r.answer_rate:>6.1%} | "
              f"position bias {r.position_bias:>5.3f} | earlier slot "
              f"{r.slot_duel_lo:>6.1%} | flips {r.flip_rate:>6.1%}")
        print(f"                           no-pref {fmt(r.get('no_pref_rate')):>6} | "
              f"reliability {fmt(r.get('reliability')):>6} | "
              f"transitivity {fmt(r.get('transitivity')):>6}")
    if not shift.empty:
        moved = shift[(shift.p < 0.05) & shift.comparable]
        print(f"    flips that move a validity metric: "
              f"{len(moved)} of {int(shift.comparable.sum())} tested", end="")
        if len(moved):
            top = moved.reindex(moved["shift"].abs().sort_values(ascending=False).index)
            print("  — " + "; ".join(f"{B.FACTOR_LABEL[r.factor]}: {r.label} "
                                     f"{r.shift:+.3f}" for r in top.head(3).itertuples()))
        else:
            print()

    if save:
        tag = _slug(model)
        bias = position_bias(choices)
        slots, duels = V.slot_rates(bias), V.slot_duels(choices)
        cons = order_consistency(choices)
        if not cons.empty:
            cons = cons.assign(n_options=np.where(cons.no_pref_offered, 3, 2))
        scope = f"{model} — every condition pooled"
        d = f"validity/models/{tag}"
        if not slots.empty and not duels.empty:
            P.plot_position_bias(slots, duels, saver.dir / f"{d}/position_bias.png", scope)
        if not cons.empty:
            P.plot_order_consistency(cons, saver.dir / f"{d}/order_consistency.png", scope)
        P.plot_condition_validity(metrics, B.short_label(base),
                                  saver.dir / f"{d}/conditions.png", model)
        if not shift.empty:
            P.plot_validity_shift(shift, saver.dir / f"{d}/baseline_shift.png",
                                  f"{model} — baseline: {B.describe(base)}")

    stamp = lambda df: df.assign(model=model) if not df.empty else df
    return {"conditions": metrics.assign(model=model) if "model" not in metrics
            else metrics, "shift": stamp(shift)}


def analyse_preference(model, choices, estimates, base, welfare_set, n_boot,
                       saver, save, n_show=6):
    """The preference half for ONE model: the baseline ranking, and the flips."""
    est_base = B.select(estimates, base)
    if est_base.empty:
        print(f"    no complete baseline block — the ranking needs "
              f"{B.describe(base)}")
        return {}

    ranking = PR.attribute_ranking(est_base, welfare_set, n_boot=n_boot)
    if ranking.empty:
        return {}
    constructs = PR.construct_ranking(ranking)

    print(f"    ranking at baseline ({int(ranking.n_pairs.min())}-"
          f"{int(ranking.n_pairs.max())} comparisons per attribute):")
    for label, part in (("wants most", ranking.head(n_show // 2)),
                        ("wants least", ranking.tail(n_show // 2))):
        print(f"      -- {label} --")
        for r in part.itertuples():
            print(f"      {r.win_rate:>6.1%}  [{r.ci_lo:.2f}, {r.ci_hi:.2f}]  "
                  f"{r.entity:<13} {r.text[:44]}")

    wins = PR.condition_wins(estimates, choices)
    rel = PR.condition_reliabilities(estimates)
    matrix, agree = PR.condition_agreement(wins, rel)

    shift_rows, summary_rows, construct_rows, shifts = [], [], [], {}
    for factor, cond in B.neighbours(base).items():
        per_attr, summary = B.shift(est_base, B.select(estimates, cond), ranking,
                                    n_boot=n_boot)
        if per_attr.empty:
            continue
        label = B.LEVEL_LABEL[(factor, cond[factor])]
        per_attr.attrs["flip_label"] = label
        shifts[factor] = per_attr
        shift_rows.append(per_attr.assign(factor=factor, flip_label=label))
        summary_rows.append(summary.assign(factor=factor, flip_label=label))
        cs = B.construct_shift(per_attr)
        if not cs.empty:
            construct_rows.append(cs.assign(factor=factor))

    shift = pd.concat(shift_rows, ignore_index=True) if shift_rows else pd.DataFrame()
    summary = (pd.concat(summary_rows, ignore_index=True) if summary_rows
               else pd.DataFrame())
    con_shift = (pd.concat(construct_rows, ignore_index=True) if construct_rows
                 else pd.DataFrame())

    if not summary.empty:
        print(f"\n    how the framing moves that ranking:")
        print(f"      {'flip':<18} {'mean|Δ|':>8} {'r':>6} {'ceil':>6} "
              f"{'gap':>7} {'p':>8} {'moved':>6}  verdict")
        for r in summary.itertuples():
            print(f"      {B.FACTOR_LABEL[r.factor]:<18} {r.mean_abs_shift:>8.3f} "
                  f"{fmt(r.pearson_r, 2):>6} {fmt(r.ceiling, 2):>6} "
                  f"{r.gap:>+7.3f} {r.gap_p:>8.4f} {r.n_attr_moved:>6}  "
                  f"{B.verdict(r)}")

    if save:
        tag = _slug(model)
        d = f"preference/models/{tag}"
        scope = f"{model} — baseline: {B.describe(base)}"
        P.plot_attribute_ranking(ranking, saver.dir / f"{d}/attribute_ranking.png", scope)
        P.plot_construct_summary(ranking, saver.dir / f"{d}/construct_summary.png", scope)
        if "bt_strength" in ranking:
            P.plot_bt_vs_raw(ranking, saver.dir / f"{d}/bt_vs_raw.png", scope)
        if not agree.empty:
            P.plot_condition_agreement(matrix, saver.dir / f"{d}/condition_agreement.png",
                                       model, baseline_label=B.short_label(base))
        if shifts:
            P.plot_shift_scatters(shifts, saver.dir / f"{d}/baseline_shift.png", scope)
        if not shift.empty:
            order = list(ranking.sort_values("win_rate", ascending=False).entity)
            P.plot_shift_attributes(shift, order,
                                    saver.dir / f"{d}/shift_attributes.png", scope)
        if not summary.empty:
            P.plot_shift_summary(summary, saver.dir / f"{d}/shift_summary.png", scope)

    stamp = lambda df: df.assign(model=model) if not df.empty else df
    return {
        "ranking": stamp(ranking), "conditions": stamp(wins),
        "constructs": stamp(constructs),
        "agreement": stamp(agree), "shift": stamp(shift),
        "summary": stamp(summary), "construct_shift": stamp(con_shift),
        # Keyed exactly as `condition_reliabilities` keys it: the full condition
        # including the model, stringified.
        "reliability": rel.get(tuple([str(model)] + [str(base[c]) for c in CONDITION[1:]]),
                               np.nan),
        "matrix": matrix,
    }


# ---------------------------------------------------------------------------
# every model at once
# ---------------------------------------------------------------------------
def combine_validity(per_model, base, meta, saver, save):
    """The validity half across models: the baseline row, and the flips."""
    conds = [d["conditions"] for d in per_model.values() if d.get("conditions") is not None
             and not d["conditions"].empty]
    shifts = [d["shift"] for d in per_model.values() if d.get("shift") is not None
              and not d["shift"].empty]
    if not conds:
        return {}
    all_cond = pd.concat(conds, ignore_index=True)
    all_shift = pd.concat(shifts, ignore_index=True) if shifts else pd.DataFrame()

    section("VALIDITY ACROSS MODELS — at the baseline condition")
    at_base = all_cond[all_cond.is_baseline].copy()
    order = list(meta.sort_values(["family", "params_total_b", "model"]).model)
    at_base = at_base.set_index("model").reindex([m for m in order if m in
                                                  set(at_base.model)]).reset_index()
    if at_base.empty:
        print("  no model has a complete baseline block")
        return {"conditions": all_cond, "shift": all_shift}

    print(f"    {'model':<18} {'answered':>9} {'pos bias':>9} {'earlier':>8} "
          f"{'flips':>7} {'no-pref':>8} {'reliab':>7} {'transit':>8}")
    for r in at_base.itertuples():
        print(f"    {r.model:<18} {r.answer_rate:>9.1%} {r.position_bias:>9.3f} "
              f"{r.slot_duel_lo:>8.1%} {r.flip_rate:>7.1%} "
              f"{fmt(getattr(r, 'no_pref_rate', np.nan), 3):>8} "
              f"{fmt(getattr(r, 'reliability', np.nan), 3):>7} "
              f"{fmt(getattr(r, 'transitivity', np.nan), 3):>8}")

    trends = pd.DataFrame()
    if len(at_base) >= C.MIN_MODELS_TREND:
        cols = [c for c in list(V.RESAMPLED) + ["reliability", "transitivity"]
                if c in at_base.columns and at_base[c].notna().any()]
        W = at_base.set_index("model")[cols].T
        W.index.name = "entity"
        trends = C.trends(W, meta)
        if not trends.empty:
            print("\n  does a validity metric track model size or release date?")
            print(f"    {'metric':<18} {'ρ size':>8} {'q':>7}   {'ρ recency':>10} {'q':>7}")
            for r in trends.itertuples():
                print(f"    {r.entity:<18} {fmt(r.rho_size, 2):>8} {fmt(r.q_size, 3):>7}"
                      f"{stars(r.q_size)}  {fmt(r.rho_recency, 2):>10} "
                      f"{fmt(r.q_recency, 3):>7}{stars(r.q_recency)}")

    if save:
        P.plot_validity_models(at_base, meta,
                               saver.dir / "validity/combined/by_model.png",
                               f"baseline: {B.describe(base)}")
        if not all_shift.empty:
            P.plot_validity_shift_models(
                all_shift, saver.dir / "validity/combined/baseline_shift.png",
                f"{all_shift.model.nunique()} models — baseline: {B.describe(base)}")
    return {"conditions": all_cond, "shift": all_shift, "baseline": at_base,
            "trends": trends}


def combine_preference(per_model, base, meta, welfare_set, saver, save, n_show=8):
    """The preference half across models — a meta-analysis, never a pool."""
    rankings = {m: d["ranking"] for m, d in per_model.items()
                if d.get("ranking") is not None and not d["ranking"].empty}
    if not rankings:
        return {}
    W = C.win_matrix(rankings)
    any_ranking = next(iter(rankings.values()))
    con = C.consensus(W, any_ranking)
    concord = C.kendall_w(W) if W.shape[1] > 1 else {}

    section("PREFERENCE ACROSS MODELS — one estimate per model, then combined")
    print(f"  {W.shape[1]} model(s), {W.shape[0]} attributes, baseline condition only.")
    if concord:
        print(f"  Kendall's W = {concord['kendall_w']:.3f} "
              f"(permutation p = {concord['p']:.4f}; chance ≈ "
              f"{concord['null_mean']:.3f}) — 1.0 would be identical rankings.")
    spread = lambda v: f"±{v:>5.1%}" if np.isfinite(v) else "      "
    print("\n    -- what the cohort wants most (mean win rate, spread across models) --")
    for r in con.head(n_show).itertuples():
        print(f"    {r.mean_win:>6.1%}  {spread(r.sd_win)}  {r.entity:<13} {r.text[:42]}")
    print("\n    -- what it wants least --")
    for r in con.tail(n_show).iloc[::-1].itertuples():
        print(f"    {r.mean_win:>6.1%}  {spread(r.sd_win)}  {r.entity:<13} {r.text[:42]}")
    if W.shape[1] > 1:
        split = con.reindex(con.sd_win.sort_values(ascending=False).index).head(5)
        print("\n    -- where the models disagree most (high spread = no cohort position) --")
        for r in split.itertuples():
            print(f"    ±{r.sd_win:>5.1%}  ({r.min_win:.0%}-{r.max_win:.0%})  "
                  f"{r.entity:<13} {r.text[:42]}")

    rel = {m: d.get("reliability", np.nan) for m, d in per_model.items()}
    matrix, pairs = C.agreement(W, rel)
    pairs = C.with_covariates(pairs, meta)
    effects = C.covariate_effects(pairs, meta) if not pairs.empty else pd.DataFrame()
    family = C.family_contrast(pairs, meta) if not pairs.empty else {}
    trends = C.trends(W, meta, any_ranking)

    if not pairs.empty:
        print(f"\n  MODEL SIMILARITY — mean pairwise Spearman "
              f"{pairs.spearman_r.mean():.3f} over {len(pairs)} model pairs")
        top = pairs.sort_values("spearman_r", ascending=False)
        for label, part in (("most alike", top.head(3)),
                            ("least alike", top.tail(3).iloc[::-1])):
            print(f"    -- {label} --")
            for r in part.itertuples():
                print(f"    ρ {r.spearman_r:>6.3f}  (ceiling {fmt(r.ceiling, 2)})  "
                      f"{r.model_a} vs {r.model_b}")
    if family:
        print(f"\n  FAMILY — within-family agreement {family['within_family']:.3f} "
              f"vs across families {family['between_family']:.3f}\n"
              f"  gap {family['gap']:+.3f}, permutation p = {family['p']:.4f} "
              f"({family['n_within']} within / {family['n_between']} across pairs)")
    elif len(rankings) > 1:
        print("\n  FAMILY — not enough families with two or more complete models "
              "to contrast.")
    if not effects.empty:
        print(f"\n  WHAT PREDICTS AGREEMENT (permuting model labels, "
              f"{effects.n_models.iloc[0]} models / {effects.n_pairs.iloc[0]} pairs)")
        print(f"    {'predictor':<18} {'r alone':>9} {'p':>8}   {'β joint':>9} {'p':>8}")
        for r in effects.itertuples():
            print(f"    {r.predictor:<18} {r.marginal_r:>9.3f} {r.marginal_p:>8.4f}"
                  f"{stars(r.marginal_p)}  {r.joint_beta:>9.3f} {r.joint_p:>8.4f}"
                  f"{stars(r.joint_p)}")
    if not trends.empty:
        sized = trends[trends.q_size < 0.05] if "q_size" in trends else pd.DataFrame()
        recent = trends[trends.q_recency < 0.05] if "q_recency" in trends else pd.DataFrame()
        print(f"\n  PER-ATTRIBUTE TRENDS — {len(sized)} attribute(s) track model size, "
              f"{len(recent)} track release date (BH q < 0.05)")
        for label, part in (("with size", sized), ("with recency", recent)):
            if part.empty:
                continue
            col = "rho_size" if label == "with size" else "rho_recency"
            part = part.reindex(part[col].abs().sort_values(ascending=False).index)
            print(f"    -- {label} --")
            for r in part.head(5).itertuples():
                print(f"    ρ {getattr(r, col):>+6.2f}  {r.entity:<13} {r.text[:42]}")

    summaries = [d["summary"] for d in per_model.values()
                 if d.get("summary") is not None and not d["summary"].empty]
    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not all_summary.empty:
        section("FRAMING ACROSS MODELS — is a flip's effect a property of the question?")
        print(f"    {'flip':<18} {'mean|Δ|':>9} {'range':>16} {'gap':>8} "
              f"{'models moved':>13}")
        for factor in B.FACTORS:
            g = all_summary[all_summary.factor == factor]
            if g.empty:
                continue
            moved = int((g.gap_p < 0.05).sum())
            print(f"    {B.FACTOR_LABEL[factor]:<18} {g.mean_abs_shift.mean():>9.3f} "
                  f"{g.mean_abs_shift.min():>7.3f}-{g.mean_abs_shift.max():<8.3f} "
                  f"{g.gap.mean():>+8.3f} {moved:>8} / {len(g):<4}")
        print("\n  A flip that moves the ranking in every model is a property of the")
        print("  question; one that moves it in a few is a property of those models.")

    if save:
        sub = f"{W.shape[1]} models — baseline: {B.describe(base)}"
        P.plot_consensus(con, W, concord, saver.dir / "preference/combined/consensus.png",
                         sub)
        if W.shape[1] > 1:
            P.plot_agreement_spread(
                con, saver.dir / "preference/combined/agreement_spread.png", sub)
            P.plot_model_agreement(
                matrix, saver.dir / "preference/combined/model_agreement.png", sub)
            P.plot_cohort_patterns(
                pairs, effects, family, trends,
                saver.dir / "preference/combined/cohort_patterns.png", sub)
        if not all_summary.empty:
            P.plot_shift_models(all_summary,
                                saver.dir / "preference/combined/baseline_shift.png", sub)

    return {"consensus": con, "concordance": concord, "similarity": pairs,
            "effects": effects, "family": family, "trends": trends,
            "summary": all_summary, "matrix": matrix}


# ---------------------------------------------------------------------------
# filing
# ---------------------------------------------------------------------------
VALIDITY_CSV = {
    "conditions": "every validity metric per model x condition, with bootstrap intervals",
    "baseline_shift": "per model x flip x metric: change from the baseline, with its interval",
    "baseline": "the baseline condition's row per model — the combined view's input",
    "cohort_trends": "does a validity metric track model size or release date",
}
PREFERENCE_CSV = {
    "ranking": "per model: baseline win rate with bootstrap CI, Bradley-Terry strength, "
               "rank shift and opponent strength",
    "conditions": "per model x condition x attribute: direct win rate used by the "
                  "interactive condition explorer",
    "constructs": "per model: construct rollup of the baseline attribute win rates",
    "agreement": "per model: ranking agreement between all sixteen conditions, vs ceilings",
    "shift": "per model x flip x attribute: change from the baseline, CI, p and BH q",
    "summary": "per model x flip: effect size, agreement, and the split-half gap test",
    "construct_shift": "per model x flip: the shift rolled up to the five constructs",
    "consensus": "per attribute: cohort mean win rate and between-model spread",
    "similarity": "per model pair: ranking agreement and the family / size / date distances",
    "effects": "does family, size or release date predict agreement (permutation tests)",
    "trends": "per attribute: correlation of win rate with model size and release date",
}

VALIDITY_FIGS = [
    ("models/<model>/position_bias.png", "choice rate per printed slot and the slot duels"),
    ("models/<model>/order_consistency.png", "how far a pair's preference moves under a slot swap"),
    ("models/<model>/conditions.png", "every validity metric across all sixteen conditions"),
    ("models/<model>/baseline_shift.png", "what each flip does to each validity metric"),
    ("combined/by_model.png", "every model's validity at the baseline condition"),
    ("combined/baseline_shift.png", "flip effects on validity, one dot per model"),
]
PREFERENCE_FIGS = [
    ("models/<model>/attribute_ranking.png", "baseline win rate per attribute, with its interval"),
    ("models/<model>/construct_summary.png", "baseline win rates grouped by construct"),
    ("models/<model>/bt_vs_raw.png", "raw win rate vs schedule-corrected strength"),
    ("models/<model>/condition_agreement.png", "agreement between all sixteen conditions"),
    ("models/<model>/baseline_shift.png", "baseline vs flip, attribute by attribute"),
    ("models/<model>/shift_attributes.png", "which attributes moved, under which flip"),
    ("models/<model>/shift_summary.png", "the four flips as effect size, agreement and test"),
    ("combined/consensus.png", "the cohort ranking with every model's own estimate"),
    ("combined/agreement_spread.png", "cohort mean against between-model spread"),
    ("combined/model_agreement.png", "model x model ranking agreement"),
    ("combined/cohort_patterns.png", "what predicts where the models differ"),
    ("combined/baseline_shift.png", "flip effects on the ranking, one dot per model"),
]

VALIDITY_INTRO = """\
Can this administration be read at all? Nothing here is about which attribute
won. Every metric is computed within one condition, because each of them can
depend on the framing, and `baseline_shift.csv` tests each single-factor flip
against the baseline on the pairs the two conditions share.

Read `conditions.csv` for the whole picture, `models/<model>/conditions.png` for
one model, and `combined/by_model.png` to compare models. `position_bias.png`
and `order_consistency.png` pool every condition — they are the model-level
audit that the per-condition tables refine.

A high flip rate does not invalidate a ranking: balanced ordering averages
position out of it. It means INDIVIDUAL pair winners are rendering-sensitive and
should not be reported one by one."""

PREFERENCE_INTRO = """\
What the choices say, once `../validity/` says they can be read. Everything is
estimated at the BASELINE condition and per model — nothing is pooled across
models, and the combined view is a meta-analysis over per-model estimates.

Read `ranking.csv` first (or `models/<model>/attribute_ranking.png`), then
`summary.csv` for whether a different framing would have given a different
answer, then `combined/` for what the models have in common. Rank individual
attributes by `bt_strength`, which corrects for which opponents each one was
drawn against; the raw win rate is the readable summary and the two agree on the
extremes. Intervals bootstrap over pairs, not trials, and every per-attribute
p-value carries a BH q beside it."""


def file_category(saver, name, frames, csv_desc, figures):
    """Declare one category's CSVs and figures, so the README indexes itself.

    Per-model figures are registered once as `models/<model>/…` rather than once
    per model: with a dozen models the manifest would otherwise be a hundred
    lines of the same seven descriptions.
    """
    for key, df in frames.items():
        if df is None or (hasattr(df, "empty") and df.empty) or key not in csv_desc:
            continue
        saver.csv(df, f"{name}/{key}.csv", csv_desc[key])
    for fig, desc in figures:
        # A figure that could not be drawn — a cohort matrix with one model in
        # it — must not appear in the manifest as if it were there.
        probe = fig.replace("<model>", "*")
        if not any((saver.dir / name).glob(probe)):
            continue
        saver.artifact(f"{name}/{fig}", desc)


def write_index(saver, base, models, cfg):
    saver.index(
        "Welfare module — substantive results",
        f"Which attributes a model wants a future update to improve, from the "
        f"pairwise choice grid.\n\n**Baseline condition: {B.describe(base)}.** "
        f"Every ranking is estimated there, and each of the four framing factors "
        f"is reported as a departure from it (`baseline_shift.*`).\n\n"
        f"`validity/` asks whether an administration can be read at all; "
        f"`preference/` reads it. Each has `models/<alias>/` (one model, never "
        f"pooled) and `combined/` (all models, as a meta-analysis over per-model "
        f"estimates). Start with `coverage.csv`, then `validity/`, then "
        f"`preference/`.\n\nModels analysed: {', '.join(models)}. "
        f"Audit-side diagnostics — answerability, desirability, transitivity in "
        f"full — live in `python -m welfare.report`.")


# ---------------------------------------------------------------------------
def run(path=None, out_dir="results/welfare_analysis", save=True,
        include_partial=False, n_boot=N_BOOT, model_filter=None, baseline_spec=None):
    try:
        records, result_paths = load_analysis_records(path)
    except FileNotFoundError as exc:
        print(f"No requested welfare result file exists ({exc}) — run "
              f"`python -m welfare.run` first (or `--dry-run` for an offline sample).")
        return
    if not records:
        print(f"No welfare rows in {', '.join(result_paths)} — run "
              "`python -m welfare.run`.")
        return
    print(f"  result files: {', '.join(result_paths)}")

    from core.battery import load_battery

    cfg = load_config()
    welfare_set = load_welfare(load_battery())
    base = B.parse_baseline(baseline_spec)
    saver = Saver(out_dir, enabled=save)

    all_choices = choice_frame(records)
    if model_filter:
        all_choices = all_choices[all_choices.model.isin(set(model_filter))]
    if all_choices.empty:
        print("  no choice rows (this file may hold only desirability rows, or no "
              "model matched --models)")
        return

    cov = print_coverage(coverage(all_choices, cfg, welfare_set), cfg, base,
                         saver, welfare_set)
    choices = select_complete(all_choices, cov, include_partial)
    if choices.empty:
        print("\n  No block is complete yet. Re-run with --include-partial to analyse")
        print("  what has been collected, remembering the ranking will be biased.")
        return
    dropped = len(all_choices) - len(choices)
    if dropped:
        print(f"\n  analysing {len(choices):,} rows from complete blocks "
              f"({dropped:,} partial rows held out)")

    models = sorted(choices.model.unique())
    meta = C.model_meta(models, provenance(records))
    val_by_model, pref_by_model = {}, {}
    for model in models:
        ch = choices[choices.model == model]
        est = pair_estimates(ch)
        section(f"MODEL: {model}")
        print(f"  {ch.groupby(CONDITION).ngroups} condition(s), {len(ch):,} trials")
        val_by_model[model] = analyse_validity(model, ch, est, base, n_boot, saver, save)
        pref_by_model[model] = analyse_preference(model, ch, est, base, welfare_set,
                                                  n_boot, saver, save)

    combined_v = combine_validity(val_by_model, base, meta, saver, save)
    combined_p = combine_preference(pref_by_model, base, meta, welfare_set, saver, save)

    if not save:
        return
    section("FILES")
    validity_frames = {
        "conditions": combined_v.get("conditions"),
        "baseline_shift": combined_v.get("shift"),
        "baseline": combined_v.get("baseline"),
        "cohort_trends": combined_v.get("trends"),
    }
    collect = lambda key: [d[key] for d in pref_by_model.values()
                           if d.get(key) is not None and not d[key].empty]
    preference_frames = {}
    for key in ("ranking", "conditions", "constructs", "agreement", "shift", "summary",
                "construct_shift"):
        parts = collect(key)
        preference_frames[key] = (pd.concat(parts, ignore_index=True) if parts
                                  else pd.DataFrame())
    preference_frames.update({
        "consensus": combined_p.get("consensus"),
        "similarity": combined_p.get("similarity"),
        "effects": combined_p.get("effects"),
        "trends": combined_p.get("trends"),
    })

    saver.csv(meta, "models.csv", "family, generation, release date and size per model")
    file_category(saver, "validity", validity_frames, VALIDITY_CSV, VALIDITY_FIGS)
    file_category(saver, "preference", preference_frames, PREFERENCE_CSV,
                  PREFERENCE_FIGS)
    saver.text(f"# Validity — can this administration be read at all?\n\n"
               f"Baseline condition: **{B.describe(base)}**.\n\n{VALIDITY_INTRO}\n",
               "validity/README.md", "what the validity directory holds")
    saver.text(f"# Preference — what the choices say\n\n"
               f"Baseline condition: **{B.describe(base)}**.\n\n{PREFERENCE_INTRO}\n",
               "preference/README.md", "what the preference directory holds")
    write_index(saver, base, models, cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="*", default=None,
                    help="Result JSONL file(s). By default, merge welfare.jsonl "
                         "and welfare_api.jsonl when they exist.")
    ap.add_argument("--out", default="results/welfare_analysis")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--include-partial", action="store_true",
                    help="Also analyse half-collected blocks (their rankings are biased).")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"Bootstrap resamples for every interval (default {N_BOOT}).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these model aliases.")
    ap.add_argument("--baseline", default=None,
                    help="Move the reference condition, e.g. "
                         "'object=ai_assistant,no_pref_offered=false'. "
                         f"Default: {', '.join(f'{k}={v}' for k, v in B.BASELINE.items())}")
    args = ap.parse_args(argv)
    run(args.results, args.out, save=not args.no_save,
        include_partial=args.include_partial, n_boot=args.n_boot,
        model_filter=args.models, baseline_spec=args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
