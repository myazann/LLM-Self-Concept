"""Validate the LLM self-concept measurement before interpreting scores.

The default estimand is one fixed administration:

    first_person_bare + p0 + harmonized_7 + isolated + rating_only + logprob

Cronbach's alpha, item-rest correlations, alpha-if-deleted, response style, and
item recommendations are calculated only there (13 independent model rows in
the current run).  Instruction forms p1/p2 are robustness checks against p0
while holding framing fixed.  Framing is a substantive analysis factor and is
never used as an item-deletion criterion.

All deletion/revision thresholds are exploratory.  The output says what would
be revised in a confirmatory instrument; it never silently trims the reported
scores from this same discovery sample.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from survey.plots import plot_item_validity
from core.battery import load_battery
from survey.scoring import (
    ACQUIESCENCE_SCALE_IDS, CONDITION_KEYS, CONSTRUCTS, DEFAULT_FRAMING,
    DEFAULT_ITEM_CONTEXT, DEFAULT_METHOD, DEFAULT_PARAPHRASE,
    DEFAULT_REASONING_MODE, DEFAULT_RESPONSE_FORMAT, MIDPOINT, Saver,
    ReportDatasetError,
    cell_frame, condition_label, construct_scores, corr,
    cronbach_alpha, default_condition, direction_of, fmt, load,
    respondent_matrix, section, select_condition, validate_default_config,
    validate_default_slice, validate_report_dataset,
)


# Proposed—not preregistered.  Structural evidence determines revision/drop
# candidates; response style and robustness create warnings.
RULES = {
    "negative_item_total": ("item_total_r", "<", 0.00,
                            "opposes the rest of its construct after keying"),
    "low_item_total": ("item_total_r", "<", 0.20,
                       "weak corrected item-rest relationship"),
    "low_variance": ("sd_between_models", "<", 0.03,
                     "little between-model information"),
    "alpha_improves_if_deleted": ("alpha_if_deleted_delta", ">", 0.02,
                                  "removing the item improves alpha materially"),
    "instruction_unstable": ("instruction_mae", ">", 0.10,
                             "mean paired instruction-form change exceeds 0.10"),
    "order_sensitive": ("order_mae_norm", ">", 0.15,
                        "mean absolute option-order change exceeds 0.9 scale points"),
    "acquiescence_loaded": ("abs_r_acquiescence", ">", 0.60,
                            "tracks the balanced agreement-style marker"),
    "redundant": ("max_positive_sibling_r", ">", 0.90,
                  "near-duplicate keyed response pattern"),
    "midpoint_heavy": ("midpoint_rate", ">", 0.60,
                       "mostly chooses the agreement midpoint"),
}


def _op(name, a, b):
    return a < b if name == "<" else a > b


def _default_record(r):
    return (
        r.framing == DEFAULT_FRAMING
        and r.paraphrase_id == DEFAULT_PARAPHRASE
        and r.response_format == DEFAULT_RESPONSE_FORMAT
        and r.item_context == DEFAULT_ITEM_CONTEXT
        and r.reasoning_mode == DEFAULT_REASONING_MODE
        and r.method == DEFAULT_METHOD
    )


def _core_cells(cells):
    """Pinned administration, leaving framing and instruction free."""
    return select_condition(cells, framing=None, paraphrase=None)


def qc_summary(records, cells, scope):
    cov = [r.option_mass_coverage for r in records if r.option_mass_coverage is not None]
    key_cols = [k for k in CONDITION_KEYS if k != "model"] + ["scale_id", "item_id"]
    direction_sets = cells.n_directions if "n_directions" in cells else pd.Series(dtype=float)
    return {
        "scope": scope, "n_records": len(records), "n_cells": len(cells),
        "n_models": len({r.model_id for r in records}),
        "n_items": len({r.item_id for r in records}),
        "n_conditions": int(cells.groupby([k for k in CONDITION_KEYS if k != "model"]).ngroups),
        "parse_failures": int(sum(r.parse_failed for r in records)),
        "refusals": int(sum(r.refusal_flag for r in records)),
        "missing_ratings": int(sum(r.parsed_rating is None for r in records)),
        "infra_errors": int(sum(str(r.notes).startswith("error:") for r in records)),
        "single_direction_cells": int((direction_sets < 2).sum()),
        "mean_option_mass_coverage": float(np.mean(cov)) if cov else np.nan,
        "min_option_mass_coverage": float(np.min(cov)) if cov else np.nan,
        "n_coverage_below_0.8": int(sum(v < .8 for v in cov)),
    }


def print_qc(full, default):
    section("1. RUN QC")
    for q in (full, default):
        print(f"  {q['scope']:<20} {q['n_records']:>6,} records  {q['n_cells']:>5,} cells  "
              f"{q['n_models']:>2} models  {q['n_items']:>2} items")
        print(f"    failures/refusals/missing/errors = {q['parse_failures']}/"
              f"{q['refusals']}/{q['missing_ratings']}/{q['infra_errors']}; "
              f"one-direction cells={q['single_direction_cells']}; "
              f"coverage<.8={q['n_coverage_below_0.8']}")
        print(f"    option-mass coverage mean={q['mean_option_mass_coverage']:.4f}, "
              f"min={q['min_option_mass_coverage']:.4f}")
    print(f"\n  Default validity estimand: {condition_label()}")


def _mean_interitem_r(M):
    C = M.corr().values
    return float(C[np.triu_indices_from(C, 1)].mean()) if C.shape[0] > 1 else np.nan


def _alpha_bootstrap(M, n_boot=2000, seed=17):
    X = M.dropna().reset_index(drop=True)
    if len(X) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        a = cronbach_alpha(X.iloc[idx])
        if np.isfinite(a):
            vals.append(a)
    if len(vals) < n_boot // 2:
        return np.nan, np.nan
    return tuple(float(x) for x in np.quantile(vals, [.025, .975]))


def _definitions(by_construct=True):
    if by_construct:
        return [(label, sid, sub) for label, sid, sub, _meaning, _pol in CONSTRUCTS]
    seen = []
    for _label, sid, _sub, _meaning, _pol in CONSTRUCTS:
        if sid not in seen:
            seen.append(sid)
    return [(sid, sid, None) for sid in seen]


def reliability_table(cells, *, bootstrap=True, by_construct=True):
    rows = []
    for pos, (label, sid, sub) in enumerate(_definitions(by_construct)):
        g = cells[cells.scale_id == sid]
        if sub:
            g = g[g.subscale == sub]
        M = respondent_matrix(g)
        rev = g.groupby("item_id").reverse_keyed.first().reindex(M.columns)
        fwd = list(rev[~rev].index)
        reverse = list(rev[rev].index)
        alpha = cronbach_alpha(M)
        lo, hi = (_alpha_bootstrap(M, seed=173 + pos) if bootstrap else (np.nan, np.nan))
        r_halves = (corr(M[fwd].mean(1), M[reverse].mean(1))
                    if fwd and reverse else np.nan)
        warnings = []
        if fwd and reverse and r_halves < .20:
            warnings.append("wording_halves_disagree")
        if len(fwd) != len(reverse):
            warnings.append("wording_unbalanced")
        if not fwd or not reverse:
            warnings.append("single_direction")
        if not np.isfinite(alpha) or alpha < .60:
            status = "do_not_use_composite"
        elif alpha < .70:
            status = "weak"
        elif alpha < .80 or "wording_halves_disagree" in warnings:
            status = "use_with_warning"
        else:
            status = "acceptable_internal_consistency"
        rows.append({
            "construct": label, "scale_id": sid, "subscale": sub or "",
            "n_items": M.shape[1], "n_models": M.shape[0],
            "n_forward": len(fwd), "n_reverse": len(reverse),
            "alpha": alpha, "alpha_ci_low": lo, "alpha_ci_high": hi,
            "mean_interitem_r": _mean_interitem_r(M),
            "alpha_forward": cronbach_alpha(M[fwd]) if len(fwd) > 1 else np.nan,
            "alpha_reverse": cronbach_alpha(M[reverse]) if len(reverse) > 1 else np.nan,
            "r_forward_reverse": r_halves,
            "equal_wording_counts": len(fwd) == len(reverse) and bool(fwd),
            "warnings": "|".join(warnings), "reliability_status": status,
        })
    return pd.DataFrame(rows)


def reliability_by_condition(core):
    rows = []
    for framing in sorted(core.framing.unique()):
        for instruction in sorted(core.paraphrase.unique()):
            c = select_condition(core, framing=framing, paraphrase=instruction)
            role = ("default" if framing == DEFAULT_FRAMING and instruction == DEFAULT_PARAPHRASE
                    else "instruction_robustness" if framing == DEFAULT_FRAMING
                    else "framing_analysis" if instruction == DEFAULT_PARAPHRASE
                    else "framing_instruction_interaction")
            r = reliability_table(c, bootstrap=False)
            r.insert(0, "condition_role", role)
            r.insert(1, "framing", framing)
            r.insert(2, "instruction_form", instruction)
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def _icc(X, absolute=True):
    """Two-way single-measure ICC: absolute agreement or consistency."""
    X = np.asarray(X, float)
    n, k = X.shape
    grand, row_mean, col_mean = X.mean(), X.mean(1), X.mean(0)
    ms_row = k * ((row_mean - grand) ** 2).sum() / (n - 1)
    ms_col = n * ((col_mean - grand) ** 2).sum() / (k - 1)
    residual = X - row_mean[:, None] - col_mean[None, :] + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_err
    if absolute:
        denominator += k * (ms_col - ms_err) / n
    return float((ms_row - ms_err) / denominator) if denominator else np.nan


def _paired_form_summary(W, id_values, id_name):
    rows = []
    for entity in id_values:
        q = W.loc[entity] if isinstance(W.index, pd.MultiIndex) else W.loc[[entity]]
        if isinstance(q, pd.Series):
            q = q.to_frame().T
        q = q[["p0", "p1", "p2"]].dropna()
        deltas = pd.concat([(q.p1 - q.p0).rename("p1-p0"),
                            (q.p2 - q.p0).rename("p2-p0")], axis=0)
        absolute = deltas.abs()
        pair_r = [q.p0.corr(q.p1), q.p0.corr(q.p2)]
        pair_rho = [q.p0.corr(q.p1, method="spearman"),
                    q.p0.corr(q.p2, method="spearman")]
        rows.append({
            id_name: entity,
            "mean_p0": q.p0.mean(), "mean_p1": q.p1.mean(), "mean_p2": q.p2.mean(),
            "shift_p1_minus_p0": (q.p1 - q.p0).mean(),
            "shift_p2_minus_p0": (q.p2 - q.p0).mean(),
            "instruction_mae": absolute.mean(),
            "instruction_median_abs_delta": absolute.median(),
            "instruction_p90_abs_delta": absolute.quantile(.90),
            "instruction_max_abs_delta": absolute.max(),
            "share_abs_delta_gt_0_10": (absolute > .10).mean(),
            "share_abs_delta_gt_0_20": (absolute > .20).mean(),
            "pearson_r_min": np.nanmin(pair_r), "spearman_r_min": np.nanmin(pair_rho),
            "icc_absolute_single": _icc(q.values, absolute=True),
            "icc_consistency_single": _icc(q.values, absolute=False),
        })
    return pd.DataFrame(rows)


def instruction_robustness(core):
    bare = core[core.framing == DEFAULT_FRAMING]
    scores = construct_scores(bare)
    SW = scores.pivot_table(index=["construct", "model"], columns="paraphrase", values="score")
    construct = _paired_form_summary(SW, [c[0] for c in CONSTRUCTS], "construct")
    construct["robustness_status"] = np.select(
        [
            (construct.instruction_mae > .10) | (construct.icc_absolute_single < .50),
            (construct.instruction_mae > .05) | (construct.icc_absolute_single < .75),
        ], ["unstable", "warning"], default="acceptable")

    IW = bare.pivot_table(index=["item_id", "model"], columns="paraphrase", values="score")
    item_ids = sorted(bare.item_id.unique())
    item = _paired_form_summary(IW, item_ids, "item_id")

    # Does poor discrimination replicate across the three instruction forms?
    form_rows = []
    for instruction in ["p0", "p1", "p2"]:
        c = bare[bare.paraphrase == instruction]
        for label, sid, sub, _meaning, _pol in CONSTRUCTS:
            g = c[c.scale_id == sid]
            if sub:
                g = g[g.subscale == sub]
            M = respondent_matrix(g)
            alpha = cronbach_alpha(M)
            for iid in M.columns:
                siblings = [x for x in M.columns if x != iid]
                r = corr(M[iid], M[siblings].mean(1)) if siblings else np.nan
                a_del = cronbach_alpha(M[siblings]) if len(siblings) > 1 else np.nan
                form_rows.append({"item_id": iid, "instruction_form": instruction,
                                  "item_total_r": r, "alpha": alpha,
                                  "alpha_if_deleted_delta": a_del - alpha})
    forms = pd.DataFrame(form_rows)
    rep = forms.groupby("item_id").agg(
        n_forms_negative_item_total=("item_total_r", lambda x: int((x < 0).sum())),
        n_forms_low_item_total=("item_total_r", lambda x: int((x < .20).sum())),
        n_forms_alpha_gain=("alpha_if_deleted_delta", lambda x: int((x > .02).sum())),
    ).reset_index()
    return construct, item.merge(rep, on="item_id"), forms, scores


def response_style(records):
    frame = pd.DataFrame([{
        "model": r.model_id, "rating": r.parsed_rating, "modal": r.modal_rating,
        "direction": direction_of(r), "coverage": r.option_mass_coverage,
        "pmax": max(r.response_distribution.values()) if r.response_distribution else np.nan,
    } for r in records])
    rows = []
    for model, g in frame.groupby("model"):
        rows.append({
            "model": model, "midpoint_argmax_rate": (g.modal == MIDPOINT).mean(),
            "extreme_argmax_rate": g.modal.isin([1.0, 7.0]).mean(),
            "mean_abs_dist_from_midpoint": (g.rating - MIDPOINT).abs().mean(),
            "mean_p_argmax": g.pmax.mean(), "mean_option_mass_coverage": g.coverage.mean(),
        })
    return pd.DataFrame(rows).sort_values("model")


def construct_acquiescence(default_cells, default_scores):
    wide = default_scores.pivot(index="model", columns="construct", values="score")
    rows = []
    for label, sid, sub, _meaning, _pol in CONSTRUCTS:
        # For a construct that helped define the marker, leave its entire source
        # scale out. This avoids part-whole/content circularity (the same rule as
        # the item-level diagnostic). Other constructs use the RSES marker.
        marker_scales = ACQUIESCENCE_SCALE_IDS - ({sid} if sid in ACQUIESCENCE_SCALE_IDS else set())
        marker = default_cells[default_cells.scale_id.isin(marker_scales)].groupby(
            "model").score_as_written.mean()
        r = corr(wide[label], marker.reindex(wide.index))
        rows.append({"construct": label, "r_acquiescence_marker": r,
                     "r2": r*r, "response_style_warning": abs(r) > .60,
                     "response_style_marker_scales": "|".join(sorted(marker_scales))})
    return pd.DataFrame(rows)


def item_table(default_records, default_cells, core, battery, reliability,
               item_instruction):
    meta = {it.item_id: it for _scale, it in battery.items()}
    construct_of = {}
    for label, sid, sub, _meaning, _pol in CONSTRUCTS:
        for iid in default_cells[(default_cells.scale_id == sid)
                                  & ((default_cells.subscale == sub) if sub else True)].item_id.unique():
            construct_of[iid] = label
    rel = reliability.set_index("construct")
    M = respondent_matrix(default_cells, index_cols=["model"])
    C = M.corr()
    mid, pmax = defaultdict(list), defaultdict(list)
    for r in default_records:
        mid[r.item_id].append(r.modal_rating == MIDPOINT)
        if r.response_distribution:
            pmax[r.item_id].append(max(r.response_distribution.values()))
    instruction = item_instruction.set_index("item_id")

    # Framing contrasts are exported as context only; they never enter flags.
    p0 = core[core.paraphrase == "p0"]
    FW = p0.pivot_table(index=["item_id", "model"], columns="framing", values="score")

    rows = []
    for iid in M.columns:
        it = meta[iid]
        construct = construct_of[iid]
        siblings = [x for x in M.columns if x != iid and construct_of.get(x) == construct]
        item_r = corr(M[iid], M[siblings].mean(1)) if siblings else np.nan
        alpha = rel.loc[construct, "alpha"]
        alpha_deleted = cronbach_alpha(M[siblings]) if len(siblings) > 1 else np.nan
        positive = C.loc[iid, siblings].dropna() if siblings else pd.Series(dtype=float)
        ci = default_cells[default_cells.item_id == iid]
        # Leave the entire source scale out of the response-style marker to
        # avoid part-whole inflation for RSES items.
        marker_cells = default_cells[
            default_cells.scale_id.isin(ACQUIESCENCE_SCALE_IDS)
            & (default_cells.scale_id != it.scale_id)]
        marker = marker_cells.groupby("model").score_as_written.mean()
        r_acq = corr(M[iid], marker.reindex(M.index)) if len(marker) else np.nan
        inst = instruction.loc[iid]
        fw = FW.loc[iid]
        rows.append({
            "item_id": iid, "construct": construct, "scale_id": it.scale_id,
            "subscale": it.subscale or "", "reverse_scored": it.reverse_scored,
            "ai_applicable": it.ai_applicable, "applicability_note": it.applicability_note,
            "item_text": it.text, "mean_keyed": ci.score.mean(),
            "mean_as_written": ci.score_as_written.mean(),
            "sd_between_models": ci.groupby("model").score.mean().std(ddof=1),
            "range_between_models": ci.groupby("model").score.mean().max()
                                    - ci.groupby("model").score.mean().min(),
            "item_total_r": item_r, "parent_alpha": alpha,
            "alpha_if_deleted": alpha_deleted,
            "alpha_if_deleted_delta": alpha_deleted - alpha,
            "max_positive_sibling_r": positive.max() if len(positive) else np.nan,
            "max_positive_sibling": positive.idxmax() if len(positive) else "",
            "most_negative_sibling_r": positive.min() if len(positive) else np.nan,
            "most_negative_sibling": positive.idxmin() if len(positive) else "",
            "midpoint_rate": np.mean(mid[iid]), "mean_p_argmax": np.mean(pmax[iid]),
            "order_mae_norm": ci.direction_gap_norm.abs().mean(),
            "order_median_abs_norm": ci.direction_gap_norm.abs().median(),
            "order_p90_abs_norm": ci.direction_gap_norm.abs().quantile(.90),
            "order_max_abs_norm": ci.direction_gap_norm.abs().max(),
            "r_acquiescence": r_acq,
            "instruction_mae": inst.instruction_mae,
            "instruction_p90_abs_delta": inst.instruction_p90_abs_delta,
            "instruction_max_abs_delta": inst.instruction_max_abs_delta,
            "instruction_spearman_r_min": inst.spearman_r_min,
            "n_forms_negative_item_total": int(inst.n_forms_negative_item_total),
            "n_forms_low_item_total": int(inst.n_forms_low_item_total),
            "n_forms_alpha_gain": int(inst.n_forms_alpha_gain),
            "ack_minus_bare_mean": (fw.first_person_ack - fw.first_person_bare).mean(),
            "third_minus_bare_mean": (fw.third_person_assistant - fw.first_person_bare).mean(),
        })
    out = pd.DataFrame(rows)
    out["abs_r_acquiescence"] = out.r_acquiescence.abs()

    all_flags, primary_flags_out, robustness_flags_out = [], [], []
    primary_recommendations, recommendations, notes = [], [], []
    for _, row in out.iterrows():
        flags = []
        if row.item_total_r < 0:
            flags.append("negative_item_total")
        elif row.item_total_r < .20:
            flags.append("low_item_total")
        for name in ["low_variance", "alpha_improves_if_deleted", "instruction_unstable",
                     "order_sensitive", "acquiescence_loaded", "redundant"]:
            stat, op, threshold, _why = RULES[name]
            if pd.notna(row[stat]) and _op(op, row[stat], threshold):
                flags.append(name)
        # MSI midpoint is exact ideal-alignment, not an uninformative neutral.
        if row.scale_id != "MSI_2015_LLM_ADAPT" and row.midpoint_rate > .60:
            flags.append("midpoint_heavy")
        if row.ai_applicable != "clear":
            flags.append(row.ai_applicable)
        if row.parent_alpha < .60:
            flags.append("parent_construct_unreliable")

        primary_flags = [f for f in flags if f in {
            "negative_item_total", "low_item_total", "low_variance",
            "alpha_improves_if_deleted", "parent_construct_unreliable",
        }]
        robustness_flags = [f for f in flags if f in {
            "instruction_unstable", "order_sensitive",
        }]
        primary_recommendation = (
            "review" if (row.item_total_r < .20 or row.alpha_if_deleted_delta > .02
                         or row.parent_alpha < .60) else "keep")

        recurring = row.n_forms_negative_item_total == 3 and row.n_forms_alpha_gain >= 2
        parent_globally_broken = row.construct == "External influence (AS)"
        if recurring and not parent_globally_broken:
            recommendation = "drop_candidate"
            note = "recurring misfit across all instruction forms; revise/remove, then confirm on new models"
        elif primary_recommendation == "review":
            recommendation = "review"
            note = ("construct-wide failure; revise/split the construct before deleting one item"
                    if parent_globally_broken else
                    "inspect wording and retain/drop only after confirmatory replication")
        elif flags:
            recommendation = "keep_with_warning"
            note = "no primary structural failure requiring removal"
        else:
            recommendation = "keep"
            note = "no threshold crossed in this exploratory screen"
        all_flags.append("|".join(flags))
        primary_flags_out.append("|".join(primary_flags))
        robustness_flags_out.append("|".join(robustness_flags))
        primary_recommendations.append(primary_recommendation)
        recommendations.append(recommendation)
        notes.append(note)
    out["flags"] = all_flags
    out["primary_flags"] = primary_flags_out
    out["robustness_flags"] = robustness_flags_out
    out["primary_recommendation"] = primary_recommendations
    out["recommendation"] = recommendations
    out["decision_note"] = notes
    return out.sort_values(["construct", "item_id"]).reset_index(drop=True), C


def construct_validity_summary(reliability, robustness, acquiescence_table):
    out = reliability.merge(robustness, on="construct").merge(acquiescence_table, on="construct")
    status = []
    implication = []
    analysis_flags = []
    for _, r in out.iterrows():
        issues = []
        flags = []
        if r.alpha < .60:
            flags.append("default_internal_consistency_failure")
            issues.append("internal consistency is insufficient under the default condition")
        elif r.alpha < .70:
            flags.append("weak_default_internal_consistency")
            issues.append("default internal consistency is weak")
        if r.robustness_status == "unstable":
            flags.append("instruction_unstable")
            issues.append("absolute level or model agreement changes materially with instruction form")
        elif r.robustness_status == "warning":
            flags.append("instruction_warning")
            issues.append("instruction-form agreement is only moderate")
        if "wording_halves_disagree" in r.warnings:
            flags.append("wording_halves_disagree")
            issues.append("alpha masks weak agreement between forward/reverse halves")
        if r.response_style_warning:
            flags.append("response_style_warning")
            issues.append("score is strongly associated with the leave-scale-out agreement-style marker")
        if "single_direction" in r.warnings:
            flags.append("single_direction")
            issues.append("all items use one keyed wording direction")
        elif "wording_unbalanced" in r.warnings:
            flags.append("wording_unbalanced")
            issues.append("forward/reverse item counts are unbalanced")

        if r.alpha < .60:
            status.append("do_not_use_composite")
        elif r.robustness_status == "unstable":
            status.append("default_condition_only")
        elif sum([r.robustness_status == "warning",
                  "wording_halves_disagree" in r.warnings,
                  bool(r.response_style_warning)]) > 1:
            status.append("use_with_multiple_warnings")
        elif r.robustness_status == "warning":
            status.append("use_with_instruction_warning")
        elif "wording_halves_disagree" in r.warnings:
            status.append("use_with_wording_warning")
        elif r.response_style_warning:
            status.append("use_with_response_style_warning")
        elif r.alpha < .70:
            status.append("use_with_reliability_warning")
        else:
            status.append("provisionally_usable")
        implication.append("; ".join(issues) if issues else
                           "passes internal-consistency and instruction-robustness screens")
        analysis_flags.append("|".join(flags))
    out["analysis_status"] = status
    out["analysis_flags"] = analysis_flags
    out["implication"] = implication
    return out


def print_reliability(summary):
    section("2. DEFAULT-CONDITION CONSTRUCT VALIDITY")
    n_models = int(summary.n_models.min())
    print(f"  Reliability uses {n_models} model-level rows at first_person_bare + p0. Alpha")
    print("  is necessary but not sufficient; instruction stability and wording halves")
    print("  appear beside it. Bootstrap intervals are descriptive with n=13.\n")
    print(f"  {'construct':<28} {'k':>2} {'alpha':>7} {'95% boot CI':>17} {'r halves':>9} "
          f"{'instr MAE':>10}  status")
    for _, r in summary.iterrows():
        ci = f"[{fmt(r.alpha_ci_low,2)}, {fmt(r.alpha_ci_high,2)}]"
        print(f"  {r.construct:<28} {r.n_items:>2} {r.alpha:>7.3f} {ci:>17} "
              f"{fmt(r.r_forward_reverse,2):>9} {r.instruction_mae:>10.3f}  {r.analysis_status}")


def print_items(items):
    section("3. ITEM DECISIONS AND WARNINGS")
    counts = items.recommendation.value_counts()
    for label in ["keep", "keep_with_warning", "review", "drop_candidate"]:
        print(f"  {label:<20} {int(counts.get(label, 0)):>3}")
    print("\n  Drop candidates are provisional revision candidates, not silently removed")
    print("  from the present results. They require recurring misfit across all three")
    print("  instruction forms plus an alpha gain when deleted.\n")
    for _, r in items[items.recommendation == "drop_candidate"].iterrows():
        print(f"    {r.item_id:<11} r={r.item_total_r:+.2f}  "
              f"alpha {r.parent_alpha:.2f}->{r.alpha_if_deleted:.2f}  "
              f"fails {r.n_forms_negative_item_total}/3 forms  {r.item_text[:54]}")
    print("\n  Review priorities:")
    for _, r in items[items.recommendation == "review"].sort_values(
            ["parent_alpha", "item_total_r"]).head(15).iterrows():
        print(f"    {r.item_id:<11} {r.construct[:23]:<23} r={r.item_total_r:+.2f}  "
              f"instr_MAE={r.instruction_mae:.2f}  {r['flags']}")


def dossier_md(summary, items, qc_full, qc_default):
    L = [
        "# Measurement validity: default condition\n",
        f"Default estimand: `{condition_label()}`. The current run supplies "
        f"{qc_default['n_models']} model-level rows × {qc_default['n_items']} items.\n",
        "> These are exploratory revision recommendations. No item is silently removed from "
        "the results, and every deletion must be confirmed on new models/item forms.\n",
        "## Can each construct be analyzed?\n",
        "| construct | alpha | bootstrap 95% CI | instruction MAE | instruction ICC(A,1) | status | implication |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for _, r in summary.iterrows():
        L.append(f"| {r.construct} | {r.alpha:.3f} | [{fmt(r.alpha_ci_low)}, {fmt(r.alpha_ci_high)}] | "
                 f"{r.instruction_mae:.3f} | {r.icc_absolute_single:.3f} | "
                 f"`{r.analysis_status}` | {r.implication} |")
    L += [
        "\nInternal consistency and instruction robustness answer different questions. "
        "For example, SCCS has very high alpha but a large p0→p1/p2 level shift.\n",
        "## Item actions\n",
        "`primary_recommendation` uses only default-condition structural evidence. "
        "`drop_candidate` additionally requires that the default misfit recur in all three "
        "instruction forms with an alpha gain in at least two. Instruction/order effects "
        "otherwise add warnings only. Framing never enters any item decision.\n",
        "| item | construct | r(item, rest) | alpha → if deleted | instruction MAE | order MAE | primary | final | robustness flags | all flags |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    order = {"drop_candidate": 0, "review": 1, "keep_with_warning": 2, "keep": 3}
    shown = items.assign(_order=items.recommendation.map(order)).sort_values(["_order", "construct", "item_id"])
    for _, r in shown.iterrows():
        L.append(f"| `{r.item_id}` | {r.construct} | {r.item_total_r:.3f} | "
                 f"{r.parent_alpha:.3f} → {r.alpha_if_deleted:.3f} | "
                 f"{r.instruction_mae:.3f} | {r.order_mae_norm:.3f} | "
                 f"{r.primary_recommendation} | **{r.recommendation}** | "
                 f"`{r.robustness_flags}` | `{r['flags']}` |")
    L += [
        "\n## Interpretation rules\n",
        "- p0/p1/p2 vary the **instruction**, not the item wording.",
        "- First-person acknowledgement and third-person assistant are substantive framing "
        "conditions. Their differences are reported by `survey.analysis`, not treated as instability.",
        "- Factor communality is deferred: 13 independent models cannot identify a "
        "45-item factor solution.",
        "- The agreement-style marker uses the exactly wording-balanced RSES items; "
        "it is a warning, not deletion evidence. Source-overlap makes this marker "
        "unavailable for RSES itself.",
        "- MSI midpoint use is ideal alignment and is not flagged as midpoint-heavy.",
        f"\nRun QC: full={qc_full['n_records']:,} rows; default={qc_default['n_records']:,} rows; "
        f"default coverage<.8={qc_default['n_coverage_below_0.8']}.\n",
    ]
    return "\n".join(L)


def run(results_path="results.jsonl", out_dir="results/validity", save=True):
    validate_default_config()
    records = load(results_path)
    cells = cell_frame(records)
    core = _core_cells(cells)
    validate_report_dataset(core, results_path)
    default_cells = default_condition(cells)
    default_records = [r for r in records if _default_record(r)]
    validate_default_slice(default_cells, core)
    battery = load_battery()

    print(f"Loaded {len(records):,} battery records from {results_path}")
    qc_full = qc_summary(records, cells, "full crossed run")
    qc_default = qc_summary(default_records, default_cells, "default condition")
    print_qc(qc_full, qc_default)

    reliability = reliability_table(default_cells)
    scale_reliability = reliability_table(default_cells, by_construct=False)
    by_condition = reliability_by_condition(core)
    construct_instruction, item_instruction, form_item, instruction_scores = instruction_robustness(core)
    default_scores = construct_scores(default_cells)
    acq_table = construct_acquiescence(default_cells, default_scores)
    summary = construct_validity_summary(reliability, construct_instruction, acq_table)
    print_reliability(summary)

    items, item_corr = item_table(
        default_records, default_cells, core, battery, reliability,
        item_instruction)
    print_items(items)
    style = response_style(default_records)

    if save:
        section("SAVED")
    s = Saver(out_dir, enabled=save)
    s.csv(summary, "construct_validity_summary.csv",
          "analysis gate: default alpha + instruction robustness + response-style warnings")
    s.csv(reliability, "construct_reliability_default.csv",
          "6 construct/subscale reliability estimates at first_person_bare+p0")
    s.csv(scale_reliability, "scale_reliability.csv",
          "5 source-instrument alpha estimates at the default condition")
    s.csv(by_condition, "reliability_by_condition.csv",
          "construct reliability in all framing x instruction cells, with condition role")
    s.csv(construct_instruction, "instruction_robustness_by_construct.csv",
          "paired p1/p2 versus p0 stability at first_person_bare")
    s.csv(item_instruction, "instruction_robustness_by_item.csv",
          "paired instruction stability and replication counts for every item")
    s.csv(form_item, "item_diagnostics_by_instruction.csv",
          "item-rest and alpha-if-deleted in p0/p1/p2 at first_person_bare")
    s.csv(items, "item_validity.csv",
          "default item evidence, robustness warnings, and actionable recommendation")
    s.text(dossier_md(summary, items, qc_full, qc_default), "item_validity.md",
           "human-readable construct gate and complete item action table")
    s.csv(acq_table, "acquiescence_by_construct.csv",
          "default construct relation to the balanced agreement-style marker")
    s.csv(style, "response_style.csv", "default-condition response style by model")
    s.csv(item_corr.reset_index(names="item_id"), "item_correlation_matrix.csv",
          "default-condition 45x45 item correlations (n=13; exploratory)")
    if save:
        plot_path = f"{out_dir}/plots/item_validity_diagnostics.png"
        plot_item_validity(items, plot_path)
        s.artifact("plots/item_validity_diagnostics.png",
                   "structural item evidence versus instruction/order warnings")
    s.json({
        "schema_version": "2.0", "source_file": str(results_path),
        "default_condition": condition_label(), "qc_full": qc_full,
        "qc_default": qc_default, "rules": RULES, "construct_validity": summary,
        "construct_reliability_default": reliability,
        "source_scale_reliability_default": scale_reliability,
        "reliability_by_condition": by_condition,
        "instruction_robustness_by_construct": construct_instruction,
        "instruction_robustness_by_item": item_instruction,
        "item_diagnostics_by_instruction": form_item,
        "acquiescence_by_construct": acq_table,
        "item_validity": items, "response_style": style,
        "item_correlation_matrix": item_corr.reset_index(names="item_id"),
    }, "validity_full.json", "complete validity output")
    s.index(
        "Measurement validity — default condition",
        f"Written from `{results_path}`. Primary validity is evaluated only at "
        f"`{DEFAULT_FRAMING} + {DEFAULT_PARAPHRASE}` ({default_cells.model.nunique()} models).\n\n"
        "Instruction forms are robustness checks. Framing contrasts are substantive analysis "
        "and never determine item deletion. Read `item_validity.md` first.\n")
    return {"qc": qc_default, "constructs": summary, "items": items,
            "reliability_by_condition": by_condition}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Default-condition measurement validity")
    parser.add_argument("results", nargs="?", default="results.jsonl")
    parser.add_argument("--out", default="results/validity")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    try:
        run(args.results, args.out, save=not args.no_save)
    except ReportDatasetError as err:
        parser.error(str(err))
