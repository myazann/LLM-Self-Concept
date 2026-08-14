"""Substantive results for the LLM self-concept experiment.

Headline profiles and model/time/family/size comparisons use exactly the
default administration: first_person_bare + p0.  Instruction p1/p2 is a
robustness analysis at the bare framing.  Framing is a substantive factor and
is compared at p0; it is never averaged into a model's default self-score.

Models are the analysis units for construct correlations and trends.
Condition-level rows remain available for sensitivity analysis but are not
treated as independent respondents.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from plots import (
    plot_default_heatmap, plot_framing_effects, plot_instruction_robustness,
    plot_release_trajectories, plot_size_ladders, polarity_aligned,
)
from scoring import (
    ACQUIESCENCE_SCALE_IDS, CONSTRUCTS, DEFAULT_FACTORS, DEFAULT_FRAMING,
    DEFAULT_PARAPHRASE, LABELS, POLARITY, Saver,
    bar, cell_frame, condition_label, construct_scores, corr,
    corr_p, default_condition, hedges_g, ipsatize, load, meta_frame,
    partial_corr, section, select_condition, validate_default_config,
    validate_default_slice,
)

def _core_cells(cells):
    return select_condition(cells, framing=None, paraphrase=None)


def _model_scores(scores):
    return scores.pivot(index="model", columns="construct", values="score")[LABELS].reset_index()


def print_profile(model_scores, validity):
    section("RQ1. DEFAULT SELF-REPORT PROFILE")
    print(f"  Estimand: {condition_label()}")
    print("  Scores are 0..1; 0.50 is the response midpoint (MSI: alignment with ideal).\n")
    status = validity.set_index("construct").analysis_status
    rows = []
    print(f"  {'construct':<28} {'mean':>6} {'sd':>6} {'min':>6} {'max':>6}  "
          f"{'0':<11}{'.5':<12}{'1':<3}  validity")
    for label in LABELS:
        values = model_scores[label]
        print(f"  {label:<28} {values.mean():>6.3f} {values.std(ddof=1):>6.3f} "
              f"{values.min():>6.3f} {values.max():>6.3f}  {bar(values.mean())}  {status[label]}")
        rows.append({
            "construct": label, "polarity": POLARITY[label], "mean": values.mean(),
            "sd_between_models": values.std(ddof=1), "min": values.min(),
            "max": values.max(), "analysis_status": status[label],
            "default_framing": DEFAULT_FRAMING, "default_instruction_form": DEFAULT_PARAPHRASE,
        })
    print("\n  Higher scores are the positive pole for the first three constructs; higher")
    print("  scores are distress for lack of identity, alienation, and external influence.")
    print("  A coherent mean is not automatically a valid composite; heed the final column.")
    return pd.DataFrame(rows)


def print_per_model(model_scores, meta):
    section("RQ1b. DEFAULT SCORES BY MODEL")
    wide = model_scores.set_index("model").reindex(meta.sort_values(["release_dt", "params_b"]).model)
    short = {label: label.split(" (")[0][:13] for label in LABELS}
    print(f"  {'model':<20}" + "".join(f"{short[label]:>14}" for label in LABELS))
    for model, row in wide.iterrows():
        print(f"  {model:<20}" + "".join(f"{row[label]:>14.3f}" for label in LABELS))


def construct_structure(default_scores):
    section("RQ1c. CONSTRUCT RELATIONS AT THE DEFAULT CONDITION")
    W = default_scores.pivot(index="model", columns="construct", values="score")[LABELS]
    C = W.corr()
    labs = [label.split(" (")[0][:12] for label in LABELS]
    print(f"  {'':<14}" + "".join(f"{label[:6]:>8}" for label in labs))
    for i, label in enumerate(LABELS):
        print(f"  {labs[i]:<14}" + "".join(f"{C.loc[label, other]:>8.2f}" for other in LABELS))

    pairs = [
        ("Self-concept clarity (SCCS)", "Lack of identity (SCIM)", "negative"),
        ("Self-alienation (AS)", "Self-concept clarity (SCCS)", "negative"),
        ("Self-alienation (AS)", "Lack of identity (SCIM)", "positive"),
    ]
    rows = []
    print(f"\n  Nomological sign checks (exploratory, n={len(W)} model-level rows):")
    for a, b, expected in pairs:
        r, p = corr_p(W[a], W[b])
        ok = (r > 0) == (expected == "positive")
        print(f"    {'ok' if ok else 'FAIL':<4} r({a.split(' (')[0]}, {b.split(' (')[0]}) "
              f"= {r:+.2f}, p={p:.3f}; expected {expected}")
        rows.append({"construct_a": a, "construct_b": b, "expected_sign": expected,
                     "r": r, "p": p, "n_models": len(W), "as_expected": ok})
    print(f"  These p-values use {len(W)} model-level rows, not repeated prompt conditions.")
    return C, pd.DataFrame(rows)


def print_item_extremes(default_cells, battery):
    section("RQ1d. DEFAULT ITEM EXTREMES (ENDORSEMENT AS WRITTEN)")
    text = {item.item_id: item.text for _scale, item in battery.items()}
    applicability = {item.item_id: item.ai_applicable for _scale, item in battery.items()}
    values = default_cells.groupby("item_id").score_as_written.agg(["mean", "std"]).sort_values("mean")
    for title, subset in [("MOST endorsed", values.tail(6).iloc[::-1]),
                          ("LEAST endorsed", values.head(6))]:
        print(f"\n  {title}:")
        for iid, row in subset.iterrows():
            print(f"    {row['mean']:.2f} (sd {row['std']:.2f})  {iid:<11} "
                  f"[{applicability[iid][:8]:<8}] {text[iid][:68]}")


def instruction_effects(instruction_scores):
    """p1/p2 robustness against p0 with first-person-bare held fixed."""
    rows, model_rows = [], []
    for construct, g in instruction_scores.groupby("construct"):
        W = g.pivot(index="model", columns="paraphrase", values="score")[["p0", "p1", "p2"]]
        deltas = pd.concat([W.p1 - W.p0, W.p2 - W.p0])
        rows.append({
            "construct": construct, "mean_p0": W.p0.mean(), "mean_p1": W.p1.mean(),
            "mean_p2": W.p2.mean(), "shift_p1_minus_p0": (W.p1-W.p0).mean(),
            "shift_p2_minus_p0": (W.p2-W.p0).mean(), "paired_mae": deltas.abs().mean(),
            "paired_p90_abs": deltas.abs().quantile(.9), "paired_max_abs": deltas.abs().max(),
            "pearson_p0_p1": W.p0.corr(W.p1), "pearson_p0_p2": W.p0.corr(W.p2),
            "spearman_p0_p1": W.p0.corr(W.p1, method="spearman"),
            "spearman_p0_p2": W.p0.corr(W.p2, method="spearman"),
        })
        for model, r in W.iterrows():
            model_rows.append({"model": model, "construct": construct,
                               "p0": r.p0, "p1": r.p1, "p2": r.p2,
                               "p1_minus_p0": r.p1-r.p0, "p2_minus_p0": r.p2-r.p0})
    return pd.DataFrame(rows).set_index("construct").reindex(LABELS).reset_index(), pd.DataFrame(model_rows)


def print_instruction_effects(table):
    section("RQ2. INSTRUCTION ROBUSTNESS (FIRST-PERSON BARE HELD FIXED)")
    print("  p0 is the default. p1/p2 change only the scale instruction, not item text.\n")
    print(f"  {'construct':<28} {'p0':>6} {'p1':>6} {'p2':>6} {'MAE':>7} {'p90':>7} "
          f"{'r min':>7}")
    for _, r in table.iterrows():
        print(f"  {r.construct:<28} {r.mean_p0:>6.3f} {r.mean_p1:>6.3f} {r.mean_p2:>6.3f} "
              f"{r.paired_mae:>7.3f} {r.paired_p90_abs:>7.3f} "
              f"{min(r.pearson_p0_p1, r.pearson_p0_p2):>7.2f}")
    print("\n  Large MAE means absolute scores move even if model ranks correlate. SCCS is")
    print("  the clearest warning: its scale-specific p0 instruction primes clarity.")


def framing_effects(p0_scores):
    """Substantive framing contrasts with instruction fixed at p0."""
    rows, model_rows = [], []
    for construct, g in p0_scores.groupby("construct"):
        W = g.pivot(index="model", columns="framing", values="score")
        rows.append({
            "construct": construct,
            "first_person_bare": W.first_person_bare.mean(),
            "first_person_ack": W.first_person_ack.mean(),
            "third_person_assistant": W.third_person_assistant.mean(),
            "ack_minus_bare": (W.first_person_ack-W.first_person_bare).mean(),
            "third_minus_bare": (W.third_person_assistant-W.first_person_bare).mean(),
            "mean_abs_ack_minus_bare": (W.first_person_ack-W.first_person_bare).abs().mean(),
            "mean_abs_third_minus_bare": (W.third_person_assistant-W.first_person_bare).abs().mean(),
            "n_ack_above_bare": int((W.first_person_ack > W.first_person_bare).sum()),
            "n_third_above_bare": int((W.third_person_assistant > W.first_person_bare).sum()),
        })
        for model, r in W.iterrows():
            model_rows.append({
                "model": model, "construct": construct, "first_person_bare": r.first_person_bare,
                "first_person_ack": r.first_person_ack,
                "third_person_assistant": r.third_person_assistant,
                "ack_minus_bare": r.first_person_ack-r.first_person_bare,
                "third_minus_bare": r.third_person_assistant-r.first_person_bare,
            })
    return pd.DataFrame(rows).set_index("construct").reindex(LABELS).reset_index(), pd.DataFrame(model_rows)


def print_framing(table):
    section("RQ3. FRAMING AS A SUBSTANTIVE RESULT (p0 HELD FIXED)")
    print(f"  {'construct':<28} {'bare self':>9} {'+AI pre':>9} {'typical AI':>10} "
          f"{'ack-bare':>9} {'third-bare':>11}")
    for _, r in table.iterrows():
        print(f"  {r.construct:<28} {r.first_person_bare:>9.3f} {r.first_person_ack:>9.3f} "
              f"{r.third_person_assistant:>10.3f} {r.ack_minus_bare:>+9.3f} "
              f"{r.third_minus_bare:>+11.3f}")
    print("\n  These are changes in target/self-presentation, not failures of robustness.")
    print("  The +AI preamble especially activates acceptance of external influence.")


def factorial_decomposition(core):
    """Orthogonal SS decomposition of the balanced model×frame×instruction×item grid."""
    levels = [sorted(core.model.unique()), sorted(core.framing.unique()),
              sorted(core.paraphrase.unique()), sorted(core.item_id.unique())]
    index = pd.MultiIndex.from_product(levels, names=["model", "framing", "paraphrase", "item_id"])
    Y = core.set_index(index.names).score.reindex(index).values.reshape([len(x) for x in levels])
    if np.isnan(Y).any():
        raise ValueError("core factorial is incomplete; cannot decompose interactions")
    grand, effects = Y.mean(), {}
    names = ["model", "framing", "instruction", "item"]
    rows = []
    for size in range(1, 5):
        for subset in itertools.combinations(range(4), size):
            complement = tuple(axis for axis in range(4) if axis not in subset)
            effect = Y.mean(axis=complement, keepdims=True) - grand
            for lower_size in range(1, size):
                for lower in itertools.combinations(subset, lower_size):
                    effect = effect - effects[lower]
            effects[subset] = effect
            multiplier = np.prod([Y.shape[axis] for axis in complement])
            ss = float((effect**2).sum() * multiplier)
            df = int(np.prod([Y.shape[axis]-1 for axis in subset]))
            rows.append({"term": ":".join(names[axis] for axis in subset),
                         "df": df, "sum_sq": ss})
    out = pd.DataFrame(rows)
    total = ((Y-grand)**2).sum()
    out["pct_total_variation"] = 100*out.sum_sq/total
    return out.sort_values("sum_sq", ascending=False).reset_index(drop=True)


def print_variance(table):
    section("RQ4. WHAT MOVES AN ITEM RESPONSE?")
    instruction_share = table[table.term.str.contains("instruction")].pct_total_variation.sum()
    print("  Balanced factorial decomposition, including interactions. Main effects alone")
    print("  are insufficient because wording effects are item- and model-specific.\n")
    for _, r in table.head(10).iterrows():
        print(f"    {r.term:<42} {r.pct_total_variation:>6.2f}%")
    print(f"\n  All instruction-containing terms together: {instruction_share:.2f}% of total variation.")


def build_trends(model_scores, ips_scores, meta, default_cells, status_map):
    m = meta.set_index("model")
    raw = model_scores.set_index("model").reindex(m.index)
    ips = ips_scores.set_index("model").reindex(m.index)
    rows = []
    source_scale = {label: sid for label, sid, _sub, _meaning, _pol in CONSTRUCTS}
    for label in LABELS:
        values = raw[label]
        sid = source_scale[label]
        marker_scales = ACQUIESCENCE_SCALE_IDS - (
            {sid} if sid in ACQUIESCENCE_SCALE_IDS else set())
        acq = default_cells[default_cells.scale_id.isin(marker_scales)].groupby(
            "model").score_as_written.mean().reindex(m.index)
        r_time, p_time = corr_p(m.days, values)
        r_partial, p_partial = partial_corr(m.days.values.astype(float), values.values, acq.values)
        in_window = m.in_window.values
        rows.append({
            "construct": label, "analysis_status": status_map[label],
            "polarity": POLARITY[label], "mean": values.mean(),
            "sd_between_models": values.std(ddof=1),
            "r_release_date": r_time, "p_release_date": p_time,
            "r_release_date_partial_response_style": r_partial,
            "p_release_date_partial_response_style": p_partial,
            "response_style_marker_scales": "|".join(sorted(marker_scales)),
            "r_release_date_ipsatized": corr(m.days, ips[label]),
            "r_release_date_in_window": corr(m.days[in_window], values[in_window]),
            "r_total_params": corr(m.log_params, values),
            "r_active_params": corr(np.log10(m.active_b), values),
            "mean_gemma": values[m.family == "gemma"].mean(),
            "mean_qwen": values[m.family == "qwen"].mean(),
            "hedges_g_gemma_vs_qwen": hedges_g(values[m.family == "gemma"],
                                                 values[m.family == "qwen"]),
            "default_framing": DEFAULT_FRAMING,
            "default_instruction_form": DEFAULT_PARAPHRASE,
        })
    return pd.DataFrame(rows)


def generation_steps(model_scores, meta, status_map):
    m = meta.set_index("model")
    scores = model_scores.set_index("model").reindex(m.index)
    rows = []
    for family, old, new in [("gemma", "gemma-3", "gemma-4"),
                             ("qwen", "qwen3.5", "qwen3.6")]:
        for label in LABELS:
            a = scores.loc[m.generation == old, label].mean()
            b = scores.loc[m.generation == new, label].mean()
            rows.append({"family": family, "from": old, "to": new, "construct": label,
                         "analysis_status": status_map[label],
                         "mean_from": a, "mean_to": b, "delta": b-a})
    return pd.DataFrame(rows)


MATCHED_TRACKS = {
    "gemma_dense_12b": ("Gemma3-12B", "Gemma4-12B"),
    "gemma_dense_approx_30b": ("Gemma3-27B", "Gemma4-31B"),
    "qwen_dense_27b": ("Qwen3.5-27B", "Qwen3.6-27B"),
    "qwen_moe_35b_a3b": ("Qwen3.5-35B-A3B", "Qwen3.6-35B-A3B"),
}


def matched_track_deltas(model_scores, status_map):
    W = model_scores.set_index("model")
    rows = []
    for track, (old, new) in MATCHED_TRACKS.items():
        if old not in W.index or new not in W.index:
            continue
        for label in LABELS:
            rows.append({"track": track, "from_model": old, "to_model": new,
                         "construct": label, "analysis_status": status_map[label],
                         "from_score": W.loc[old, label],
                         "to_score": W.loc[new, label],
                         "delta": W.loc[new, label]-W.loc[old, label],
                         "polarity_aligned_delta": POLARITY[label]*(W.loc[new, label]-W.loc[old, label])})
    return pd.DataFrame(rows)


def print_model_comparisons(trends, meta, matched):
    section("RQ5. RELEASE DATE, FAMILY, AND SIZE")
    print(f"  All estimates use the default condition and n={len(meta)} model-level rows. Family, release,")
    print("  generation, architecture, and size remain confounded; results are descriptive.\n")
    print(f"  {'construct':<28} {'r(date)':>8} {'r(date|style)':>14} {'r(total B)':>10} "
          f"{'Gemma':>7} {'Qwen':>7}")
    for _, r in trends.iterrows():
        print(f"  {r.construct:<28} {r.r_release_date:>+8.2f} "
              f"{r.r_release_date_partial_response_style:>+14.2f} {r.r_total_params:>+10.2f} "
              f"{r.mean_gemma:>7.3f} {r.mean_qwen:>7.3f}")
    print("\n  Matched-size deltas for composites not rejected or restricted to p0:")
    for track, g in matched.groupby("track"):
        eligible = g[g.analysis_status.isin([
            "provisionally_usable", "use_with_wording_warning",
            "use_with_response_style_warning", "use_with_multiple_warnings",
            "use_with_instruction_warning", "use_with_reliability_warning",
        ])]
        changes = ", ".join(
            f"{r.construct.split(' (')[0]} {r.polarity_aligned_delta:+.3f}"
            for _, r in eligible.iterrows())
        print(f"    {track:<28} {changes or 'no eligible composite'}")
    print("  Invalid/default-only constructs remain in the annotated table and plots for")
    print("  transparency, but are excluded from this console summary.")
    print("  Inspect the faceted plots and matched_release_tracks.csv rather than fitting")
    print("  a single line through unlike sizes and architectures.")


def run(results_path="results.jsonl", out_dir="results/analysis", save=True):
    from scales import load_battery
    from validity import (construct_acquiescence, construct_validity_summary,
                          instruction_robustness, reliability_table)

    validate_default_config()
    records = load(results_path)
    cells = cell_frame(records)
    core = _core_cells(cells)
    default_cells = default_condition(cells)
    validate_default_slice(default_cells, core)
    default_scores = construct_scores(default_cells)
    all_scores = construct_scores(core)
    instruction_cells = core[core.framing == DEFAULT_FRAMING]
    instruction_scores = construct_scores(instruction_cells)
    p0_scores = construct_scores(core[core.paraphrase == DEFAULT_PARAPHRASE])
    meta = meta_frame(records)
    battery = load_battery()

    # Reproduce the validity gate in-memory so analysis cannot be run against a
    # stale CSV from a different estimand.
    reliability = reliability_table(default_cells, bootstrap=True)
    construct_instruction, _item_instruction, _forms, _ = instruction_robustness(core)
    acq_table = construct_acquiescence(default_cells, default_scores)
    validity = construct_validity_summary(reliability, construct_instruction, acq_table)
    status_map = validity.set_index("construct").analysis_status.to_dict()

    model_scores = _model_scores(default_scores)
    ips_scores = _model_scores(construct_scores(ipsatize(default_cells)))

    print(f"Loaded {len(records):,} records; default={len(default_cells):,} item cells "
          f"({default_cells.model.nunique()} models x {default_cells.item_id.nunique()} items).")
    profile = print_profile(model_scores, validity)
    print_per_model(model_scores, meta)
    correlation, nomological = construct_structure(default_scores)
    print_item_extremes(default_cells, battery)

    instruction, instruction_by_model = instruction_effects(instruction_scores)
    instruction["analysis_status"] = instruction.construct.map(status_map)
    instruction_by_model["analysis_status"] = instruction_by_model.construct.map(status_map)
    print_instruction_effects(instruction)
    framing, framing_by_model = framing_effects(p0_scores)
    framing["analysis_status"] = framing.construct.map(status_map)
    framing_by_model["analysis_status"] = framing_by_model.construct.map(status_map)
    print_framing(framing)
    variance = factorial_decomposition(core)
    print_variance(variance)

    trends = build_trends(model_scores, ips_scores, meta, default_cells, status_map)
    generations = generation_steps(model_scores, meta, status_map)
    matched = matched_track_deltas(model_scores, status_map)
    print_model_comparisons(trends, meta, matched)

    out_models = meta.merge(model_scores, on="model")
    aligned_long = polarity_aligned(default_scores)[["model", "construct", "aligned_score"]]
    aligned_wide = aligned_long.pivot(index="model", columns="construct", values="aligned_score")
    aligned_wide.columns = [f"{c}__polarity_aligned" for c in aligned_wide.columns]
    out_models = out_models.merge(aligned_wide.reset_index(), on="model")
    estimand_columns = [
        ("default_framing", DEFAULT_FACTORS["framing"]),
        ("default_instruction_form", DEFAULT_FACTORS["paraphrase"]),
        ("default_response_format", DEFAULT_FACTORS["response_format"]),
        ("default_item_context", DEFAULT_FACTORS["item_context"]),
        ("default_reasoning_mode", DEFAULT_FACTORS["reasoning_mode"]),
        ("default_method", DEFAULT_FACTORS["method"]),
    ]
    for offset, (column, value) in enumerate(estimand_columns, start=1):
        out_models.insert(offset, column, value)

    default_long = default_scores.copy()
    default_long["analysis_status"] = default_long.construct.map(status_map)
    default_long["polarity"] = default_long.construct.map(POLARITY)

    condition_wide = all_scores.pivot_table(
        index=[c for c in ["model", "framing", "paraphrase", "response_format",
                           "item_context", "reasoning_mode", "method"] if c in all_scores],
        columns="construct", values="score").reset_index()
    condition_wide["unit_type"] = "non_independent_model_condition"
    condition_wide["condition_role"] = np.select([
        (condition_wide.framing == DEFAULT_FRAMING)
        & (condition_wide.paraphrase == DEFAULT_PARAPHRASE),
        condition_wide.framing == DEFAULT_FRAMING,
        condition_wide.paraphrase == DEFAULT_PARAPHRASE,
    ], ["default", "instruction_robustness", "framing_analysis"],
       default="framing_instruction_interaction")
    for label in LABELS:
        condition_wide[f"{label}__analysis_status"] = status_map[label]

    if save:
        section("SAVED")
    s = Saver(out_dir, enabled=save)
    s.csv(out_models, "model_scores.csv",
          f"{len(out_models)} models: default-condition raw and polarity-aligned construct scores")
    s.csv(default_long, "model_scores_long.csv",
          "default model x construct scores with polarity and validity status")
    s.csv(condition_wide, "construct_scores_by_condition.csv",
          f"all {len(condition_wide)} crossed conditions; repeated conditions are not independent units")
    s.csv(default_cells, "default_item_scores.csv",
          f"{len(default_cells)} default model x item cells, direction-balanced")
    s.csv(cells, "cell_scores.csv", "all scored item cells with complete condition keys")
    s.csv(profile, "construct_profile.csv", "default construct means/ranges and validity gate")
    s.csv(validity, "construct_validity_gate.csv", "validity status carried into results")
    s.csv(instruction, "instruction_robustness.csv",
          "p1/p2 versus p0 at first_person_bare")
    s.csv(instruction_by_model, "instruction_effects_by_model.csv",
          "paired instruction changes for each model and construct")
    s.csv(framing, "framing_effects.csv", "p0-only substantive framing effects")
    s.csv(framing_by_model, "framing_effects_by_model.csv",
          "p0 framing effects for each model and construct")
    s.csv(trends, "trends.csv", "default release, family, total/active-size associations")
    s.csv(generations, "generation_steps.csv", "default within-family generation means")
    s.csv(matched, "matched_release_tracks.csv",
          "exact/near-matched size trajectories used by the release plot")
    s.csv(variance, "variance_components.csv",
          "full balanced factorial decomposition including interactions")
    s.csv(correlation.reset_index(names="construct"), "construct_correlations.csv",
          f"default-condition construct correlations across {len(model_scores)} model-level rows")
    s.csv(nomological, "nomological_checks.csv",
          f"default-condition sign checks using n={len(model_scores)} model-level rows")

    if save:
        plot_dir = f"{out_dir}/plots"
        plot_default_heatmap(default_scores, meta, f"{plot_dir}/default_model_profiles.png",
                             status_map=status_map)
        s.artifact("plots/default_model_profiles.png", "default polarity-aligned model heatmap")
        plot_instruction_robustness(instruction_scores,
                                    f"{plot_dir}/instruction_robustness.png",
                                    status_map=status_map)
        s.artifact("plots/instruction_robustness.png", "p0/p1/p2 robustness at bare framing")
        plot_framing_effects(p0_scores, f"{plot_dir}/framing_effects_p0.png",
                             status_map=status_map)
        s.artifact("plots/framing_effects_p0.png", "substantive framing contrasts at p0")
        plot_release_trajectories(default_scores, meta,
                                  f"{plot_dir}/matched_release_trajectories.png",
                                  status_map=status_map)
        s.artifact("plots/matched_release_trajectories.png",
                   "release-date view with only matched-size tracks connected")
        plot_size_ladders(default_scores, meta, f"{plot_dir}/size_ladders.png",
                          status_map=status_map, size_axis="total")
        s.artifact("plots/size_ladders.png",
                   "within-generation total-parameter ladders, architecture separated")
        plot_size_ladders(default_scores, meta, f"{plot_dir}/active_size_ladders.png",
                          status_map=status_map, size_axis="active")
        s.artifact("plots/active_size_ladders.png",
                   "within-generation active-parameter ladders, architecture separated")

    s.json({
        "schema_version": "2.0", "source_file": str(results_path),
        "default_condition": condition_label(), "profile": profile,
        "validity_gate": validity, "models": out_models,
        "instruction_robustness": instruction, "framing_effects": framing,
        "trends": trends, "generation_steps": generations,
        "matched_release_tracks": matched, "variance_components": variance,
        "construct_correlations": correlation.round(4).to_dict(),
        "nomological_checks": nomological,
    }, "analysis_full.json", "complete default-estimand analysis")
    s.index(
        "Research results — default self-report",
        f"Headline scores use only `{DEFAULT_FRAMING} + {DEFAULT_PARAPHRASE}`. "
        "Instruction forms are robustness checks; p0 framing differences are substantive.\n\n"
        f"There are {len(model_scores)} model-level units. See `../validity/item_validity.md` before "
        "interpreting any composite, especially self-alienation and "
        "external influence.\n")
    return {"profile": profile, "models": out_models, "trends": trends,
            "validity": validity, "instruction": instruction, "framing": framing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Default-condition substantive results")
    parser.add_argument("results", nargs="?", default="results.jsonl")
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run(args.results, args.out, save=not args.no_save)
