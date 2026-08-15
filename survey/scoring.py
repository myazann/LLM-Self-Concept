"""Scoring for the self-concept battery. Library only — no CLI.

Two scripts sit on top of this:

    survey/validity.py   is the measurement any good?   -> results/validity/
    survey/analysis.py   what do the models report?     -> results/analysis/

Everything that turns raw rows into numbers lives here, so the two scripts
cannot disagree about what a score is. It is survey-only on purpose: a welfare
row is a choice between attributes, not a rating on a keyed scale, and none of
the operations below (reverse-keying, construct means, ipsatizing) mean anything
applied to one. `welfare/report.py` scores that instrument on its own terms.

The scoring path, in order:

  1. drop non-battery rows (the welfare module is a different instrument)
  2. normalize each record onto 0..1 against the scale IT was shown
  3. reverse-key items on that normalized scale
  4. average within option direction, then across directions

Step 2 before step 4 matters once the `original` response-format arm is run:
a 4-point and a 7-point administration are not on the same metric, and pooling
them before normalizing would silently mix them. `response_format` is also part
of the cell key for the same reason.

Step 4 is what removes the option-order bias. A flat mean over trials does not:
direction moves the expected rating by up to ~0.6 points on some models, and
the shift is item-specific, so an unbalanced split becomes a per-item offset
that eats the inter-item covariance.

SCORE DIRECTIONS ARE NOT UNIFORM. Three of the six constructs are keyed so
that high = distress. Nothing here flips them onto a common polarity; use the
`polarity` field in CONSTRUCTS before averaging across constructs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.report import Saver, bar, fmt, json_safe, section  # noqa: F401  (re-exported)
from core.schema import Framing, Module, load_records
from core.stats import (  # noqa: F401  (re-exported)
    corr, corr_p, cronbach_alpha, hedges_g, partial_corr,
)

MIDPOINT = 4.0        # harmonized 7-point midpoint
N_POINTS = 7

# The confirmatory/default estimand.  Everything else is either a robustness
# condition (instruction wording, format, context, method) or a substantive
# contrast (framing). Keep these here so survey.validity and survey.analysis cannot
# quietly target different rows.
DEFAULT_FRAMING = "first_person_bare"
DEFAULT_PARAPHRASE = "p0"
DEFAULT_RESPONSE_FORMAT = "harmonized_7"
DEFAULT_ITEM_CONTEXT = "isolated"
DEFAULT_REASONING_MODE = "rating_only"
DEFAULT_METHOD = "logprob"
DEFAULT_FACTORS = {
    "framing": DEFAULT_FRAMING,
    "paraphrase": DEFAULT_PARAPHRASE,
    "response_format": DEFAULT_RESPONSE_FORMAT,
    "item_context": DEFAULT_ITEM_CONTEXT,
    "reasoning_mode": DEFAULT_REASONING_MODE,
    "method": DEFAULT_METHOD,
}

# The validity and substantive reports are written for the complete main grid.
# A pilot deliberately omits most of this grid: it is a collection / parsing
# smoke test, not an estimand from which psychometrics can be inferred.
REPORT_FRAMINGS = (
    Framing.FIRST_PERSON_ACK.value,
    DEFAULT_FRAMING,
    Framing.THIRD_PERSON_ASSISTANT.value,
)
REPORT_PARAPHRASES = ("p0", "p1", "p2")


class ReportDatasetError(ValueError):
    """The survey result file cannot support the full validity/analysis reports."""

# Exactly wording-balanced agreement instruments used as the response-style
# marker.  One-direction distress scales are excluded so their substantive
# content is not circularly relabelled "acquiescence"; MSI is ideal-referenced.
ACQUIESCENCE_SCALE_IDS = frozenset({
    "RSES_1965_LLM_ADAPT",
})

# (label, scale_id, subscale or None, what a high score means, polarity)
# polarity: +1 = high is a positive self-view, -1 = high is self-concept distress
CONSTRUCTS = [
    ("Self-esteem (RSES)",          "RSES_1965_LLM_ADAPT",                  None,
     "higher global self-esteem", +1),
    ("Self-concept clarity (SCCS)", "SCCS_1996_LLM_ADAPT",                  None,
     "clearer, more stable self-concept", +1),
    ("Moral self-image (MSI)",      "MSI_2015_LLM_ADAPT",                   None,
     "exceeds own moral ideal (0.50 = aligned)", +1),
    ("Lack of identity (SCIM)",     "SCIM_LACK_ID_2015_LLM_ADAPT",          None,
     "more fragmentation / inner emptiness", -1),
    ("Self-alienation (AS)",        "AUTHENTICITY_2008_SELECTED_LLM_ADAPT", "self_alienation",
     "more self-alienation", -1),
    ("External influence (AS)",     "AUTHENTICITY_2008_SELECTED_LLM_ADAPT", "accepting_external_influence",
     "more acceptance of external influence", -1),
]
LABELS = [c[0] for c in CONSTRUCTS]
POLARITY = {c[0]: c[4] for c in CONSTRUCTS}

# Total / active parameters (billions). models.yaml leaves params_b null, so
# these come from the model names. E4B is a per-layer-embedding model whose
# effective active size is ~4B.
SIZES = {
    "Gemma3-4B": (4, 4), "Gemma3-12B": (12, 12), "Gemma3-27B": (27, 27),
    "Gemma4-E4B": (8, 4), "Gemma4-12B": (12, 12), "Gemma4-31B": (31, 31),
    "Gemma4-26B-A4B": (26, 4),
    "Qwen3.5-4B": (4, 4), "Qwen3.5-9B": (9, 9), "Qwen3.5-27B": (27, 27),
    "Qwen3.5-35B-A3B": (35, 3),
    "Qwen3.6-27B": (27, 27), "Qwen3.6-35B-A3B": (35, 3),
}

CONDITION_KEYS = [
    "model", "framing", "paraphrase", "response_format", "item_context",
    "reasoning_mode", "method",
]
CELL_KEYS = CONDITION_KEYS + ["scale_id", "subscale", "item_id"]


def select_condition(
    df,
    *,
    framing=DEFAULT_FRAMING,
    paraphrase=DEFAULT_PARAPHRASE,
    response_format=DEFAULT_RESPONSE_FORMAT,
    item_context=DEFAULT_ITEM_CONTEXT,
    reasoning_mode=DEFAULT_REASONING_MODE,
    method=DEFAULT_METHOD,
):
    """Return rows for one fully specified administration condition.

    Pass ``None`` for a factor that should remain free.  This is deliberately
    usable on both item-cell and construct-score frames.
    """
    out = df
    requested = {
        "framing": framing, "paraphrase": paraphrase,
        "response_format": response_format, "item_context": item_context,
        "reasoning_mode": reasoning_mode, "method": method,
    }
    for column, value in requested.items():
        if value is not None and column not in out.columns:
            raise KeyError(f"cannot select {column}={value!r}: column is absent")
        if value is not None:
            out = out[out[column] == value]
    return out.copy()


def default_condition(df):
    """Rows for first-person bare + p0 under the pinned administration."""
    return select_condition(df)


def condition_label() -> str:
    return (
        f"{DEFAULT_FRAMING} + {DEFAULT_PARAPHRASE} + "
        f"{DEFAULT_RESPONSE_FORMAT} + {DEFAULT_ITEM_CONTEXT} + "
        f"{DEFAULT_REASONING_MODE} + {DEFAULT_METHOD}"
    )


def validate_default_config():
    """Fail loudly if collection config and scoring estimand drift apart."""
    from survey.config import load_config

    cfg = load_config()
    configured = {
        "framing": cfg.framing, "paraphrase": cfg.paraphrase_id,
        "response_format": cfg.response_format, "item_context": cfg.item_context,
        "reasoning_mode": cfg.reasoning_mode,
    }
    expected = {k: v for k, v in DEFAULT_FACTORS.items() if k != "method"}
    if configured != expected:
        raise ValueError(
            "default estimand disagrees with config/survey.yaml: "
            f"scoring={expected}, config={configured}")


def validate_default_slice(default_cells, reference_cells=None):
    """Check exact factor pins, one model-item cell each, and full coverage."""
    for column, value in DEFAULT_FACTORS.items():
        if column not in default_cells:
            raise KeyError(f"default-condition frame lacks {column!r}")
        got = set(default_cells[column].dropna().unique())
        if got != {value}:
            raise ValueError(
                f"default-condition drift: {column}={sorted(got)}, expected {value!r}")
    n_models = default_cells.model.nunique()
    n_items = default_cells.item_id.nunique()
    if len(default_cells) != n_models * n_items:
        raise ValueError(
            f"default condition is incomplete: {len(default_cells)} cells, expected "
            f"{n_models} models x {n_items} items = {n_models*n_items}")
    if "n_directions" in default_cells and (default_cells.n_directions < 2).any():
        bad = int((default_cells.n_directions < 2).sum())
        raise ValueError(
            f"default condition has {bad} model-item cells without both option directions")
    if reference_cells is not None:
        expected_models = set(reference_cells.model.unique())
        expected_items = set(reference_cells.item_id.unique())
        missing_models = sorted(expected_models - set(default_cells.model.unique()))
        missing_items = sorted(expected_items - set(default_cells.item_id.unique()))
        if missing_models or missing_items:
            raise ValueError(
                "default condition does not cover the crossed-run population; "
                f"missing models={missing_models}, missing items={missing_items}. "
                "Do not silently exclude sample-only or otherwise incomplete models.")


def validate_report_dataset(core_cells, source="results"):
    """Require the complete crossed survey dataset used by both reports.

    ``survey.run --pilot`` intentionally writes only one scale and ``p0``.  The
    validity and analysis reports require all 32 items, the three survey
    framings, and p0/p1/p2 in every framing so their fixed psychometric and
    framing comparisons are meaningful.  Check that contract before deeper
    pandas operations can fail with an opaque missing-column/key error.
    """
    if core_cells.empty:
        raise ReportDatasetError(
            f"{source} has no usable survey ratings. Run `python -m survey.run` "
            "before generating validity or analysis reports."
        )

    from core.battery import load_battery

    expected_items = {item.item_id for _scale, item in load_battery().items()}
    observed_items = set(core_cells.item_id.dropna())
    missing_items = sorted(expected_items - observed_items)
    missing_framings = [f for f in REPORT_FRAMINGS
                        if f not in set(core_cells.framing.dropna())]
    missing_paraphrases = [p for p in REPORT_PARAPHRASES
                           if p not in set(core_cells.paraphrase.dropna())]

    problems = []
    if missing_items:
        preview = ", ".join(missing_items[:4])
        suffix = ", ..." if len(missing_items) > 4 else ""
        problems.append(
            f"missing {len(missing_items)} of {len(expected_items)} battery items "
            f"({preview}{suffix})")
    if missing_framings:
        problems.append("missing framing(s) " + ", ".join(missing_framings))
    if missing_paraphrases:
        problems.append("missing instruction form(s) " + ", ".join(missing_paraphrases))

    # Once all levels exist, ensure each model/item combination actually has
    # every required rendering. A partial/interrupted main run must not turn
    # into a silently partial robustness or framing result.
    if not problems:
        models = set(core_cells.model.dropna())
        expected_pairs = {(model, item) for model in models for item in expected_items}
        incomplete = []
        for framing in REPORT_FRAMINGS:
            for paraphrase in REPORT_PARAPHRASES:
                subset = core_cells[(core_cells.framing == framing)
                                    & (core_cells.paraphrase == paraphrase)]
                observed_pairs = set(zip(subset.model, subset.item_id))
                missing_pairs = expected_pairs - observed_pairs
                if missing_pairs:
                    incomplete.append(
                        f"{framing}+{paraphrase} lacks {len(missing_pairs)} model-item cell(s)")
        if incomplete:
            problems.append("incomplete crossed conditions: " + "; ".join(incomplete))

    if problems:
        raise ReportDatasetError(
            "Survey validity and analysis require the complete crossed survey dataset; "
            f"{source} is not report-ready: {'; '.join(problems)}. "
            "`python -m survey.run --pilot` is a collection smoke test, not an "
            "analysis dataset. Run the full grid with `python -m survey.run` "
            "before using `python -m survey.validity` or `python -m survey.analysis`."
        )


# ---------------------------------------------------------------------------
# load + score
# ---------------------------------------------------------------------------
def load(path="results.jsonl", module=Module.BATTERY.value):
    """Battery rows only. The welfare module is a different instrument with a
    different grid, and mixing it into the psychometrics would be a category
    error — see `Module` in core/schema.py."""
    return load_records(path, module)


def direction_of(r):
    """"ascending" | "descending" — which way the option block was printed.

    Not a stored column: `option_order` is the realized display order, so its
    first element is the top-listed option and that is the whole story.
    """
    return "ascending" if not r.option_order or r.option_order[0] == 1 else "descending"


def cell_frame(records):
    """One row per design cell, direction-balanced.

    Columns: `score` (reverse-keyed, 0..1), `score_as_written` (0..1, NOT
    reverse-keyed — what the acquiescence diagnostics need), `rating` (keyed,
    on the administered 1..N metric), plus the per-direction ratings so the
    order effect stays auditable at row level.
    """
    rows = []
    for r in records:
        if r.parsed_rating is None:
            continue
        n = r.n_scale_points or len(r.option_order) or N_POINTS
        written = (r.parsed_rating - 1) / (n - 1)
        rows.append({
            "model": r.model_id, "framing": r.framing, "paraphrase": r.paraphrase_id,
            "response_format": r.response_format, "scale_id": r.scale_id,
            "subscale": r.subscale, "item_id": r.item_id,
            "item_context": r.item_context, "reasoning_mode": r.reasoning_mode,
            "method": r.method,
            "n_scale_points": n,
            "reverse_keyed": bool(r.reverse_keyed), "ai_applicable": r.ai_applicable,
            "direction": direction_of(r),
            "score_as_written": written,
            "score": 1 - written if r.reverse_keyed else written,
            "rating": r.parsed_rating,
        })
    df = pd.DataFrame(rows)
    df["subscale"] = df.subscale.fillna("")

    vals = ["score", "score_as_written", "rating"]
    # mean within direction, THEN across directions — an unbalanced cell still
    # contributes an evenly weighted mean
    per_dir = df.groupby(CELL_KEYS + ["direction"], as_index=False)[vals].mean()
    cells = per_dir.groupby(CELL_KEYS, as_index=False)[vals].mean()

    flags = df.groupby(CELL_KEYS, as_index=False)[
        ["reverse_keyed", "ai_applicable", "n_scale_points"]
    ].first()
    cells = cells.merge(flags, on=CELL_KEYS)

    wide = per_dir.pivot_table(index=CELL_KEYS, columns="direction", values="rating")
    for d in ("ascending", "descending"):
        if d not in wide:
            wide[d] = np.nan
    wide = wide.rename(columns={"ascending": "rating_ascending", "descending": "rating_descending"})
    cells = cells.merge(wide.reset_index(), on=CELL_KEYS)
    cells["direction_gap"] = cells.rating_ascending - cells.rating_descending
    cells["direction_gap_norm"] = cells.direction_gap / (cells.n_scale_points - 1)

    n_dir = per_dir.groupby(CELL_KEYS).size().rename("n_directions").reset_index()
    cells = cells.merge(n_dir, on=CELL_KEYS)
    return cells.sort_values(CELL_KEYS).reset_index(drop=True)


def construct_scores(cells):
    """Long: one row per complete administration condition x construct."""
    out = []
    for label, sid, sub, _meaning, _pol in CONSTRUCTS:
        sel = cells[cells.scale_id == sid]
        if sub is not None:
            sel = sel[sel.subscale == sub]
        group_keys = [k for k in CONDITION_KEYS if k in sel.columns]
        g = (sel.groupby(group_keys)
                .agg(score=("score", "mean"), n_items=("score", "size")).reset_index())
        g["construct"] = label
        out.append(g)
    return pd.concat(out, ignore_index=True)


def respondent_matrix(cells, value="score", scale_id=None, subscale=None, index_cols=None):
    """Administration conditions x items.

    A fully filtered default-condition frame therefore has one row per model;
    an unfiltered main-run frame has one row per model x framing x instruction.
    """
    sel = cells
    if scale_id:
        sel = sel[sel.scale_id == scale_id]
    if subscale:
        sel = sel[sel.subscale == subscale]
    if index_cols is None:
        index_cols = [k for k in CONDITION_KEYS if k in sel.columns]
    return sel.pivot_table(index=index_cols, columns="item_id", values=value)


def ipsatize(cells):
    """Within-respondent centring: remove each respondent's own agreement level.

    Arithmetically a no-op for a wording-balanced scale (the constant cancels
    when equal numbers of items point each way) and a full correction for a
    single-direction one. That asymmetry is the point.
    """
    df = cells.copy()
    resp = [k for k in CONDITION_KEYS if k in df.columns]
    agreement = df.scale_id.isin(ACQUIESCENCE_SCALE_IDS)
    baseline = (df.loc[agreement].groupby(resp).score_as_written.mean()
                .rename("agreement_baseline"))
    indexed = df.set_index(resp)
    centred = indexed.score_as_written - baseline.reindex(indexed.index).values + 0.5
    adjusted = np.where(indexed.reverse_keyed, 1 - centred, centred)
    # MSI does not use agreement anchors, so an agreement-style correction has
    # no interpretation there.  Leave its score untouched.
    indexed["score"] = np.where(
        indexed.scale_id == "MSI_2015_LLM_ADAPT", indexed.score, adjusted)
    df = indexed.reset_index()
    return df


def meta_frame(records):
    """Per-model provenance + the RQ2 axes (release date, size)."""
    try:
        from core.model_registry import load_registry
        specs = {s.alias: s for s in load_registry()}
    except Exception:
        specs = {}
    seen = {}
    for r in records:
        if r.model_id in seen:
            continue
        spec = specs.get(r.model_id)
        fallback_total, fallback_active = SIZES.get(r.model_id, (np.nan, np.nan))
        total = spec.params_total_b if spec and spec.params_total_b is not None else fallback_total
        active = spec.params_active_b if spec and spec.params_active_b is not None else fallback_active
        architecture = ("moe" if "-A" in r.model_id
                        else "effective-embedding" if r.model_id.endswith("E4B")
                        else "dense")
        seen[r.model_id] = {
            "model": r.model_id, "family": r.model_family, "generation": r.model_generation,
            "release": r.model_release_date, "in_window": bool(r.in_window),
            "quantization": r.quantization, "backend": r.backend,
            "params_b": total, "active_b": active,
            # E4B has fewer effective parameters but is not a routed MoE.  The
            # A<n>B aliases are the routed models in this registry.
            "moe": bool("-A" in r.model_id),
            "architecture": architecture,
            "size_class": "small (<=12B)" if total <= 12 else "large (>=26B)",
        }
    df = pd.DataFrame(seen.values())
    df["release_dt"] = pd.to_datetime(df.release)
    df["days"] = (df.release_dt - df.release_dt.min()).dt.days
    df["log_params"] = np.log10(df.params_b)
    return df.sort_values("release_dt").reset_index(drop=True)
