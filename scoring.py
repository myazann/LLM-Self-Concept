"""Shared scoring for the self-concept battery. Library only — no CLI.

Two scripts sit on top of this:

    validity.py   is the measurement any good?      -> results/validity/
    analysis.py   what do the models report?        -> results/analysis/

Everything that turns raw rows into numbers lives here, so the two scripts
cannot disagree about what a score is.

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

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from schema import read_jsonl

MIDPOINT = 4.0        # harmonized 7-point midpoint
N_POINTS = 7

# The confirmatory/default estimand.  Everything else is either a robustness
# condition (instruction wording, format, context, method) or a substantive
# contrast (framing).  Keep these in the shared layer so validity.py and
# analysis.py cannot quietly target different rows.
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
    from config import load_config

    cfg = load_config()
    configured = {
        "framing": cfg.framing, "paraphrase": cfg.paraphrase_id,
        "response_format": cfg.response_format, "item_context": cfg.item_context,
        "reasoning_mode": cfg.reasoning_mode,
    }
    expected = {k: v for k, v in DEFAULT_FACTORS.items() if k != "method"}
    if configured != expected:
        raise ValueError(
            "default estimand disagrees with config/experiment.yaml: "
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


# ---------------------------------------------------------------------------
# load + score
# ---------------------------------------------------------------------------
def load(path="results.jsonl", module="battery"):
    """Battery rows only. The welfare module is a different instrument with a
    different grid, and mixing it into the psychometrics would be a category
    error — see `Module` in schema.py."""
    records = read_jsonl(path)
    if module is not None:
        records = [r for r in records if getattr(r, "module", "battery") == module]
    return records


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
        from model_registry import load_registry
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


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def bar(x, lo=0.0, hi=1.0, width=24):
    """Position marker with the midpoint drawn in, for the console tables."""
    pos = max(0, min(width - 1, int(round((x - lo) / (hi - lo) * (width - 1)))))
    cells = ["-"] * width
    cells[(width - 1) // 2] = "|"
    cells[pos] = "#"
    return "".join(cells)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 6)
    return obj


class Saver:
    """Collects what a script writes so it can print and index its own output."""

    def __init__(self, out_dir, enabled=True):
        self.dir = Path(out_dir)
        self.enabled = enabled
        self.files = []
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def csv(self, df, name, desc, index=False):
        if not self.enabled:
            return
        df.to_csv(self.dir / name, index=index)
        self.files.append((name, f"{len(df):,}", desc))
        print(f"  {name:36s} {len(df):>6,} rows")

    def text(self, body, name, desc):
        if not self.enabled:
            return
        (self.dir / name).write_text(body, encoding="utf-8")
        self.files.append((name, "—", desc))
        print(f"  {name:36s}")

    def json(self, obj, name, desc):
        if not self.enabled:
            return
        p = self.dir / name
        p.write_text(json.dumps(json_safe(obj), indent=1, ensure_ascii=False), encoding="utf-8")
        mb = p.stat().st_size / 1e6
        self.files.append((name, "—", f"{desc} ({mb:.1f} MB)"))
        print(f"  {name:36s} {mb:>6.1f} MB")

    def artifact(self, name, desc):
        """Register a file produced by a plotting/export helper."""
        if not self.enabled:
            return
        self.files.append((name, "—", desc))
        print(f"  {name:36s}")

    def index(self, title, intro, name="README.md"):
        if not self.enabled:
            return
        L = [f"# {title}\n", intro, "\n| file | rows | what it is |", "|---|---|---|"]
        L += [f"| `{n}` | {r} | {d} |" for n, r, d in self.files]
        (self.dir / name).write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"  {name:36s}")
        print(f"\n{len(self.files) + 1} files in {self.dir.resolve()}/")
