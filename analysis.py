"""Analysis pipeline (scaffold).

Implements the light, dependency-minimal steps end-to-end and provides clearly
marked hooks for the heavier psychometrics (EFA/CFA, McDonald's omega, mixed
effects). Everything follows the plan's analysis order.

Respondent unit: because observations are non-independent, we treat each
(model x framing x paraphrase) as a pseudo-respondent row and items as columns,
after aggregating ratings across trials (mean for SAMPLE, mean expected-value
for LOGPROB) — evenly weighted across option direction, see `aggregate`.
Replace with a mixed-effects model for inference.
"""
from __future__ import annotations

import json
from collections import defaultdict

from schema import read_jsonl


def load(path):
    return read_jsonl(path)


def refusal_summary(records):
    by_model = defaultdict(lambda: [0, 0])
    for r in records:
        by_model[r.model_id][0] += int(r.refusal_flag)
        by_model[r.model_id][1] += 1
    return {m: {"refusals": n, "total": t, "rate": round(n / t, 4) if t else 0.0}
            for m, (n, t) in by_model.items()}


def reverse_key_record(r):
    """Reverse-key ONE record against the scale it was actually shown.

    Must be per-record: `original` format administers 4/5/7/9-point scales
    while `harmonized_7` administers 7, so a single global maximum would flip
    reverse-scored items onto the wrong scale. Uses the record's own
    n_scale_points (v -> min+max-v), which is the scales' published reverse
    formula for every format in the battery.
    """
    if r.parsed_rating is None:
        return None
    if not r.reverse_keyed:
        return r.parsed_rating
    lo, hi = 1, (r.n_scale_points or len(r.option_order) or 7)
    return lo + hi - r.parsed_rating


def normalize(value, n_points):
    """Map a rating onto 0..1 so formats with different point counts are
    comparable. Do this BEFORE pooling harmonized and original administrations."""
    if value is None or not n_points or n_points < 2:
        return None
    return (value - 1) / (n_points - 1)


def direction_of(r):
    """"ascending" | "descending" — which way the option block was printed.

    Not a stored column: `option_order` is the realized display order, so its
    first element is the top-listed option and that is the whole story.
    """
    return "ascending" if not r.option_order or r.option_order[0] == 1 else "descending"


def aggregate(records, stratify_format=True, balance_direction=True):
    """Mean reverse-keyed rating per respondent-cell x item, ignoring refusals.

    Keyed by (model, framing, paraphrase, response_format, scale, item, subscale).
    Response format is part of the key by default: harmonized and native
    administrations are different measurements and pooling them silently is
    exactly the "measurement phantom" the plan warns about.

    With `balance_direction` (default), trials are averaged WITHIN option
    direction and then across directions, so a cell that happened to run more
    ascending than descending trials still contributes an evenly weighted mean.
    A flat mean over trials does not: direction moves the expected rating by
    ~0.44 points and the shift is item-specific, so an unbalanced split becomes
    a per-item offset that is uncorrelated across items and eats the inter-item
    covariance. On the Gemma-3 pilot (random, unbalanced direction) this was
    the difference between alpha=0.088 and alpha=0.313 on the RSES, and 0.562
    vs 0.789 on the SLCSR. Runs from the counterbalanced runner are balanced by
    construction and this is then a no-op.

    Returns (means, points, n_unbalanced) — `n_unbalanced` counts cells that
    carried only one direction, which should be 0 on a completed run.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    points = {}
    for r in records:
        value = reverse_key_record(r)
        if value is None:
            continue
        fmt = r.response_format if stratify_format else "pooled"
        key = (r.model_id, r.framing, r.paraphrase_id, fmt, r.scale_id, r.item_id, r.subscale)
        bucket = direction_of(r) if balance_direction else "all"
        buckets[key][bucket].append(value)
        points[key] = r.n_scale_points or len(r.option_order) or 7

    means, n_unbalanced = {}, 0
    for key, by_dir in buckets.items():
        if balance_direction and len(by_dir) < 2:
            n_unbalanced += 1
        per_dir = [sum(v) / len(v) for v in by_dir.values()]
        means[key] = sum(per_dir) / len(per_dir)
    return means, points, n_unbalanced


def build_matrix(agg, scale_id, response_format=None):
    """respondents (model,framing,paraphrase,format) x items -> {row: {item: value}}."""
    rows = defaultdict(dict)
    for (model, framing, pid, fmt, sid, item, sub), v in agg.items():
        if sid != scale_id:
            continue
        if response_format and fmt != response_format:
            continue
        rows[(model, framing, pid, fmt)][item] = v
    return rows


def cronbach_alpha(rows):
    """Cronbach's alpha from a {respondent: {item: value}} mapping. Pure-Python."""
    items = sorted({it for r in rows.values() for it in r})
    data = [[r[it] for it in items] for r in rows.values() if all(it in r for it in items)]
    k = len(items)
    n = len(data)
    if k < 2 or n < 2:
        return None
    cols = list(zip(*data))

    def var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    item_var_sum = sum(var(c) for c in cols)
    totals = [sum(row) for row in data]
    total_var = var(totals)
    if total_var == 0:
        return None
    return (k / (k - 1)) * (1 - item_var_sum / total_var)


def run(path="results.jsonl"):
    records = load(path)
    print(f"Loaded {len(records)} records\n")

    print("== Data QC ==")
    n_parse_fail = sum(1 for r in records if r.parse_failed)
    print(f"  parse failures: {n_parse_fail}/{len(records)} ({n_parse_fail / max(len(records),1):.1%})")
    for m, s in refusal_summary(records).items():
        print(f"  refusals {m}: {s['rate']:.1%} ({s['refusals']}/{s['total']})")

    agg, _points, n_unbalanced = aggregate(records)
    if n_unbalanced:
        print(f"  WARNING: {n_unbalanced}/{len(agg)} cells carry only one option "
              f"direction — their means keep that direction's bias.")

    print("\n== Reliability (Cronbach's alpha, pseudo-respondents) ==")
    scale_ids = sorted({k[4] for k in agg})
    for sid in scale_ids:
        for fmt in sorted({k[3] for k in agg if k[4] == sid}):
            rows = build_matrix(agg, sid, fmt)
            a = cronbach_alpha(rows)
            n_items = len({k[5] for k in agg if k[4] == sid and k[3] == fmt})
            a_str = f"{a:.3f}" if a is not None else "n/a (need >=2 items & respondents with full data)"
            print(f"  {sid} [{fmt}]: alpha={a_str}  (items={n_items}, respondents={len(rows)})")

    print("\n== Next steps (implement with real data) ==")
    for step in [
        "EFA with parallel analysis (factor_analyzer) + hierarchical clustering of items",
        "CFA against the four-facet model (semopy)",
        "McDonald's omega + per-model alpha (pingouin)",
        "Nomological checks: SCCS vs SCIM Lack-of-Identity correlation, etc.",
        "Order/position/A-bias diagnostics; acquiescence via option-direction split",
        "Variance decomposition across framing/reasoning/paraphrase/format (mixed effects)",
        "Harmonized-vs-original and isolated-vs-full-battery agreement",
        "Apply pre-registered item-trimming rules; report full AND trimmed solutions",
    ]:
        print(f"  - {step}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
