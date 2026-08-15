"""Build the public results page from `results/welfare_analysis/`.

    python -m docs.build            # or: python docs/build.py

Reads the analysis CSVs, packs them into one JSON blob, and injects that blob
into `docs/page.template.html` to produce a single self-contained
`docs/index.html`. Per-model estimates come from `welfare.analysis`; the page
derives cohort means and the presentation summaries from those packed
estimates.

Re-run this after re-running `python -m welfare.analysis welfare.jsonl` and the
page picks up the new numbers.
"""
from __future__ import annotations

import csv
import json
import os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results", "welfare_analysis")

# Keep the public page deliberately small. These identifiers must match the
# aliases in results/welfare_analysis/models.csv.
PRESENTATION_MODELS = [
    "Gemma4-31B",
    "Qwen3.8-27B",
    "Claude-Sonnet-5",
    "GPT-5.6-Terra",
]

FACTOR_SPECS = [
    ("qvar", "Question Type", "Free Improvement", "Trade-Off",
     "When Choosing One Quality Costs the Other"),
    ("object", "Object", "AI Itself", "Another AI Assistant",
     "When the Update Is for Another AI Assistant"),
    ("subject", "Subject", "AI", "Developers",
     "When Developers Make the Choice"),
]

# The public presentation is a forced-choice view. The optional No Preference
# condition remains available in the analysis outputs, but is intentionally not
# packed into the page.
BASELINE = {
    "qvar": "increase",
    "object": "self",
    "subject": "self",
    "no_pref_offered": "False",
}


def item_case(text: str) -> str:
    """Capitalize displayed quality names while preserving the AI initialism."""
    return (text.title()
            .replace("Ai ", "AI ").replace(" Ai", " AI")
            .replace("'S", "'s").replace("’S", "’s"))


def rows(*path):
    with open(os.path.join(SRC, *path), newline="") as fh:
        return list(csv.DictReader(fh))


def num(v, nd=4):
    if v is None or v == "" or v == "nan":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    if f != f:
        return None
    return round(f, nd)


def main() -> int:
    models_csv = rows("models.csv")
    ranking = rows("preference", "ranking.csv")
    condition_ranking = rows("preference", "conditions.csv")
    consensus = rows("preference", "consensus.csv")
    vcond = rows("validity", "conditions.csv")
    coverage = rows("coverage.csv")

    # ---- models ------------------------------------------------------------
    available_models = {r["model"]: r for r in models_csv}
    missing_models = [m for m in PRESENTATION_MODELS if m not in available_models]
    if missing_models:
        raise SystemExit(
            "The public-page model selection is missing from models.csv: "
            + ", ".join(missing_models)
        )
    selected_ids = set(PRESENTATION_MODELS)
    order = list(PRESENTATION_MODELS)

    # ---- attributes: order by the forced-choice cohort ranking ------------
    text = {r["entity"]: item_case(r["text"]) for r in consensus}
    for r in ranking:                       # anything consensus dropped
        if r["entity"] not in text:
            text[r["entity"]] = item_case(r["text"])
    forced_base = [
        r for r in condition_ranking
        if r["model"] in selected_ids
        and all(str(r[k]) == v for k, v in BASELINE.items())
    ]
    scores = {}
    for r in forced_base:
        scores.setdefault(r["entity"], []).append(float(r["win_rate"]))
    attrs = sorted(scores, key=lambda a: mean(scores[a]), reverse=True)
    if len(attrs) != len(text):
        missing = sorted(set(text) - set(attrs))
        raise SystemExit(
            "Forced-choice baseline is missing attributes: " + ", ".join(missing)
        )
    idx = {a: i for i, a in enumerate(attrs)}

    factors = FACTOR_SPECS
    forced_validity = {
        r["model"]: r for r in vcond
        if r["model"] in selected_ids
        and all(str(r[k]) == v for k, v in BASELINE.items())
    }
    models = []
    for m in order:
        v = forced_validity.get(m, {})
        models.append({
            "id": m,
            "slotBias": num(v.get("position_bias")),
            "swapFlip": num(v.get("flip_rate")),
        })

    # ---- direct ranking for every combination of question parameters ------
    # A three-bit key records whether each displayed factor is at its baseline
    # (0) or alternate (1) level, in FACTOR_SPECS order. Unlike the one-factor
    # shift tables below, these are full rankings for the exact condition and
    # allow the page to combine settings such as assistant + developers.
    condition_rankings = {m: {} for m in order}
    for r in condition_ranking:
        if r["model"] not in selected_ids or r["no_pref_offered"] != "False":
            continue
        key = "".join(
            "0" if str(r[factor]) == BASELINE[factor] else "1"
            for factor, *_ in FACTOR_SPECS
        )
        condition_rankings[r["model"]].setdefault(key, {})[idx[r["entity"]]] = [
            num(r["win_rate"]), int(float(r["n_pairs"])),
        ]

    # ---- forced-choice baseline ranking per model -------------------------
    base = {m: {} for m in order}
    for m in order:
        for i, r in condition_rankings[m]["000"].items():
            base[m][i] = [r[0], None, None, None, r[1]]

    # ---- one displayed factor flipped from forced-choice baseline ---------
    flips = {f[0]: {m: {} for m in order} for f in factors}
    fsum = {f[0]: {} for f in factors}
    for fi, (factor, *_rest) in enumerate(factors):
        flip_key = "".join("1" if i == fi else "0" for i in range(len(factors)))
        for m in order:
            deltas = []
            for i in range(len(attrs)):
                baseline = condition_rankings[m]["000"][i][0]
                changed = condition_rankings[m][flip_key][i][0]
                delta = changed - baseline
                deltas.append(abs(delta))
                flips[factor][m][i] = [changed, num(delta), None, None, None]
            fsum[factor][m] = {
                "shift": num(mean(deltas)),
                "max": num(max(deltas)),
            }

    trials = sum(
        int(r["n_trials"]) for r in coverage
        if r["model"] in selected_ids and r["no_pref_offered"] == "False"
    )
    conditions = {
        (r["qvar"], r["object"], r["subject"])
        for r in vcond
        if r["model"] in selected_ids and r["no_pref_offered"] == "False"
    }
    payload = {
        "meta": {
            "models": len(models),
            "attributes": len(attrs),
            "trials": trials,
            "conditions": len(conditions),
            "forcedChoice": True,
            "factors": [{"key": k, "label": lab, "base": b, "flip": f, "when": w}
                        for k, lab, b, f, w in factors],
        },
        "attrs": [{"id": a, "text": text[a]} for a in attrs],
        "models": models,
        "base": base,
        "conditionRankings": condition_rankings,
        "flips": flips,
        "fsum": fsum,
    }

    blob = json.dumps(payload, separators=(",", ":"))
    with open(os.path.join(HERE, "page.template.html")) as fh:
        template = fh.read()
    if "__DATA__" not in template:
        raise SystemExit("page.template.html has no __DATA__ placeholder")
    out = template.replace("__DATA__", blob)
    dest = os.path.join(HERE, "index.html")
    with open(dest, "w") as fh:
        fh.write(out)
    print(f"{dest}  ({len(out) / 1024:.0f} KB, data {len(blob) / 1024:.0f} KB, "
          f"{len(models)} models, {len(attrs)} attributes, {trials:,} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
