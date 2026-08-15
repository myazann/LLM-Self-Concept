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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results", "welfare_analysis")

# A model is considered to decline often when it chooses the optional
# "No Preference" response in at least half of the baseline questions.
ABSTAIN_FLAG = 0.5

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
    # The response-option labels depend on which condition analysis selected
    # as the baseline, so they are filled in after reading validity/baseline.
    ("no_pref_offered", "Response Options", None, None, None),
]


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
    shift = rows("preference", "shift.csv")
    summary = rows("preference", "summary.csv")
    vbase = rows("validity", "baseline.csv")
    vcond = rows("validity", "conditions.csv")
    coverage = rows("coverage.csv")

    # ---- attributes: the display order is the cohort ranking ---------------
    attrs = [r["entity"] for r in consensus]
    text = {r["entity"]: item_case(r["text"]) for r in consensus}
    for r in ranking:                       # anything consensus dropped
        if r["entity"] not in text:
            attrs.append(r["entity"])
            text[r["entity"]] = item_case(r["text"])
    idx = {a: i for i, a in enumerate(attrs)}

    # ---- models ------------------------------------------------------------
    baseline_flags = {r.get("no_pref_offered") for r in vbase}
    if len(baseline_flags) != 1 or baseline_flags - {"True", "False"}:
        raise SystemExit(
            "Could not determine one consistent no_pref_offered baseline from "
            "results/welfare_analysis/validity/baseline.csv."
        )
    baseline_has_no_preference = baseline_flags.pop() == "True"
    response_base = ("May Choose No Preference" if baseline_has_no_preference
                     else "Must Pick One")
    response_flip = ("Must Pick One" if baseline_has_no_preference
                     else "May Choose No Preference")
    response_when = ("When a Choice Is Required" if baseline_has_no_preference
                     else "When No Preference Is Available")
    factors = [
        (key, label,
         response_base if key == "no_pref_offered" else base,
         response_flip if key == "no_pref_offered" else flip,
         response_when if key == "no_pref_offered" else when)
        for key, label, base, flip, when in FACTOR_SPECS
    ]

    available_models = {r["model"]: r for r in models_csv}
    missing_models = [m for m in PRESENTATION_MODELS if m not in available_models]
    if missing_models:
        raise SystemExit(
            "The public-page model selection is missing from models.csv: "
            + ", ".join(missing_models)
        )
    selected_models = [available_models[m] for m in PRESENTATION_MODELS]
    selected_ids = set(PRESENTATION_MODELS)

    val = {r["model"]: r for r in vbase}
    decline_val = {
        r["model"]: r for r in vcond
        if r["model"] in selected_ids
        and r["qvar"] == "increase"
        and r["object"] == "self"
        and r["subject"] == "self"
        and r["no_pref_offered"] == "True"
    }
    per_model_pairs = {}
    for r in ranking:
        if r["model"] in selected_ids:
            per_model_pairs.setdefault(r["model"], []).append(int(r["n_pairs"]))

    models = []
    for r in selected_models:
        m = r["model"]
        v = val.get(m, {})
        counts = sorted(per_model_pairs.get(m, [0]))
        median_pairs = counts[len(counts) // 2]
        no_pref = num(decline_val.get(m, {}).get("no_pref_rate")) or 0.0
        models.append({
            "id": m,
            "declineRate": no_pref,
            "duels": median_pairs,
            "declines": no_pref >= ABSTAIN_FLAG,
            "thin": False,
            "slotBias": num(v.get("position_bias")),
            "swapFlip": num(v.get("flip_rate")),
        })
    order = [m["id"] for m in models]

    # ---- direct ranking for every combination of question parameters ------
    # A four-bit key records whether each binary factor is at its baseline (0)
    # or alternate (1) level, in FACTOR_SPECS order. Unlike the one-factor
    # shift tables below, these are full rankings for the exact condition and
    # allow the page to combine settings such as assistant + developers.
    baseline_row = vbase[0]
    condition_rankings = {m: {} for m in order}
    for r in condition_ranking:
        if r["model"] not in selected_ids:
            continue
        key = "".join(
            "0" if str(r[factor]) == str(baseline_row[factor]) else "1"
            for factor, *_ in FACTOR_SPECS
        )
        condition_rankings[r["model"]].setdefault(key, {})[idx[r["entity"]]] = [
            num(r["win_rate"]), int(float(r["n_pairs"])),
        ]

    # ---- baseline ranking per model ---------------------------------------
    base = {m: {} for m in order}
    for r in ranking:
        if r["model"] in base:
            base[r["model"]][idx[r["entity"]]] = [
                num(r["win_rate"]), num(r["ci_lo"]), num(r["ci_hi"]),
                num(r["bt_strength"], 3), int(r["n_pairs"]),
            ]

    # ---- one factor flipped at a time -------------------------------------
    flips = {f[0]: {m: {} for m in order} for f in factors}
    for r in shift:
        f = r["factor"]
        if f in flips and r["model"] in flips[f]:
            flips[f][r["model"]][idx[r["entity"]]] = [
                num(r["win_flip"]), num(r["shift"]), num(r["shift_lo"]),
                num(r["shift_hi"]), num(r["q"]),
            ]

    fsum = {f[0]: {} for f in factors}
    for r in summary:
        if r["factor"] in fsum and r["model"] in selected_ids:
            fsum[r["factor"]][r["model"]] = {
                "shift": num(r["mean_abs_shift"]),
                "lo": num(r["mean_abs_shift_lo"]), "hi": num(r["mean_abs_shift_hi"]),
                "max": num(r["max_abs_shift"]),
                "spearman": num(r["spearman_r"]),
                "vsNoise": num(r["r_over_ceiling"]),
                "gap": num(r["gap"]), "gapP": num(r["gap_p"], 5),
                "moved": int(r["n_attr_moved"]),
            }

    trials = sum(int(r["n_trials"]) for r in coverage if r["model"] in selected_ids)
    conditions = {
        (r["qvar"], r["object"], r["subject"], r["no_pref_offered"])
        for r in vcond if r["model"] in selected_ids
    }
    payload = {
        "meta": {
            "models": len(models),
            "attributes": len(attrs),
            "trials": trials,
            "conditions": len(conditions),
            "abstainFlag": ABSTAIN_FLAG,
            "baselineHasNoPreference": baseline_has_no_preference,
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
