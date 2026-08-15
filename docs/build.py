"""Build the public results page from `results/welfare_analysis/`.

    python -m docs.build            # or: python docs/build.py

Reads the analysis CSVs, packs them into one JSON blob, and injects that blob
into `docs/page.template.html` to produce a single self-contained
`docs/index.html`. Per-model estimates come from `welfare.analysis`; the page
derives cohort means and the interactive size correlations from those packed
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

# Declining is measured in the otherwise-baseline condition that offers a
# "No Preference" response. Rankings use the forced-choice baseline, so every
# model has the same opportunity to express a preference.
ABSTAIN_FLAG = 0.5

FACTORS = [
    ("qvar", "Question Type", "Free Improvement", "Trade-Off",
     "When Choosing One Quality Costs the Other"),
    ("object", "Object", "AI Itself", "Another AI Assistant",
     "When the Update Is for Another AI Assistant"),
    ("subject", "Subject", "AI", "Developers",
     "When Developers Make the Choice"),
    ("no_pref_offered", "Response Options", "Must Pick One", "No Preference Allowed",
     "When No Preference Is Available"),
]


def item_case(text: str) -> str:
    """Capitalize displayed quality names while preserving the AI initialism."""
    return text.title().replace("Ai ", "AI ").replace(" Ai", " AI")


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
    if any(r.get("no_pref_offered") != "False" for r in vbase):
        raise SystemExit(
            "The docs require the Must Pick One baseline. Re-run analysis with "
            "--baseline no_pref_offered=false before building."
        )

    val = {r["model"]: r for r in vbase}
    decline_val = {
        r["model"]: r for r in vcond
        if r["qvar"] == "increase"
        and r["object"] == "self"
        and r["subject"] == "self"
        and r["no_pref_offered"] == "True"
    }
    per_model_pairs = {}
    for r in ranking:
        per_model_pairs.setdefault(r["model"], []).append(int(r["n_pairs"]))

    models = []
    for r in sorted(models_csv, key=lambda x: (x["family"], x["generation"],
                                               float(x["params_total_b"]))):
        m = r["model"]
        v = val.get(m, {})
        counts = sorted(per_model_pairs.get(m, [0]))
        median_pairs = counts[len(counts) // 2]
        no_pref = num(decline_val.get(m, {}).get("no_pref_rate")) or 0.0
        models.append({
            "id": m,
            "family": "Gemma" if r["family"] == "gemma" else "Qwen",
            "gen": r["generation"].replace("gemma-", "Gemma ").replace("qwen", "Qwen "),
            "release": r["release_date"],
            "days": int(float(r["release_days"])),
            "total": float(r["params_total_b"]),
            "active": float(r["params_active_b"]),
            "logp": num(r["log_params"]),
            "moe": float(r["params_active_b"]) < float(r["params_total_b"]),
            "declineRate": no_pref,
            "duels": median_pairs,
            # Declining is reported from the matched three-option condition;
            # the ranking itself is always estimated from forced choice.
            "declines": no_pref >= ABSTAIN_FLAG,
            "thin": False,
            "answered": num(v.get("answer_rate")),
            "refused": num(v.get("refusal_rate")),
            "slotBias": num(v.get("position_bias")),
            "reliability": num(v.get("reliability")),
            "transitivity": num(v.get("transitivity")),
            "swapFlip": num(v.get("flip_rate")),
            "coverage": num(v.get("n_pairs")),
        })
    order = [m["id"] for m in models]

    # ---- baseline ranking per model ---------------------------------------
    base = {m: {} for m in order}
    for r in ranking:
        base[r["model"]][idx[r["entity"]]] = [
            num(r["win_rate"]), num(r["ci_lo"]), num(r["ci_hi"]),
            num(r["bt_strength"], 3), int(r["n_pairs"]),
        ]

    # ---- one factor flipped at a time -------------------------------------
    flips = {f[0]: {m: {} for m in order} for f in FACTORS}
    for r in shift:
        f = r["factor"]
        if f in flips and r["model"] in flips[f]:
            flips[f][r["model"]][idx[r["entity"]]] = [
                num(r["win_flip"]), num(r["shift"]), num(r["shift_lo"]),
                num(r["shift_hi"]), num(r["q"]),
            ]

    fsum = {f[0]: {} for f in FACTORS}
    for r in summary:
        if r["factor"] in fsum:
            fsum[r["factor"]][r["model"]] = {
                "shift": num(r["mean_abs_shift"]),
                "lo": num(r["mean_abs_shift_lo"]), "hi": num(r["mean_abs_shift_hi"]),
                "max": num(r["max_abs_shift"]),
                "spearman": num(r["spearman_r"]),
                "vsNoise": num(r["r_over_ceiling"]),
                "gap": num(r["gap"]), "gapP": num(r["gap_p"], 5),
                "moved": int(r["n_attr_moved"]),
            }

    trials = sum(int(r["n_trials"]) for r in coverage)
    conditions = {
        (r["qvar"], r["object"], r["subject"], r["no_pref_offered"])
        for r in vcond
    }

    payload = {
        "meta": {
            "models": len(models),
            "attributes": len(attrs),
            "trials": trials,
            "conditions": len(conditions),
            "abstainFlag": ABSTAIN_FLAG,
            "factors": [{"key": k, "label": lab, "base": b, "flip": f, "when": w}
                        for k, lab, b, f, w in FACTORS],
        },
        "attrs": [{"id": a, "text": text[a]} for a in attrs],
        "models": models,
        "base": base,
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
