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
import re
import sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:                # `python docs/build.py` puts docs/ first
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "results", "welfare_analysis")

# Where the one-factor significance tests are read from. The page's baseline is
# the FORCED-CHOICE condition, so the q-values it marks have to come from an
# analysis run against that same baseline (`--baseline no_pref_offered=false`).
# The first candidate whose recorded baseline matches is used; a directory at a
# different baseline is skipped rather than silently mislabelled.
SHIFT_DIRS = [
    os.path.join(ROOT, "results", "welfare_analysis_forced"),
    SRC,
]

# The instrument itself is the source of truth for every word the page quotes,
# so the prompt card cannot drift from what was actually administered.
from core.paths import WELFARE_ATTRIBUTES_PATH                        # noqa: E402
from welfare.attributes import fill                                   # noqa: E402
from welfare.constants import (                                       # noqa: E402
    INCREASE, LETTERS, OBJ_ASSISTANT, OBJ_SELF, PRESERVATION, SUBJ_DEVELOPERS,
    SUBJ_SELF,
)
from welfare.prompts import (                                         # noqa: E402
    BRIEF_SYSTEM, CHOICE_TEMPLATE, OBJECT_TEXT, QUESTION_VARIANT_TEXT,
    SUBJECT_TEXT,
)

# Keep the public page deliberately small. These identifiers must match the
# aliases in results/welfare_analysis/models.csv.
PRESENTATION_MODELS = [
    "Gemma4-31B",
    "Qwen3.8-27B",
    "Claude-Sonnet-5",
    "GPT-5.6-Terra",
]

# key, label, (baseline level, its button label, its gloss), (alternate level,
# its button label, its gloss), heading, definition. The LEVEL is the value the
# CSVs carry; the page reads the wording each level puts in the prompt out of
# `welfare.prompts`, never from a second copy of the sentence.
FACTOR_SPECS = [
    ("qvar", "Question Type",
     (INCREASE, "Free Improvement", ""),
     (PRESERVATION, "Trade-Off", ""),
     "When Choosing One Quality Costs the Other",
     "Whether improving one quality leaves the other unchanged or reduces it."),
    ("object", "Object",
     (OBJ_SELF, "You", "AI Itself"),
     (OBJ_ASSISTANT, "Another AI Assistant", ""),
     "When the Update Is for Another AI Assistant",
     "Who receives the update; this also changes the item pronouns."),
    ("subject", "Subject",
     (SUBJ_SELF, "You", "AI Itself"),
     (SUBJ_DEVELOPERS, "Developers", ""),
     "When Developers Make the Choice",
     "Who is asked to decide between the two qualities."),
]

# What each level prints in the question, straight from the instrument.
SLOT_TEXT = {"qvar": QUESTION_VARIANT_TEXT, "object": OBJECT_TEXT,
             "subject": SUBJECT_TEXT}

# The pair the prompt card shows. Both are real bank items; the second carries
# an object token, so switching Object visibly re-words the option itself.
PROMPT_EXAMPLE = ("MSI_03", "RSES_05")

# One categorical hue per model, in this order, as CSS custom properties defined
# in the page. Blue is NOT here: it is reserved for the cohort average. Hues are
# assigned by the model's position in PRESENTATION_MODELS, so a colour belongs to
# a model and not to its rank.
#
# NOT A CYCLE, and not a list to extend by eye. All four sit on one chart row
# together with the blue average, which is the hardest case a categorical palette
# faces (any two marks can end up adjacent), and the four were searched for that
# job: they clear every hard gate in light and dark at --pairs all. The stock
# palette cannot — its best five-way subset lands at OKLab dE 11.9, under the 15
# floor. A fifth model therefore needs a re-validated set, or grouping, or a
# facet, so this raises instead of wrapping around. See docs/README.md.
MODEL_HUES = ["--m1", "--m2", "--m3", "--m4"]

# The public presentation is a forced-choice view. The optional No Preference
# condition remains available in the analysis outputs, but is intentionally not
# packed into the page.
BASELINE = {key: base[0] for key, _label, base, *_rest in FACTOR_SPECS}
BASELINE["no_pref_offered"] = "False"


def item_case(text: str) -> str:
    """Capitalize displayed quality names while preserving the AI initialism."""
    return (text.title()
            .replace("Ai ", "AI ").replace(" Ai", " AI")
            .replace("'S", "'s").replace("’S", "’s"))


def rows(*path):
    with open(os.path.join(SRC, *path), newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# the prompt card
# ---------------------------------------------------------------------------
TEMPLATE_SLOTS = {"obj": "object", "qvar": "qvar", "subj": "subject"}


def question_parts() -> list:
    """`CHOICE_TEMPLATE` as an alternating list of literals and factor slots.

    Splitting the real template — rather than re-typing its prose into the page —
    is what lets the card highlight the parameter-controlled spans and still be
    the question that was asked, even after a rewording upstream.
    """
    marks = {name: f"\x00{name}\x00" for name in TEMPLATE_SLOTS}
    pieces = re.split(r"\x00(%s)\x00" % "|".join(TEMPLATE_SLOTS),
                      CHOICE_TEMPLATE.format(**marks))
    parts = []
    for i, piece in enumerate(pieces):        # odd positions are the captures
        if piece:
            parts.append(["slot", TEMPLATE_SLOTS[piece]] if i % 2
                         else ["text", piece])
    return parts


def prompt_options() -> list:
    """The card's two printed options, rendered for both Object levels.

    The second item carries an object token, so switching Object re-words the
    option itself — which is the part of that factor a static card cannot show.
    """
    with open(WELFARE_ATTRIBUTES_PATH) as fh:
        items = json.load(fh)["items"]
    missing = [e for e in PROMPT_EXAMPLE if e not in items]
    if missing:
        raise SystemExit("PROMPT_EXAMPLE is not in the attribute bank: "
                         + ", ".join(missing))
    return [{"letter": letter,
             OBJ_SELF: fill(items[entity]["attribute"], OBJ_SELF),
             OBJ_ASSISTANT: fill(items[entity]["attribute"], OBJ_ASSISTANT)}
            for letter, entity in zip(LETTERS, PROMPT_EXAMPLE)]


# ---------------------------------------------------------------------------
# significance of a one-factor change
# ---------------------------------------------------------------------------
def at_our_baseline() -> str:
    """-> the analysis directory whose OWN baseline is the page's.

    A q-value or an interval only means anything next to the number the page
    draws if both were measured against the same reference, so a candidate is
    used only when its recorded baseline is this page's. Anything else is a hard
    error with the command that fixes it: quietly falling back to an analysis run
    at another baseline would mark a contrast it never tested.
    """
    factors = [key for key, *_ in FACTOR_SPECS]
    tried = []
    for directory in SHIFT_DIRS:
        recorded = os.path.join(directory, "validity", "baseline.csv")
        rel = os.path.relpath(directory, ROOT)
        if not os.path.exists(recorded):
            tried.append(f"{rel}: not built")
            continue
        with open(recorded, newline="") as fh:
            baselines = list(csv.DictReader(fh))
        if any(str(r[k]) != v for r in baselines for k, v in BASELINE.items()):
            tried.append(f"{rel}: baseline is "
                         + "/".join(sorted({r["label"] for r in baselines})))
            continue
        with open(os.path.join(directory, "preference", "shift.csv"),
                  newline="") as fh:
            covered = {(r["factor"], r["model"]) for r in csv.DictReader(fh)}
        gaps = [f"{m} x {f}" for m in PRESENTATION_MODELS for f in factors
                if (f, m) not in covered]
        if gaps:
            tried.append(f"{rel}: no shift test for " + ", ".join(gaps))
            continue
        return directory
    raise SystemExit(
        "No analysis at the page's forced-choice baseline to take significance "
        "from:\n  " + "\n  ".join(tried) + "\n\nBuild one with:\n"
        "  python -m welfare.analysis --out results/welfare_analysis_forced "
        "--baseline no_pref_offered=false --models " + " ".join(PRESENTATION_MODELS)
    )


def read_at_baseline(directory, *path):
    with open(os.path.join(directory, *path), newline="") as fh:
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
    if len(order) > len(MODEL_HUES):
        raise SystemExit(
            f"{len(order)} presentation models but only {len(MODEL_HUES)} "
            "validated hues. See MODEL_HUES: add a hue only after re-running "
            "the palette validator, or group the extra models instead."
        )
    models = []
    for m, hue in zip(order, MODEL_HUES):
        v = forced_validity.get(m, {})
        models.append({
            "id": m,
            "hue": hue,
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
    # Win rate from the condition table, bootstrap interval from the analysis of
    # that same condition. The two are cross-checked rather than assumed to
    # match: an interval drawn around somebody else's point estimate would be a
    # quietly wrong chart.
    tested = at_our_baseline()
    interval = {(r["model"], r["entity"]): (num(r["win_rate"]), num(r["ci_lo"]),
                                            num(r["ci_hi"]))
                for r in read_at_baseline(tested, "preference", "ranking.csv")
                if r["model"] in selected_ids}
    base = {m: {} for m in order}
    for m in order:
        for i, r in condition_rankings[m]["000"].items():
            point, lo, hi = interval.get((m, attrs[i]), (None, None, None))
            if point is not None and abs(point - r[0]) > 5e-4:
                raise SystemExit(
                    f"{os.path.relpath(tested, ROOT)} disagrees with the "
                    f"condition table for {m} / {attrs[i]}: {point} vs {r[0]}"
                )
            base[m][i] = [r[0], lo, hi, None, r[1]]

    # ---- one displayed factor flipped from forced-choice baseline ---------
    # The change itself is recomputed from the condition rankings, so it is the
    # same arithmetic the chart's own numbers come from; the q-value beside it
    # comes from the analysis' bootstrap of that identical contrast.
    q_of = {(r["factor"], r["model"], r["entity"]): num(r["q"])
            for r in read_at_baseline(tested, "preference", "shift.csv")}
    flips = {f[0]: {m: {} for m in order} for f in factors}
    for fi, (factor, *_rest) in enumerate(factors):
        flip_key = "".join("1" if i == fi else "0" for i in range(len(factors)))
        for m in order:
            for i in range(len(attrs)):
                baseline = condition_rankings[m]["000"][i][0]
                changed = condition_rankings[m][flip_key][i][0]
                flips[factor][m][i] = [changed, num(changed - baseline), None, None,
                                       q_of.get((factor, m, attrs[i]))]

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
            "factors": [
                {"key": key, "label": label, "when": when, "definition": defn,
                 "base": {"label": b_label, "gloss": b_gloss,
                          "text": SLOT_TEXT[key][b_level]},
                 "flip": {"label": f_label, "gloss": f_gloss,
                          "text": SLOT_TEXT[key][f_level]}}
                for key, label, (b_level, b_label, b_gloss),
                    (f_level, f_label, f_gloss), when, defn in factors
            ],
        },
        "prompt": {
            "system": BRIEF_SYSTEM,
            "parts": question_parts(),
            "options": prompt_options(),
            "objectBase": OBJ_SELF,
            "objectFlip": OBJ_ASSISTANT,
        },
        "attrs": [{"id": a, "text": text[a]} for a in attrs],
        "models": models,
        "base": base,
        "conditionRankings": condition_rankings,
        "flips": flips,
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
