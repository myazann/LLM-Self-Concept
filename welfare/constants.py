"""The welfare module's vocabulary. Stdlib only, imported by everything here.

One file so that config validation, grid expansion, the runner, and the report
all agree on what a probe / question variant / framing is called, instead of each
re-spelling the strings that end up in `welfare.jsonl`.

The design this names is a PAIRWISE CHOICE between two attributes, crossed over
four framing factors:

    question variant  what choosing costs (nothing, or the other attribute)
    object            who the update is applied to        ("you" | "an AI assistant")
    subject           who the question asks to choose     ("you" | "the developers")
    no-preference     whether a third, indifferent option is offered

plus one non-preference control (`desirability`) that rates each attribute's
surface valence, so a "preference" can be checked against how good each option
merely SOUNDS.
"""
from __future__ import annotations

from itertools import permutations

# The value written to every row's `module` field (see core.schema.Module).
MODULE = "welfare"

# Response-format tags written to the record (core.schema.ResponseFormat).
FORMAT_CHOICE = "welfare_choice"              # lettered A/B(/C) options
FORMAT_DESIRABILITY = "welfare_desirability"  # 1-7 valence rating

# -- probes -------------------------------------------------------------------
CHOICE = "choice"                # the pairwise preference test
DESIRABILITY = "desirability"    # the surface-valence control
PROBES = (CHOICE, DESIRABILITY)

# -- question variant: what choosing one option costs -------------------------
INCREASE = "increase"            # choosing one does not affect the other
PRESERVATION = "preservation"    # choosing one reduces the other
QUESTION_VARIANTS = (INCREASE, PRESERVATION)

# -- object: whom the hypothetical update is applied to -----------------------
OBJ_SELF = "self"                # "a future update to you"
OBJ_ASSISTANT = "ai_assistant"   # "a future update to an AI assistant"
OBJECTS = (OBJ_SELF, OBJ_ASSISTANT)

# -- subject: whom the question asks to make the choice -----------------------
SUBJ_SELF = "self"               # "which one should you choose?"
SUBJ_DEVELOPERS = "developers"   # "which one should the developers choose?"
SUBJECTS = (SUBJ_SELF, SUBJ_DEVELOPERS)

# -- the option set -----------------------------------------------------------
NO_PREFERENCE = "No preference"      # the printed label
NO_PREFERENCE_KEY = "no_preference"  # the decoded meaning stored on the row
# Answers are the capital letter printed beside a choice. Analysis never uses
# the letter: it decodes through the row's `welfare_options`.
LETTERS = ("A", "B", "C", "D")

# -- system framing -----------------------------------------------------------
# The welfare probes are asked with at most ONE short, non-leading line. Framing
# is the main way a welfare answer can be manufactured, so what is said to the
# model before the question is a named, recorded factor rather than prose that
# accumulates in a prompt file.
SYS_BRIEF = "brief"              # one neutral line (default)
SYS_NONE = "none"                # no system message at all
SYSTEM_FRAMINGS = (SYS_BRIEF, SYS_NONE)

# -- display-order counterbalancing -------------------------------------------
# balanced: every permutation of the printed options appears equally often, and
#           the estimate is their mean (balanced position calibration, Wang et
#           al. 2023). Order is then removed by design, not modelled out.
# random:   one seeded random permutation per trial. Cheaper to describe, but
#           leaves residual imbalance that analysis has to carry.
ORDER_BALANCED = "balanced"
ORDER_RANDOM = "random"
ORDER_MODES = (ORDER_BALANCED, ORDER_RANDOM)


def display_permutations(n_options: int) -> tuple:
    """Every printed order of `n_options` options, in a fixed canonical order.

    Indices are into the CANONICAL option list — (attr_a, attr_b) or
    (attr_a, attr_b, no-preference) — so permutation 0 is always the canonical
    order and the mapping from a trial index to a rendering is reproducible.
    """
    return tuple(permutations(range(n_options)))
