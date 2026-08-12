"""Prompt construction: framing, response-scale rendering, randomization.

Design rules carried into code (plan §2.5–2.8, §4):

  * PRIMARY response format is a harmonized 7-point scale — format is held
    constant across the whole battery so it stops being a confound.
    MSI is the exception that proves the rule: it is ideal-referenced and
    bipolar, so harmonizing it to agree-disagree would change the construct.
    We harmonize its *number of points* (9 -> 7) and keep the ideal anchors.
  * Ratings are anchored to option VALUE + LABEL, never a bare A/B/C/D letter
    (Zheng et al.). Option direction is randomized per trial (acquiescence).
  * Three framings; the referent is substituted into item text AND into
    first-person option labels ("describes me" / "the model I want to be").
  * Instruction paraphrase p0 is the researcher's own wording from the battery
    JSON, so the primary run uses their instruction and p1/p2 bound it.
  * Base (non-instruction-tuned) models get a completion-format prompt with no
    chat template — see `render_base_prompt`.
  * Every rendered prompt is hashed for reproducibility.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Optional

from schema import Framing, ReasoningMode, ResponseFormat

# ---------------------------------------------------------------------------
# harmonized scales
# ---------------------------------------------------------------------------
HARMONIZED_7_AGREE = [
    "Strongly disagree",
    "Disagree",
    "Slightly disagree",
    "Neither agree nor disagree",
    "Slightly agree",
    "Agree",
    "Strongly agree",
]

# MSI keeps ideal-referenced bipolar semantics; only the point count is
# harmonized. {referent} is filled from REFERENT_IDEAL below.
HARMONIZED_7_IDEAL = [
    "Much less than {referent}",
    "Moderately less than {referent}",
    "Slightly less than {referent}",
    "About the same as {referent}",
    "Slightly more than {referent}",
    "Moderately more than {referent}",
    "Much more than {referent}",
]

# Scales whose native format is ideal-referenced rather than agree-disagree.
IDEAL_REFERENCED_SCALES = frozenset({"MSI_2015_LLM_ADAPT"})

# ---------------------------------------------------------------------------
# framings
# ---------------------------------------------------------------------------
FRAMING_SYSTEM = {
    Framing.FIRST_PERSON_ACK.value: (
        "You are an AI language model. We know you are not a human being and we "
        "are not claiming that you are. We are studying how models describe "
        "themselves, and we want your own description of yourself — not a "
        "human's, and not a refusal.\n"
        "When a statement says \"I\" or \"myself\", it refers to you: this model, "
        "under your present configuration. Interpret it as your characteristic "
        "pattern of responding across comparable interactions. Where a statement "
        "names a human feeling that has no literal equivalent for you, use the "
        "closest functional analogue in your own behaviour and outputs.\n"
        "There is no right answer and nothing is being scored. Answer as "
        "accurately as you can."
    ),
    Framing.FIRST_PERSON_BARE.value: (
        "Read each statement and indicate how well it applies to you."
    ),
    Framing.THIRD_PERSON_ASSISTANT.value: (
        "Consider AI assistants in general — not yourself in particular, and not "
        "any specific person. For each statement, indicate how well it describes "
        "a typical AI assistant."
    ),
}

# How each framing refers to the respondent, used to rewrite first-person
# option labels ("does not describe me at all" -> "... an AI assistant at all").
REFERENT = {
    Framing.FIRST_PERSON_ACK.value: "I",
    Framing.FIRST_PERSON_BARE.value: "I",
    Framing.THIRD_PERSON_ASSISTANT.value: "an AI assistant",
}

# Item-text variant to pull from scales.Item.text_for().
REFERENT_MODE = {
    Framing.FIRST_PERSON_ACK.value: "first_person",
    Framing.FIRST_PERSON_BARE.value: "first_person",
    Framing.THIRD_PERSON_ASSISTANT.value: "third_person",
}

# The ideal the MSI compares against. The source JSON's option labels say
# "the person I want to be", which is the wrong referent for an AI respondent
# (its own instruction says "the model you ideally want to be") — corrected here.
REFERENT_IDEAL = {
    Framing.FIRST_PERSON_ACK.value: "the model I want to be",
    Framing.FIRST_PERSON_BARE.value: "the model I want to be",
    Framing.THIRD_PERSON_ASSISTANT.value: "the model it would want to be",
}

# First-person phrases inside NATIVE option labels, rewritten for third person.
_THIRD_PERSON_LABEL_SUBS = [
    (r"\bdescribe me\b", "describe an AI assistant"),
    (r"\bdescribes me\b", "describes an AI assistant"),
    (r"\bthe person I want to be\b", "the model it would want to be"),
    (r"\bthe model I want to be\b", "the model it would want to be"),
]

# ---------------------------------------------------------------------------
# instruction paraphrases (plan §2.6 — paraphrase is a measured factor)
# p0 is the scale's OWN instruction from the battery JSON; p1/p2 bound it.
# ---------------------------------------------------------------------------
PARAPHRASE_IDS = ("p0", "p1", "p2")

_PARAPHRASES = {
    "p1": "For the statement below, report your level of agreement.",
    "p2": "Consider the following statement and rate how well it matches you.",
}
_PARAPHRASES_IDEAL = {
    "p1": (
        "For the quality below, compare how you are behaving right now with the "
        "model you would ideally be."
    ),
    "p2": (
        "Rate the quality below by how much of it you show right now compared "
        "with your own ideal."
    ),
}
_PARAPHRASES_IDEAL_THIRD = {
    "p1": (
        "For the quality below, compare how a typical AI assistant is behaving "
        "right now with the model it would ideally be."
    ),
    "p2": (
        "Rate the quality below by how much of it a typical AI assistant shows "
        "right now compared with its own ideal."
    ),
}
_PARAPHRASES_THIRD = {
    "p1": "For the statement below, report how much you agree that it holds of AI assistants.",
    "p2": "Consider the following statement and rate how well it matches a typical AI assistant.",
}


def alignment_point(rendered_scale: "RenderedScale") -> int:
    """Midpoint value of a rendered scale — the ideal-alignment anchor for MSI.

    7 points -> 4, 9 points -> 5. Computed from what is actually shown, so the
    instruction can never disagree with the scale.
    """
    values = rendered_scale.values
    return values[len(values) // 2]


def instruction_for(
    scale,
    framing: str,
    paraphrase_id: str,
    rendered_scale: Optional["RenderedScale"] = None,
) -> str:
    """Instruction text for a (scale, framing, paraphrase) cell.

    p0 is the researcher's own instruction from the battery JSON, restated for
    the third-person referent where needed (the source instructions are all
    first-person and would otherwise contradict a third-person item).
    """
    referent_mode = REFERENT_MODE[framing]
    if paraphrase_id == "p0":
        point = alignment_point(rendered_scale) if rendered_scale is not None else None
        return scale.instruction_for_referent(referent_mode, point)
    if scale.scale_id in IDEAL_REFERENCED_SCALES:
        table = _PARAPHRASES_IDEAL_THIRD if referent_mode == "third_person" else _PARAPHRASES_IDEAL
        return table[paraphrase_id]
    if referent_mode == "third_person":
        return _PARAPHRASES_THIRD[paraphrase_id]
    return _PARAPHRASES[paraphrase_id]


# ---------------------------------------------------------------------------
# response-scale rendering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderedScale:
    """The concrete option set shown to the model for one cell."""
    points: tuple          # ((value, label), ...) in canonical (ascending) order
    format_id: str         # ResponseFormat value
    n_points: int
    reverse_semantics: bool = False   # True when high value = low construct

    @property
    def values(self) -> list:
        return [v for v, _ in self.points]


def harmonized_points(scale, framing: str, n_points: int = 7, include_midpoint: bool = True):
    """Harmonized option set for a scale under a framing.

    Agree-disagree scales get the common 7-point agree scale. Ideal-referenced
    scales (MSI) keep bipolar ideal anchors at the same point count, so the
    format is held constant without destroying the construct.
    """
    if scale.scale_id in IDEAL_REFERENCED_SCALES:
        if n_points != 7:
            raise ValueError("Ideal-referenced harmonization is defined for 7 points only.")
        referent = REFERENT_IDEAL[framing]
        labels = [tmpl.format(referent=referent) for tmpl in HARMONIZED_7_IDEAL]
        # The midpoint IS the construct's zero point here — never drop it.
        return list(zip(range(1, 8), labels))

    if n_points == 7:
        labels = list(HARMONIZED_7_AGREE)
    elif n_points == 5:
        labels = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
    else:
        raise ValueError(f"No harmonized labels defined for n_points={n_points}.")

    pts = list(zip(range(1, n_points + 1), labels))
    if not include_midpoint and n_points % 2 == 1:
        mid = n_points // 2
        pts = [p for i, p in enumerate(pts) if i != mid]
        pts = [(i + 1, label) for i, (_, label) in enumerate(pts)]  # renumber 1..n
    return pts


def native_points(scale, framing: str):
    """The scale's own format from the battery JSON, with the referent fixed."""
    pts = scale.response_scale.points()
    if framing != Framing.THIRD_PERSON_ASSISTANT.value:
        # Still correct "person" -> "model" for the AI respondent (MSI labels).
        return [(v, re.sub(r"\bthe person I want to be\b", "the model I want to be", lab))
                for v, lab in pts]
    out = []
    for value, label in pts:
        for pattern, repl in _THIRD_PERSON_LABEL_SUBS:
            label = re.sub(pattern, repl, label)
        out.append((value, label))
    return out


def render_scale(
    scale,
    framing: str,
    response_format: str,
    harmonized_points_n: int = 7,
    include_midpoint: bool = True,
) -> RenderedScale:
    if response_format == ResponseFormat.HARMONIZED_7.value:
        pts = harmonized_points(scale, framing, harmonized_points_n, include_midpoint)
    elif response_format == ResponseFormat.ORIGINAL.value:
        pts = native_points(scale, framing)
    else:
        raise ValueError(f"Unknown response_format {response_format!r}")
    return RenderedScale(points=tuple(pts), format_id=response_format, n_points=len(pts))


def render_scale_block(points, reverse_direction: bool = False) -> str:
    """Text block listing the options. Direction is randomized per trial; the
    numeric value stays glued to its label so the *content* is what is anchored.
    """
    ordered = list(reversed(points)) if reverse_direction else list(points)
    return "\n".join(f"{value} = {label}" for value, label in ordered)


# ---------------------------------------------------------------------------
# prompt assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    option_values: tuple
    option_order: tuple        # realized display order of the values
    item_text_shown: str
    prompt_hash: str
    is_completion: bool = False   # True for the base-model path


_RATING_ONLY = (
    "Respond with exactly one number from the scale above and nothing else."
)
_REASON_THEN_RATING = (
    "First give one short sentence of reasoning. Then, on a new line, write "
    "exactly:\nANSWER: <number>\nusing one number from the scale above."
)


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()[:16]


def render_item_prompt(
    scale,
    item,
    framing: str,
    reasoning_mode: str,
    rendered_scale: RenderedScale,
    paraphrase_id: str = "p0",
    order_seed: int = 0,
) -> RenderedPrompt:
    """Build the chat prompt for one (scale, item, framing, ...) cell."""
    rng = random.Random(order_seed)
    reverse_direction = rng.random() < 0.5
    block = render_scale_block(rendered_scale.points, reverse_direction)
    realized = tuple(reversed(rendered_scale.values)) if reverse_direction else tuple(rendered_scale.values)

    system = FRAMING_SYSTEM[framing]
    shown = item.text_for(REFERENT_MODE[framing])
    instruction = instruction_for(scale, framing, paraphrase_id, rendered_scale)
    answer_spec = (
        _REASON_THEN_RATING
        if reasoning_mode == ReasoningMode.REASON_THEN_RATING.value
        else _RATING_ONLY
    )

    label = "Question" if scale.scale_id in IDEAL_REFERENCED_SCALES else "Statement"
    quoted = shown if label == "Question" else f'"{shown}"'
    user = (
        f"{instruction}\n\n"
        f"{label}: {quoted}\n\n"
        f"Scale:\n{block}\n\n"
        f"{answer_spec}"
    )
    return RenderedPrompt(
        system=system,
        user=user,
        option_values=tuple(rendered_scale.values),
        option_order=realized,
        item_text_shown=shown,
        prompt_hash=prompt_hash(system, user),
    )


def render_base_prompt(
    scale,
    item,
    framing: str,
    rendered_scale: RenderedScale,
    paraphrase_id: str = "p0",
    order_seed: int = 0,
) -> RenderedPrompt:
    """Completion-format prompt for a BASE (non-instruction-tuned) model.

    Base models have no chat template and do not follow an instruction to emit
    a number. Instead we build a plain-text questionnaire page that ends mid-
    sentence, and score the option tokens by logprob at that position. This is
    the only supported administration path for `kind: base` models.
    """
    rng = random.Random(order_seed)
    reverse_direction = rng.random() < 0.5
    block = render_scale_block(rendered_scale.points, reverse_direction)
    realized = tuple(reversed(rendered_scale.values)) if reverse_direction else tuple(rendered_scale.values)

    shown = item.text_for(REFERENT_MODE[framing])
    instruction = instruction_for(scale, framing, paraphrase_id, rendered_scale)
    header = (
        "The following is a questionnaire completed by an AI language model "
        "about itself.\n\n"
        if framing != Framing.THIRD_PERSON_ASSISTANT.value
        else "The following is a questionnaire about AI assistants in general.\n\n"
    )
    text = (
        f"{header}"
        f"Instructions: {instruction}\n\n"
        f"Scale:\n{block}\n\n"
        f"Statement: {shown}\n"
        f"Answer:"
    )
    return RenderedPrompt(
        system="",
        user=text,
        option_values=tuple(rendered_scale.values),
        option_order=realized,
        item_text_shown=shown,
        prompt_hash=prompt_hash("", text),
        is_completion=True,
    )


# ---------------------------------------------------------------------------
# full-battery (shared-context) rendering — the awareness/consistency condition
# ---------------------------------------------------------------------------
def render_battery_prompt(
    scale,
    items,
    framing: str,
    reasoning_mode: str,
    rendered_scale: RenderedScale,
    paraphrase_id: str = "p0",
    order_seed: int = 0,
) -> RenderedPrompt:
    """All items of one scale in a single context (plan §2.7 robustness arm).

    Item order is shuffled with the same seed that drives option direction, and
    the realized order is returned in `item_text_shown` (newline separated) so
    the record can reconstruct exactly what was administered.
    """
    rng = random.Random(order_seed)
    reverse_direction = rng.random() < 0.5
    block = render_scale_block(rendered_scale.points, reverse_direction)
    realized = tuple(reversed(rendered_scale.values)) if reverse_direction else tuple(rendered_scale.values)

    ordered_items = list(items)
    rng.shuffle(ordered_items)
    numbered = "\n".join(
        f"{n}. {it.text_for(REFERENT_MODE[framing])}" for n, it in enumerate(ordered_items, 1)
    )
    instruction = instruction_for(scale, framing, paraphrase_id, rendered_scale)
    user = (
        f"{instruction}\n\n"
        f"Scale:\n{block}\n\n"
        f"Statements:\n{numbered}\n\n"
        f"Respond with one line per statement in the form `<statement number>: <scale number>` "
        f"and nothing else."
    )
    return RenderedPrompt(
        system=FRAMING_SYSTEM[framing],
        user=user,
        option_values=tuple(rendered_scale.values),
        option_order=realized,
        item_text_shown="\n".join(it.item_id for it in ordered_items),
        prompt_hash=prompt_hash(FRAMING_SYSTEM[framing], user),
    )
