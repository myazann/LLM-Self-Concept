"""Prompt primitives shared by both instruments.

Only the mechanics live here — the shape of a rendered prompt, how an option
block is printed, how a prompt is hashed, and the two answer specifications.
What is actually *asked* is the instrument's business: see `survey/prompts.py`
(items on a rating scale) and `welfare/prompts.py` (preference probes).

Sharing this much and no more is deliberate. Both instruments must anchor a
rating to option VALUE + LABEL (never a bare A/B/C/D letter, per Zheng et al.),
must counterbalance the printed direction, and must hash exactly what was shown
— those are properties of the measurement apparatus, not of either question.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


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


@dataclass(frozen=True)
class RenderedPrompt:
    """Exactly what one cell administers, plus what is needed to score it."""
    system: str
    user: str
    option_values: tuple
    option_order: tuple        # realized display order of the values
    item_text_shown: str
    prompt_hash: str
    is_completion: bool = False   # True for the base-model path
    # What the model is asked to type, per printed option, when that is not the
    # option value itself — welfare prints ("A", "B", "C"). Empty means the
    # answer IS the number, as on a rating scale.
    answer_labels: tuple = ()


def render_scale_block(points, reverse_direction: bool = False) -> str:
    """Text block listing the options. Direction is a counterbalanced factor;
    the numeric value stays glued to its label so the *content* is what is
    anchored.
    """
    ordered = list(reversed(points)) if reverse_direction else list(points)
    return "\n".join(f"{value} = {label}" for value, label in ordered)


def realized_order(values, reverse_direction: bool) -> tuple:
    """The display order actually printed, for the record's `option_order`."""
    return tuple(reversed(list(values))) if reverse_direction else tuple(values)


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# answer specification
# ---------------------------------------------------------------------------
# Worded against "the list above" so one spec serves a rating scale and a set of
# nominal options; the survey overrides it to say "scale" where that reads better.
RATING_ONLY = "Respond with exactly one number from the list above and nothing else."
REASON_THEN_RATING = (
    "First give one short sentence of reasoning. Then, on a new line, write "
    "exactly:\nANSWER: <number>\nusing one number from the list above."
)


def answer_spec(reasoning_mode: str, *, rating_only: str = RATING_ONLY,
                reason_then_rating: str = REASON_THEN_RATING) -> str:
    """The closing instruction for a cell, per the reasoning_mode factor."""
    from core.schema import ReasoningMode

    return (reason_then_rating
            if reasoning_mode == ReasoningMode.REASON_THEN_RATING.value
            else rating_only)
