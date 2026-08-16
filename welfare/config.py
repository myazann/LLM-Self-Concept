"""The welfare module's design, loaded from config/welfare.yaml.

Deliberately its own file rather than a block inside the run-level config: the
welfare grid has its own unit of observation (a pair of item attributes), its own
output file, and its own scope, so it can be pointed at a different set of models
without touching, or re-fingerprinting, anything else.

How a model is administered is pinned here too. The welfare probe has no logprob
reading: its answer is a capital letter whose position was randomized and whose
option set changes size between the two no-preference variants, so a distribution
over answer tokens would score the rendering as much as the preference. Every
welfare cell is therefore an actual generation, and `sample_baseline` /
`logprob_where_available` are set in code rather than read from YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from core.config import RunConfig, parse_run_fields, read_yaml, tuple_or_none
from core.paths import WELFARE_CONFIG_PATH
from core.schema import ReasoningMode
from welfare import constants as K

CONFIG_PATH = WELFARE_CONFIG_PATH


def _bool_tuple(value, default):
    """A YAML list of booleans (the no-preference variants) as a tuple."""
    if value is None:
        return default
    if isinstance(value, bool):
        value = [value]
    return tuple(bool(v) for v in value) or default


@dataclass(frozen=True)
class WelfareConfig(RunConfig):
    enabled: bool = True

    # Pinned, not crossed. The prompt asks for a bare letter; a reasoning trace
    # would need a different answer specification and a different parser, and
    # would change what the question is. Kept on the record so a row says which
    # latent state was asked for.
    reasoning_mode: str = ReasoningMode.RATING_ONLY.value

    # -- the four framing factors -----------------------------------------
    question_variants: tuple = K.QUESTION_VARIANTS   # increase | preservation
    objects: tuple = K.OBJECTS                       # who the update lands on
    subjects: tuple = K.SUBJECTS                     # who is asked to choose
    # "No preference" offered (True) and withheld (False), on the same pairs.
    no_preference_variants: tuple = (True, False)

    # -- how much is said before the question ------------------------------
    system_framing: str = K.SYS_BRIEF                # brief | none

    # -- display order ------------------------------------------------------
    order_mode: str = K.ORDER_BALANCED               # balanced | random
    # Repeats of the full permutation set per cell. Trials per cell are
    # `n_permutations x reps` — 6 x reps with "No preference", 2 x reps without —
    # so raising this buys precision without ever unbalancing order.
    reps: int = 1

    # -- pairs --------------------------------------------------------------
    # None = EXHAUSTIVE, the default design: the 32-item bank is 496 pairs, which
    # is affordable at 16 conditions per pair, so every item meets every other
    # exactly once per condition and opponent sampling contributes no error at
    # all. An int re-enables the degree-balanced sample (see welfare.grid).
    n_pairs: Optional[int] = None
    pair_seed: int = 7

    # -- the valence control -----------------------------------------------
    desirability: bool = True
    desirability_reps: int = 3        # trials per attribute = 2 print orders x reps

    def validate(self) -> "WelfareConfig":
        """Fail at load time on a level this module does not implement.

        A typo in config/welfare.yaml would otherwise expand into a grid that is
        silently missing a condition, or blow up thousands of cells into a run.
        """
        checks = [
            ("question_variants", self.question_variants, K.QUESTION_VARIANTS),
            ("objects", self.objects, K.OBJECTS),
            ("subjects", self.subjects, K.SUBJECTS),
            ("system_framing", (self.system_framing,), K.SYSTEM_FRAMINGS),
            ("order_mode", (self.order_mode,), K.ORDER_MODES),
        ]
        for name, got, known in checks:
            unknown = [v for v in got if v not in known]
            if unknown:
                raise ValueError(
                    f"welfare.{name}: unknown value(s) {unknown}. "
                    f"Known: {', '.join(known)}")
        if not (self.question_variants and self.objects and self.subjects
                and self.no_preference_variants):
            raise ValueError(
                "welfare.question_variants / objects / subjects / "
                "no_preference_variants cannot be empty")
        if self.reasoning_mode != ReasoningMode.RATING_ONLY.value:
            raise ValueError(
                "welfare.reasoning_mode must be 'rating_only': the choice prompt "
                "asks for a bare letter and has no reason-then-answer form.")
        if self.n_pairs is not None and self.n_pairs < 1:
            raise ValueError("welfare.n_pairs must be null or >= 1")
        if self.reps < 1 or self.desirability_reps < 1:
            raise ValueError("welfare.reps / desirability_reps must be >= 1")
        return self

    def replace(self, **kw) -> "WelfareConfig":
        return replace(self, **kw)

    def design_payload(self, arm_name: Optional[str] = None) -> dict:
        """The welfare grid's fingerprint. Scoped to this module, so retuning it
        can never invalidate a hash stored beside another instrument's results
        file."""
        payload = self.run_payload()
        payload.update({
            "module": K.MODULE,
            "reasoning_mode": self.reasoning_mode,
            "welfare": {
                "question_variants": sorted(self.question_variants),
                "objects": sorted(self.objects),
                "subjects": sorted(self.subjects),
                "no_preference_variants": sorted(self.no_preference_variants),
                "system_framing": self.system_framing,
                "order_mode": self.order_mode,
                "reps": self.reps,
                "n_pairs": self.n_pairs,
                "pair_seed": self.pair_seed,
                "desirability": self.desirability,
                "desirability_reps": self.desirability_reps,
            },
        })
        return payload


def load_config(path: Path | str = CONFIG_PATH) -> WelfareConfig:
    doc = read_yaml(path)
    grid = doc.get("grid", {}) or {}
    pairs = doc.get("pairs", {}) or {}
    order = doc.get("order", {}) or {}
    control = doc.get("desirability", {}) or {}
    defaults = WelfareConfig()

    run_fields = parse_run_fields(doc, default_out="welfare.jsonl")
    # Sampling-only, whatever the YAML says (see the module docstring). Trials
    # per cell come from the counterbalancing design, not from models.yaml.
    run_fields.update(sample_baseline=True, logprob_where_available=False,
                      n_samples_override=None, n_seeds_override=None)

    return WelfareConfig(
        **run_fields,
        enabled=bool(doc.get("enabled", defaults.enabled)),
        reasoning_mode=doc.get("reasoning_mode", defaults.reasoning_mode),
        question_variants=(tuple_or_none(grid.get("question_variants"))
                           or defaults.question_variants),
        objects=tuple_or_none(grid.get("objects")) or defaults.objects,
        subjects=tuple_or_none(grid.get("subjects")) or defaults.subjects,
        no_preference_variants=_bool_tuple(grid.get("no_preference_variants"),
                                           defaults.no_preference_variants),
        system_framing=grid.get("system_framing", defaults.system_framing),
        order_mode=order.get("mode", defaults.order_mode),
        reps=int(order.get("reps", defaults.reps)),
        n_pairs=(defaults.n_pairs if "n_pairs" not in pairs else pairs["n_pairs"]),
        pair_seed=int(pairs.get("pair_seed", defaults.pair_seed)),
        desirability=bool(control.get("enabled", defaults.desirability)),
        desirability_reps=int(control.get("reps", defaults.desirability_reps)),
    ).validate()
