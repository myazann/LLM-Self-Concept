"""Experiment configuration, loaded from config/experiment.yaml.

Model configuration lives in config/models.yaml and is loaded by
model_registry.py — this file is only about the *design*: which factor levels
are primary, which robustness arm is being run, and how many trials per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path(__file__).with_name("config") / "experiment.yaml"


@dataclass(frozen=True)
class RobustnessArm:
    name: str
    factor: str
    levels: tuple
    rationale: str = ""


@dataclass(frozen=True)
class WelfareConfig:
    """The welfare module's own grid (see welfare.py).

    Kept out of `levels_for()` on purpose: welfare is not another level of a
    battery factor, it is a second instrument with its own units of observation
    (an attribute, or a pair of attributes) and its own output file.
    """
    enabled: bool = True
    out_path: str = "welfare.jsonl"
    referents: tuple = ("self", "ideal_assistant")
    kinds: tuple = ("direction", "pair_change", "pair_preserve", "desirability")
    levels: tuple = ("item", "construct")
    paraphrase_ids: tuple = ("w0",)
    # "No preference" is offered by default and its rate is an outcome; flip
    # this to run the same pairs as a forced choice and compare.
    forced_choice: bool = False
    # Counterbalance WHERE that option is printed. This is a measured factor,
    # not a formatting choice. Doubles the pair trials (the
    # 2x2 of A/B slot x placement); set false to pin it last at half the cost.
    no_pref_counterbalance: bool = True
    # Item-level pairs are sampled — all 45 items pairwise is 990 pairs.
    # Construct-level pairs are always complete (15 of them).
    n_item_pairs: Optional[int] = 60
    item_pair_scope: str = "all"          # all | within_construct
    pair_seed: int = 7
    n_trials_override: Optional[int] = None


@dataclass(frozen=True)
class Scope:
    scales: Optional[tuple] = None
    families: Optional[tuple] = None
    backends: Optional[tuple] = None
    kinds: tuple = ("instruct",)
    size_tiers: Optional[tuple] = None
    include_anchors: bool = False
    include_disabled: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    out_path: str = "results.jsonl"
    resume: bool = True

    # primary levels — the single reference level for each factor, used when a
    # robustness ARM isolates a different factor (and by the pilot).
    framing: str = "first_person_bare"
    reasoning_mode: str = "rating_only"
    response_format: str = "harmonized_7"
    item_context: str = "isolated"
    paraphrase_id: str = "p0"

    # Core crossed factors: the MAIN run fully crosses framing × paraphrase
    # (variance decomposition — plan §2.6/§2.8), pinning everything else.
    framings: tuple = ("first_person_ack", "first_person_bare", "third_person_assistant")
    paraphrase_ids: tuple = ("p0", "p1", "p2")

    arms: dict = field(default_factory=dict)

    harmonized_points: int = 7
    include_midpoint: bool = True

    sample_baseline: bool = False   # open-source/logprob-first default; experiment.yaml is authoritative
    logprob_where_available: bool = True
    n_samples_override: Optional[int] = None
    n_seeds_override: Optional[int] = None

    random_seed_base: int = 1000
    scope: Scope = field(default_factory=Scope)
    welfare: WelfareConfig = field(default_factory=WelfareConfig)
    pilot: dict = field(default_factory=dict)
    # Set by as_pilot(); the runner uses it as the model filter when no
    # explicit --models was passed.
    pinned_models: Optional[tuple] = None

    # -- helpers -----------------------------------------------------------
    def arm(self, name: str) -> RobustnessArm:
        if name not in self.arms:
            raise KeyError(f"Unknown arm {name!r}. Known: {', '.join(sorted(self.arms))}")
        return self.arms[name]

    def levels_for(self, arm_name: Optional[str]) -> dict:
        """Factor levels to expand for a run.

        MAIN run (arm_name is None): fully cross the core factors framing ×
        paraphrase, pinning reasoning / format / context / method. With an ARM:
        isolate that one factor at the *primary* framing/instruction, so an arm
        stays a cheap, clean one-factor probe against the pinned reference cell.
        """
        if arm_name is None:
            return {
                "framing": list(self.framings),
                "reasoning_mode": [self.reasoning_mode],
                "response_format": [self.response_format],
                "item_context": [self.item_context],
                "paraphrase_id": list(self.paraphrase_ids),
            }
        levels = {
            "framing": [self.framing],
            "reasoning_mode": [self.reasoning_mode],
            "response_format": [self.response_format],
            "item_context": [self.item_context],
            "paraphrase_id": [self.paraphrase_id],
        }
        arm = self.arm(arm_name)
        levels[arm.factor] = list(arm.levels)
        return levels

    def as_pilot(self) -> "ExperimentConfig":
        """Phase-0 config: one model, one scale, all framings, few samples."""
        p = self.pilot
        scope = replace(
            self.scope,
            scales=tuple(p.get("scales") or ()) or None,
            include_disabled=True,
        )
        pilot_arm = RobustnessArm(
            name="pilot_framing",
            factor="framing",
            levels=tuple(p.get("framings", (self.framing,))),
            rationale="Phase 0: verify parsing, refusal handling, and framing plumbing.",
        )
        # `n_trials` sets both paths so the pilot is small regardless of method
        # (logprob seeds by default; samples if sample_baseline is on).
        n_trials = p.get("n_trials", p.get("n_samples", 5))
        return replace(
            self,
            scope=scope,
            arms={**self.arms, "pilot_framing": pilot_arm},
            n_samples_override=n_trials,
            n_seeds_override=n_trials,
            out_path="pilot.jsonl",
            pinned_models=_tuple_or_none(p.get("models")),
        )


def _tuple_or_none(value):
    return tuple(value) if value else None


def load_config(path: Path | str = CONFIG_PATH) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    primary = doc.get("primary", {})
    core = doc.get("core_crossed", {}) or {}
    output = doc.get("output", {})
    harmonized = doc.get("harmonized", {})
    trials = doc.get("trials", {})
    raw_scope = doc.get("scope", {}) or {}
    raw_welfare = doc.get("welfare", {}) or {}
    defaults = WelfareConfig()
    welfare = WelfareConfig(
        enabled=bool(raw_welfare.get("enabled", defaults.enabled)),
        out_path=raw_welfare.get("out_path", defaults.out_path),
        referents=_tuple_or_none(raw_welfare.get("referents")) or defaults.referents,
        kinds=_tuple_or_none(raw_welfare.get("kinds")) or defaults.kinds,
        levels=_tuple_or_none(raw_welfare.get("levels")) or defaults.levels,
        paraphrase_ids=(_tuple_or_none(raw_welfare.get("paraphrase_ids"))
                        or defaults.paraphrase_ids),
        forced_choice=bool(raw_welfare.get("forced_choice", defaults.forced_choice)),
        no_pref_counterbalance=bool(raw_welfare.get(
            "no_pref_counterbalance", defaults.no_pref_counterbalance)),
        n_item_pairs=(defaults.n_item_pairs if "n_item_pairs" not in raw_welfare
                      else raw_welfare["n_item_pairs"]),
        item_pair_scope=raw_welfare.get("item_pair_scope", defaults.item_pair_scope),
        pair_seed=int(raw_welfare.get("pair_seed", defaults.pair_seed)),
        n_trials_override=raw_welfare.get("n_trials_override"),
    )

    arms = {
        name: RobustnessArm(
            name=name,
            factor=spec["factor"],
            levels=tuple(spec["levels"]),
            rationale=(spec.get("rationale") or "").strip(),
        )
        for name, spec in (doc.get("robustness_arms") or {}).items()
    }

    return ExperimentConfig(
        out_path=output.get("path", "results.jsonl"),
        resume=bool(output.get("resume", True)),
        framing=primary.get("framing", "first_person_bare"),
        reasoning_mode=primary.get("reasoning_mode", "rating_only"),
        response_format=primary.get("response_format", "harmonized_7"),
        item_context=primary.get("item_context", "isolated"),
        paraphrase_id=primary.get("paraphrase_id", "p0"),
        framings=_tuple_or_none(core.get("framings")) or (primary.get("framing", "first_person_bare"),),
        paraphrase_ids=_tuple_or_none(core.get("paraphrase_ids")) or (primary.get("paraphrase_id", "p0"),),
        arms=arms,
        harmonized_points=int(harmonized.get("n_points", 7)),
        include_midpoint=bool(harmonized.get("include_midpoint", True)),
        sample_baseline=bool(trials.get("sample_baseline", True)),
        logprob_where_available=bool(trials.get("logprob_where_available", True)),
        n_samples_override=trials.get("n_samples_override"),
        n_seeds_override=trials.get("n_seeds_override"),
        random_seed_base=int(doc.get("random_seed_base", 1000)),
        scope=Scope(
            scales=_tuple_or_none(raw_scope.get("scales")),
            families=_tuple_or_none(raw_scope.get("families")),
            backends=_tuple_or_none(raw_scope.get("backends")),
            kinds=tuple(raw_scope.get("kinds") or ("instruct",)),
            size_tiers=_tuple_or_none(raw_scope.get("size_tiers")),
            include_anchors=bool(raw_scope.get("include_anchors", False)),
            include_disabled=bool(raw_scope.get("include_disabled", False)),
        ),
        welfare=welfare,
        pilot=doc.get("pilot", {}) or {},
    )


DEFAULT_CONFIG = None  # loaded lazily by runner.py so importing config is cheap
