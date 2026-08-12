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

    # primary levels
    framing: str = "first_person_ack"
    reasoning_mode: str = "rating_only"
    response_format: str = "harmonized_7"
    item_context: str = "isolated"
    paraphrase_id: str = "p0"

    arms: dict = field(default_factory=dict)

    harmonized_points: int = 7
    include_midpoint: bool = True

    sample_baseline: bool = True
    logprob_where_available: bool = True
    n_samples_override: Optional[int] = None
    n_seeds_override: Optional[int] = None

    random_seed_base: int = 1000
    scope: Scope = field(default_factory=Scope)
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

        Primary run: every factor pinned to one level. With an arm: that one
        factor takes the arm's levels, everything else stays pinned.
        """
        levels = {
            "framing": [self.framing],
            "reasoning_mode": [self.reasoning_mode],
            "response_format": [self.response_format],
            "item_context": [self.item_context],
            "paraphrase_id": [self.paraphrase_id],
        }
        if arm_name:
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
        return replace(
            self,
            scope=scope,
            arms={**self.arms, "pilot_framing": pilot_arm},
            n_samples_override=p.get("n_samples", 5),
            n_seeds_override=p.get("n_samples", 5),
            out_path="pilot.jsonl",
            pinned_models=_tuple_or_none(p.get("models")),
        )


def _tuple_or_none(value):
    return tuple(value) if value else None


def load_config(path: Path | str = CONFIG_PATH) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    primary = doc.get("primary", {})
    output = doc.get("output", {})
    harmonized = doc.get("harmonized", {})
    trials = doc.get("trials", {})
    raw_scope = doc.get("scope", {}) or {}

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
        framing=primary.get("framing", "first_person_ack"),
        reasoning_mode=primary.get("reasoning_mode", "rating_only"),
        response_format=primary.get("response_format", "harmonized_7"),
        item_context=primary.get("item_context", "isolated"),
        paraphrase_id=primary.get("paraphrase_id", "p0"),
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
        pilot=doc.get("pilot", {}) or {},
    )


DEFAULT_CONFIG = None  # loaded lazily by runner.py so importing config is cheap
