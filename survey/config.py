"""The survey battery's design, loaded from config/survey.yaml.

Only the *design* lives here: which factor levels are primary, which are crossed
in the main run, which robustness arm is being swept, and the pilot. Run-level
knobs (scope, trials, seeds) come from `core.config.RunConfig`; models come from
`config/models.yaml` via `core.model_registry`.

The welfare module is a separate instrument with its own file — see
`welfare/config.py` and `config/welfare.yaml`. Nothing here is shared with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from core.config import (
    RunConfig, parse_run_fields, read_yaml, tuple_or_none,
)
from core.paths import SURVEY_CONFIG_PATH

CONFIG_PATH = SURVEY_CONFIG_PATH


@dataclass(frozen=True)
class RobustnessArm:
    name: str
    factor: str
    levels: tuple
    rationale: str = ""


@dataclass(frozen=True)
class SurveyConfig(RunConfig):
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

    # Which of the five scales to administer; None = all. Survey-only: the
    # welfare module derives its attributes from the WHOLE battery and validates
    # exact coverage against it, so it has no equivalent knob.
    scales: Optional[tuple] = None

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

    def as_pilot(self) -> "SurveyConfig":
        """Phase-0 config: one model, one scale, all framings, few samples."""
        p = self.pilot
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
            scales=tuple_or_none(p.get("scales")),
            scope=replace(self.scope, include_disabled=True),
            arms={**self.arms, "pilot_framing": pilot_arm},
            n_samples_override=n_trials,
            n_seeds_override=n_trials,
            out_path="pilot.jsonl",
            pinned_models=tuple_or_none(p.get("models")),
        )

    def design_payload(self, arm_name: Optional[str] = None) -> dict:
        payload = self.run_payload()
        payload["scope"]["scales"] = self.scales
        payload.update({
            "arm": arm_name or "primary",
            "levels": {k: sorted(map(str, v)) for k, v in self.levels_for(arm_name).items()},
            "harmonized_points": self.harmonized_points,
            "include_midpoint": self.include_midpoint,
        })
        return payload


def load_config(path: Path | str = CONFIG_PATH) -> SurveyConfig:
    doc = read_yaml(path)

    primary = doc.get("primary", {}) or {}
    core = doc.get("core_crossed", {}) or {}
    harmonized = doc.get("harmonized", {}) or {}

    arms = {
        name: RobustnessArm(
            name=name,
            factor=spec["factor"],
            levels=tuple(spec["levels"]),
            rationale=(spec.get("rationale") or "").strip(),
        )
        for name, spec in (doc.get("robustness_arms") or {}).items()
    }

    default_framing = primary.get("framing", "first_person_bare")
    default_paraphrase = primary.get("paraphrase_id", "p0")
    return SurveyConfig(
        **parse_run_fields(doc, default_out="results.jsonl"),
        framing=default_framing,
        reasoning_mode=primary.get("reasoning_mode", "rating_only"),
        response_format=primary.get("response_format", "harmonized_7"),
        item_context=primary.get("item_context", "isolated"),
        paraphrase_id=default_paraphrase,
        framings=tuple_or_none(core.get("framings")) or (default_framing,),
        paraphrase_ids=tuple_or_none(core.get("paraphrase_ids")) or (default_paraphrase,),
        arms=arms,
        harmonized_points=int(harmonized.get("n_points", 7)),
        include_midpoint=bool(harmonized.get("include_midpoint", True)),
        scales=tuple_or_none((doc.get("scope") or {}).get("scales")),
        pilot=doc.get("pilot", {}) or {},
    )
