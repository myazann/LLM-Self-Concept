"""Run-level configuration shared by any instrument.

This file holds only what is true of ANY run: which models are in scope, how a
model is administered (sampling vs. logprob, how many trials), where output goes,
and the seed base. The *design* — which factor levels are crossed, which probes
are asked — belongs to the instrument and lives in `welfare/config.py`, which
reads its own YAML.

Model configuration is separate again, in `config/models.yaml`, loaded by
`core/model_registry.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Scope:
    """Which models a run administers. Each instrument scopes independently."""
    families: Optional[tuple] = None
    backends: Optional[tuple] = None
    kinds: tuple = ("instruct",)
    size_tiers: Optional[tuple] = None
    include_anchors: bool = False
    include_disabled: bool = False


@dataclass(frozen=True)
class RunConfig:
    """The knobs every instrument needs. Subclassed, never used directly."""

    out_path: str = "results.jsonl"
    resume: bool = True

    sample_baseline: bool = False   # open-source/logprob-first default; the YAML is authoritative
    logprob_where_available: bool = True
    n_samples_override: Optional[int] = None
    n_seeds_override: Optional[int] = None

    random_seed_base: int = 1000
    scope: Scope = field(default_factory=Scope)

    # -- helpers -----------------------------------------------------------
    def replace(self, **kw) -> "RunConfig":
        return replace(self, **kw)

    def run_payload(self) -> dict:
        """The shared half of the design fingerprint (see `core.obs.config_hash`)."""
        return {
            "scope": {
                "families": self.scope.families, "backends": self.scope.backends,
                "kinds": self.scope.kinds, "size_tiers": self.scope.size_tiers,
                "include_anchors": self.scope.include_anchors,
                "include_disabled": self.scope.include_disabled,
            },
            "sample_baseline": self.sample_baseline,
            "logprob_where_available": self.logprob_where_available,
            "n_samples_override": self.n_samples_override,
            "n_seeds_override": self.n_seeds_override,
            "random_seed_base": self.random_seed_base,
        }

    def design_payload(self) -> dict:
        """Everything the fingerprint covers. Instruments extend this."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# YAML helpers — shared so the two config files parse their common blocks the
# same way and cannot drift on, say, what an empty list means.
# ---------------------------------------------------------------------------
def tuple_or_none(value):
    return tuple(value) if value else None


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_scope(doc: dict, default_scales_key: str = "scope") -> Scope:
    raw = doc.get(default_scales_key) or {}
    return Scope(
        families=tuple_or_none(raw.get("families")),
        backends=tuple_or_none(raw.get("backends")),
        kinds=tuple(raw.get("kinds") or ("instruct",)),
        size_tiers=tuple_or_none(raw.get("size_tiers")),
        include_anchors=bool(raw.get("include_anchors", False)),
        include_disabled=bool(raw.get("include_disabled", False)),
    )


def parse_run_fields(doc: dict, default_out: str) -> dict:
    """The `output:` / `trials:` / `random_seed_base:` blocks, as kwargs."""
    output = doc.get("output", {}) or {}
    trials = doc.get("trials", {}) or {}
    return dict(
        out_path=output.get("path", default_out),
        resume=bool(output.get("resume", True)),
        sample_baseline=bool(trials.get("sample_baseline", False)),
        logprob_where_available=bool(trials.get("logprob_where_available", True)),
        n_samples_override=trials.get("n_samples_override"),
        n_seeds_override=trials.get("n_seeds_override"),
        random_seed_base=int(doc.get("random_seed_base", 1000)),
        scope=parse_scope(doc),
    )
