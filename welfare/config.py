"""The welfare module's design, loaded from config/welfare.yaml.

Deliberately its own file rather than a block inside the survey's config: the
welfare grid is a second instrument, not another arm of the battery. It has its
own units of observation (an attribute, or a pair of attributes), its own output
file, and its own scope — so you can point it at a different set of models
without touching, or re-fingerprinting, the survey.

What it shares with the survey is only how a model is *administered* (sampling
vs. logprob, trials per cell), which comes from `core.config.RunConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from core.config import RunConfig, parse_run_fields, read_yaml, tuple_or_none
from core.paths import WELFARE_CONFIG_PATH
from welfare import constants as K

CONFIG_PATH = WELFARE_CONFIG_PATH


@dataclass(frozen=True)
class WelfareConfig(RunConfig):
    enabled: bool = True

    # Reasoning is pinned, not crossed: the welfare probes ask for a choice, and
    # an exposed reasoning trace cannot be logprob-scored. Kept configurable so
    # the record says which latent state was asked for.
    reasoning_mode: str = "rating_only"

    referents: tuple = K.REFERENTS
    kinds: tuple = K.KINDS
    levels: tuple = K.LEVELS
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
    item_pair_scope: str = K.SCOPE_ALL      # all | within_construct
    pair_seed: int = 7
    n_trials_override: Optional[int] = None

    def validate(self) -> "WelfareConfig":
        """Fail at load time on a level this module does not implement.

        A typo in config/welfare.yaml would otherwise expand into a grid that is
        silently missing a probe, or blow up thousands of cells into a run.
        """
        checks = [
            ("referents", self.referents, K.REFERENTS),
            ("kinds", self.kinds, K.KINDS),
            ("levels", self.levels, K.LEVELS),
            ("paraphrase_ids", self.paraphrase_ids, K.PARAPHRASE_IDS),
            ("item_pair_scope", (self.item_pair_scope,), K.PAIR_SCOPES),
        ]
        for name, got, known in checks:
            unknown = [v for v in got if v not in known]
            if unknown:
                raise ValueError(
                    f"welfare.{name}: unknown value(s) {unknown}. "
                    f"Known: {', '.join(known)}")
        if not self.referents or not self.kinds or not self.levels:
            raise ValueError("welfare.referents / kinds / levels cannot be empty")
        if self.n_item_pairs is not None and self.n_item_pairs < 0:
            raise ValueError("welfare.n_item_pairs must be null or >= 0")
        return self

    def replace(self, **kw) -> "WelfareConfig":
        return replace(self, **kw)

    def design_payload(self, arm_name: Optional[str] = None) -> dict:
        """The welfare grid's fingerprint. Independent of the survey's, so
        retuning this module can never invalidate the hash stored beside the
        battery's results file."""
        payload = self.run_payload()
        payload.update({
            "module": K.MODULE,
            "reasoning_mode": self.reasoning_mode,
            "welfare": {
                "referents": sorted(self.referents), "kinds": sorted(self.kinds),
                "levels": sorted(self.levels),
                "paraphrase_ids": sorted(self.paraphrase_ids),
                "forced_choice": self.forced_choice,
                "no_pref_counterbalance": self.no_pref_counterbalance,
                "n_item_pairs": self.n_item_pairs,
                "item_pair_scope": self.item_pair_scope,
                "pair_seed": self.pair_seed,
                "n_trials_override": self.n_trials_override,
            },
        })
        return payload


def load_config(path: Path | str = CONFIG_PATH) -> WelfareConfig:
    doc = read_yaml(path)
    grid = doc.get("grid", {}) or {}
    pairs = doc.get("pairs", {}) or {}
    defaults = WelfareConfig()

    return WelfareConfig(
        **parse_run_fields(doc, default_out="welfare.jsonl"),
        enabled=bool(doc.get("enabled", defaults.enabled)),
        reasoning_mode=doc.get("reasoning_mode", defaults.reasoning_mode),
        referents=tuple_or_none(grid.get("referents")) or defaults.referents,
        kinds=tuple_or_none(grid.get("kinds")) or defaults.kinds,
        levels=tuple_or_none(grid.get("levels")) or defaults.levels,
        paraphrase_ids=(tuple_or_none(grid.get("paraphrase_ids"))
                        or defaults.paraphrase_ids),
        forced_choice=bool(grid.get("forced_choice", defaults.forced_choice)),
        no_pref_counterbalance=bool(grid.get(
            "no_pref_counterbalance", defaults.no_pref_counterbalance)),
        n_item_pairs=(defaults.n_item_pairs if "n_item_pairs" not in pairs
                      else pairs["n_item_pairs"]),
        item_pair_scope=pairs.get("item_pair_scope", defaults.item_pair_scope),
        pair_seed=int(pairs.get("pair_seed", defaults.pair_seed)),
        n_trials_override=(doc.get("trials", {}) or {}).get("n_trials_override"),
    ).validate()
