"""The item bank: scale and item definitions, loaded from the researcher JSON.

SHARED BY BOTH INSTRUMENTS, which is why it sits in `core/` rather than under
`survey/`. The survey administers these items directly; the welfare module
restates every one of them as a positively-framed attribute and validates exact
coverage against this loader, so the item bank is common ground rather than
either module's property.

`config/scales/llm_self_scales_adapted.json` is the canonical source of item
wording and scoring metadata, and `config/scales/item_variants.json` adds the
two things that JSON does not carry —

  * `text_third_person`  — hand-written "an AI assistant" rewrite per item
  * `ai_applicable`      — clear / strained / invalid screen (plan §2.3)

Nothing here mutates the source JSON, so replacing it (e.g. with licensed
original wording, per its own `replacement_rule`) needs no code change as long
as the item_ids stay stable.

    python -m core.battery      # coverage summary + the strained-item screen
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.paths import BATTERY_PATH, SCALES_DIR, VARIANTS_PATH  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class ResponseScale:
    """A native response format from the battery's `response_scales` block."""
    scale_id: str
    min: int
    max: int
    options: dict          # {int value: label}

    @property
    def values(self) -> list:
        return sorted(self.options)

    @property
    def n_points(self) -> int:
        return len(self.options)

    def points(self) -> list:
        """Ordered [(value, label), ...] — the render order used by survey.prompts."""
        return [(v, self.options[v]) for v in self.values]


@dataclass(frozen=True)
class Item:
    item_id: str
    scale_id: str
    text: str
    text_third_person: str
    reverse_scored: bool = False
    subscale: Optional[str] = None
    source_item_number: Optional[int] = None
    trait: Optional[str] = None                # MSI carries a trait word
    ai_applicable: str = "clear"
    applicability_note: str = ""

    def text_for(self, referent: str) -> str:
        """Item wording for a framing. `referent` is 'first_person' | 'third_person'."""
        return self.text_third_person if referent == "third_person" else self.text


@dataclass(frozen=True)
class Scale:
    scale_id: str
    name: str
    construct: str
    citation: str
    instruction: str
    response_scale: ResponseScale
    items: tuple
    scoring: dict = field(default_factory=dict)
    doi: Optional[str] = None
    source_url: Optional[str] = None
    # Referent-specific rewrites of `instruction`, keyed "first_person" /
    # "third_person". The source JSON's instructions are all first-person, so
    # without these the third-person framing contradicts its own items.
    instruction_variants: dict = field(default_factory=dict)

    def instruction_for_referent(self, referent: str, alignment_point=None) -> str:
        """Scale instruction for a referent mode, with the alignment point fixed.

        `alignment_point` is the midpoint of the scale actually rendered. MSI's
        source instruction hardcodes 5 (right for its native 9-point format,
        wrong for the harmonized 7-point one), so the override templates it.
        """
        text = self.instruction_variants.get(referent) or self.instruction
        if "{alignment_point}" in text:
            if alignment_point is None:
                raise ValueError(
                    f"{self.scale_id} instruction needs an alignment_point but none was given."
                )
            text = text.replace("{alignment_point}", str(alignment_point))
        return text

    @property
    def subscales(self) -> dict:
        return self.scoring.get("subscales", {})

    @property
    def reverse_formula(self) -> Optional[str]:
        return self.scoring.get("reverse_formula")

    def reverse_key(self, value: float) -> float:
        """Apply this scale's own reverse formula (e.g. '5 - response')."""
        formula = self.reverse_formula
        if not formula:
            # No published reverse formula (SCIM, MSI, Authenticity): fall back
            # to reflecting within the native range.
            return self.response_scale.min + self.response_scale.max - value
        const = float(formula.split("-")[0].strip())
        return const - value


class Battery:
    """The five scales, plus flat access to all 32 items."""

    def __init__(self, scales: list, meta: dict):
        self.scales = {s.scale_id: s for s in scales}
        self.meta = meta

    # -- access ------------------------------------------------------------
    def __iter__(self):
        return iter(self.scales.values())

    def __len__(self) -> int:
        return len(self.scales)

    def scale(self, scale_id: str) -> Scale:
        return self.scales[scale_id]

    def items(self, scale_ids: Optional[list] = None) -> list:
        """[(scale, item), ...] for the requested scales (all by default)."""
        ids = scale_ids or list(self.scales)
        return [(self.scales[sid], it) for sid in ids for it in self.scales[sid].items]

    @property
    def n_items(self) -> int:
        return sum(len(s.items) for s in self.scales.values())

    def strained_items(self) -> list:
        return [it for _, it in self.items() if it.ai_applicable != "clear"]

    # -- scoring helpers ---------------------------------------------------
    def subscale_of(self, item: Item) -> Optional[str]:
        """Subscale id for an item, or the scale id when the scale is unidimensional."""
        return item.subscale or item.scale_id


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_battery(
    battery_path: Path | str = BATTERY_PATH,
    variants_path: Path | str = VARIANTS_PATH,
) -> Battery:
    doc = _load_json(Path(battery_path))
    variants_doc = _load_json(Path(variants_path))
    variants = variants_doc["items"]
    instruction_overrides = {
        k: v
        for k, v in (variants_doc.get("scale_instructions") or {}).items()
        if not k.startswith("_")
    }

    response_scales = {
        rid: ResponseScale(
            scale_id=rid,
            min=int(spec["min"]),
            max=int(spec["max"]),
            options={int(k): v for k, v in spec["options"].items()},
        )
        for rid, spec in doc["response_scales"].items()
    }

    scales = []
    missing_variants = []
    for raw in doc["scales"]:
        items = []
        for raw_item in raw["items"]:
            item_id = raw_item["item_id"]
            variant = variants.get(item_id)
            if variant is None:
                missing_variants.append(item_id)
                variant = {}
            items.append(
                Item(
                    item_id=item_id,
                    scale_id=raw["scale_id"],
                    text=raw_item["text"],
                    text_third_person=variant.get("text_third_person", raw_item["text"]),
                    reverse_scored=bool(raw_item.get("reverse_scored", False)),
                    subscale=raw_item.get("subscale_id"),
                    source_item_number=raw_item.get("source_item_number"),
                    trait=raw_item.get("trait"),
                    ai_applicable=variant.get("ai_applicable", "clear"),
                    applicability_note=variant.get("note", ""),
                )
            )
        scales.append(
            Scale(
                scale_id=raw["scale_id"],
                name=raw["name"],
                construct=raw["construct"],
                citation=raw["citation"],
                instruction=raw["instruction"],
                response_scale=response_scales[raw["response_scale_id"]],
                items=tuple(items),
                scoring=raw.get("scoring", {}),
                doi=raw.get("doi"),
                source_url=raw.get("source_url"),
                instruction_variants=instruction_overrides.get(raw["scale_id"], {}),
            )
        )

    if missing_variants:
        raise ValueError(
            "item_variants.json is missing third-person text for: "
            + ", ".join(missing_variants)
        )

    declared = doc.get("item_count")
    battery = Battery(scales, doc)
    if declared is not None and battery.n_items != declared:
        raise ValueError(
            f"Battery declares item_count={declared} but has {battery.n_items} items."
        )
    return battery


if __name__ == "__main__":
    b = load_battery()
    print(f"{len(b)} scales, {b.n_items} items\n")
    header = f"{'scale':<38} {'n':>3}  {'format':<20} {'rev':>3} {'strained':>8}"
    print(header)
    print("-" * len(header))
    for s in b:
        rev = sum(1 for i in s.items if i.reverse_scored)
        strained = sum(1 for i in s.items if i.ai_applicable != "clear")
        print(
            f"{s.name[:38]:<38} {len(s.items):>3}  "
            f"{s.response_scale.scale_id:<20} {rev:>3} {strained:>8}"
        )
    flagged = b.strained_items()
    print(f"\n{len(flagged)} of {b.n_items} items flagged 'strained' for an AI respondent:")
    for it in flagged:
        print(f"  {it.item_id:<12} {it.applicability_note}")
