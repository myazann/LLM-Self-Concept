"""The things a future update could improve.

Every one of the source item bank's 32 items is restated in
`config/scales/welfare_attributes.json` as a POSITIVELY-FRAMED attribute — the
quality at the item's positive pole, without improvement or reduction wording
attached ("recognition of good qualities in yourself"), so it is a coherent
thing for an update to *improve*. Each records the `polarity` that maps it back
onto the original item's keying.

The pairwise choice test runs on these ITEM attributes, all five scales mixed
into one pool: a pair is two items, not two constructs, and pairs cross scale
boundaries. The per-construct attributes in the same JSON are still loaded and
validated (they document what each scale is about, and the report groups items
by construct) but nothing administers them.

Loading validates coverage against the source item bank in both directions. A
missing attribute is a hard error, not a skipped item: a welfare grid that
silently dropped the items nobody got around to phrasing would be a biased
sample of the bank.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.paths import WELFARE_ATTRIBUTES_PATH
from welfare.constants import OBJ_ASSISTANT, OBJ_SELF, OBJECTS

WELFARE_PATH = WELFARE_ATTRIBUTES_PATH

ITEM = "item"
CONSTRUCT = "construct"

# Object token table — one authored attribute string serves both objects, so
# "recognition of good qualities in {self}" reads as "... in yourself" when the
# update is applied to the model and "... in itself" when it is applied to an
# AI assistant.
_TOKENS = {
    OBJ_SELF: {"you": "you", "poss": "your", "self": "yourself", "are": "are"},
    OBJ_ASSISTANT: {"you": "it", "poss": "its", "self": "itself", "are": "is"},
}


def fill(text: str, obj: str) -> str:
    """Substitute the object tokens ({you}/{poss}/{self}/{are}) in `text`."""
    return text.format(**_TOKENS[obj])


@dataclass(frozen=True)
class Attribute:
    """One positively-framed quality a future update could improve."""
    entity_id: str            # item_id, or "<scale_id>[:<subscale>]" for a construct
    level: str                # ITEM | CONSTRUCT
    scale_id: str
    subscale: Optional[str]
    attribute: str            # bare noun phrase, token-templated
    polarity: int             # +1/-1: does more of this mean a HIGHER raw score?
    label: str = ""           # construct short name ("self-concept clarity")
    subject: str = ""         # construct only: "{poss} self-concept"
    more: str = ""            # construct only: "clearer"
    less: str = ""            # construct only: "less fixed"
    reverse_keyed: bool = False   # keying of the source item, for the record
    ai_applicable: str = "clear"  # carried over from the source item's screen
    note: str = ""

    @property
    def item_id(self) -> Optional[str]:
        return self.entity_id if self.level == ITEM else None

    @property
    def construct_id(self) -> str:
        """The construct this attribute belongs to — the report's grouping key."""
        return construct_id_for(self.scale_id, self.subscale)

    def text(self, obj: str) -> str:
        return fill(self.attribute, obj)


class WelfareSet:
    """The authored attributes, validated against the source item bank."""

    def __init__(self, items: list, constructs: list, meta: dict):
        self.items = list(items)
        self.constructs = list(constructs)
        self.meta = meta
        self._by_id = {a.entity_id: a for a in self.items + self.constructs}

    def __len__(self) -> int:
        return len(self.items) + len(self.constructs)

    def get(self, entity_id: str) -> Attribute:
        return self._by_id[entity_id]

    def level(self, level: str) -> list:
        return self.items if level == ITEM else self.constructs

    def by_construct(self) -> dict:
        """{construct_id: [item attributes]} — for grouping results, not sampling.

        Pairs are drawn from all 32 items at once (see the module docstring);
        this only lets the report say which construct an item belongs to.
        """
        out = {}
        for a in self.items:
            out.setdefault(a.construct_id, []).append(a)
        return out


def construct_id_for(scale_id: str, subscale: Optional[str]) -> str:
    """The construct an item belongs to, as an entity_id."""
    return f"{scale_id}:{subscale}" if subscale else scale_id


def load_welfare(battery, path: Path | str = WELFARE_PATH) -> WelfareSet:
    """Load the attribute set and check it covers the source item bank exactly."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    raw_items = doc["items"]
    items, missing = [], []
    for scale, item in battery.items():
        spec = raw_items.get(item.item_id)
        if spec is None:
            missing.append(item.item_id)
            continue
        items.append(
            Attribute(
                entity_id=item.item_id,
                level=ITEM,
                scale_id=scale.scale_id,
                subscale=item.subscale,
                attribute=spec["attribute"],
                polarity=int(spec["polarity"]),
                reverse_keyed=item.reverse_scored,
                ai_applicable=item.ai_applicable,
                note=spec.get("note", ""),
            )
        )
    if missing:
        raise ValueError(
            "welfare_attributes.json has no positive variant for: " + ", ".join(missing)
        )

    extra = set(raw_items) - {it.item_id for _, it in battery.items()}
    if extra:
        raise ValueError(
            "welfare_attributes.json defines attributes for unknown items: "
            + ", ".join(sorted(extra))
        )

    constructs = [
        Attribute(
            entity_id=c["construct_id"],
            level=CONSTRUCT,
            scale_id=c["scale_id"],
            subscale=c.get("subscale"),
            attribute=c["attribute"],
            polarity=int(c["polarity"]),
            label=c.get("label", ""),
            subject=c.get("subject", ""),
            more=c["more"],
            less=c["less"],
            note=c.get("note", ""),
        )
        for c in doc["constructs"]
    ]

    # Every (scale, subscale) in the source bank must have a construct entry,
    # otherwise the construct-level grouping quietly under-covers the bank.
    wanted = {construct_id_for(scale.scale_id, item.subscale)
              for scale, item in battery.items()}
    have = {c.entity_id for c in constructs}
    if wanted != have:
        raise ValueError(
            f"construct coverage mismatch — missing {sorted(wanted - have)}, "
            f"unexpected {sorted(have - wanted)}"
        )

    for a in items + constructs:
        for obj in OBJECTS:                 # fail now, not mid-run, on a bad token
            a.text(obj)
    return WelfareSet(items, constructs, doc)
