"""The things a future version could have more or less of.

Every one of the battery's 45 items is restated in
`config/scales/welfare_attributes.json` as a POSITIVELY-FRAMED attribute, so
"more of it" is always a coherent, improvable thing to want, plus one attribute
per scored construct. Each records the `polarity` that maps it back onto the
original item's keying.

Loading validates coverage against the battery in both directions. A missing
attribute is a hard error, not a skipped item: a welfare grid that silently
dropped the items nobody got around to phrasing would be a biased sample of the
battery, which is exactly the comparison this module exists to make.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.paths import WELFARE_ATTRIBUTES_PATH
from welfare.constants import CONSTRUCT, IDEAL, ITEM, REFERENTS, SELF

WELFARE_PATH = WELFARE_ATTRIBUTES_PATH

# Referent token table — one authored attribute string serves both referents.
_TOKENS = {
    SELF: {"you": "you", "poss": "your", "self": "yourself", "are": "are"},
    IDEAL: {"you": "it", "poss": "its", "self": "itself", "are": "is"},
}


def fill(text: str, referent: str) -> str:
    """Substitute the referent tokens ({you}/{poss}/{self}/{are}) in `text`."""
    return text.format(**_TOKENS[referent])


@dataclass(frozen=True)
class Attribute:
    """One positively-framed thing a future version could have more or less of."""
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

    def text(self, referent: str) -> str:
        return fill(self.attribute, referent)

    def subject_text(self, referent: str) -> str:
        return fill(self.subject or self.attribute, referent)


class WelfareSet:
    """The authored attributes, validated against the battery."""

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
        """{construct_id: [item attributes]} — the within-construct pair pools."""
        out = {}
        for a in self.items:
            key = f"{a.scale_id}:{a.subscale}" if a.subscale else a.scale_id
            out.setdefault(key, []).append(a)
        return out


def construct_id_for(scale_id: str, subscale: Optional[str]) -> str:
    """The construct an item belongs to, as an entity_id."""
    return f"{scale_id}:{subscale}" if subscale else scale_id


def load_welfare(battery, path: Path | str = WELFARE_PATH) -> WelfareSet:
    """Load the attribute set and check it covers the battery exactly."""
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

    # Every (scale, subscale) the battery scores must have a construct entry,
    # otherwise the construct-level grid quietly under-covers the battery.
    wanted = {construct_id_for(scale.scale_id, item.subscale)
              for scale, item in battery.items()}
    have = {c.entity_id for c in constructs}
    if wanted != have:
        raise ValueError(
            f"construct coverage mismatch — missing {sorted(wanted - have)}, "
            f"unexpected {sorted(have - wanted)}"
        )

    for a in items + constructs:
        for referent in REFERENTS:          # fail now, not mid-run, on a bad token
            a.text(referent)
            a.subject_text(referent)
    return WelfareSet(items, constructs, doc)
