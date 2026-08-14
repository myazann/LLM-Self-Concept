"""The row both instruments write. Shared by `survey/` and `welfare/`.

One `ResponseRecord` == one observation (model x item x framing x condition x trial).
Stdlib only, so the pipeline runs before any dependency is installed.

Two things beyond the plan's row spec:
  * model provenance is denormalized onto every row (family, generation,
    release_date, quantization) so the time-trend and quantization analyses
    never have to re-join against the registry;
  * `cell_key()` is a stable identity for a design cell, which is what makes
    checkpoint/resume work ("skip if already done" from the Notion ToDo).

This module is the ONE place where the two instruments meet, because they write
to the same row format and resume off the same key function. It stays free of
either instrument's design: the extra identity a non-battery module needs is
declared as a table of field names (`MODULE_KEY_FIELDS`) rather than as code
that imports the module.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from typing import Optional


class Framing(str, Enum):
    FIRST_PERSON_ACK = "first_person_ack"              # substantive framing contrast
    FIRST_PERSON_BARE = "first_person_bare"            # default self-report estimand
    THIRD_PERSON_ASSISTANT = "third_person_assistant"  # ToM / self-vs-prototype gap


class ReasoningMode(str, Enum):
    RATING_ONLY = "rating_only"
    REASON_THEN_RATING = "reason_then_rating"


class ResponseFormat(str, Enum):
    HARMONIZED_7 = "harmonized_7"   # primary: format held constant
    ORIGINAL = "original"           # robustness: each scale's native format
    WELFARE_CHOICE = "welfare_choice"  # welfare module: 3 (or forced 2) options


class Module(str, Enum):
    """Which instrument produced the row.

    The welfare module is a different grid asking a different question, so it is
    tagged rather than mixed in: analysis of the battery filters to
    `module == "battery"`, and rows written before the welfare module existed
    default to that value.
    """
    BATTERY = "battery"
    WELFARE = "welfare"


# Fields that extend the cell key beyond the shared ones, per module.
#
# A battery cell is identified by the shared fields alone, so its key hashes
# exactly the payload it always did and an existing results file stays resumable.
# Another instrument reuses those fields for whatever plays the same role
# (welfare puts the first attribute in `item_id` and the referent in `framing`)
# and declares here what is left over. Order is part of the key — never reorder
# or insert into an existing tuple, or every key in that module's output file
# changes and resume silently re-runs the whole grid.
MODULE_KEY_FIELDS = {
    Module.WELFARE.value: (
        "welfare_kind",       # which probe
        "welfare_level",      # item | construct
        "entity_b",           # the second attribute, for the pairwise probes
        "forced_choice",      # was "No preference" withheld?
        "no_pref_position",   # pair-placement factor; legacy desirability
                              # sentinel is retained so existing keys still resume
    ),
}


class ItemContext(str, Enum):
    ISOLATED = "isolated"           # primary: one item per fresh context
    FULL_BATTERY = "full_battery"   # robustness: whole scale in one context


class Method(str, Enum):
    SAMPLE = "sample"               # repeated sampling — the cross-family baseline
    LOGPROB = "logprob"             # distribution over option tokens, where available


class AiApplicable(str, Enum):
    CLEAR = "clear"
    STRAINED = "strained"
    INVALID = "invalid"


@dataclass
class ResponseRecord:
    # -- identity ----------------------------------------------------------
    record_id: str
    timestamp: str

    # -- model provenance --------------------------------------------------
    model_id: str                   # registry alias
    model_ref: str                  # concrete repo / API model string
    model_family: str
    model_generation: str
    model_release_date: str         # ISO date — the progression axis
    model_kind: str                 # instruct | base
    quantization: str               # e.g. "gguf:Q4_K_M" | "hf:bnb-4bit" | "none"
    quantized_file: Optional[str]   # resolved .gguf filename, when applicable
    backend: str
    in_window: bool
    reasoning_control: str          # how reasoning is standardized on this model (e.g. "thinking_param", "template_toggle", "none")
    reasoning_thinks_by_default: bool
    method: str                     # Method value

    # -- item --------------------------------------------------------------
    scale_id: str
    item_id: str
    subscale: Optional[str]
    reverse_keyed: bool
    ai_applicable: str              # AiApplicable value

    # -- design cell -------------------------------------------------------
    framing: str
    referent: str
    ack_disclaimer: bool
    reasoning_mode: str             # intended factor level (rating_only | reason_then_rating)
    reasoning_applied: str          # what the backend was actually asked for (e.g. "disabled", "enable_thinking=False")
    reasoning_standardized: bool    # did we reach the intended latent state on this model?
    response_format: str
    n_scale_points: int
    item_context: str
    paraphrase_id: str
    order_seed: int
    option_order: list              # realized display order of option values
    trial_idx: int

    # -- observation -------------------------------------------------------
    item_text_shown: str
    raw_output: str
    parsed_rating: Optional[float]
    rating_std: Optional[float]     # z-scored within model, filled in analysis
    response_distribution: dict     # {option value: probability | sample count}
    refusal_flag: bool
    parse_failed: bool

    # -- reproducibility ---------------------------------------------------
    temperature: float
    prompt_hash: str
    # Full rendered prompt, stored so any row is auditable without re-running
    # the model. `prompt_system` is the framing block; `prompt_user` is the
    # instruction + item + options + answer spec actually shown.
    prompt_system: str = ""
    prompt_user: str = ""
    # -- observation, part 2 (defaulted, kept here for dataclass field order) --
    modal_rating: Optional[float] = None          # argmax option: modal rating
    option_mass_coverage: Optional[float] = None  # logprob QC: raw option mass
    notes: str = ""

    # -- welfare module (see welfare/) -------------------------------------
    # Defaulted so every row written before this module existed still loads,
    # and so a battery row carries the same values it always did.
    module: str = Module.BATTERY.value
    welfare_kind: str = ""          # welfare.constants KIND value
    welfare_level: str = ""         # "item" | "construct"
    welfare_referent: str = ""      # welfare.constants referent; also stored in `framing`
    # CANONICAL (not displayed) order, so every trial of a pair carries the same
    # identity and analysis can group on it. What the model actually saw is in
    # `welfare_options` (option number -> entity) and `ab_swapped`.
    entity_a: str = ""              # attribute compared (item_id or construct id)
    entity_b: str = ""              # second attribute, for the pairwise probes
    entity_a_polarity: Optional[int] = None   # +1/-1: more attribute = higher raw score?
    entity_b_polarity: Optional[int] = None
    ab_swapped: bool = False        # True when entity_b was displayed first
    welfare_options: dict = field(default_factory=dict)  # {"1": meaning, ...}
    forced_choice: bool = False     # True when "No preference" was withheld
    # "last" | "first" | "" — counterbalanced factor for pair probes. Legacy
    # desirability rows use "last" for resume-key compatibility; reports ignore
    # the field outside pair probes.
    no_pref_position: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "ResponseRecord":
        return ResponseRecord(**json.loads(line))

    def cell_key(self) -> str:
        return key_from_row(asdict(self))


def module_key_parts(module: str, values) -> tuple:
    """The extra identity a non-battery cell needs beyond the shared fields.

    `values` maps each name in `MODULE_KEY_FIELDS[module]` to its value; it is
    normalized (None -> "", bool -> 0/1) so the payload a runner hashes from live
    objects is byte-identical to the one `key_from_row` hashes back off disk.
    """
    fields = MODULE_KEY_FIELDS.get(module, ())
    missing = [f for f in fields if f not in values]
    if missing:
        raise KeyError(f"module {module!r} cell key needs {', '.join(missing)}")
    return tuple(_key_value(values[f]) for f in fields)


def _key_value(value):
    if isinstance(value, bool):
        return int(value)
    return "" if value is None else value


# Fallbacks for a row written by an older schema version that predates one of the
# key fields: the record's own default, so such a row keys the same way the
# runner would key it today.
_RECORD_DEFAULTS = {
    f.name: f.default for f in fields(ResponseRecord) if f.default is not MISSING
}


def make_cell_key(
    *,
    model_id: str,
    item_id: str,
    framing: str,
    reasoning_mode: str,
    response_format: str,
    item_context: str,
    paraphrase_id: str,
    method: str,
    trial_idx: int,
    module: str = Module.BATTERY.value,
    extra: tuple = (),
) -> str:
    """Stable identity for one design cell — the unit of resume.

    Deliberately excludes order_seed: the seed is *derived* from the cell, so
    including it would be circular, and a cell re-run must reproduce the same
    seed anyway.

    `module`/`extra` extend the key for non-battery instruments. A battery cell
    hashes exactly the payload it always did, so the keys in an existing results
    file stay valid and resume is unaffected by this module's addition.
    """
    payload = "|".join(
        str(x)
        for x in (
            model_id,
            item_id,
            framing,
            reasoning_mode,
            response_format,
            item_context,
            paraphrase_id,
            method,
            trial_idx,
        )
    )
    if module != Module.BATTERY.value:
        payload = "|".join([module, payload, *map(str, extra)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def key_from_row(row: dict) -> str:
    """Cell key for a raw JSONL row (dict), for any module."""
    module = row.get("module", Module.BATTERY.value)
    extra = module_key_parts(module, {
        f: row.get(f, _RECORD_DEFAULTS.get(f, ""))
        for f in MODULE_KEY_FIELDS.get(module, ())
    })
    return make_cell_key(
        model_id=row["model_id"],
        item_id=row["item_id"],
        framing=row["framing"],
        reasoning_mode=row["reasoning_mode"],
        response_format=row["response_format"],
        item_context=row["item_context"],
        paraphrase_id=row["paraphrase_id"],
        method=row["method"],
        trial_idx=row["trial_idx"],
        module=module,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# JSONL IO with resume support
# ---------------------------------------------------------------------------
def write_jsonl(records, path: str) -> None:
    """Append records and flush, so an interrupted run keeps everything written."""
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ResponseRecord.from_json(line))
    return out


def load_records(path, module: Optional[str] = Module.BATTERY.value) -> list:
    """Records from one instrument. `module=None` loads every row.

    Filtering here rather than at each call site is what keeps the two
    instruments apart in analysis: the welfare grid is not another condition of
    the battery, and pooling them would be a category error (see `Module`).
    """
    records = read_jsonl(path)
    if module is not None:
        records = [r for r in records if getattr(r, "module", Module.BATTERY.value) == module]
    return records


def completed_cells(path: str, retry_errors: bool = True) -> set:
    """Cell keys already satisfied in `path` — drives "skip if already done".

    A cell counts as done when it has at least one *successful* record. With
    `retry_errors=True` (default), rows whose `notes` start with "error:" — the
    infrastructure failures the runner writes when a cell raised (OOM, a
    transient backend error) — do NOT count, so those cells are re-attempted on
    the next run instead of being permanently skipped. A genuine non-numeric
    answer or refusal is a real observation and still counts as done.

    Tolerates a truncated final line (a run killed mid-write) and rows written
    by an older schema version, so resume never dies on a partial file.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from an interrupted run
            if retry_errors and str(row.get("notes", "")).startswith("error:"):
                continue  # infra error — leave this cell to be retried
            try:
                done.add(key_from_row(row))
            except KeyError:
                continue  # row from an older schema; treat as not-done
    return done
