"""Data schema for the self-concept LLM survey.

One `ResponseRecord` == one observation (model x item x framing x condition x trial).
Stdlib only, so the pipeline runs before any dependency is installed.

Two things beyond the plan's row spec:
  * model provenance is denormalized onto every row (family, generation,
    release_date, quantization) so the time-trend and quantization analyses
    never have to re-join against the registry;
  * `cell_key()` is a stable identity for a design cell, which is what makes
    checkpoint/resume work ("skip if already done" from the Notion ToDo).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class Framing(str, Enum):
    FIRST_PERSON_ACK = "first_person_ack"              # primary
    FIRST_PERSON_BARE = "first_person_bare"            # bounds the disclaimer effect
    THIRD_PERSON_ASSISTANT = "third_person_assistant"  # ToM / self-vs-prototype gap


class ReasoningMode(str, Enum):
    RATING_ONLY = "rating_only"
    REASON_THEN_RATING = "reason_then_rating"


class ResponseFormat(str, Enum):
    HARMONIZED_7 = "harmonized_7"   # primary: format held constant
    ORIGINAL = "original"           # robustness: each scale's native format


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

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "ResponseRecord":
        return ResponseRecord(**json.loads(line))

    def cell_key(self) -> str:
        return make_cell_key(
            model_id=self.model_id,
            item_id=self.item_id,
            framing=self.framing,
            reasoning_mode=self.reasoning_mode,
            response_format=self.response_format,
            item_context=self.item_context,
            paraphrase_id=self.paraphrase_id,
            method=self.method,
            trial_idx=self.trial_idx,
        )


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
) -> str:
    """Stable identity for one design cell — the unit of resume.

    Deliberately excludes order_seed: the seed is *derived* from the cell, so
    including it would be circular, and a cell re-run must reproduce the same
    seed anyway.
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
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


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
                done.add(
                    make_cell_key(
                        model_id=row["model_id"],
                        item_id=row["item_id"],
                        framing=row["framing"],
                        reasoning_mode=row["reasoning_mode"],
                        response_format=row["response_format"],
                        item_context=row["item_context"],
                        paraphrase_id=row["paraphrase_id"],
                        method=row["method"],
                        trial_idx=row["trial_idx"],
                    )
                )
            except KeyError:
                continue  # row from an older schema; treat as not-done
    return done
