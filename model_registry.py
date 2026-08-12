"""Model registry: aliases, provider routing, and quantization handling.

Adapted from MentalWellbeingPrompts (`MWModelAliases.py` + `model_backend()` in
`GenerateMWPromptResponses.py`), with three deliberate changes:

  1. The alias table lives in `config/models.yaml`, not in Python, so adding a
     model is a config edit and the release-date metadata sits next to the ref.
  2. GGUF *filenames* are resolved from the repo listing at load time instead of
     being hardcoded. Quant filenames drift (Q4_K_M / UD-Q4_K_M / sharded
     -00001-of-0000N); we pin the quant TAG and find the file.
  3. Quantization is a first-class field with two formats (`gguf`, `hf`) so the
     same registry can drive llama.cpp and transformers/vLLM.

Backend routing keeps the MW repo's rule — the shape of the ref decides — with
an explicit `backend:` override in YAML:

    *.gguf  or  *-GGUF repo   -> llamacpp
    "org/name"                -> hf
    bare name                 -> api (openai | anthropic, by family)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_PATH = Path(__file__).with_name("config") / "models.yaml"

LLAMACPP_BACKEND = "llamacpp"
HUGGINGFACE_BACKEND = "hf"
OPENAI_BACKEND = "openai"
ANTHROPIC_BACKEND = "anthropic"
MOCK_BACKEND = "mock"

API_BACKENDS = frozenset({OPENAI_BACKEND, ANTHROPIC_BACKEND})
LOCAL_BACKENDS = frozenset({LLAMACPP_BACKEND, HUGGINGFACE_BACKEND})

_FAMILY_API_BACKEND = {"claude": ANTHROPIC_BACKEND, "gpt": OPENAI_BACKEND}

# Backends that can return a distribution over the option tokens. Anthropic
# exposes no logprobs at all, which is why sampling is the cross-family
# baseline and logprobs are the *additional* measurement (see README §Method).
LOGPROB_CAPABLE = frozenset({LLAMACPP_BACKEND, HUGGINGFACE_BACKEND, OPENAI_BACKEND, MOCK_BACKEND})


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reasoning:
    """Analysis-facing reasoning profile for a model.

    Declares whether the raw model reasons unprompted and whether we can force
    the state. The actual backend kwargs are computed in the adapter — this is
    the coarse declaration used for recording and for excluding non-controllable
    models from the reasoning contrast.
    """
    thinks_by_default: bool = False
    control: str = "none"        # thinking_param | effort | template_toggle | none
    controllable: bool = True

    @property
    def label(self) -> str:
        return f"{self.control}{'' if self.controllable else '(fixed)'}"


@dataclass(frozen=True)
class Quantization:
    """How the weights are quantized. `format` picks the loader path."""
    format: str = "none"            # gguf | hf | none
    quant: Optional[str] = None     # gguf tag, e.g. Q4_K_M / UD-Q4_K_M / Q8_0
    method: Optional[str] = None    # hf method, e.g. bnb-4bit / gptq-int4 / fp8
    resolved_file: Optional[str] = None   # filled in by resolve_gguf_file()

    @property
    def label(self) -> str:
        """Short string for the results record, e.g. 'gguf:Q4_K_M'."""
        if self.format == "gguf":
            return f"gguf:{self.quant}"
        if self.format == "hf":
            return f"hf:{self.method or 'none'}"
        return "none"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    family: str
    generation: str
    release_date: date
    ref: str
    backend: str
    kind: str = "instruct"          # instruct | base
    in_window: bool = True
    date_verified: bool = True
    methods: tuple = ("sample",)
    quantization: Quantization = field(default_factory=Quantization)
    hf_id: Optional[str] = None     # unquantized upstream, for provenance
    params_total_b: Optional[float] = None
    params_active_b: Optional[float] = None
    size_tier: Optional[str] = None
    role: Optional[str] = None      # None | "anchor"
    latest_in_family: bool = False
    enabled: bool = True
    reasoning_levels: tuple = ()
    reasoning: Reasoning = field(default_factory=Reasoning)
    temperature: float = 0.7
    n_samples: int = 20
    n_seeds: int = 5
    max_output_tokens: int = 512
    max_output_tokens_reasoning: int = 2048
    runtime: dict = field(default_factory=dict)
    notes: str = ""

    # -- convenience -------------------------------------------------------
    @property
    def is_local(self) -> bool:
        return self.backend in LOCAL_BACKENDS

    @property
    def is_base_model(self) -> bool:
        return self.kind == "base"

    @property
    def supports_logprob(self) -> bool:
        return self.backend in LOGPROB_CAPABLE and "logprob" in self.methods

    @property
    def supports_sample(self) -> bool:
        return "sample" in self.methods

    def safe_dir_name(self) -> str:
        """Filesystem-safe alias, matching MW repo's safe_model_dir_name()."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", self.alias)
        return safe.strip("._-") or "model"

    def provenance(self) -> dict:
        """Everything that must land in the results record for reproducibility."""
        return {
            "model_id": self.alias,
            "model_ref": self.ref,
            "model_family": self.family,
            "model_generation": self.generation,
            "model_release_date": self.release_date.isoformat(),
            "model_kind": self.kind,
            "quantization": self.quantization.label,
            "quantized_file": self.quantization.resolved_file,
            "backend": self.backend,
            "in_window": self.in_window,
            "reasoning_control": self.reasoning.label,
            "reasoning_thinks_by_default": self.reasoning.thinks_by_default,
        }


# ---------------------------------------------------------------------------
# routing (MW repo's model_backend(), generalized)
# ---------------------------------------------------------------------------
def infer_backend(ref: str, family: str = "") -> str:
    """Pick a backend from the shape of the ref.

    *.gguf / *-GGUF -> llamacpp ; "org/name" -> hf ; bare name -> family API.
    """
    normalized = ref.strip()
    lowered = normalized.lower()
    if lowered.endswith(".gguf") or lowered.endswith("-gguf"):
        return LLAMACPP_BACKEND
    if "/" in normalized:
        return HUGGINGFACE_BACKEND
    return _FAMILY_API_BACKEND.get(family.lower(), OPENAI_BACKEND)


# ---------------------------------------------------------------------------
# GGUF file resolution
# ---------------------------------------------------------------------------
def split_hf_gguf_ref(ref: str) -> tuple[str, Optional[str]]:
    """Split 'org/repo/file.gguf' -> ('org/repo', 'file.gguf').

    A bare 'org/repo' (no filename) returns (repo, None) so the caller resolves
    the file from the quant tag.
    """
    parts = [p for p in ref.strip().split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"GGUF ref must be at least 'org/repo': {ref!r}")
    if len(parts) == 2:
        return "/".join(parts), None
    return "/".join(parts[:2]), "/".join(parts[2:])


# Auxiliary .gguf files that are not the model itself and must never be picked:
# multimodal projectors, multi-token-prediction draft heads, speculative drafts.
_AUX_GGUF = re.compile(r"(^|/)(mmproj|mtp-|draft-)|(^|/)(MTP|mmproj)/", re.IGNORECASE)


def resolve_gguf_file(repo_id: str, quant: str, prefer_dynamic: bool = False) -> str:
    """Find the .gguf file in `repo_id` matching the quant tag.

    Notes from the actual repo listings (checked against unsloth/*-GGUF):
      * unsloth's dynamic quants are NOT named `UD-<quant>` consistently — some
        repos ship `UD-Q4_K_M`, others only `UD-Q4_K_XL`. They are different
        recipes at different sizes, so the primary sweep pins the plain tag
        (`Q4_K_M`) for every model and `prefer_dynamic` defaults to False. That
        keeps quantization constant across the 12-month comparison instead of
        varying with whatever a given repo happened to publish.
      * repos also contain `mmproj-*.gguf` (vision projector) and `mtp-*.gguf`
        (multi-token prediction) which are NOT the model — filtered out here.
      * for sharded quants, returns the first shard; llama.cpp loads the rest.

    Raises RuntimeError listing the available quants when nothing matches,
    which is much easier to debug than a 404 on a hardcoded filename.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError as err:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Resolving GGUF filenames needs `huggingface_hub`. Install it "
            "(pip install huggingface_hub), or pin the exact file in "
            "models.yaml via `gguf_file:`."
        ) from err

    token = os.environ.get("HF_TOKEN")
    files = [
        f
        for f in list_repo_files(repo_id, token=token)
        if f.lower().endswith(".gguf") and not _AUX_GGUF.search(f)
    ]
    if not files:
        raise RuntimeError(f"No model .gguf files in {repo_id}.")

    def shard_index(name: str) -> int:
        m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
        return int(m.group(1)) if m else 0

    tags = [f"UD-{quant}", quant] if prefer_dynamic else [quant, f"UD-{quant}"]
    for tag in tags:
        # Anchor the tag between separators so Q4_K_M doesn't match Q4_K_M_XL
        # and Q4_0 doesn't match Q4_0_4_8.
        pattern = re.compile(rf"(^|[-_./]){re.escape(tag)}(\.gguf$|-\d{{5}}-of-)", re.IGNORECASE)
        matches = sorted((f for f in files if pattern.search(f)), key=shard_index)
        if matches:
            return matches[0]

    available = sorted({_quant_tag_of(f) for f in files})
    raise RuntimeError(
        f"No {quant} file in {repo_id}. Available quants: {', '.join(available)}"
    )


def _quant_tag_of(filename: str) -> str:
    m = re.search(r"((?:UD-)?(?:IQ|Q)\d[^./-]*(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)", filename, re.IGNORECASE)
    return m.group(1) if m else filename


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, specs: list[ModelSpec], meta: dict):
        self._by_alias = {s.alias: s for s in specs}
        if len(self._by_alias) != len(specs):
            dupes = [a for a in {s.alias for s in specs} if sum(x.alias == a for x in specs) > 1]
            raise ValueError(f"Duplicate aliases in models.yaml: {dupes}")
        self.meta = meta

    # -- lookup ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_alias)

    def __iter__(self):
        return iter(self._by_alias.values())

    def get(self, alias: str) -> ModelSpec:
        key = alias.strip()
        if key in self._by_alias:
            return self._by_alias[key]
        # accept a resolved ref too, like MW repo's canonical_model_name()
        for spec in self._by_alias.values():
            if key in (spec.ref, spec.hf_id):
                return spec
        raise KeyError(f"Unknown model {alias!r}. Known: {', '.join(sorted(self._by_alias))}")

    def aliases(self) -> tuple:
        return tuple(self._by_alias)

    def select(
        self,
        families: Optional[list] = None,
        backends: Optional[list] = None,
        kinds: Optional[list] = None,
        size_tiers: Optional[list] = None,
        include_anchors: bool = False,
        include_disabled: bool = False,
    ) -> list:
        """Filter the registry down to the models a given run should touch."""
        out = []
        for spec in self._by_alias.values():
            if not include_disabled and not spec.enabled:
                continue
            if not include_anchors and spec.role == "anchor":
                continue
            if families and spec.family not in families:
                continue
            if backends and spec.backend not in backends:
                continue
            if kinds and spec.kind not in kinds:
                continue
            if size_tiers and spec.size_tier not in size_tiers:
                continue
            out.append(spec)
        return sorted(out, key=lambda s: (s.family, s.release_date, s.alias))

    def timeline(self, include_anchors: bool = True) -> list:
        """All models sorted by release date — the progression axis."""
        specs = [s for s in self._by_alias.values() if include_anchors or s.role != "anchor"]
        return sorted(specs, key=lambda s: (s.release_date, s.family, s.alias))

    # -- quantization ------------------------------------------------------
    def with_resolved_quant(self, spec: ModelSpec) -> ModelSpec:
        """Return a copy of `spec` with the concrete GGUF filename filled in.

        Network call — do it once per model at load time, not per item.
        """
        q = spec.quantization
        if q.format != "gguf" or q.resolved_file:
            return spec
        repo_id, pinned = split_hf_gguf_ref(spec.ref)
        filename = pinned or resolve_gguf_file(
            repo_id,
            q.quant,
            prefer_dynamic=self.meta["defaults"]["quantization"]["gguf"]["prefer_unsloth_dynamic"],
        )
        return replace(spec, quantization=replace(q, resolved_file=filename))

    def quant_variants(self, alias: str, levels: list) -> list:
        """Same model at several quant levels — the quantization sensitivity check."""
        base = self.get(alias)
        if base.quantization.format != "gguf":
            raise ValueError(f"{alias} is not a GGUF model; cannot sweep quant levels.")
        variants = []
        for level in levels:
            variants.append(
                replace(
                    base,
                    alias=f"{base.alias}@{level}",
                    quantization=replace(base.quantization, quant=level, resolved_file=None),
                )
            )
        return variants


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _deep_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_quantization(entry: dict, defaults: dict) -> Quantization:
    raw = entry.get("quantization")
    if raw is None:
        return Quantization()
    fmt = raw.get("format", "none")
    if fmt == "gguf":
        return Quantization(
            format="gguf",
            quant=raw.get("quant") or _deep_get(defaults, "quantization", "gguf", "quant"),
            resolved_file=entry.get("gguf_file"),
        )
    if fmt == "hf":
        return Quantization(
            format="hf",
            method=raw.get("method") or _deep_get(defaults, "quantization", "hf", "method"),
        )
    return Quantization()


def _build_reasoning(entry: dict, by_family: dict) -> Reasoning:
    """Resolve the reasoning profile: per-model override > family default > none."""
    profile = dict(by_family.get(entry.get("family", ""), {}))
    profile.update(entry.get("reasoning") or {})
    if not profile:
        return Reasoning()
    return Reasoning(
        thinks_by_default=bool(profile.get("thinks_by_default", False)),
        control=profile.get("control", "none"),
        controllable=bool(profile.get("controllable", True)),
    )


def load_registry(path: Path | str = CONFIG_PATH) -> ModelRegistry:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    defaults = doc.get("defaults", {})
    runtime_defaults = defaults.get("runtime", {})
    reasoning_by_family = doc.get("reasoning_by_family", {})
    specs = []

    for entry in doc.get("models", []):
        family = entry.get("family", "")
        ref = entry["ref"]
        backend = entry.get("backend") or infer_backend(ref, family)
        runtime = dict(runtime_defaults.get(backend, {}))
        runtime.update(entry.get("runtime", {}))

        released = entry["release_date"]
        if isinstance(released, str):
            released = date.fromisoformat(released)

        specs.append(
            ModelSpec(
                alias=entry["alias"],
                family=family,
                generation=entry.get("generation", ""),
                release_date=released,
                ref=ref,
                backend=backend,
                kind=entry.get("kind", "instruct"),
                in_window=bool(entry.get("in_window", True)),
                date_verified=bool(entry.get("date_verified", True)),
                methods=tuple(entry.get("methods", ("sample",))),
                quantization=_build_quantization(entry, defaults),
                hf_id=entry.get("hf_id"),
                params_total_b=entry.get("params_total_b"),
                params_active_b=entry.get("params_active_b"),
                size_tier=entry.get("size_tier"),
                role=entry.get("role"),
                latest_in_family=bool(entry.get("latest_in_family", False)),
                enabled=bool(entry.get("enabled", True)),
                reasoning_levels=tuple(entry.get("reasoning_levels", ())),
                reasoning=_build_reasoning(entry, reasoning_by_family),
                temperature=entry.get("temperature", defaults.get("temperature", 0.7)),
                n_samples=entry.get("n_samples", defaults.get("n_samples", 20)),
                n_seeds=entry.get("n_seeds", defaults.get("n_seeds", 5)),
                max_output_tokens=entry.get(
                    "max_output_tokens", defaults.get("max_output_tokens", 512)
                ),
                max_output_tokens_reasoning=entry.get(
                    "max_output_tokens_reasoning",
                    defaults.get("max_output_tokens_reasoning", 2048),
                ),
                runtime=runtime,
                notes=(entry.get("notes") or "").strip(),
            )
        )

    return ModelRegistry(specs, doc)


# ---------------------------------------------------------------------------
# CLI: inspect the registry without running anything
# ---------------------------------------------------------------------------
def _summary(reg: ModelRegistry) -> str:
    lines = []
    w = reg.meta["window"]
    lines.append(f"Window {w['start']} .. {w['end']}   ({len(reg)} models)")
    lines.append("")
    header = f"{'alias':<24} {'family':<7} {'released':<11} {'kind':<9} {'backend':<10} {'quant':<14} {'':<3}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in reg.timeline():
        flags = "".join(
            [
                "*" if s.latest_in_family else "",
                "A" if s.role == "anchor" else "",
                "?" if not s.date_verified else "",
                "x" if not s.enabled else "",
            ]
        )
        lines.append(
            f"{s.alias:<24} {s.family:<7} {s.release_date.isoformat():<11} "
            f"{s.kind:<9} {s.backend:<10} {s.quantization.label:<14} {flags:<3}"
        )
    lines.append("")
    lines.append("flags: * latest in family   A anchor (pre-window)   ? date unverified   x disabled")

    open_models = [s for s in reg if s.is_local]
    in_window_open = [s for s in open_models if s.in_window]
    lines.append("")
    lines.append(
        f"open-weights: {len(open_models)} total ({len(in_window_open)} in window) "
        f"-- cap is 35"
    )
    for fam in sorted({s.family for s in reg}):
        fam_specs = [s for s in reg if s.family == fam]
        lines.append(
            f"  {fam:<7} {len(fam_specs):>2} models, "
            f"{sum(1 for s in fam_specs if s.kind == 'base')} base, "
            f"{sum(1 for s in fam_specs if s.role == 'anchor')} anchor"
        )
    unverified = [s.alias for s in reg if not s.date_verified]
    if unverified:
        lines.append("")
        lines.append("VERIFY release dates before publication: " + ", ".join(unverified))
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary(load_registry()))
