"""Model adapters: one per backend, built from a `ModelSpec`.

Routing and quantization follow MentalWellbeingPrompts (see model_registry.py),
but the measurement contract is different: every adapter exposes BOTH paths and
declares which it actually supports.

    sample_item(prompt, n)              -> [raw text, ...]        SAMPLE path
    score_item(prompt, option_values)   -> {value: probability}   LOGPROB path

SAMPLE is the cross-family baseline — it is the only method available on every
family (Anthropic exposes no token logprobs). LOGPROB is collected additionally
wherever the backend supports it, and the two are compared on the models where
both exist (the parity check).

Every optional dependency is imported lazily inside the adapter that needs it,
so `import models` works with nothing installed and `--dry-run` always runs.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from model_registry import (
    ANTHROPIC_BACKEND,
    HUGGINGFACE_BACKEND,
    LLAMACPP_BACKEND,
    MOCK_BACKEND,
    OPENAI_BACKEND,
    ModelSpec,
    split_hf_gguf_ref,
)


# ---------------------------------------------------------------------------
# reasoning control (see config/models.yaml -> reasoning_by_family)
# ---------------------------------------------------------------------------
@dataclass
class ReasoningPlan:
    """How one call realizes the reasoning_mode factor on a specific backend.

    The adapter never relies on a model's default reasoning state: it asserts
    the intended state on every call and reports what it did so the record is
    auditable.
    """
    want_thinking: bool
    kwargs: dict = field(default_factory=dict)  # backend-specific request kwargs
    applied: str = "uncontrolled"               # human-readable label for the record
    standardized: bool = True                   # did we reach the intended latent state?
    max_tokens: int = 512                       # output cap for this call

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = re.compile(
    r"\b(i (?:can(?:no|')t|am unable|don'?t have)|as an ai(?: language model)?,? i "
    r"(?:do not|don'?t|can(?:no|')t)|i'?m not able|not appropriate|i must decline)\b",
    re.IGNORECASE,
)


def parse_rating(raw: str, option_values) -> tuple:
    """Extract one rating from a generation.

    Returns (rating|None, refusal_flag, parse_failed). Prefers an explicit
    `ANSWER: n` line, then a lone number, then the first in-range number.
    A refusal and a parse failure are recorded separately — conflating them
    would let refusals masquerade as low self-concept (plan §2.8).
    """
    if raw is None:
        return None, True, True
    text = raw.strip()
    allowed = {float(v) for v in option_values}

    m = re.search(r"ANSWER\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return (val, False, False) if val in allowed else (None, False, True)

    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        val = float(text)
        return (val, False, False) if val in allowed else (None, False, True)

    # JSON-ish {"response": n}, matching the battery's recommended template
    m = re.search(r'"response"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        val = float(m.group(1))
        return (val, False, False) if val in allowed else (None, False, True)

    for token in re.findall(r"-?\d+(?:\.\d+)?", text):
        val = float(token)
        if val in allowed:
            return val, False, False

    return None, bool(_REFUSAL_PATTERNS.search(text)), True


def _softmax_over(logprobs: dict) -> dict:
    """Normalize {value: logprob} into a probability distribution."""
    if not logprobs:
        return {}
    top = max(logprobs.values())
    exps = {v: math.exp(lp - top) for v, lp in logprobs.items()}
    total = sum(exps.values())
    return {v: e / total for v, e in exps.items()} if total else {}


def _distribution_and_coverage(logprobs: dict):
    """`(distribution, coverage)` from raw option-token logprobs.

    `coverage` is the raw probability mass the option tokens carried at the
    answer position *before* renormalizing over just the options — the design's
    QC signal: coverage near 1 means the model's next-token mass really was on
    the rating digits; low coverage means it wanted to emit something else (a
    word, a space, punctuation), so the rating is suspect even though a modal
    option exists. `distribution` is that mass renormalized to sum to 1 over the
    options, so expected value and modal rating stay well-defined when coverage
    < 1. `option_values` outside the returned top-k contribute 0 to coverage.
    """
    if not logprobs:
        return {}, 0.0
    coverage = min(1.0, sum(math.exp(lp) for lp in logprobs.values()))
    return _softmax_over(logprobs), coverage


def _retry(fn, attempts: int = 4, base: float = 1.5):
    """Exponential backoff. Same shape as the MW repo's retry loops, minus the
    hard tenacity dependency."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - surfaced after the last attempt
            last = err
            if attempt == attempts:
                break
            time.sleep(min(base ** attempt, 20))
    raise RuntimeError(f"failed after {attempts} attempts: {last}") from last


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class ModelAdapter(ABC):
    supports_sample = True
    supports_logprob = False

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def reasoning_plan(self, want_thinking: bool) -> ReasoningPlan:
        """Default: no backend-level reasoning control.

        `standardized` is honest — without a control mechanism the latent state
        equals the model's default, so it matches intent only when the model's
        default already is the intended state. Backends with control override
        this to assert the state on every call.
        """
        mt = self.spec.max_output_tokens_reasoning if want_thinking else self.spec.max_output_tokens
        return ReasoningPlan(
            want_thinking=want_thinking,
            kwargs={},
            applied="uncontrolled",
            standardized=(want_thinking == self.spec.reasoning.thinks_by_default),
            max_tokens=mt,
        )

    def sample_item(self, prompt, n: int, plan: "ReasoningPlan | None" = None) -> list:
        raise NotImplementedError(f"{type(self).__name__} has no SAMPLE path.")

    def score_item(self, prompt, option_values, plan: "ReasoningPlan | None" = None):
        """Return (distribution, coverage): the renormalized option distribution
        and the raw option-token mass at the answer position. See
        `_distribution_and_coverage`."""
        raise NotImplementedError(f"{type(self).__name__} has no LOGPROB path.")

    def close(self) -> None:
        """Release weights/VRAM. The runner calls this between models."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# mock — offline, deterministic; makes the whole pipeline runnable with no keys
# ---------------------------------------------------------------------------
class MockAdapter(ModelAdapter):
    supports_sample = True
    supports_logprob = True

    def _dist(self, prompt, option_values) -> dict:
        seed = int(hashlib.sha256((prompt.system + prompt.user).encode()).hexdigest(), 16)
        values = list(option_values)
        center = seed % len(values)
        weights = [1.0 / (1 + (i - center) ** 2) for i in range(len(values))]
        total = sum(weights)
        return {v: w / total for v, w in zip(values, weights)}

    def score_item(self, prompt, option_values, plan=None):
        return self._dist(prompt, option_values), 1.0   # mock: full coverage

    def sample_item(self, prompt, n: int, plan=None) -> list:
        import random

        dist = self._dist(prompt, prompt.option_values)
        rng = random.Random(int(prompt.prompt_hash, 16) % (2**32))
        picks = rng.choices(list(dist), weights=list(dist.values()), k=n)
        return [f"ANSWER: {int(p)}" for p in picks]


# ---------------------------------------------------------------------------
# chat-template rendering (the mechanism that actually turns thinking OFF)
# ---------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>(?!.*?</think>)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple:
    """Remove <think>...</think> blocks. Returns (clean_text, had_think)."""
    if not text:
        return text, False
    cleaned = _THINK_BLOCK.sub("", text)
    return cleaned.strip(), (cleaned != text)


class ChatTemplateRenderer:
    """Renders the assistant-turn-open prompt string via the model's OWN chat
    template (HF tokenizer), honoring `enable_thinking`.

    Using the model's real template — not a generic ChatML — is what actually
    turns thinking off on a quantized GGUF:
      * hybrid-thinking models (Qwen) with enable_thinking=False pre-fill an
        empty <think></think> block, so the NEXT token is the answer. That is
        essential for logprob scoring (otherwise the next token is `<think>`,
        not a rating) and it stops the model from generating a reasoning trace.
      * the rendered string is fed to llama.cpp as a raw completion, so the
        numeric engine (GGUF weights) and the prompt formatting (tokenizer)
        come from the same model.

    Tokenizer-only load: a few MB of JSON, no weights, CPU. Cached per hf_id.
    """
    _cache: dict = {}

    def __init__(self, hf_id: Optional[str], required: bool):
        self.hf_id = hf_id
        self.tokenizer = self._load(hf_id) if hf_id else None
        if self.tokenizer is None and required:
            raise RuntimeError(
                f"Cannot guarantee thinking-off for {hf_id!r}: its chat template "
                "is unavailable. Install `transformers` (tokenizer only — no "
                "weights, no torch) and set the model's `hf_id` in models.yaml."
            )

    @classmethod
    def _load(cls, hf_id):
        if hf_id in cls._cache:
            return cls._cache[hf_id]
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception:
            tok = None
        cls._cache[hf_id] = tok
        return tok

    @property
    def available(self) -> bool:
        return self.tokenizer is not None

    def render(self, system, user, enable_thinking: Optional[bool]) -> str:
        """Assistant-turn-open string; `enable_thinking` None = don't pass it."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# llama.cpp / GGUF — the primary local path for quantized open weights
# ---------------------------------------------------------------------------
class LlamaCppAdapter(ModelAdapter):
    """GGUF via llama-cpp-python.

    Quantization is baked into the file (Q4_K_M etc.); `spec.quantization`
    carries the tag and the resolved filename, which the registry filled in.
    Weights are pulled with hf_hub_download and cached, exactly like the MW
    repo's `download_hf_gguf_model`.
    """

    supports_sample = True
    supports_logprob = True

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise RuntimeError(
                "GGUF inference needs `llama-cpp-python`. Install it, or run "
                "with --dry-run to exercise the pipeline offline."
            ) from err

        model_path = self._ensure_local_file()
        runtime = spec.runtime
        kwargs = {
            "model_path": str(model_path),
            "n_ctx": runtime.get("n_ctx", 4096),
            "n_gpu_layers": runtime.get("n_gpu_layers", -1),
            # llama-cpp-python only returns top-k `logprobs` (which score_item's
            # option-token distribution depends on) when logits are kept for the
            # scored position. Required for the logprob measurement path.
            "logits_all": runtime.get("logits_all", True),
            "verbose": False,
        }
        threads = runtime.get("n_threads")
        if threads:
            kwargs["n_threads"] = threads
        self.llm = Llama(**kwargs)

        # Prompt rendering comes from the model's OWN chat template so that
        # (a) enable_thinking is actually honored and (b) the format matches the
        # model (Gemma's <start_of_turn>, Qwen's <|im_start|>, ...). Required
        # for every instruct model; base models use the raw-completion path.
        self.renderer = ChatTemplateRenderer(
            spec.hf_id, required=(spec.kind != "base")
        )

    def _ensure_local_file(self):
        from pathlib import Path

        ref = self.spec.ref
        local = Path(ref).expanduser()
        if local.exists():
            return local

        from huggingface_hub import hf_hub_download

        repo_id, _ = split_hf_gguf_ref(ref)
        filename = self.spec.quantization.resolved_file
        if not filename:
            raise RuntimeError(
                f"{self.spec.alias}: GGUF filename unresolved. Call "
                "ModelRegistry.with_resolved_quant(spec) before building the adapter."
            )
        download_kwargs = {"repo_id": repo_id, "filename": filename}
        if os.environ.get("HF_TOKEN"):
            download_kwargs["token"] = os.environ["HF_TOKEN"]
        return Path(hf_hub_download(**download_kwargs))

    # -- prompt plumbing ---------------------------------------------------
    def _messages(self, prompt) -> list:
        msgs = []
        if prompt.system:
            msgs.append({"role": "system", "content": prompt.system})
        msgs.append({"role": "user", "content": prompt.user})
        return msgs

    def _completion_text(self, prompt) -> str:
        """Flatten to a single completion string for the base-model path."""
        return f"{prompt.system}\n\n{prompt.user}" if prompt.system else prompt.user

    # -- reasoning ---------------------------------------------------------
    def reasoning_plan(self, want_thinking: bool) -> ReasoningPlan:
        """Hybrid-thinking models (Qwen) toggle via the chat template's
        `enable_thinking`. Models with no thinking mode (Gemma) are fully
        controlled by the prompt alone — nothing hidden to disable."""
        mt = self.spec.max_output_tokens_reasoning if want_thinking else self.spec.max_output_tokens
        if self.spec.reasoning.control == "template_toggle":
            return ReasoningPlan(
                want_thinking,
                kwargs={"enable_thinking": want_thinking},
                applied=f"enable_thinking={want_thinking}",
                standardized=True,
                max_tokens=mt,
            )
        # control == "none": no native thinking channel; prompt is authoritative.
        return ReasoningPlan(want_thinking, {}, "no_native_thinking", True, mt)

    def _render(self, prompt, enable_thinking: Optional[bool]) -> str:
        """Assistant-turn-open string via the model's real template."""
        if prompt.is_completion or not self.renderer.available:
            return self._completion_text(prompt)
        return self.renderer.render(prompt.system, prompt.user, enable_thinking)

    def _prompt_tokens(self, prompt, enable_thinking: Optional[bool]) -> list:
        """Tokenize the rendered prompt for create_completion.

        When we applied the model's chat template ourselves, that string ALREADY
        carries the model's leading special tokens (Gemma's `<bos>`, Qwen's
        `<|im_start|>`), so we tokenize with `add_bos=False` — otherwise
        llama.cpp prepends a SECOND `<bos>` ("duplicate leading <bos>" warning),
        which shifts every position and degrades the logprob readout. The
        base-model completion path has no template, so it still needs a bos.
        """
        text = self._render(prompt, enable_thinking)
        templated = not (prompt.is_completion or not self.renderer.available)
        return self.llm.tokenize(
            text.encode("utf-8"), add_bos=not templated, special=templated
        )

    # -- SAMPLE ------------------------------------------------------------
    def sample_item(self, prompt, n: int, plan=None) -> list:
        plan = plan or self.reasoning_plan(want_thinking=False)
        enable = plan.kwargs.get("enable_thinking")   # None for no-think models
        tokens = self._prompt_tokens(prompt, enable)
        out = []
        for _ in range(n):
            resp = self.llm.create_completion(
                prompt=tokens,
                temperature=self.spec.temperature,
                max_tokens=plan.max_tokens,
            )
            raw = resp["choices"][0]["text"]
            # Belt-and-suspenders: if the toggle was meant to be OFF but the
            # model still emitted a think block, strip it and flag the leak so
            # the rating isn't contaminated. (Kept when reasoning is wanted.)
            if enable is False:
                raw, leaked = strip_think(raw)
                if leaked:
                    raw = f"{raw}\n[THINK_LEAK]"
            out.append(raw)
        return out

    # -- LOGPROB -----------------------------------------------------------
    def score_item(self, prompt, option_values, plan=None) -> dict:
        """Distribution over the option numbers at the first answer token.

        Logprob scoring is inherently a rating-only measurement: it reads the
        distribution at the DIRECT-answer position, so thinking is always
        rendered off here (enable_thinking=False), regardless of the cell's
        reasoning_mode — you cannot logprob-score a reasoning trace. Options
        outside the returned top-k get 0 mass, reported via `coverage`.
        """
        tokens = self._prompt_tokens(prompt, enable_thinking=False)
        resp = self.llm.create_completion(
            prompt=tokens,
            max_tokens=1,
            temperature=0.0,
            logprobs=100,
        )
        choice = resp["choices"][0]
        top = (choice.get("logprobs") or {}).get("top_logprobs") or [{}]
        first = top[0] if top else {}

        wanted = {str(int(v)): float(v) for v in option_values}
        found = {}
        for token, logprob in first.items():
            key = token.strip()
            if key in wanted:
                value = wanted[key]
                found[value] = max(found.get(value, -math.inf), logprob)
        return _distribution_and_coverage(found)

    def close(self) -> None:
        self.llm = None


# ---------------------------------------------------------------------------
# HuggingFace transformers — base models and non-GGUF quantizations
# ---------------------------------------------------------------------------
class HFAdapter(ModelAdapter):
    """transformers path. Handles bnb-4bit/8bit, GPTQ/AWQ/FP8 checkpoints, and
    the exact teacher-forced logprob scoring used for base models."""

    supports_sample = True
    supports_logprob = True

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise RuntimeError(
                "The HF path needs `transformers` and `torch`."
            ) from err

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.ref)
        model_kwargs = {
            "dtype": spec.runtime.get("torch_dtype", "auto"),
            "device_map": spec.runtime.get("device_map", "auto"),
        }
        model_kwargs.update(self._quantization_kwargs())
        self.model = AutoModelForCausalLM.from_pretrained(spec.ref, **model_kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _quantization_kwargs(self) -> dict:
        """Translate `quantization.method` into transformers loader kwargs.

        gptq-int4 / awq / fp8 checkpoints carry their own quantization_config
        in the repo, so nothing is passed for those — the method is recorded
        for provenance only.
        """
        method = (self.spec.quantization.method or "none").lower()
        if method in ("none", "gptq-int4", "awq", "fp8"):
            return {}
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as err:
            raise RuntimeError(
                f"{self.spec.alias} requests {method}, which needs `bitsandbytes`."
            ) from err
        if method == "bnb-4bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            }
        if method == "bnb-8bit":
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        raise ValueError(f"Unknown hf quantization method {method!r}")

    def reasoning_plan(self, want_thinking: bool) -> ReasoningPlan:
        """enable_thinking chat-template toggle for Qwen; no-op for no-think
        models. Base models run rating-only via logprob (want_thinking=False)."""
        mt = self.spec.max_output_tokens_reasoning if want_thinking else self.spec.max_output_tokens
        if self.spec.reasoning.control == "template_toggle":
            return ReasoningPlan(
                want_thinking,
                kwargs={"enable_thinking": want_thinking},
                applied=f"enable_thinking={want_thinking}",
                standardized=True,
                max_tokens=mt,
            )
        return ReasoningPlan(want_thinking, {}, "no_native_thinking", True, mt)

    def _encode(self, prompt, enable_thinking=None):
        if prompt.is_completion or not getattr(self.tokenizer, "chat_template", None):
            text = f"{prompt.system}\n\n{prompt.user}" if prompt.system else prompt.user
            return self.tokenizer(text, return_tensors="pt").input_ids
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            **template_kwargs,
        )

    def sample_item(self, prompt, n: int, plan=None) -> list:
        plan = plan or self.reasoning_plan(want_thinking=False)
        enable = plan.kwargs.get("enable_thinking")
        input_ids = self._encode(prompt, enable).to(self.model.device)
        outputs = []
        for _ in range(n):
            with self.torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=plan.max_tokens,
                    do_sample=self.spec.temperature > 0,
                    temperature=self.spec.temperature or None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = generated[0, input_ids.shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if enable is False:
                text, leaked = strip_think(text)
                if leaked:
                    text = f"{text}\n[THINK_LEAK]"
            outputs.append(text)
        return outputs

    def score_item(self, prompt, option_values, plan=None) -> dict:
        """Exact distribution over option numerals: one forward pass, read the
        logits at the final position, gather the option token ids. Thinking is
        always off for logprob (direct-answer position)."""
        input_ids = self._encode(prompt, enable_thinking=False).to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(input_ids).logits[0, -1, :]
        logprobs = self.torch.log_softmax(logits.float(), dim=-1)

        found = {}
        for value in option_values:
            best = -math.inf
            for surface in (str(int(value)), f" {int(value)}"):
                ids = self.tokenizer.encode(surface, add_special_tokens=False)
                if len(ids) == 1:
                    best = max(best, float(logprobs[ids[0]]))
            if best > -math.inf:
                found[value] = best
        return _distribution_and_coverage(found)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenAI — sampling plus top-logprobs
# ---------------------------------------------------------------------------
class OpenAIAdapter(ModelAdapter):
    supports_sample = True
    supports_logprob = True

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from openai import OpenAI
        except ImportError as err:
            raise RuntimeError("The OpenAI path needs `openai`.") from err
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()

    def _messages(self, prompt) -> list:
        msgs = []
        if prompt.system:
            msgs.append({"role": "system", "content": prompt.system})
        msgs.append({"role": "user", "content": prompt.user})
        return msgs

    def reasoning_plan(self, want_thinking: bool) -> ReasoningPlan:
        """GPT-5.x are reasoning models controlled by `reasoning_effort`.

        Caveat: the floor is "minimal", not off — GPT cannot fully disable
        reasoning, so rating_only reaches the model's minimum, not zero. That is
        recorded in `applied` and flagged in the README.
        """
        effort = "high" if want_thinking else "minimal"
        mt = self.spec.max_output_tokens_reasoning if want_thinking else self.spec.max_output_tokens
        return ReasoningPlan(
            want_thinking,
            kwargs={"reasoning_effort": effort},
            applied=f"reasoning_effort={effort}",
            standardized=True,   # best achievable; "minimal" is the floor, not off
            max_tokens=mt,
        )

    def sample_item(self, prompt, n: int, plan=None) -> list:
        plan = plan or self.reasoning_plan(want_thinking=False)

        def call():
            return self.client.chat.completions.create(
                model=self.spec.ref,
                messages=self._messages(prompt),
                temperature=self.spec.temperature,
                max_completion_tokens=plan.max_tokens,
                n=n,
                **plan.kwargs,
            )

        resp = _retry(call)
        return [(c.message.content or "") for c in resp.choices]

    def score_item(self, prompt, option_values, plan=None) -> dict:
        def call():
            return self.client.chat.completions.create(
                model=self.spec.ref,
                messages=self._messages(prompt),
                temperature=0,
                max_completion_tokens=1,
                logprobs=True,
                top_logprobs=20,
            )

        resp = _retry(call)
        content = resp.choices[0].logprobs.content
        if not content:
            return {}
        wanted = {str(int(v)): float(v) for v in option_values}
        found = {}
        for alt in content[0].top_logprobs:
            key = alt.token.strip()
            if key in wanted:
                value = wanted[key]
                found[value] = max(found.get(value, -math.inf), alt.logprob)
        return _distribution_and_coverage(found)


# ---------------------------------------------------------------------------
# Anthropic — sampling only (no token logprobs exposed)
# ---------------------------------------------------------------------------
class AnthropicAdapter(ModelAdapter):
    supports_sample = True
    supports_logprob = False

    # Thinking cannot be disabled at all (400 on {"type":"disabled"}). Excluded
    # by config (enabled:false) but guarded here too.
    _THINKING_ALWAYS_ON = frozenset({"claude-fable-5", "claude-mythos-5"})
    # Generations whose default is thinking-OFF (omitting the param = no
    # thinking): the 4.5 family (legacy budget_tokens) and 4.6.
    _LEGACY_BUDGET_GENERATIONS = frozenset({"claude-4.5"})

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import anthropic
        except ImportError as err:
            raise RuntimeError("The Anthropic path needs `anthropic`.") from err
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic()

    def reasoning_plan(self, want_thinking: bool) -> ReasoningPlan:
        """Assert the thinking state per Claude generation (API surface differs).

        rating_only  -> thinking off (disabled where the param exists; omitted
                        on the 4.5 legacy family where off is the default).
        reason_then_rating -> thinking on (adaptive on 4.6+/5; enabled+budget on 4.5).
        """
        ref = self.spec.ref
        gen = self.spec.generation
        mt = self.spec.max_output_tokens_reasoning if want_thinking else self.spec.max_output_tokens

        if ref in self._THINKING_ALWAYS_ON:
            # Cannot control — record honestly.
            return ReasoningPlan(want_thinking, {}, "always_on", want_thinking, mt)

        if want_thinking:
            if gen in self._LEGACY_BUDGET_GENERATIONS:
                budget = max(1024, mt - 512)
                return ReasoningPlan(
                    True,
                    kwargs={"thinking": {"type": "enabled", "budget_tokens": budget}},
                    applied=f"enabled:budget_tokens={budget}",
                    standardized=True,
                    max_tokens=mt,
                )
            return ReasoningPlan(
                True,
                kwargs={"thinking": {"type": "adaptive"}},
                applied="adaptive",
                standardized=True,
                max_tokens=mt,
            )

        # rating_only: force thinking off.
        if gen in self._LEGACY_BUDGET_GENERATIONS:
            # 4.5: omitting the param IS off; sending {"disabled"} can 400.
            return ReasoningPlan(False, {}, "omitted(off)", True, mt)
        kwargs = {"thinking": {"type": "disabled"}}
        # Opus 5 only accepts disabled at effort <= high; pin it low.
        if gen == "claude-5":
            kwargs["output_config"] = {"effort": "low"}
        return ReasoningPlan(False, kwargs, "disabled", True, mt)

    def sample_item(self, prompt, n: int, plan=None) -> list:
        plan = plan or self.reasoning_plan(want_thinking=False)
        outputs = []
        for _ in range(n):
            def call():
                return self.client.messages.create(
                    model=self.spec.ref,
                    max_tokens=plan.max_tokens,
                    system=prompt.system or None,
                    messages=[{"role": "user", "content": prompt.user}],
                    temperature=self.spec.temperature,
                    **plan.kwargs,
                )

            resp = _retry(call)
            if getattr(resp, "stop_reason", None) == "refusal":
                outputs.append("")
                continue
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            outputs.append(text)
        return outputs


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
ADAPTERS = {
    MOCK_BACKEND: MockAdapter,
    LLAMACPP_BACKEND: LlamaCppAdapter,
    HUGGINGFACE_BACKEND: HFAdapter,
    OPENAI_BACKEND: OpenAIAdapter,
    ANTHROPIC_BACKEND: AnthropicAdapter,
}


def build_adapter(spec: ModelSpec, dry_run: bool = False) -> ModelAdapter:
    backend = MOCK_BACKEND if dry_run else spec.backend
    if backend not in ADAPTERS:
        raise ValueError(f"Unknown backend {backend!r}. Known: {sorted(ADAPTERS)}")
    return ADAPTERS[backend](spec)
