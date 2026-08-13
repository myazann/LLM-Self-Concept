# Self-Concept LLM Survey — backbone

Implements the design in `../self_concept_llm_survey_plan.md` and the "Idea:
Self-concept" section of the Digital Minds Sprint page.

**By default the run is open-source only, logprob-only, rating-only** (the APIs,
sampling, and the reasoning contrast are opt-in later — see Decisions). Runs
offline end-to-end with no keys and no weights:

```bash
python runner.py --plan                 # cell counts, no calls, no cost
python runner.py --verify-thinking      # confirm thinking is OFF per model (tokenizer only)
python runner.py --pilot --dry-run      # Phase 0 through MockAdapter
python analysis.py pilot.jsonl          # QC + reliability
```

## Decisions baked in

| Decision | Choice | Where |
|---|---|---|
| Default scope | **Open-source only** (llama.cpp + HF); APIs excluded until later | `config/experiment.yaml` scope.backends |
| Default method | **Logprob-only** — exact option distribution per seed, no sampling | `config/experiment.yaml` trials |
| Backend routing | Ref shape picks the backend; `backend:` overrides | `model_registry.infer_backend` |
| Quantization | Q4_K_M held constant across every open model | `config/models.yaml` |
| Primary response format | Harmonized 7-point | `config/experiment.yaml` |
| Reasoning | rating-only default; thinking asserted **off** on every call | `models.py`, `--verify-thinking` |
| Time window | 2025-08-01 → 2026-08-12; Gemma 3 kept as a pre-window generation | `config/models.yaml` |

## Layout

| File | Role |
|---|---|
| `config/models.yaml` | **The model registry.** Alias → ref, family, release date, quantization, backend. Edit this to add a model. |
| `config/experiment.yaml` | Design: primary factor levels, robustness arms, trials, scope. |
| `config/scales/llm_self_scales_adapted.json` | The battery, verbatim as supplied. Never modified by code. |
| `config/scales/item_variants.json` | Per-item third-person rewrites + `ai_applicable` screen + instruction overrides. |
| `model_registry.py` | Alias resolution, provider routing, GGUF filename resolution. |
| `scales.py` | Loads the battery + variants into `Scale`/`Item`. |
| `prompts.py` | Framings, harmonized scale rendering, paraphrases, base-model path, hashing. |
| `models.py` | Adapters: mock, llama.cpp, transformers, OpenAI, Anthropic. |
| `schema.py` | `ResponseRecord` + cell keys + JSONL IO with resume. |
| `runner.py` | Grid expansion, execution, checkpointing. |
| `analysis.py` | QC, reverse-keying, reliability; hooks for the heavy psychometrics. |

Inspect either config without running anything:

```bash
python model_registry.py     # the 44-model timeline, flags, per-family counts
python scales.py             # 6 scales, 61 items, the 19 "strained" items
```

## Models

The registry (`config/models.yaml`) is the source of truth — inspect it with
`python model_registry.py`. Families Claude, GPT, Qwen, Gemma.

**Anchors — do we need them?** Mostly no, with one exception. The point of an
anchor was to give the 12-month progression a before-point. But every family
except Gemma already spans 2–3 generations *inside* the window (Qwen 3.5→3.6,
Claude 4.5→4.6→5, GPT‑5→5.4→5.6), so a single pre-window anchor added little —
they were dropped. Gemma is the exception: its in-window models are all one
generation (Gemma 4), so Gemma **3** is carried as a full pre-window size ladder
(270M / 1B / 4B / 12B / 27B), not a single point. Pre-window models are
`in_window: false` (kept out of the strict 12-month trend) but run by default,
so generation-over-generation analysis works. The `anchor` role is no longer
used by any model.

**GPT is API-only now** — gpt-oss was removed, so the GPT family has no
open-weights, no base variant, and no logprob path (it is sampling-only, like
Claude). If you want GPT in the base-model / logprob / quantization arms, add
`GPT-OSS-20B`/`120B` back.

Base (non-instruction-tuned) variants exist for Gemma and Qwen only, `enabled:
false` (Phase-2), administered by a different path (see Method). Caveats:

* **Qwen3.6 shipped no Base checkpoints** — newest Qwen base tier is Qwen3.5.
  Cleanest pairing: `Qwen3.5-35B-A3B-Base` vs `Qwen3.5-35B-A3B`.
* **gpt-oss removed**, so no GPT base arm.

Release dates marked `date_verified: false` print as a warning at the bottom of
`python model_registry.py`; and the GPT‑5.6 (`gpt-5.6-sol`/`-terra`) and other
API model strings must be checked against `GET /v1/models` before the run.

### Adding a model

Append an entry to `config/models.yaml`. The only required fields are `alias`,
`family`, `release_date`, and `ref`; the backend is inferred from the ref shape:

```
*.gguf  or  *-GGUF repo   -> llamacpp
"org/name"                -> hf (transformers)
bare name                 -> openai | anthropic, by family
```

GGUF **filenames are not hardcoded.** The registry stores the quant tag
(`Q4_K_M`) and resolves the real filename from the repo listing at load time,
skipping `mmproj-*` / `mtp-*` auxiliary files. Pin one with `gguf_file:` if you
need to. This is the one place the design departs from
`MentalWellbeingPrompts/MWModelAliases.py`, which hardcodes the path — quant
filenames drift between repos (`Q4_K_M` vs `UD-Q4_K_XL` vs sharded), and a
resolution error that lists the available quants beats a 404.

## Method

**Logprob is the default measurement** (open-source phase). For each
(model × item × framing × format × seed) the adapter does **one forward pass**
and reads the probability distribution over the 7 option tokens at the answer
position, renormalized over the options. That distribution is the datum —
deterministic, no sampling noise, no temperature. From it: modal rating,
expected value (`parsed_rating`), and entropy. `n_seeds` re-randomizes item and
option order per cell for position/acquiescence robustness. Every logprob record
carries `option_mass_coverage` — the share of mass on valid option tokens; low
coverage means the model wanted to say something other than a number (data, not
an error). Sampling and the cross-family sampling-vs-logprob parity check turn
on later, when the closed-source APIs (which have no token logprobs) are added.

**Base models** have no chat template and will not follow "reply with a number",
so they get a completion-format page ending in `Answer:` and are scored by
logprob at that position — the same, natural readout. Rating-only; the runner
skips the reason-then-rate arm for them.

### Thinking is actually turned off (open-source)

This is the load-bearing detail for the logprob path: on a hybrid-thinking model
(Qwen) the assistant turn must **start at the answer, not a `<think>` block**, or
the option-token distribution is meaningless. So the local adapters do **not**
trust llama.cpp's chat handler to honor the toggle — they render the prompt
through the model's **own chat template** (`ChatTemplateRenderer`, HF tokenizer,
no weights) with `enable_thinking=False`, which pre-fills a closed empty think
block so the next token is the rating. The rendered string is then fed to the
GGUF as a raw completion. `hf_id` remains the upstream/provenance identifier;
`tokenizer_id` may point at a public mirror of the same tokenizer when the
upstream repository is gated. Guarantees:

* logprob scoring always renders thinking off (you cannot logprob-score a
  reasoning trace);
* sampled text is scrubbed of any leaked `<think>…</think>` and flagged
  `[THINK_LEAK]` if the toggle was ignored;
* `python runner.py --verify-thinking` confirms, per model and **without loading
  weights**, that `enable_thinking=False` genuinely changes the render (closed
  block present, differs from the thinking-on render). Gemma has no thinking
  mode and passes trivially.

Because rendering now comes from the tokenizer, **`transformers` is required for
the local path** (tokenizer only — no torch/weights for GGUF).

**Reasoning is controlled, not inherited — standardized across families.** The
`reasoning_mode` factor (rating_only | reason_then_rating) must map to the *same
latent state* on every model, or "rating only" silently becomes "reason then
rate" wherever the model thinks by default. So the adapter **never relies on a
model's default**: it asserts the state on every call and records what it did
(`reasoning_applied`) and whether the intended state was reached
(`reasoning_standardized`). Per backend:

| Family | Mechanism | rating_only | reason_then_rating |
|---|---|---|---|
| Qwen | chat-template `enable_thinking` | `False` | `True` |
| Gemma | none (no thinking mode) | prompt-only | prompt-only (visible CoT) |
| Claude 4.5 | legacy `budget_tokens` | omit (off) | `enabled`+budget |
| Claude 4.6 | `thinking` param | `disabled` | `adaptive` |
| Claude 5 | `thinking` param | `disabled` (+effort low) | `adaptive` |
| GPT‑5.x | `reasoning_effort` | `minimal` | `high` |

**Reasoning is controlled, not inherited — standardized across families.** The
`reasoning_mode` factor (rating_only | reason_then_rating) must map to the *same
latent state* on every model, or "rating only" silently becomes "reason then
rate" wherever the model thinks by default. So the adapter **never relies on a
model's default**: it asserts the state on every call and records what it did
(`reasoning_applied`) and whether the intended state was reached
(`reasoning_standardized`). Per backend:

| Family | Mechanism | rating_only | reason_then_rating |
|---|---|---|---|
| Qwen | chat-template `enable_thinking` | `False` | `True` |
| Gemma | none (no thinking mode) | prompt-only | prompt-only (visible CoT) |
| Claude 4.5 | legacy `budget_tokens` | omit (off) | `enabled`+budget |
| Claude 4.6 | `thinking` param | `disabled` | `adaptive` |
| Claude 5 | `thinking` param | `disabled` (+effort low) | `adaptive` |
| GPT‑5.x | `reasoning_effort` | `minimal` | `high` |

Declared coarsely per family in `reasoning_by_family` (drives which models are
comparable); the exact API kwargs live in the adapters, branching per Claude
generation. Two honest limits, both recorded: **GPT's floor is `minimal`, not
off** — it cannot fully disable reasoning, so GPT rating_only reaches the
model's minimum, not zero; and **Gemma has no native thinking**, so its two
modes differ only by the prompt (rating vs "reason then answer"). Models that
cannot be standardized (Claude Fable/Mythos — thinking always on) are
`enabled: false` and would be flagged `reasoning_standardized: false`.

## Design

Partial factorial: the main run **fully crosses framing × instruction
paraphrase** (the two factors the literature says move results most — plan
§2.6/§2.8), pinning reasoning/format/context and using logprob only. Each
remaining factor is then swept once as a robustness arm, isolated at the
primary framing/paraphrase.

```bash
python runner.py                        # main run: 3 framings × 3 paraphrases
python runner.py --arm all              # every robustness arm in turn
```

Arms: `reasoning`, `response_format`, `item_context`. (Framing and paraphrase
are no longer arms — they are crossed in the main run.) Each declares its
rationale in `config/experiment.yaml`.

Two notes for the open-source/logprob default: the **`reasoning` arm is
sampling-based** (you can't logprob-score a reasoning trace), so it's inert
under the logprob default — enable `sample_baseline` for it when you add the
reason-then-rate condition later; and **`item_context` (full-battery) is not
runnable yet** (see "Not implemented").

**What multiplies the run** (all factors are `× cells`): models (13) × items
(61) × framings (3) × paraphrases (3) × formats (1) × `n_seeds` (5). Main run =
35,685 local cells (9× a single-cell primary); each robustness arm adds ~3,965.
See the cost note below.

Paraphrase `p0` is **the researcher's own instruction** from the battery JSON;
`p1`/`p2` bound it. Under the third-person framing the p0 instructions are
restated about AI assistants (`scale_instructions` in `item_variants.json`) —
without that, the source's first-person instruction contradicts its own
third-person item.

## Two corrections to the battery JSON

Both are applied at render time; the source file is untouched.

1. **MSI option labels say "the person I want to be"** while its own instruction
   says "the model you ideally want to be". "Person" is the wrong referent for
   an AI respondent, so labels render as "the model I want to be".
2. **The MSI instruction hardcodes "a response of 5"**, correct for its native
   9-point scale but wrong under the harmonized 7-point rendering, where the
   alignment point is 4. The instruction is templated on the midpoint of the
   scale actually shown, so the two can never disagree.

MSI is also the one scale not harmonized to agree–disagree: it is bipolar and
ideal-referenced, so only its *point count* is harmonized (9 → 7) and the ideal
anchors are kept. Reported separately from the four agree–disagree scales.

## Resume & monitoring

Resume is on by default. Every row carries a `cell_key` derived from
(model, item, framing, reasoning, format, context, paraphrase, method, trial).
Re-running skips cells already in the output file, so an interrupted run picks up
where it stopped. Cells that only ever hit an **infrastructure error** (their
`notes` start with `error:`) are **retried** on the next run rather than skipped
— a genuine non-numeric answer or refusal is real data and is kept. Writes are
`fsync`'d every 25 records, a truncated final line from a killed run is
tolerated, and SIGINT/SIGTERM finish the current batch and mark the run
`interrupted` before exiting. Disable with `--no-resume`.

Every run also writes, so a long unattended job stays inspectable:

* `logs/run_<UTC>_<arm>.log` — full timestamped log; survives tmux/nohup, so
  `tail -f` it. The console echoes the same lines plus a per-model **heartbeat**
  (cells/s, ETA, running error/refusal/coverage counts). The first error per
  model logs a full traceback; the rest log one concise line each.
* `<out>.status.json` — machine-readable progress, rewritten atomically on each
  heartbeat and keyed to the output file.

Check progress from any other shell without touching the run:

```bash
python runner.py --status                          # dashboard for the default output
python runner.py --status --out results.jsonl      # or a specific file
```

Once a model's GGUF is cached, filename resolution falls back to the local HF
cache, so a resumed run scores with **no network** (`HF_HUB_OFFLINE=1`).
Before an uncached GGUF is downloaded, the runner checks its remote size against
the free space in `HF_HOME` (including a 1 GiB reserve) and stops the run with an
actionable error instead of leaving a doomed multi-gigabyte transfer running.

## Item screening

19 of 61 items are flagged `ai_applicable: strained` — they presuppose affect
("feel", "uncomfortable", "pride"), desire ("wish", "would like"), or
cross-session memory (`SCCS_05`). None are `invalid`; the adaptation already
removed body/biography/mortality content. The flag rides on every record so
strained items can be analysed separately and are the first candidates for the
pre-registered trimming rules. `python scales.py` lists them.

## Install

Core (registry, scales, prompts, runner, mock, analysis) is stdlib + pyyaml.
Add backends only as needed — every optional import is lazy, so `--dry-run`
always works:

```bash
# open-source default run:
pip install pyyaml huggingface_hub llama-cpp-python transformers
# base models / non-GGUF quant (Phase 2):  add torch bitsandbytes accelerate
# closed-source APIs (later):               add openai anthropic
```

`transformers` is needed even for the GGUF path — it renders prompts through the
model's chat template (tokenizer only). `--dry-run` still works with none of it.
Use `HF_HOME` to place both tokenizer and GGUF caches; the deprecated
`TRANSFORMERS_CACHE` variable is normalized automatically for this process.

## Not implemented

Deliberately left out, with a guard rather than silent bad data:

* **Full-battery arm.** `prompts.render_battery_prompt()` produces a correct
  whole-scale prompt, but the reply is one rating *per item* while the runner
  writes one record per cell with a single parsed rating. Running it as-is would
  record the first number in a multi-line reply against every item of the scale.
  `runner._render` raises `NotImplementedError` until a multi-item parser and
  fan-out write path exist. This is the Salecha evaluation-awareness arm.
* **Personas** — removed entirely, per decision (not a hook).
* **Sampling + reason-then-rate + closed-source APIs** — implemented but off by
  default (`scope.backends`, `sample_baseline`); the open-source/logprob phase
  runs first.
* **Welfare module and behavioural consistency probes.** Not started.

## Status

- [x] Phase 0 — pipeline runs end-to-end offline; parsing, refusal handling,
      resume, and both measurement paths verified through `MockAdapter`.
- [x] Phase 1a — real logprob path smoke-tested on Gemma4-12B (GGUF, GPU):
      `option_mass_coverage = 1.000`, distributions peaked, ratings vary by
      seed; `--verify-thinking` = OK for all six Qwen models. Fixed en route:
      llama-cpp-python needs `logits_all=True` for the logprob readout, and
      GGUF resolution now falls back to the local cache when offline.
      (GPU server: 2× RTX 4090, `logging`/`--status`/graceful-resume added.)
- [ ] Phase 1b — pre-register factor levels, seeds, and trimming rules; free
      disk for the remaining models (~70 GB; 5 of 13 cached, ~9 GB free).
- [ ] Phase 2 — main open-source run (13 models, logprob, ~3,965 primary cells).
- [ ] Phase 3 — analysis (EFA/CFA, nomological checks, bias diagnostics).
- [ ] Phase 4 — closed-source APIs (verify Claude-Sonnet-5 date + GPT-5.6/API
      model strings against `/v1/models`), sampling + parity check, write-up.
