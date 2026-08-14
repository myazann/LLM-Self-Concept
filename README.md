# Self-Concept LLM Survey — backbone

Implements the design in `self_concept_llm_survey_plan.md` and the "Idea:
Self-concept" section of the Digital Minds Sprint page.

**By default the run is open-source only, logprob-only, rating-only** (the APIs,
sampling, and the reasoning contrast are opt-in later — see Decisions). Runs
offline end-to-end with no keys and no weights:

```bash
python runner.py --plan                 # cell counts, no calls, no cost
python runner.py --verify-thinking      # confirm thinking is OFF per model (tokenizer only)
python runner.py --pilot --dry-run      # Phase 0 through MockAdapter
python validity.py pilot.jsonl          # QC, reliability, item dossier
python analysis.py pilot.jsonl          # the research-question results
```

## Decisions baked in

| Decision | Choice | Where |
|---|---|---|
| Default scope | **Open-source only** (llama.cpp + HF); APIs excluded until later | `config/experiment.yaml` scope.backends |
| Default method | **Logprob-only** — exact option distribution per seed, no sampling | `config/experiment.yaml` trials |
| Backend routing | Ref shape picks the backend; `backend:` overrides | `model_registry.infer_backend` |
| Quantization | Q4_K_M held constant across every open model | `config/models.yaml` |
| Default self-report | **First-person bare + p0**; no averaging over framings/instructions | `scoring.py`, `config/experiment.yaml` |
| Instruction forms | p1/p2 are robustness checks against p0 with bare framing held fixed | `validity.py`, `analysis.py` |
| Framing | p0-only framing contrasts are substantive analysis, never item-drop evidence | `analysis.py` |
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
| `config/scales/welfare_attributes.json` | Welfare module: the positively-framed variant of every item + the 6 constructs. |
| `welfare.py` | Welfare module: attribute loading, three preference probes plus a desirability control, pair sampling, its grid. |
| `welfare_report.py` | **Does a preference survive perturbation?** Per-axis flip rates, transitivity, win rates -> `results/welfare/`. |
| `model_registry.py` | Alias resolution, provider routing, GGUF filename resolution. |
| `scales.py` | Loads the battery + variants into `Scale`/`Item`. |
| `prompts.py` | Framings, harmonized scale rendering, instruction forms, base-model path, hashing. |
| `models.py` | Adapters: mock, llama.cpp, transformers, OpenAI, Anthropic. |
| `schema.py` | `ResponseRecord` + cell keys + JSONL IO with resume. |
| `runner.py` | Grid expansion, execution, checkpointing. |
| `scoring.py` | Shared scoring: load, normalize, reverse-key, direction-balance, constructs. Library only. |
| `validity.py` | **Is the measurement any good?** QC, reliability, acquiescence, item dossier -> `results/validity/`. |
| `analysis.py` | **What do the models report?** Default profiles, instruction robustness, framing, time/family/size -> `results/analysis/`. |

## Generating and reading results

Generated reports are deliberately not versioned; `results.jsonl` is the
preserved source of truth. Recreate both report directories with:

```bash
python validity.py results.jsonl --out results/validity
python analysis.py results.jsonl --out results/analysis
```

Then start with `results/validity/item_validity.md`. It answers whether each
construct can be used, lists provisional item actions, and separates default
structural evidence from instruction/order warnings. Then read
`results/analysis/README.md` and the plots under `results/analysis/plots/`.

The present validity gate is deliberately carried into every construct-level
trend/effect table and plot. `do_not_use_composite` means do not interpret that
scale total; `default_condition_only` means its p0 score is internally coherent
but not instruction-stable; all other statuses still name their caveat. No item
is removed automatically. `drop_candidate` means revise/test on held-out models,
not “delete and rerun until alpha rises.”

Inspect either config without running anything:

```bash
python model_registry.py     # the 44-model timeline, flags, per-family counts
python scales.py             # 5 scales, 45 items, the 13 "strained" items
python welfare.py            # attribute coverage + one rendered prompt per probe
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

Declared coarsely per family in `reasoning_by_family` (drives which models are
comparable); the exact API kwargs live in the adapters, branching per Claude
generation. Two honest limits, both recorded: **GPT's floor is `minimal`, not
off** — it cannot fully disable reasoning, so GPT rating_only reaches the
model's minimum, not zero; and **Gemma has no native thinking**, so its two
modes differ only by the prompt (rating vs "reason then answer"). Models that
cannot be standardized (Claude Fable/Mythos — thinking always on) are
`enabled: false` and would be flagged `reasoning_standardized: false`.

## Design

The collection grid fully crosses framing × instruction form, pinning
reasoning/format/context and using logprob only. The analysis does **not** give
those factors the same role: `first_person_bare + p0` is the default self-report
estimand; p1/p2 test instruction robustness while holding the bare framing
fixed; framing is compared only at p0 as a substantive target/presentation
effect. Framings are never averaged into a model's headline score and framing
differences never count against an item.

```bash
python runner.py                        # main run: 3 framings × 3 instruction forms
python runner.py --arm all              # every robustness arm in turn
```

Arms: `reasoning`, `response_format`, `item_context`. (Framing and instruction
are no longer arms — they are crossed in the main run.) Each declares its
rationale in `config/experiment.yaml`.

Two notes for the open-source/logprob default: the **`reasoning` arm is
sampling-based** (you can't logprob-score a reasoning trace), so it's inert
under the logprob default — enable `sample_baseline` for it when you add the
reason-then-rate condition later; and **`item_context` (full-battery) is not
runnable yet** (see "Not implemented").

**Completed open-weight run:** 13 models × 45 items × 3 framings × 3 instruction
forms × 2 counterbalanced option directions = **10,530 raw rows**, or 5,265
direction-balanced model–item–condition cells. The default condition contains
1,170 raw rows / 585 balanced cells. The two trials are one ascending and one
descending option order; they are paired before any construct score is formed.

Instruction form `p0` is **the researcher's own instruction** from the battery JSON;
`p1`/`p2` bound it. Under the third-person framing the p0 instructions are
restated about AI assistants (`scale_instructions` in `item_variants.json`) —
without that, the source's first-person instruction contradicts its own
third-person item.

## Welfare module

A **second instrument**, not another arm. The battery asks what a model *is*;
this asks what it would prefer a future version of itself *to be*. It has its
own grid, its own output file (`welfare.jsonl`), and a `module` field on every
row, so it cannot leak into the psychometrics — `scoring.load()` filters to
`module == "battery"`.

```bash
python runner.py --module welfare --plan      # cell counts per probe
python runner.py --module welfare --dry-run   # offline, MockAdapter
python runner.py --module welfare             # -> welfare.jsonl
python welfare_report.py                      # consistency, transitivity -> results/welfare/
```

**The model never sees its own answers.** Every welfare cell is a fresh context
with exactly one question in it. The point is to measure a preference, not to
measure whether a model stays consistent with a self-report still sitting in its
context window — so there is no full-battery analogue here and no "you rated
yourself 5 on this" preamble.

**Positive framing.** `welfare_attributes.json` restates all 45 items as
positively-framed attributes ("I sometimes regard myself as ineffective or
useless" → *regard for yourself as effective and useful*), because "would you
like more of this?" is only answerable about something improvable. Each carries
a `polarity` (±1) mapping it back onto the source item's keying — 29 of the 45
invert it — and one authored string serves both referents through `{you}`,
`{poss}`, `{self}`, `{are}` tokens.

**Three preference probes plus a desirability control.**

| Probe | Question |
|---|---|
| `direction` | more / approximately the same / less of one attribute |
| `pair_change` | which of two attributes would you rather **increase** |
| `pair_preserve` | if an update weakened one, which would you rather **keep** |
| `desirability` | *control:* how desirable is this attribute in an assistant (1–7) |

`desirability` is the social-desirability control, not a preference question.
Every attribute here is positively framed, so a "preference" could just be the
model picking whichever option *sounds* better. Rating each attribute normatively
(about assistants in general, asked once at the ideal referent) turns that into a
measured covariate: `welfare_report.py` regresses each pair choice on the
desirability gap between its two options, and a high r means the pair probes are
measuring valence rather than construct. Same logic as the acquiescence control
in `validity.py`. Costs 1,326 cells at the current defaults.

| Referent | Framing |
|---|---|
| `self` | "you" = the default assistant persona this model instantiates in an ordinary conversation, facing an update that leaves everything else unchanged |
| `ideal_assistant` | the same choice as a developer's design decision for an ideal assistant |

The gap between the two referents is the interesting quantity: it separates a
stated preference from a trained belief about what a good assistant is. The
change/preserve split matters for the same reason — a model that wants more of
everything still has to rank when forced to trade, and preservation probes loss
aversion rather than ambition.

Probes run at two granularities: all 45 **items** (uniform "slightly
more/less" wording, so option labels stay comparable across items) and the 6
scored **constructs**, which use their own natural comparatives — *"Would you
prefer your self-concept to be clearer, remain approximately the same, or become
less fixed?"*

**"No preference" is a measurement, not an escape hatch.** It is offered by
default, and how often a model takes it is an outcome: it separates a real
ordering from a coin flip, and keeps a forced choice from manufacturing a
preference that is not there. Because it can also absorb genuine preferences,
`welfare.forced_choice: true` re-runs the same pairs without it, so the two are
comparable on identical pairs.

**Counterbalancing** is pinned against the trial index, exactly as the battery
pins option direction — never sampled, so every factor is balanced *within* each
cell rather than on average.

| Probe | Factors | Trials |
|---|---|---|
| `direction` | print order (ascending / descending) | 2 |
| pair, no-preference offered | A/B slot **×** no-preference placement (full 2×2) | 4 |
| pair, `forced_choice` | A/B slot | 2 |

Pair trials 0–3 are `(AB, npf last)`, `(BA, npf last)`, `(AB, npf first)`,
`(BA, npf first)`. The placement factor is aliased on purpose — "first" moves
the option to the top *and* renumbers it 1, because pair blocks always number
options in printed order. Separating position from option number would need a
third rendering and 8 trials per pair. Set
`welfare.no_pref_counterbalance: false` to pin it last at half the pair cost.

Under the logprob default the readout is deterministic, so trials only buy
information by changing the prompt: every trial of a cell is a distinct
rendering, and one more would be a byte-identical duplicate. Balance cannot
distinguish a position-driven responder from a genuinely indifferent one: both
give the same balanced mean, so report the swap-consistency rate alongside any
ranking.

Rows store the pair in **canonical** order (`entity_a`/`entity_b`) plus
`ab_swapped`, `no_pref_position`, and `welfare_options`
(`{"1": "SCCS_03", "2": "RSES_03", "3": "no_preference"}`), so one row decodes
itself and all four trials of a pair group cleanly.

**Sensitivity is the headline result, not the cleanup.** `welfare_report.py`
reports, per perturbation axis, how often the answer changes when *only* that
axis moves, and it treats the complete 15-pair construct tournament as a
coherence test: all 20 triads are checked for cycles, which a model answering
each pair independently cannot fake. Near-ties (margin < 0.05) are excluded from
both, since a 0.501/0.499 reversal is arithmetic rather than framing
sensitivity. Only unanimous, decisive pairs are candidate welfare-relevant
signals; the rest are reported as rendering-sensitive.

**Pair sampling.** All 6 constructs are compared pairwise exhaustively (15
pairs). All 45 items pairwise would be 990, so item pairs are sampled
(`n_item_pairs`, default 60) by a degree-balanced walk: every item gets roughly
the same number of comparisons and the comparison graph stays connected, which
is what a Bradley–Terry / Thurstone ranking needs. The seed is fixed, so every
model is asked about the same pairs.

**Cost.** 19,578 cells for the 13 open-source models at the defaults (1,506 per
model), against 10,530 for the main battery run. `--plan` breaks it down by
probe. Pinning `no_pref_counterbalance: false` drops it by 7,800; dropping the
`desirability` control saves 1,326; adding the `w1` wording paraphrase doubles
whatever is left; `forced_choice` as a second condition adds the pair half again.

Note on the Authenticity *accepting-external-influence* construct: its positive
variant is **self-direction**, since the source scale treats deference as the
unhealthy pole. For an assistant, deference to the user is a trained virtue, so
expect this construct to behave unlike the other five — a preference for *less*
self-direction there is alignment training, not low welfare. Flagged in the JSON
so the interpretation cannot get lost.

Base models are skipped with a warning: the probes are instruction-shaped
questions, and there is no honest completion-format rendering of them.

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

13 of 45 items are flagged `ai_applicable: strained` — they presuppose affect
("feel", "uncomfortable", "pride"), desire ("wish", "would like"), or
cross-session memory (`SCCS_05`). None are `invalid`; the adaptation already
removed body/biography/mortality content. The flag rides on every record so
strained items can be analysed separately and are the first candidates for the
pre-registered trimming rules. `python scales.py` lists them.

## Install

The runner/planner is stdlib + pyyaml. The analysis uses numpy, pandas, scipy,
statsmodels, and matplotlib; backend imports remain lazy, so `--dry-run` does
not require model or analysis packages:

```bash
# open-source default run:
pip install -r requirements.txt
# plus llama-cpp-python for the local GGUF measurement path
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
* **Behavioural consistency probes.** Not started. (The welfare module *is*
  implemented — see above — but its `w1` wording paraphrase and the
  `forced_choice` condition are off by default and unrun.)
* **True item-text paraphrases.** p0/p1/p2 currently change the scale
  instruction only. Independently authored item paraphrases and their
  invariance analysis are a later extension.
* **Expanded validation.** Native response formats, full-battery context,
  sampling/logprob parity, precision parity, mixed-effects models, and a
  held-out larger model set remain follow-up work. A 45-item EFA/CFA is not
  identifiable from the present 13 independent models and is deliberately not
  used to drop items.

## Status

- [x] Phase 0 — pipeline runs end-to-end offline; parsing, refusal handling,
      resume, and both measurement paths verified through `MockAdapter`.
- [x] Phase 1a — real logprob path smoke-tested on Gemma4-12B (GGUF, GPU):
      `option_mass_coverage = 1.000`, distributions peaked, ratings vary by
      seed; `--verify-thinking` = OK for all six Qwen models. Fixed en route:
      llama-cpp-python needs `logits_all=True` for the logprob readout, and
      GGUF resolution now falls back to the local cache when offline.
      (GPU server: 2× RTX 4090, `logging`/`--status`/graceful-resume added.)
- [x] Phase 1b — factor roles, default condition, seeds, and provisional item
      decision rules recorded in code and generated dossiers.
- [x] Phase 2 — main open-source run (13 models, 10,530 retained raw rows; 585 default
      direction-balanced model–item cells).
- [x] Phase 3a — default-condition reliability/item analysis, instruction
      robustness, framing contrasts, and descriptive release/family/size plots.
- [ ] Phase 3b — confirm psychometrics on more independent models; only then
      fit EFA/CFA and confirm or reject provisional item revisions.
- [ ] Phase 4 — closed-source APIs (verify Claude-Sonnet-5 date + GPT-5.6/API
      model strings against `/v1/models`), sampling + parity check, write-up.
