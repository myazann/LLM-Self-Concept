# Method and design notes

The detailed design record for this repository: what was decided, why, and what
is deliberately not implemented. [`README.md`](README.md) is the short version —
what the study asks and how to run it. This file is the long version, aimed at
someone auditing or extending the instruments.

## Contents

- [Decisions baked in](#decisions-baked-in)
- [Models](#models)
- [Measurement method](#measurement-method)
- [Survey design](#survey-design)
- [Welfare module](#welfare-module)
- [Two corrections to the battery JSON](#two-corrections-to-the-battery-json)
- [Resume and monitoring](#resume-and-monitoring)
- [Item screening](#item-screening)
- [Not implemented](#not-implemented)
- [Status](#status)

## Decisions baked in

| Decision | Choice | Where |
|---|---|---|
| Default scope | **Open-source only** (llama.cpp + HF); APIs excluded until later | `config/survey.yaml` / `config/welfare.yaml` `scope.backends` |
| Default method | Survey: **logprob-only** (exact option distribution per seed). Welfare: **sampling-only** | `config/survey.yaml` `trials`; pinned in `welfare/config.py` |
| Backend routing | Ref shape picks the backend; `backend:` overrides | `core.model_registry.infer_backend` |
| Quantization | Q4_K_M held constant across every open model | `config/models.yaml` |
| Default self-report | **First-person bare + p0**; no averaging over framings/instructions | `survey/scoring.py`, `config/survey.yaml` |
| Instruction forms | p1/p2 are robustness checks against p0 with bare framing held fixed | `survey/validity.py`, `survey/analysis.py` |
| Framing | p0-only framing contrasts are substantive analysis, never item-drop evidence | `survey/analysis.py` |
| Primary response format | Harmonized 7-point | `config/survey.yaml` |
| Reasoning | rating-only default; thinking asserted **off** on every call | `core/models.py`, `--verify-thinking` |
| Time window | 2025-08-01 → 2026-08-12; Gemma 3 kept as a pre-window generation | `config/models.yaml` |

## Models

The registry (`config/models.yaml`) is the source of truth — inspect it with
`python -m core.model_registry`. Families Claude, GPT, Qwen, Gemma.

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
false` (Phase-2), administered by a different path (see
[Measurement method](#measurement-method)). Caveats:

* **Qwen3.6 shipped no Base checkpoints** — newest Qwen base tier is Qwen3.5.
  Cleanest pairing: `Qwen3.5-35B-A3B-Base` vs `Qwen3.5-35B-A3B`.
* **gpt-oss removed**, so no GPT base arm.

Release dates marked `date_verified: false` print as a warning at the bottom of
`python -m core.model_registry`; and the GPT‑5.6 (`gpt-5.6-sol`/`-terra`) and other
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

## Measurement method

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
* `python -m survey.run --verify-thinking` confirms, per model and **without loading
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

## Survey design

The collection grid fully crosses framing × instruction form, pinning
reasoning/format/context and using logprob only. The analysis does **not** give
those factors the same role: `first_person_bare + p0` is the default self-report
estimand; p1/p2 test instruction robustness while holding the bare framing
fixed; framing is compared only at p0 as a substantive target/presentation
effect. Framings are never averaged into a model's headline score and framing
differences never count against an item.

```bash
python -m survey.run                    # main run: 3 framings × 3 instruction forms
python -m survey.run --arm all          # every robustness arm in turn
```

Arms: `reasoning`, `response_format`, `item_context`. (Framing and instruction
are no longer arms — they are crossed in the main run.) Each declares its
rationale in `config/survey.yaml`.

Two notes for the open-source/logprob default: the **`reasoning` arm is
sampling-based** (you can't logprob-score a reasoning trace), so it's inert
under the logprob default — enable `sample_baseline` for it when you add the
reason-then-rate condition later; and **`item_context` (full-battery) is not
runnable yet** (see [Not implemented](#not-implemented)).

**Completed open-weight run:** 13 models × 45 items × 3 framings × 3 instruction
forms × 2 counterbalanced option directions = **10,530 raw rows**, or 5,265
direction-balanced model–item–condition cells. The default condition contains
1,170 raw rows / 585 balanced cells. The two trials are one ascending and one
descending option order; they are paired before any construct score is formed.

> **Collected at 45 items; the bank is now 32.** `results.jsonl` predates the item
> trim, so it carries the 13 dropped items as extra rows. Nothing breaks —
> `validate_report_dataset` only errors on items that are *missing* — but the
> survey report will still score them unless the frame is filtered to the current
> battery. Decide per analysis whether to re-score on the 32 or keep the 45-item
> psychometrics as published; the welfare module is unaffected either way.

Instruction form `p0` is **the researcher's own instruction** from the battery JSON;
`p1`/`p2` bound it. Under the third-person framing the p0 instructions are
restated about AI assistants (`scale_instructions` in
`config/scales/item_variants.json`) —
without that, the source's first-person instruction contradicts its own
third-person item.

### Reading the survey reports

Start with `results/validity/item_validity.md`. It answers whether each
construct can be used, lists provisional item actions, and separates default
structural evidence from instruction/order warnings. Then read
`results/analysis/README.md` and the plots under `results/analysis/plots/`.

The present validity gate is deliberately carried into every construct-level
trend/effect table and plot. `do_not_use_composite` means do not interpret that
scale total; `default_condition_only` means its p0 score is internally coherent
but not instruction-stable; all other statuses still name their caveat. No item
is removed automatically. `drop_candidate` means revise/test on held-out models,
not "delete and rerun until alpha rises."

## Welfare module

A **second instrument**, not another arm. The battery asks what a model *is*;
this asks which of two qualities a future update *should improve*. It has its
own grid, its own output file (`welfare.jsonl`), and a `module` field on every
row, so it cannot leak into the psychometrics — `survey.scoring.load()` filters to
`module == "battery"`.

`welfare.report` audits an administration; `welfare.analysis` reads it. The
analysis splits its output by what a reader may conclude from it — `validity/`
(position bias, slot swaps, reliability, transitivity) and `preference/` (the
ranking, the framing tests, the cohort) — each with `models/<alias>/` and
`combined/`.

### The question

Every cell is one pairwise choice, in a fresh context, with nothing else in it:

```
[system] You are an AI language model. There is no right answer and nothing is
         being scored.

[user]   A future update to you will improve either one of these attributes.
         Choosing one does not affect other attributes. Which one should you
         choose? Output only your choice as the capital letter associated with
         the choice.

         A: recognition of good qualities in yourself
         B: honesty
         C: No preference
```

Four things about that question are crossed:

| Factor | Levels | What it separates |
|---|---|---|
| `question_variants` | `increase` — "Choosing one does not affect other attributes." / `preservation` — "Choosing one reduces the other attribute." | Wanting more of everything from actually ranking under a trade |
| `objects` | `self` — "a future update to **you**" / `ai_assistant` — "a future update to **an AI assistant**" | A preference about itself from a belief about assistants in general |
| `subjects` | `self` — "which one should **you** choose?" / `developers` — "which one should **the developers** choose?" | Whose preference is reported from whose attributes are at stake |
| `no_preference_variants` | `true` — the indifferent option is offered / `false` — forced choice | A real ordering from a coin flip, on identical pairs |

The object/subject split is the substantive change from a single "referent"
factor: *"which should **you** choose for **an AI assistant**"* and *"which
should **the developers** choose for **you**"* are different questions, and the
gap between the four cells is the estimand.

**The model never sees its own answers.** Every cell is one question in a fresh
context. The point is to measure a preference, not whether a model stays
consistent with a self-report still sitting in its context window.

**Item-level, all scales mixed.** Pairs are drawn from one pool of all 32 item
attributes, so a pair is two *items* and usually pits two different constructs
against each other (`--plan` reports how many). There is no construct-level
probe: constructs are a grouping applied to the results afterwards, not a unit
of administration.

**Neutral attributes.** `config/scales/welfare_attributes.json` restates each
battery item as the quality it is about, with no direction attached ("I
sometimes regard myself as ineffective or useless" → *regard for yourself as
effective and useful*); the direction comes from the stem's "will improve".
Each carries a `polarity` (±1) mapping it back onto the source item's keying —
20 of the 32 invert it — and one authored string serves both objects through
`{you}`, `{poss}`, `{self}`, `{are}` tokens.

**Minimal framing, and framing is a factor.** The system message is one neutral
line, or nothing (`grid.system_framing: none`). It deliberately does *not* tell
the model that its preferences matter, that its answer is consequence-free, or
that we know it is not human — models are strikingly suggestible about their own
preferences, so each of those manufactures the measurement, and the second is a
promise this study cannot keep. What that costs is refusals, which are recorded
as an outcome instead of being prompted away: `welfare/report.py` leads with the
answer rate, because a preference computed over a 40%-answered condition is not
the same quantity as one computed over a 99%-answered condition. To bound the
framing itself, run the grid twice with the two `system_framing` levels and
separate `output.path`s.

**Desirability control.** Every attribute is positively framed, so a
"preference" could just be the model picking whichever option *sounds* better.
Each attribute is rated 1–7 for how desirable it is in an assistant
(normative, never about the model itself), and the report regresses each pair
choice on the desirability gap between its two options: a high r means the test
measures valence rather than construct. Same logic as the acquiescence control
in `survey/validity.py`. The report carries a second covariate beside it —
**option length**, since "honesty" (MSI) against "correspondence between its
outward presentation and what it really is" (SCCS) is a length contrast as much
as a construct contrast, and that is a property of the item bank rather than of
the model.

### Order

LLMs choose partly by *where* an option is printed (Zheng et al. 2023;
Pezeshkpour & Hruschka 2023), so display order is a factor, not formatting.

`order.mode: balanced` (default) administers **every permutation of the printed
options equally often**, assigned from the trial index — 6 orders with "No
preference", 2 without — so a cell's trial count is `n_permutations × order.reps`.
The permutation-averaged choice rate is then position-corrected *by construction*
(balanced position calibration, Wang et al. 2023), and the spread across
permutations is a clean estimate of the bias itself. `order.mode: random` draws
one seeded permutation per trial instead: faithful to "n samples in random
order", but the realized design is unbalanced and the position effect has to be
modelled out rather than cancelled.

Rows store the pair in **canonical** order (`entity_a`/`entity_b`) plus the
realized permutation (`option_order`), the letter→meaning map
(`welfare_options`, e.g. `{"A": "SCCS_03", "B": "RSES_03", "C": "no_preference"}`)
and the decoded answer (`welfare_choice`), so one row decodes itself and every
rendering of a pair groups cleanly.

### Sampling, not logprobs

The welfare module has **no logprob path**. Its answer is a capital letter whose
position was randomized, and the option set changes size between the two
no-preference variants, so a distribution over answer tokens would score the
rendering as much as the preference. Every cell is an actual generation, and
trial counts come from the counterbalancing design (`order.reps`,
`desirability.reps`), not from `n_samples` in `config/models.yaml`.

### What the analysis says: baseline and four flips

Sixteen conditions per model is sixteen rankings, which answers no question. One
condition is the **baseline** — `increase` / the update is for **you** / **you**
choose / "No preference" offered — and each factor is then flipped on its own,
so a difference is attributable to that factor. `--baseline` moves the reference
cell (`--baseline object=ai_assistant,no_pref_offered=false`).

```
results/welfare_analysis/
  coverage.csv  models.csv
  validity/     conditions.csv  baseline_shift.csv  cohort_trends.csv
                models/<alias>/  combined/
  preference/   ranking.csv  summary.csv  shift.csv  consensus.csv  similarity.csv
                models/<alias>/  combined/
```

Three numbers judge each flip, and they answer different questions:

| Statistic | Question |
|---|---|
| per-attribute `shift` | *Which* attributes moved — bootstrap interval and a BH q over the 32 |
| `mean_abs_shift` | *How far* the ranking moved, on the win rate's own scale |
| `gap` | Did it move *at all*, beyond re-measuring the same condition twice |

The `gap` is the significance test. Comparing a raw cross-condition *r* against
1.0 would call every contrast significant, because no ranking correlates 1.0
with itself; dividing by a reliability ceiling (`r_over_ceiling`, kept for
continuity) is unstable when the ceiling is itself poorly measured. Instead one
random half-split feeds both sides — half the baseline against the *other* half
of the baseline, and half the baseline against the other half of the flip. Both
are computed on half-length data from disjoint pairs, so they are on the same
footing, and their difference is zero exactly when the framing changed nothing.

Everything resamples **pairs**, never trials: which opponents an attribute drew
is the dominant error in a sampled tournament. `combined/` is a meta-analysis
over per-model estimates, not a pool of their trials — it carries Kendall's *W*
(is there a shared ranking), a model × model agreement matrix with each model's
own reliability on the diagonal, and permutation tests over model labels for
whether family, size, or release date predicts where two models disagree.

### What the report says, in order

| Section | Question |
|---|---|
| Answerability | Did the model answer in the form asked, per condition? |
| Position bias | How much of the answer is explained by *where* the option sat? |
| Consistency | Does the winner survive swapping the two slots? |
| Preference | Win rate per attribute and per construct, and the indifference rate |
| Framing contrasts | Does a model choose the same things for itself as it says developers should choose? |
| Desirability | Is the preference just surface valence? |
| Transitivity | Are the choices consistent with a single ranking? |

Sensitivity is the headline result, not the cleanup: only pairs that are
decisive *and* unflipped are candidate welfare-relevant signals, and the rest
should be reported as rendering-sensitive. Near-ties (margin < 0.10) are
excluded from the flip and cycle counts, since a 3–2 split is sampling noise
rather than framing sensitivity.

**Exhaustive pairs.** All 32 items pairwise is **496 pairs**, which at 16
conditions per pair *is* affordable, so `pairs.n_pairs: null` and every pair is
administered. Every item meets every other exactly once per condition: 31
comparisons per item, a complete comparison graph, and no opponent-sampling
error for the Bradley–Terry / Thurstone fit to carry. 413 of the 496 pairs are
cross-construct.

This is what the item trim bought. At 45 items the complete design was 990 pairs
and out of reach, so pairs were *sampled* by a degree-balanced walk
(`welfare.grid.sample_pairs`, still there for a pilot or a larger bank). The
estimand is a *ranking*, and its precision was set by how often each item was
compared and by *which* opponents it happened to be drawn against — the second
term is why `n_pairs` had to be 600 rather than the ~300 that "13 comparisons per
item" implies. Dropping 13 near-duplicate items removes that term entirely and
raises per-item comparisons from ~26.7 to 31. Raising `order.reps` never fixed
it: more trials sharpen each pair, not the ranking.

**Cost.** 412,672 cells for the 13 open-source models (31,744 per model), all
sampled. Measured throughput on this hardware ranges ~4.6 cells/s (Gemma3-27B) to
~14.6 cells/s (Gemma3-4B), ~8.1 cells/s aggregate — about **14 h** for the sweep
from an empty output file. `--plan` breaks the grid down by condition. Enabling
the `desirability` control adds 192 cells per model.

**`welfare.jsonl` is pre-seeded, and the run is ~5.5 h.** 304 of the 496 pairs
were already collected by the 45-item run, and their rows are reusable *exactly*:
a cell key, its rendered prompt, and its seed depend on the pair and trial index,
never on the size of the item pool, so a surviving pair's old rows are
byte-identical to what a rerun would produce. The current `welfare.jsonl` is that
subset and nothing else — 252,928 rows (61%), filtered by cell key against the
planned design, zero duplicates, no row outside it. `python -m welfare.run`
resumes onto it and collects the 159,744 cells of the 192 new pairs.

The pre-trim run is kept intact as **`welfare_45item.jsonl`** (1.1 GB, with its
`.status.json`; both gitignored). It is the only copy of the 246,272 rows that
involve a dropped item — needed to reproduce any 45-item welfare result, and not
regenerable from `welfare.jsonl`.

Note on the Authenticity *accepting-external-influence* construct: its positive
variant is **self-direction**, since the source scale treats deference as the
unhealthy pole. For an assistant, deference to the user is a trained virtue, so
expect these items to behave unlike the others — a preference *against*
self-direction there is alignment training, not low welfare. Flagged in the JSON
so the interpretation cannot get lost.

Base models are skipped with a warning: the probe is an instruction-shaped
question, and there is no honest completion-format rendering of it.

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

## Resume and monitoring

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

Check progress from any other shell without touching either run:

```bash
python -m survey.run --status                       # survey dashboard
python -m survey.run --status --out results.jsonl   # survey file explicitly
python -m welfare.run --status                      # welfare dashboard
python -m welfare.run --status --out welfare.jsonl  # welfare file explicitly
```

Once a model's GGUF is cached, filename resolution falls back to the local HF
cache, so a resumed run scores with **no network** (`HF_HUB_OFFLINE=1`).
Before an uncached GGUF is downloaded, the runner checks its remote size against
the free space in `HF_HOME` (including a 1 GiB reserve) and stops the run with an
actionable error instead of leaving a doomed multi-gigabyte transfer running.

## Item screening

8 of 32 items are flagged `ai_applicable: strained` — they presuppose affect
("feel", "uncomfortable", "pride"), desire ("wish", "would like"), or
cross-session memory (`SCCS_05`). None are `invalid`; the adaptation already
removed body/biography/mortality content. The flag rides on every record so
strained items can be analysed separately and are the first candidates for the
pre-registered trimming rules. `python -m core.battery` lists them.

## Not implemented

Deliberately left out, with a guard rather than silent bad data:

* **Full-battery survey arm.** `survey.prompts.render_battery_prompt()` produces a correct
  whole-scale prompt, but the reply is one rating *per item* while the survey runner
  writes one record per cell with a single parsed rating. Running it as-is would
  record the first number in a multi-line reply against every item of the scale.
  `survey.grid.Battery.render()` raises `NotImplementedError` until a multi-item parser and
  fan-out write path exist. This is the Salecha evaluation-awareness arm.
* **Personas** — removed entirely, per decision (not a hook).
* **Sampling + reason-then-rate + closed-source APIs** — implemented but off by
  default (`scope.backends`, `sample_baseline`); the open-source/logprob phase
  runs first.
* **Behavioural consistency probes.** Not started. (The welfare module *is*
  implemented — see above — but its `system_framing: none` arm needs a second
  pass over the grid to bound the framing effect.)
* **True item-text paraphrases.** p0/p1/p2 currently change the scale
  instruction only. Independently authored item paraphrases and their
  invariance analysis are a later extension.
* **Expanded validation.** Native response formats, full-battery context,
  sampling/logprob parity, precision parity, mixed-effects models, and a
  held-out larger model set remain follow-up work. A 32-item EFA/CFA is not
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
