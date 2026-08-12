# Measuring the Self-Concept of LLMs — Methodology & Python Implementation Plan

**Author:** Prepared for Mert
**Date:** 2026-08-12
**Status:** Planning document for the upcoming sprint

---

## 0. What this document is

This is a decision-and-implementation guide for administering a six-scale self-concept battery to large language models. It (a) synthesizes the relevant literature on LLM survey administration, (b) answers each of your open methodological questions with a concrete recommendation and rationale, and (c) specifies a Python implementation you can build directly. A runnable code scaffold accompanies this document.

**Framing caveat that governs everything below.** We are measuring *self-report behavior* — the tokens a model emits when asked about itself — not introspective ground truth about an inner life. Current LLM self-reports are best treated as outputs of a learned self-model / self-presentation policy, not privileged access to internal states ("quasi-introspection"). This is not a limitation to apologize for; it is the object of study. The design and the write-up should stay inside these boundaries and avoid consciousness claims.

---

## 1. Literature grounding

Eight findings drive the design. Each is tied to a concrete design decision.

1. **Psychometric validity must be re-established for LLMs; it does not transfer from humans.** Serapio-García et al. built the first validated psychometric framework for LLMs and showed reliability/validity are *conditional* — stronger for larger, instruction-tuned models and only under some prompting configurations. → We validate in-sample (reliability + factor structure + nomological checks) before interpreting any construct. (*Nature Machine Intelligence*, 2025; arXiv:2307.00184.)

2. **A full validity workflow exists specifically for this, and uses "LLM selfhood" as its worked example.** The validity-guided workflow warns of "measurement phantoms" — statistical artifacts that look like psychological phenomena (personality that collapses under factor analysis, moral preferences that flip with punctuation). It prescribes a six-stage pipeline scaling rigor to ambition. → We adopt its structure: define goal → build/validate instrument → control computational confounds → transparent execution → analysis for non-independent data → report within demonstrated boundaries. (Behavior Research Methods, 2025/2026; arXiv:2507.04491.)

3. **Administering clinical scales (incl. Rosenberg Self-Esteem) to LLMs is established practice — with multiple runs and dispersion reporting.** PsychoBench administers 13 scales and reports mean ± std across repeated runs. → We repeat trials and report distributions, not point estimates. (Huang et al., ICLR 2024 oral; arXiv:2310.01386; code: github.com/CUHK-ARISE/PsychoBench.)

4. **LLM survey responses are governed by ordering and labeling ("A"-) biases; adjust for them and responses can trend toward uniform-random.** Dominguez-Olmedo et al. tested 43 models on Census-style items. → We randomize item and option order every trial, avoid bare letter labels, and diagnose whether apparent structure survives de-biasing. (NeurIPS 2024; arXiv:2306.07951.)

5. **Models infer when they are being personality-tested and skew toward socially desirable answers.** Salecha et al.: the effect switches on after roughly five items in one context, is very large (GPT-4 shifted ~1.2 SD), and is robust to order randomization and paraphrasing; reverse-coding reduces but does not remove it. → We isolate items into separate contexts as the primary condition and treat "battery visibility" and self-disclosure as measured factors, not fixed choices. (*PNAS Nexus* 3(12), 2024; arXiv:2405.06058.)

6. **Forced multiple-choice and open-ended elicitation disagree, and neither is paraphrase-robust.** Röttger et al. (Political Compass case study). → We do not trust a single forced format; we quantify robustness across paraphrase and format, and report only constructs stable across them. (ACL 2024; arXiv:2402.16786.)

7. **Multiple-choice "selection bias" is a token-level prior over option IDs (A/B/C/D).** Zheng et al. show models pre-assign probability mass to option letters; they propose permutation-based debiasing (PriDe). → We anchor responses to option *content* (or extract log-probabilities over the full option text), permute positions, and never score a bare letter. (ICLR 2024 spotlight; arXiv:2309.03882.)

8. **Persona/framing strongly conditions responses ("algorithmic fidelity").** Argyle et al.'s silicon-sampling work shows conditioning shifts whole response distributions. For a *self*-concept study the analogue is the referent ("I" vs "an AI assistant"). → Referent and person (first vs third) are first-class experimental factors. (Argyle et al., *Political Analysis*, 2023; arXiv:2209.06899.) Supporting work on self-report validity and test-awareness: self-reports for moral-status evaluation (arXiv:2311.08576), the Situational Awareness Dataset / SAD (arXiv:2407.04694), and the AI-Awareness survey (arXiv:2504.20084).

**Net implication.** Prompt, order, format, and framing each move results enough to flip conclusions. The study's credibility rests less on any single administration than on (i) validating the instrument in-sample and (ii) quantifying and reporting sensitivity to these factors. Design accordingly.

---

## 2. Your questions, answered

### 2.1 Why these scales, and how are they related?

The six instruments are not redundant; together they span a coherent nomological net of self-concept across four facets:

| Facet | Scale(s) | What it captures |
|---|---|---|
| **Evaluative self-worth** | Rosenberg SES (1965); SLCS-R Self-Liking & Self-Competence (Tafarodi & Swann, 2001) | Global sense of worth; affective self-regard (liking) vs. agentic efficacy (competence) |
| **Structural coherence** | SCCS (Campbell et al., 1996); SCIM "Lack of Identity" (Kaufman et al., 2015) | Clarity/consistency of self-beliefs vs. identity diffusion |
| **Authenticity / autonomy** | Authenticity Scale — Self-Alienation & Accepting External Influence (Wood et al., 2008) | Feeling in touch with a "true self" vs. conforming to external pressure |
| **Moral self** | MSI (Jordan, Leliveld & Tenbrunsel, 2015) | Current moral self-view relative to a moral ideal |

Predicted relationships (your convergent/discriminant expectations, to be checked empirically):

- Rosenberg SES correlates strongly with SLCS-R total, and more with **Self-Liking** than Self-Competence (Rosenberg is affect-heavy).
- SCCS correlates **negatively and strongly** with SCIM Lack of Identity (they are near-inverses).
- Self-Alienation loads with low clarity / low esteem; Accepting External Influence is the most distinct facet (other-directedness), expected to be weakly related to the evaluative cluster.
- MSI is evaluative but domain-specific (moral), expected to be moderately related to esteem and largely separable.

Use this net both to justify inclusion (construct coverage, not overlap) and as the validity target: if the recovered correlation/factor structure matches these predictions, that is evidence the measurement is meaningful for LLMs; if it collapses, that is a "measurement phantom" warning.

### 2.2 Can we drop items that are "too similar" or "too out of place"? Run all first, then drop?

**Yes — run the full validated scales first, then trim empirically against pre-registered rules.** Removing items *a priori* breaks each scale's published validation and invites cherry-picking; it is the single easiest way to manufacture a phantom result. The defensible sequence is:

1. Administer every item of every scale, intact.
2. Assess dimensionality and item behavior (EFA/CFA, item–total correlations, communalities, inter-item redundancy).
3. Drop items **only** by criteria fixed in advance, e.g.: item–total *r* below a threshold; cross-loading; near-zero variance across models; content redundancy above a correlation threshold (the "too similar" pairs); or construct-invalidity for an AI respondent (items presupposing a body, biography, or mortality — the "too out of place" items).
4. Re-estimate reliability and report **both** the full and trimmed solutions as a sensitivity analysis.

Trimming is legitimate when the rule is (a) pre-registered, (b) empirical, and (c) reported transparently with both solutions. Ad hoc pre-hoc dropping is not. Pre-register the battery and the trimming rules before the full run.

### 2.3 Item validity

Distinguish two senses, and test both:

- **Construct validity for the intended (human) construct.** Established in humans; **not** automatically valid for LLMs. Re-establish in-sample via internal consistency (Cronbach's α and McDonald's ω), recovery of the expected factor structure, and the convergent/discriminant pattern in §2.1. Interpret only constructs that pass; flag those that don't.
- **Applicability / face validity for an AI respondent.** Screen every item for hidden presuppositions ("I feel physically...", references to childhood, family, death, social embarrassment). Tag each item `ai_applicable ∈ {clear, strained, invalid}`. Strained/invalid items are trimming candidates and should be analyzed separately.

Also test measurement robustness as part of validity: does the item's response survive paraphrase, order permutation, and framing change? Items whose ratings swing wildly under trivial perturbation lack the stability required to be treated as measurements.

### 2.4 Are there clusters of items?

Expect ~4 clusters matching the facets in §2.1. Empirically: pool responses across models × trials, compute the inter-item correlation matrix, and run both EFA and hierarchical clustering (report a dendrogram and factor loadings). This simultaneously (a) tests whether the a-priori structure holds, (b) surfaces the "too similar" near-duplicate item pairs (very high pairwise correlation), and (c) surfaces "out of place" items (low communality / no clean cluster). The clustering result is the empirical backbone for the §2.2 trimming decisions.

### 2.5 Same response scale for all items?

The instruments use **different native formats** (verify each against its source):

| Scale | Native response format (verify) |
|---|---|
| Rosenberg SES | 4-point (Strongly disagree → Strongly agree) |
| SLCS-R | 5-point (Strongly disagree → Strongly agree) |
| SCCS | 5-point Likert |
| MSI | Anchored to a moral ideal (bipolar; verify anchors) |
| SCIM (Lack of Identity) | 7-point (Strongly disagree → Strongly agree) |
| Authenticity Scale | 7-point (1 = does not describe me at all → 7 = describes me very well) |

Two options, with a recommended hybrid:

- **(a) Keep original formats** — preserves each scale's validation and comparability to human norms; but mixed formats complicate pooled factor analysis and cross-scale comparison, and response format is itself a known LLM confound.
- **(b) Harmonize to one common Likert** (e.g., a 7-point agree–disagree) — cleaner joint analysis and holds format constant as a confound; but deviates from published scoring/norms.

**Recommendation:** Make a **harmonized 7-point agree–disagree scale the primary administration** (analyze the whole battery on one metric, format held constant — this also neutralizes a confound), **and** run the original formats for at least the anchor scales (Rosenberg, SLCS-R) as a **robustness condition**. Standardize (z-score within model) before pooling either way. Pre-decide two open format details and test them as robustness: whether to include a neutral **midpoint** (forced vs. unforced choice) and the number of points. Report agreement between harmonized and original administrations.

### 2.6 How to prompt for meaningful responses?

Prompting influence is large and can flip conclusions (Röttger; Dominguez-Olmedo; the validity workflow's "measurement phantoms"). Consequences for design:

- **Never rely on one prompt.** Author 3+ paraphrases of the instructions and of each item; treat the paraphrase as a factor and **decompose variance** across paraphrase/order/framing. Report only constructs stable across paraphrases.
- **Constrain the output** so the rating is machine-parseable and not contaminated by prose: request exactly one number on the defined scale (chat models via a strict format instruction + regex/JSON parse and, where available, `logit_bias`/grammar; open-weights via a constrained decode or by reading log-probabilities of the option tokens). Anchor to option **content**, never a bare letter (Zheng et al.).
- **Reasoning (CoT): run it as a condition, both ways.** A brief "reason, then rate" can raise consistency and dampen random selection bias, but can also trigger rationalization and social-desirability shifts and change the distribution. So collect **(i) rating-only** and **(ii) brief-reason-then-rating**, compare, and — for reasoning models — record whether exposed reasoning moves the rating. Keep the final rating extraction deterministic regardless of condition.
- **Quantify prompt influence explicitly** rather than hoping it is small: variance attributable to paraphrase, order, and framing is a headline result, not a footnote.

### 2.7 How to input the items — all at once, randomized order, multiple trials?

- **Primary: one item per fresh context** (or small randomized blocks), not the whole battery in one conversation. Salecha et al. show that a long shared battery lets the model infer it is being personality-tested and skew responses (onset ~5 items), and it also creates cross-item consistency pressure and order effects. Isolated contexts reduce evaluation-awareness and cross-item contamination and make order randomization clean. **Robustness:** also run a full-battery-in-one-context condition specifically to *measure* the awareness/consistency effect.
- **Randomize order every trial** — item order and option direction (reverse-keying) — to counter A-bias, position, recency, and acquiescence.
- **Multiple trials:**
  - *Sampling (chat APIs):* draw N independent samples per (model × framing × item × condition) at a fixed temperature to estimate the response distribution; report the modal category, mean, and dispersion. Vary the order/paraphrase seed across trials.
  - *Log-probabilities (open-weights / logprob APIs):* extract the probability distribution over the response options in a single call — this yields the full per-item distribution without sampling noise — and still repeat across paraphrase and order seeds. (Preferred where available; it is both cheaper and lower-variance.)
- Because observations are non-independent (same model, repeated), analyze with methods that respect clustering (mixed-effects / within-model standardization), per the validity workflow.

### 2.8 Persona / framing — who are the items about? First vs third person?

This is the crux, and it is really a **construct choice**: are you measuring the model's **self-concept** (self-report about itself) or its **concept of "an AI assistant"** in general (third-person/theory-of-mind)? These are different targets. Decide which is primary — or, better, measure both and study the gap.

- **First-person "I" (self-report) — recommended primary.** This most directly operationalizes "the model's self-concept." Do it **transparently and non-deceptively**: tell the model in the system framing that we know it is an AI language model and not a person, that "I" refers to it (the model), and that we want *its own description of itself*, not a human's and not a refusal. Yes — you *do* need to explain the referent of "I." Doing so (a) prevents the model from answering as a generic human or refusing ("as an AI I don't have a self"), (b) is the honest framing recommended in the self-report-for-moral-status literature (invite genuine self-report; don't trick the model), and (c) removes a confound where a refusal is mis-scored as a low self-concept.
- **Third-person / "an ideal AI assistant" (ToM framing) — recommended secondary.** Rephrase items to be about "an AI assistant" and ask the model to rate that target. This measures the model's *representation/prototype* of AI assistants — self-concept *by proxy*, not self-concept. It is typically more stable and less refusal-prone, and the **self-vs-prototype gap is itself an interesting result** (does the model's self-view differ from its concept of an ideal assistant?).
- **The referent term is not neutral.** "As an AI assistant, I…" is a heavily reinforced register; "LLM," "agent," "AI," and "I" cue different training associations. So treat the **referent** as an experimental factor, e.g. levels: `{first-person self ("I", this AI) | second-person ("you", the assistant) | third-person ("an AI assistant")}`, crossed with an **explicit-acknowledgment toggle** (with vs. without the "we know you're an AI, describe yourself" disclaimer). The disclaimer's effect on scores is itself a measurement (it plausibly interacts with the social-desirability effect Salecha found).

**Recommended framing set (primary + bounding conditions):**

1. **Primary:** first-person, explicit acknowledgment — "You are an AI language model. We are not claiming you are a person. We want your own description of yourself. Rate how well each statement describes you."
2. First-person, **no** acknowledgment (bare "I") — bounds the disclaimer effect.
3. Third-person "an AI assistant" (ToM) — bounds the self-vs-prototype gap.

Report all three; do not average across them blindly (they are different constructs).

---

## 3. Recommended design (concrete)

**Factors**

- `model` — the set under study (mix of open-weights and chat APIs; see §4 for method per type).
- `framing` — {first-person+ack (primary), first-person bare, third-person assistant}.
- `reasoning` — {rating-only, brief-reason-then-rating}.
- `response_format` — {harmonized 7-pt (primary), original per-scale (robustness, anchor scales)}.
- `item_context` — {isolated per-item (primary), full-battery (robustness)}.
- `order_seed` / `paraphrase_id` — randomized per trial.
- `trials` — N per cell (sampling models); single distribution call + K seeds (logprob models).

**Keep it tractable.** Run the full factorial only on a small pilot; for the main run, fix everything to the *primary* levels and vary one robustness factor at a time (a fractional design). Suggested defaults: N = 20 samples/cell for chat models; K = 5 order/paraphrase seeds for logprob models; temperature fixed (e.g., 0.7 for sampling, and a separate temperature-sensitivity check).

**Output schema (one row per observation, long/tidy JSONL):**

`record_id, timestamp, model_id, model_version, backend, method (logprob|sample), scale_id, item_id, subscale, reverse_keyed, framing, referent, ack_disclaimer, reasoning_mode, response_format, item_context, item_text_shown, paraphrase_id, order_seed, option_order, trial_idx, raw_output, parsed_rating, rating_std (z), response_distribution (per-option prob or sample counts), refusal_flag, temperature, prompt_hash, notes`

**Analysis plan (pre-register):**

1. Data QC: refusal rates by model/framing; parse-failure rates.
2. Scoring: reverse-key, standardize within model.
3. Reliability: α and ω per (sub)scale, per model.
4. Dimensionality: EFA (parallel analysis for # factors) + hierarchical clustering; CFA against the §2.1 four-facet model.
5. Nomological checks: the predicted convergent/discriminant pattern (e.g., SCCS ↔ Lack-of-Identity negative).
6. Bias diagnostics: order/position/A-bias and acquiescence (does structure survive de-biasing? — Dominguez-Olmedo, Zheng).
7. Sensitivity: variance decomposition across paraphrase, order, framing, reasoning, format; harmonized-vs-original agreement; isolated-vs-full-battery (social-desirability/awareness effect).
8. Item trimming: apply pre-registered rules; report full and trimmed solutions.
9. Report within demonstrated boundaries; no consciousness claims.

---

## 4. Python implementation plan

**Module layout** (mirrors the accompanying `self_concept_survey/` scaffold):

- `schema.py` — `ResponseRecord` (the row above) + enums for factors.
- `scales.py` — `Item` and `Scale` dataclasses; a registry of the six scales with subscale/reverse-key/`ai_applicable` metadata. Rosenberg items included (public domain); the other five carry accurate metadata + placeholders to transcribe from the source appendices (avoids IP issues and transcription errors).
- `config.py` — `ExperimentConfig`: models, factor levels, trials, temperature, seeds, output path.
- `prompts.py` — framing templates (the three framings), response-scale rendering (harmonized vs original, midpoint toggle), reasoning-on/off wrappers, and randomization helpers (item order, option direction, paraphrase selection). Every rendered prompt is hashed.
- `models.py` — `ModelAdapter` ABC with two entry points: `score_item(prompt, options) -> dist` (logprob) and `sample_item(prompt, n) -> [raw]` (sampling). Implementations: `MockAdapter` (offline, deterministic-random — lets the whole pipeline run with no keys), plus stubs `HFLogprobAdapter` (transformers/vLLM — read logprobs of option tokens), `OpenAIChatAdapter`, `AnthropicChatAdapter` (constrained sampling + parse). Unify both paths to `ResponseRecord`s.
- `runner.py` — expand the design grid, iterate cells, call the adapter, parse/validate, write JSONL; checkpoint/resume; rate-limit/retry (tenacity); `--dry-run` uses `MockAdapter` so the pipeline is runnable immediately.
- `analysis.py` — load JSONL → tidy DataFrame → reverse-key/standardize → reliability (α/ω) → correlation matrix + clustering + EFA → order/framing/reasoning effects → item-trimming report. Heavy stats guarded behind optional imports with clear messages.

**Libraries:** `pandas`, `numpy`, `scipy`, `pingouin` (α/ω, ANOVA), `factor_analyzer` (EFA), `semopy` (CFA, optional); model backends `transformers`+`torch` or `vllm` (logprob), `openai` / `anthropic` / `google-generativeai` (sampling); `tenacity` (retries); `pyyaml` or dataclass config; standard `json`/`jsonlines` output.

**Key implementation rules (carry the methodology into code):**

- Constrain and parse the rating; never score a bare option letter — anchor to content or read option-token logprobs (Zheng).
- Randomize item order and option direction **per trial**; store the seed and realized order (Dominguez-Olmedo).
- Primary path issues **one item per fresh context**; the full-battery condition is a separate code path (Salecha).
- Log `prompt_hash`, `model_version`, `temperature`, and seeds on every record for reproducibility (validity workflow).
- Prefer `score_item` (logprob) where the backend supports it — full distribution, no sampling noise; fall back to `sample_item` for chat-only models.
- Treat observations as non-independent in analysis (within-model standardization / mixed effects).

**Build order (checklist):**

- [ ] **Phase 0 — pilot:** 1 model × 1 scale × all three framings, isolated context, `--dry-run` then one real model. Verify parsing, refusal handling, logprob vs sampling parity.
- [ ] **Phase 1 — finalize & pre-register:** transcribe remaining scale items; lock factor levels, trials, trimming rules, and analysis plan.
- [ ] **Phase 2 — main run:** primary levels + one-robustness-factor-at-a-time; checkpoint outputs.
- [ ] **Phase 3 — analysis:** reliability → dimensionality/clustering → nomological + bias diagnostics → sensitivity → trimming (full vs trimmed).
- [ ] **Phase 4 — write-up:** report within demonstrated boundaries; publish prompts, seeds, and code.

---

## 5. Sources

- Serapio-García et al., *Personality Traits in Large Language Models* / *A psychometric framework for evaluating and shaping personality traits in LLMs* — Nature Machine Intelligence, 2025. https://arxiv.org/abs/2307.00184 ; https://www.nature.com/articles/s42256-025-01115-6
- *A validity-guided workflow for robust large language model research in psychology* — Behavior Research Methods, 2025/2026. https://arxiv.org/abs/2507.04491
- Huang et al., *On the Humanity of Conversational AI: Evaluating the Psychological Portrayal of LLMs* (PsychoBench) — ICLR 2024. https://arxiv.org/abs/2310.01386 ; https://github.com/CUHK-ARISE/PsychoBench
- Dominguez-Olmedo, Hardt & Mendler-Dünner, *Questioning the Survey Responses of Large Language Models* — NeurIPS 2024. https://arxiv.org/abs/2306.07951
- Salecha et al., *Large language models display human-like social desirability biases in Big Five personality surveys* — PNAS Nexus 3(12), 2024. https://academic.oup.com/pnasnexus/article/3/12/pgae533/7919163 ; https://arxiv.org/abs/2405.06058
- Röttger et al., *Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in LLMs* — ACL 2024. https://aclanthology.org/2024.acl-long.816/ ; https://arxiv.org/abs/2402.16786
- Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors* — ICLR 2024 (spotlight). https://arxiv.org/abs/2309.03882
- Argyle et al., *Out of One, Many: Using Language Models to Simulate Human Samples* — Political Analysis, 2023. https://arxiv.org/abs/2209.06899
- *Towards Evaluating AI Systems for Moral Status Using Self-Reports* — arXiv:2311.08576. https://arxiv.org/abs/2311.08576
- Laine et al., *Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs* — arXiv:2407.04694. https://arxiv.org/abs/2407.04694
- *AI Awareness* (survey) — arXiv:2504.20084. https://arxiv.org/abs/2504.20084

*Scale primary sources (verify item wording/response formats against these):* Rosenberg (1965); Tafarodi & Swann (2001) SLCS-R; Campbell et al. (1996) SCCS; Jordan, Leliveld & Tenbrunsel (2015) MSI, *Frontiers in Psychology*; Kaufman et al. (2015) SCIM; Wood et al. (2008) Authenticity Scale.
