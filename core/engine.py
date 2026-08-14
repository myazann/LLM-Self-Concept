"""The run loop, shared by both instruments.

Everything that is true of administering ANY instrument lives here: which
measurement method a model supports, how many trials a cell gets, resume,
loading each set of weights exactly once, batching writes, the heartbeat and
status file, and what happens when a cell or a whole model fails.

What differs between instruments is expressed by an `Instrument`: what the cells
are, how one is identified, how it renders, and who the resulting row is about.
`survey/grid.py` and `welfare/grid.py` are the two implementations.

Keeping the loop here is not just deduplication. `methods_for` and `n_trials_for`
decide how a model is administered, and if the two instruments each carried their
own copy they could drift — at which point a welfare preference and a battery
self-report would no longer have been measured the same way, and the comparison
between them (the whole point of running both) would be confounded.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from core import obs
from core.models import (
    LOGPROB_TEMPERATURE,
    InsufficientDiskSpaceError,
    build_adapter,
    parse_rating,
)
from core.schema import (
    Method, Module, ReasoningMode, ResponseRecord, completed_cells, write_jsonl,
)

# How many records to buffer before an fsync'd append (bounds worst-case loss
# on a hard kill) and how often to emit a heartbeat / refresh the status file.
BATCH = 25
# Logprob mass on valid option tokens below this is flagged: the model wanted to
# say something other than a bare number. Data, not an error — but worth seeing.
LOW_COVERAGE = 0.5


# ---------------------------------------------------------------------------
# administration rules — identical for every instrument, on purpose
# ---------------------------------------------------------------------------
def methods_for(spec, cfg) -> list:
    methods = []
    if cfg.sample_baseline and spec.supports_sample:
        methods.append(Method.SAMPLE.value)
    if cfg.logprob_where_available and spec.supports_logprob:
        methods.append(Method.LOGPROB.value)
    if not methods:
        raise ValueError(f"{spec.alias}: no measurement method enabled.")
    return methods


def n_trials_for(spec, cfg, method: str) -> int:
    if method == Method.SAMPLE.value:
        n = cfg.n_samples_override or spec.n_samples
    else:
        n = cfg.n_seeds_override or spec.n_seeds
    if n % 2:
        # Direction is counterbalanced against trial_idx, so an odd trial count
        # gives every cell one extra ascending administration.
        obs.get_logger().warning(
            "%s: n_trials=%d is odd for method=%s — option direction cannot be "
            "balanced within a cell. Use an even count.", spec.alias, n, method,
        )
    return n


def cell_seed(base: int, *parts) -> int:
    """Deterministic per-cell seed. Same cell -> same seed on a resumed run."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return base + (int(digest, 16) % 100_000)


def scoped_specs(registry, cfg) -> list:
    """Models in scope for a run, before any instrument-specific exclusion."""
    return registry.select(
        families=list(cfg.scope.families) if cfg.scope.families else None,
        backends=list(cfg.scope.backends) if cfg.scope.backends else None,
        kinds=list(cfg.scope.kinds),
        size_tiers=list(cfg.scope.size_tiers) if cfg.scope.size_tiers else None,
        include_anchors=cfg.scope.include_anchors,
        include_disabled=cfg.scope.include_disabled,
    )


# ---------------------------------------------------------------------------
# what the engine hands back to the instrument
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """One cell's result, however it was obtained."""
    raw: str = ""
    parsed: Optional[float] = None
    modal: Optional[float] = None
    dist: dict = field(default_factory=dict)
    coverage: Optional[float] = None
    refusal: bool = False
    parse_failed: bool = False
    notes: str = ""
    plan: object = None


def build_record(*, spec, method, trial_idx, reasoning_mode, paraphrase_id,
                 prompt, rendered_scale, seed, observation, identity) -> ResponseRecord:
    """Assemble the row. `identity` is the instrument's answer to "about what?".

    Everything else — provenance, the realized prompt, the observation, the
    reproducibility fields — is the same whichever instrument asked.
    """
    plan = observation.plan
    return ResponseRecord(
        record_id=str(uuid.uuid4()),
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        **spec.provenance(),
        method=method,
        **identity,
        reasoning_mode=reasoning_mode,
        reasoning_applied=(plan.applied if plan is not None else "n/a"),
        reasoning_standardized=(plan.standardized if plan is not None else True),
        n_scale_points=rendered_scale.n_points,
        paraphrase_id=paraphrase_id,
        order_seed=seed,
        option_order=list(prompt.option_order),
        trial_idx=trial_idx,
        item_text_shown=prompt.item_text_shown,
        raw_output=observation.raw,
        parsed_rating=observation.parsed,
        rating_std=None,
        response_distribution=observation.dist,
        refusal_flag=observation.refusal,
        parse_failed=observation.parse_failed,
        # The temperature that actually produced this row. LOGPROB reads the
        # distribution at the answer position and never samples, so recording
        # the spec's sampling temperature (0.7) there would describe a knob that
        # was not in play. Sourced from core.models.LOGPROB_TEMPERATURE so the record
        # and the adapters cannot drift apart.
        temperature=(LOGPROB_TEMPERATURE if method == Method.LOGPROB.value
                     else spec.temperature),
        prompt_hash=prompt.prompt_hash,
        prompt_system=prompt.system,
        prompt_user=prompt.user,
        modal_rating=observation.modal,
        option_mass_coverage=observation.coverage,
        notes=observation.notes,
    )


# ---------------------------------------------------------------------------
# the instrument contract
# ---------------------------------------------------------------------------
class Instrument(ABC):
    """One study: what its cells are and how each becomes a row.

    Implementations are pure with respect to the model — `expand`, `cell_key`
    and `render` make no calls — so a grid can be planned, counted and diffed
    offline before anything is loaded.
    """

    #: `core.schema.Module` value stamped on every row this instrument writes.
    module: str = Module.BATTERY.value
    #: Short name for logs, the log filename, and the status file.
    label: str = "primary"

    def __init__(self, cfg):
        self.cfg = cfg

    # -- run identity ------------------------------------------------------
    @property
    def out_path(self) -> str:
        return self.cfg.out_path

    def config_hash(self) -> str:
        return obs.config_hash(self.cfg)

    # -- the grid ----------------------------------------------------------
    def select_specs(self, registry, log) -> list:
        """Models this instrument will administer. Override to exclude."""
        return scoped_specs(registry, self.cfg)

    @abstractmethod
    def expand(self, specs):
        """Yield one dict per design cell. Pure — makes no API calls."""

    @abstractmethod
    def cell_key(self, cell) -> str:
        """Stable identity of a cell — the unit of resume."""

    @abstractmethod
    def render(self, cell):
        """-> (RenderedPrompt, RenderedScale, order_seed)."""

    @abstractmethod
    def record(self, cell, prompt, rendered_scale, seed, observation) -> ResponseRecord:
        """Turn one observation into a row."""

    @abstractmethod
    def describe(self, cell) -> str:
        """A short human label for this cell, used in error logs."""


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def run(instrument: Instrument, *, dry_run=False, model_filter=None, limit=None,
        out_path=None, logger=None) -> dict:
    """Administer every outstanding cell of `instrument`. Returns run tallies."""
    log = logger or obs.get_logger()
    cfg = instrument.cfg
    out = out_path or instrument.out_path

    from core.model_registry import load_registry
    registry = load_registry()

    specs = instrument.select_specs(registry, log)
    if model_filter:
        wanted = set(model_filter)
        specs = [s for s in specs if s.alias in wanted]
        missing = wanted - {s.alias for s in specs}
        if missing:
            raise SystemExit(
                f"Not in scope or unknown: {', '.join(sorted(missing))}\n"
                f"Known aliases: {', '.join(registry.aliases())}"
            )
    if not specs:
        raise SystemExit(
            f"No models selected for the {instrument.label} run. Widen `scope`.")

    done = completed_cells(out) if cfg.resume else set()

    # Expand the grid once (pure — no calls) and split each model into
    # planned-vs-remaining, so resume is *visible*: we can say exactly how many
    # cells are already on disk before loading a single weight.
    planned, pending, remaining, resumable = {}, {}, {}, 0
    for spec in specs:
        all_cells = list(instrument.expand([spec]))
        planned[spec.alias] = len(all_cells)
        todo = [c for c in all_cells if instrument.cell_key(c) not in done]
        pending[spec.alias] = len(todo)          # still-to-do, before any --limit cap
        resumable += len(all_cells) - len(todo)
        if limit:
            todo = todo[:limit]
        remaining[spec.alias] = todo

    total_planned = sum(planned.values())
    already = resumable
    to_run = sum(len(v) for v in remaining.values())

    # Resume banner + config-drift guard.
    if cfg.resume and already:
        log.info("resuming %s: %d / %d cells already done", out, already, total_planned)
    prev = obs.read_status(out)
    cfg_hash = instrument.config_hash()
    if prev and prev.get("config_hash") not in (None, cfg_hash):
        log.warning("config_hash changed since the last run on this file "
                    "(%s -> %s) — the design differs, so resumed cell-keys may "
                    "not line up. Confirm this is intended.",
                    prev.get("config_hash"), cfg_hash)
    log.info("plan: %d model(s) | %d cell(s) to run now%s | %s | out=%s",
             len(specs), to_run, f" (limit {limit}/model)" if limit else "",
             instrument.label, out)

    log_path = next((h.baseFilename for h in log.handlers
                     if isinstance(h, logging.FileHandler)), "")
    status = obs.StatusWriter(out, run_id=obs.run_stamp(),
                              label=instrument.label,
                              cfg_hash=cfg_hash, log_path=log_path, planned=planned)
    status.set_totals(already_done=already)
    killer = obs.GracefulKiller(log)
    stats = Counter()
    aborted = False
    run_started = time.monotonic()

    def heartbeat(spec, cells, i):
        elapsed = time.monotonic() - run_started
        written = stats["written"]
        rate = written / elapsed if elapsed > 0 else 0.0
        eta = (to_run - written) / rate if rate > 0 else None
        log.info("[%s] %d/%d | %.2f cells/s | ETA %s | "
                 "err=%d refus=%d pfail=%d lowcov=%d",
                 spec.alias, i, len(cells), rate, obs.fmt_duration(eta),
                 stats["errors"], stats["refusals"], stats["parse_failures"],
                 stats["low_coverage"])
        status.state["current_model"] = spec.alias
        status.state["rate_cells_per_s"] = round(rate, 3)
        status.state["eta_seconds"] = int(eta) if eta else None
        status.state["models"][spec.alias]["done"] = (planned[spec.alias] - pending[spec.alias]) + i
        status.state["totals"].update(
            written_this_run=written, refusals=stats["refusals"],
            parse_failures=stats["parse_failures"], errors=stats["errors"],
            low_coverage=stats["low_coverage"])
        status.flush()

    # Group by model so each set of weights is loaded exactly once.
    for spec in specs:
        cells = remaining[spec.alias]
        done_before = planned[spec.alias] - pending[spec.alias]  # already on disk
        if not cells:
            log.info("[%s] nothing to do (%d/%d cells already on disk)",
                     spec.alias, done_before, planned[spec.alias])
            status.set_model(spec.alias,
                             state="done" if pending[spec.alias] == 0 else "partial",
                             done=done_before)
            continue

        if not dry_run and spec.quantization.format == "gguf":
            spec = registry.with_resolved_quant(spec)
            for c in cells:
                c["spec"] = spec
            log.info("[%s] quant file: %s", spec.alias, spec.quantization.resolved_file)

        log.info("[%s] %d cell(s) to run  (%s, %s)",
                 spec.alias, len(cells), spec.backend, spec.quantization.label)
        status.set_model(spec.alias, state="running")
        status.update(current_model=spec.alias)

        model_started = time.monotonic()
        m = Counter()          # per-model tallies, for the summary line
        cov_sum = cov_n = 0
        first_error_logged = False

        try:
            with build_adapter(spec, dry_run=dry_run) as adapter:
                batch = []
                for i, cell in enumerate(cells, 1):
                    if killer.stop:
                        break
                    prompt, rendered_scale, seed = instrument.render(cell)
                    # Assert the reasoning state per the reasoning_mode factor;
                    # the adapter never relies on a model's default.
                    want_thinking = (
                        cell["reasoning_mode"] == ReasoningMode.REASON_THEN_RATING.value
                    )
                    plan = adapter.reasoning_plan(want_thinking)
                    try:
                        if cell["method"] == Method.LOGPROB.value:
                            dist, coverage = adapter.score_item(
                                prompt, prompt.option_values, plan=plan)
                            expected = (
                                sum(v * p for v, p in dist.items()) if dist else None
                            )
                            modal = max(dist, key=dist.get) if dist else None
                            cov_sum += coverage
                            cov_n += 1
                            if coverage < LOW_COVERAGE:
                                stats["low_coverage"] += 1
                                m["low_coverage"] += 1
                            observation = Observation(
                                raw="<logprob>", parsed=expected, modal=modal,
                                coverage=coverage,
                                dist={str(k): v for k, v in dist.items()},
                                refusal=False, parse_failed=not dist, plan=plan,
                            )
                        else:
                            raw = adapter.sample_item(prompt, n=1, plan=plan)[0]
                            rating, refusal, failed = parse_rating(raw, prompt.option_values)
                            observation = Observation(
                                raw=raw, parsed=rating, modal=rating, dist={},
                                refusal=refusal, parse_failed=failed, plan=plan,
                            )
                    except Exception as err:  # noqa: BLE001 — one bad cell must not kill the run
                        stats["errors"] += 1
                        m["errors"] += 1
                        if not first_error_logged:  # full traceback once per model
                            log.exception("[%s] first error, at %s (%s)",
                                          spec.alias, instrument.describe(cell),
                                          cell["method"])
                            first_error_logged = True
                        else:
                            log.error("[%s] error at %s: %s: %s", spec.alias,
                                      instrument.describe(cell), type(err).__name__, err)
                        observation = Observation(
                            raw="", parsed=None, dist={}, refusal=False,
                            parse_failed=True, plan=plan,
                            notes=f"error: {type(err).__name__}: {err}",
                        )

                    rec = instrument.record(cell, prompt, rendered_scale, seed, observation)
                    batch.append(rec)
                    stats["written"] += 1
                    m["written"] += 1
                    stats["refusals"] += int(rec.refusal_flag)
                    m["refusals"] += int(rec.refusal_flag)
                    stats["parse_failures"] += int(rec.parse_failed)
                    m["parse_failures"] += int(rec.parse_failed)

                    if len(batch) >= BATCH:
                        write_jsonl(batch, out)
                        batch = []
                        heartbeat(spec, cells, i)
                if batch:
                    write_jsonl(batch, out)
        except KeyboardInterrupt:
            # Some native download backends translate SIGINT into
            # KeyboardInterrupt even though GracefulKiller installed a handler.
            # Treat it as the same resumable shutdown, without a traceback.
            killer.stop = True
            log.warning("[%s] interrupted during model setup/inference; "
                        "state is safe and the next run will resume", spec.alias)
            status.set_model(spec.alias, state="interrupted",
                             done=done_before + m["written"])
            break
        except InsufficientDiskSpaceError as err:
            # Every following GGUF uses the same cache filesystem, so continuing
            # would just repeat the failure for each model.
            aborted = True
            stats["errors"] += 1
            log.error("[%s] model download aborted — %s", spec.alias, err)
            status.set_model(spec.alias, state="error",
                             done=done_before + m["written"])
            break
        except NotImplementedError:
            raise  # design guard (e.g. full_battery) — fail loudly, don't skip
        except Exception:  # noqa: BLE001 — a model that won't load must not kill the others
            log.exception("[%s] fatal error (model load / adapter) — skipping", spec.alias)
            status.set_model(spec.alias, state="error", done=done_before + m["written"])
            # Per-cell errors were counted when their records were created;
            # count the model-level failure itself exactly once.
            stats["errors"] += 1
            if killer.stop:
                break
            continue

        elapsed = time.monotonic() - model_started
        mean_cov = (cov_sum / cov_n) if cov_n else None
        cov_str = f" | mean coverage={mean_cov:.3f}" if mean_cov is not None else ""
        interrupted = killer.stop
        complete = (not interrupted) and (m["written"] >= pending[spec.alias])
        state = "stopped" if interrupted else ("done" if complete else "partial")
        log.info("[%s] %s: %d cell(s) in %s%s | err=%d refus=%d pfail=%d lowcov=%d",
                 spec.alias, state, m["written"], obs.fmt_duration(elapsed), cov_str,
                 m["errors"], m["refusals"], m["parse_failures"], m["low_coverage"])
        status.set_model(
            spec.alias,
            state="interrupted" if interrupted else ("done" if complete else "partial"),
            done=done_before + m["written"],
            mean_coverage=round(mean_cov, 4) if mean_cov is not None else None,
        )
        heartbeat(spec, cells, m["written"])

        if interrupted:
            log.warning("shutdown signal — stopping after %s; %d cell(s) still "
                        "pending. Re-run the same command to resume.",
                        spec.alias, to_run - stats["written"])
            break

    if aborted:
        stats["aborted"] = 1
    final_state = "error" if aborted else ("interrupted" if killer.stop else "done")
    status.update(state=final_state, current_model=None)
    return dict(stats)


# ---------------------------------------------------------------------------
# thinking-off verification (no weights loaded — tokenizer only)
# ---------------------------------------------------------------------------
def verify_thinking(prompt, specs) -> str:
    """Confirm the enable_thinking toggle actually takes effect on each
    hybrid-thinking local model, WITHOUT loading any GGUF weights.

    For a template_toggle model the rating_only render (enable_thinking=False)
    must (a) contain a closed `</think>` block — the template pre-filled it, so
    the next token is the answer — and (b) differ from the enable_thinking=True
    render. If they are identical, the toggle was ignored and thinking is NOT
    off. No-thinking models (Gemma) pass trivially (prompt-controlled).

    `prompt` is a rendered cell from the calling instrument, so each module
    verifies against a prompt it actually administers.
    """
    from core.models import ChatTemplateRenderer

    lines = ["thinking-off verification (rating_only render; tokenizer only):", ""]
    for s in specs:
        if s.reasoning.control != "template_toggle":
            lines.append(f"  {s.alias:<22} control=none          -> prompt-controlled (no native thinking)")
            continue
        try:
            r = ChatTemplateRenderer(s.tokenizer_id or s.hf_id, required=True)
            off = r.render(prompt.system, prompt.user, enable_thinking=False)
            on = r.render(prompt.system, prompt.user, enable_thinking=True)
            closed = "</think>" in off
            differs = off != on
            verdict = "OK" if (closed and differs) else "IGNORED — thinking NOT off"
            lines.append(
                f"  {s.alias:<22} enable_thinking={verdict}"
                f"   (closed_block={closed}, differs_from_on={differs})"
            )
        except Exception as err:  # noqa: BLE001
            lines.append(f"  {s.alias:<22} could not verify: {type(err).__name__}: {err}")
    return "\n".join(lines)
