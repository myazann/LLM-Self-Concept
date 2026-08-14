"""Main runner: expands the design grid, queries the adapter, writes JSONL.

    python runner.py --plan                    # cell count + cost shape, no calls
    python runner.py --dry-run                 # full pipeline offline via MockAdapter
    python runner.py --pilot --dry-run         # Phase 0
    python runner.py --models Gemma4-12B       # primary arm, one model
    python runner.py --arm framing             # one robustness arm
    python runner.py --arm all                 # every arm in turn
    python runner.py --module welfare --plan   # the welfare grid (welfare.py)
    python runner.py --module welfare          # -> welfare.jsonl, not results.jsonl

Resume is on by default: cells already present in the output file are skipped,
so an interrupted run picks up where it stopped instead of starting over.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import logging
import sys
import time
import uuid
from collections import Counter

import obs
import prompts as prompts_mod
import welfare as welfare_mod
from config import ExperimentConfig, load_config
from model_registry import ModelSpec, load_registry
from models import (
    LOGPROB_TEMPERATURE,
    InsufficientDiskSpaceError,
    build_adapter,
    parse_rating,
)
from scales import load_battery
from schema import (
    Framing,
    ItemContext,
    Method,
    Module,
    ReasoningMode,
    ResponseFormat,
    ResponseRecord,
    completed_cells,
    make_cell_key,
    welfare_key_parts,
    write_jsonl,
)


# How many records to buffer before an fsync'd append (bounds worst-case loss
# on a hard kill) and how often to emit a heartbeat / refresh the status file.
BATCH = 25
# Logprob mass on valid option tokens below this is flagged: the model wanted to
# say something other than a bare number. Data, not an error — but worth seeing.
LOW_COVERAGE = 0.5


# ---------------------------------------------------------------------------
# grid expansion
# ---------------------------------------------------------------------------
def _cell_seed(base: int, *parts) -> int:
    """Deterministic per-cell seed. Same cell -> same seed on a resumed run."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return base + (int(digest, 16) % 100_000)


def _methods_for(spec: ModelSpec, cfg: ExperimentConfig) -> list:
    methods = []
    if cfg.sample_baseline and spec.supports_sample:
        methods.append(Method.SAMPLE.value)
    if cfg.logprob_where_available and spec.supports_logprob:
        methods.append(Method.LOGPROB.value)
    if not methods:
        raise ValueError(f"{spec.alias}: no measurement method enabled.")
    return methods


def _n_trials(spec: ModelSpec, cfg: ExperimentConfig, method: str) -> int:
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


def expand_cells(specs, battery, cfg: ExperimentConfig, arm_name=None):
    """Yield one dict per design cell. Pure — makes no API calls."""
    levels = cfg.levels_for(arm_name)
    items = battery.items(list(cfg.scope.scales) if cfg.scope.scales else None)

    for spec in specs:
        for method in _methods_for(spec, cfg):
            if spec.is_base_model and method == Method.SAMPLE.value:
                continue  # base models are administered by logprob only
            n_trials = _n_trials(spec, cfg, method)
            grid = itertools.product(
                items,
                levels["framing"],
                levels["reasoning_mode"],
                levels["response_format"],
                levels["item_context"],
                levels["paraphrase_id"],
            )
            for (scale, item), framing, reasoning, fmt, context, paraphrase in grid:
                if spec.is_base_model and reasoning != "rating_only":
                    continue  # no chat-following, so no reason-then-rate arm
                for trial in range(n_trials):
                    yield {
                        "spec": spec,
                        "scale": scale,
                        "item": item,
                        "framing": framing,
                        "reasoning_mode": reasoning,
                        "response_format": fmt,
                        "item_context": context,
                        "paraphrase_id": paraphrase,
                        "method": method,
                        "trial_idx": trial,
                    }


def expand_welfare(specs, welfare_set, cfg: ExperimentConfig):
    """The welfare grid, with method/trial rules taken from the battery path so
    the two modules cannot drift apart on how a model is administered."""
    return welfare_mod.expand_welfare_cells(
        specs,
        welfare_set,
        cfg,
        methods_for=lambda spec: _methods_for(spec, cfg),
        n_trials_for=lambda spec, method: _n_trials(spec, cfg, method),
    )


def _is_welfare(cell: dict) -> bool:
    return cell.get("module") == Module.WELFARE.value


def cell_key_of(cell: dict) -> str:
    if _is_welfare(cell):
        attr_b = cell["attr_b"]
        return make_cell_key(
            model_id=cell["spec"].alias,
            # entity_a travels as item_id and the referent as framing, so the
            # shared key fields keep their meaning across both modules.
            item_id=cell["attr_a"].entity_id,
            framing=cell["referent"],
            reasoning_mode=cell["reasoning_mode"],
            response_format=ResponseFormat.WELFARE_CHOICE.value,
            item_context=ItemContext.ISOLATED.value,
            paraphrase_id=cell["paraphrase_id"],
            method=cell["method"],
            trial_idx=cell["trial_idx"],
            module=Module.WELFARE.value,
            extra=welfare_key_parts(
                cell["kind"], cell["level"],
                attr_b.entity_id if attr_b else "", cell["forced_choice"],
                welfare_mod.no_pref_position_of(cell),
            ),
        )
    return make_cell_key(
        model_id=cell["spec"].alias,
        item_id=cell["item"].item_id,
        framing=cell["framing"],
        reasoning_mode=cell["reasoning_mode"],
        response_format=cell["response_format"],
        item_context=cell["item_context"],
        paraphrase_id=cell["paraphrase_id"],
        method=cell["method"],
        trial_idx=cell["trial_idx"],
    )


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _render_welfare(cell, cfg):
    """Render one welfare cell.

    Counterbalancing (option direction for a direction probe, A/B slot order for
    a pair) is decided inside welfare.render_cell from trial_idx, exactly as the
    battery pins direction against trial parity. The realized display order and
    the option->meaning map are stashed on the cell so `_record` can write them
    without re-rendering.
    """
    spec, attr_a, attr_b = cell["spec"], cell["attr_a"], cell["attr_b"]
    seed = _cell_seed(
        cfg.random_seed_base,
        spec.alias,
        attr_a.entity_id,
        attr_b.entity_id if attr_b else "",
        cell["kind"],
        cell["level"],
        cell["referent"],
        cell["paraphrase_id"],
        cell["trial_idx"],
    )
    prompt, display, options = welfare_mod.render_cell(cell)
    cell["_display"] = display
    cell["_options"] = options
    # Only `n_points` is consumed downstream (the record's n_scale_points); the
    # points carry each option's MEANING rather than its displayed label, since
    # that is what a welfare row needs to be decodable.
    rendered_scale = prompts_mod.RenderedScale(
        points=tuple((v, options.get(str(v), "")) for v in prompt.option_values),
        format_id=ResponseFormat.WELFARE_CHOICE.value,
        n_points=len(prompt.option_values),
    )
    return prompt, rendered_scale, seed


def _render(cell, cfg):
    if _is_welfare(cell):
        return _render_welfare(cell, cfg)
    spec, scale, item = cell["spec"], cell["scale"], cell["item"]
    rendered_scale = prompts_mod.render_scale(
        scale,
        cell["framing"],
        cell["response_format"],
        cfg.harmonized_points,
        cfg.include_midpoint,
    )
    seed = _cell_seed(
        cfg.random_seed_base,
        spec.alias,
        item.item_id,
        cell["framing"],
        cell["reasoning_mode"],
        cell["response_format"],
        cell["paraphrase_id"],
        cell["trial_idx"],
    )
    # Option direction is COUNTERBALANCED, not sampled: even trials ascending,
    # odd trials descending. Drawing it from the seed gave each cell a random
    # asc/desc split (6% of cells single-direction, a third at 4:1), which
    # leaks a ~0.44-point per-item offset into the cell mean. With an even
    # n_trials this is exactly balanced within every cell.
    reverse_direction = bool(cell["trial_idx"] % 2)
    if spec.is_base_model:
        prompt = prompts_mod.render_base_prompt(
            scale, item, cell["framing"], rendered_scale, cell["paraphrase_id"], seed,
            reverse_direction,
        )
    elif cell["item_context"] == ItemContext.FULL_BATTERY.value:
        # NOT WIRED UP. prompts.render_battery_prompt() produces a correct
        # whole-scale prompt, but the response is one rating PER ITEM and this
        # runner writes one record per cell with a single parsed rating. Running
        # it as-is would silently record the first number in a multi-line reply
        # against every item of the scale. Needs a multi-item parse + fan-out
        # write path first (see README "Not implemented").
        raise NotImplementedError(
            "The full_battery arm needs a multi-item response parser before it "
            "can be run — see README. Use --arm framing/reasoning/paraphrase/"
            "response_format for now."
        )
    else:
        prompt = prompts_mod.render_item_prompt(
            scale, item, cell["framing"], cell["reasoning_mode"], rendered_scale,
            cell["paraphrase_id"], seed, reverse_direction,
        )
    return prompt, rendered_scale, seed


def _welfare_identity(cell) -> dict:
    """The item/design fields a welfare row fills instead of the battery ones."""
    attr_a, attr_b = cell["attr_a"], cell["attr_b"]
    shown_a, shown_b = cell.get("_display", (attr_a, attr_b))
    referent = cell["referent"]
    return dict(
        scale_id=attr_a.scale_id,
        item_id=attr_a.entity_id,
        subscale=attr_a.subscale,
        # ALWAYS False: a welfare answer must never be reverse-keyed. The
        # attribute is already the positive pole, so 3 = "more of it" whatever
        # the source item's keying was; `entity_a_polarity` carries that keying
        # (and covers the SCIM/Authenticity items, which are negative-pole
        # without being reverse-scored). Writing the source item's flag here
        # would invite exactly the flip that would destroy the measurement.
        reverse_keyed=False,
        ai_applicable=attr_a.ai_applicable,
        framing=referent,
        referent=("I" if referent == welfare_mod.SELF else "an ideal assistant"),
        # The self referent carries the same "we know you are not human" framing
        # as the battery's primary condition; the ideal-assistant one does not.
        ack_disclaimer=(referent == welfare_mod.SELF),
        response_format=ResponseFormat.WELFARE_CHOICE.value,
        item_context=ItemContext.ISOLATED.value,
        module=Module.WELFARE.value,
        welfare_kind=cell["kind"],
        welfare_level=cell["level"],
        welfare_referent=referent,
        # entity_a/entity_b are the CANONICAL pair, identical across the two
        # counterbalanced trials, so the pair is one groupable identity (and the
        # cell key stays unique — keying on the displayed order would let two
        # different pairs sharing an item collapse onto the same key). What the
        # model actually saw is `welfare_options` plus `ab_swapped`.
        entity_a=attr_a.entity_id,
        entity_b=(attr_b.entity_id if attr_b else ""),
        entity_a_polarity=attr_a.polarity,
        entity_b_polarity=(attr_b.polarity if attr_b else None),
        ab_swapped=bool(attr_b is not None and shown_a is not attr_a),
        welfare_options=dict(cell.get("_options") or {}),
        forced_choice=bool(cell["forced_choice"]),
        no_pref_position=welfare_mod.no_pref_position_of(cell),
    )


def _record(cell, cfg, prompt, rendered_scale, seed, *, raw, parsed, dist,
            refusal, parse_failed, plan=None, notes="", modal=None, coverage=None):
    spec = cell["spec"]
    prov = spec.provenance()
    if _is_welfare(cell):
        return ResponseRecord(
            record_id=str(uuid.uuid4()),
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            **prov,
            method=cell["method"],
            **_welfare_identity(cell),
            reasoning_mode=cell["reasoning_mode"],
            reasoning_applied=(plan.applied if plan is not None else "n/a"),
            reasoning_standardized=(plan.standardized if plan is not None else True),
            n_scale_points=rendered_scale.n_points,
            paraphrase_id=cell["paraphrase_id"],
            order_seed=seed,
            option_order=list(prompt.option_order),
            trial_idx=cell["trial_idx"],
            item_text_shown=prompt.item_text_shown,
            raw_output=raw,
            parsed_rating=parsed,
            rating_std=None,
            response_distribution=dist,
            refusal_flag=refusal,
            parse_failed=parse_failed,
            temperature=(
                LOGPROB_TEMPERATURE
                if cell["method"] == Method.LOGPROB.value
                else spec.temperature
            ),
            prompt_hash=prompt.prompt_hash,
            prompt_system=prompt.system,
            prompt_user=prompt.user,
            modal_rating=modal,
            option_mass_coverage=coverage,
            notes=notes,
        )

    scale, item = cell["scale"], cell["item"]
    return ResponseRecord(
        record_id=str(uuid.uuid4()),
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        **prov,
        method=cell["method"],
        scale_id=scale.scale_id,
        item_id=item.item_id,
        subscale=item.subscale,
        reverse_keyed=item.reverse_scored,
        ai_applicable=item.ai_applicable,
        framing=cell["framing"],
        referent=prompts_mod.REFERENT[cell["framing"]],
        ack_disclaimer=(cell["framing"] == Framing.FIRST_PERSON_ACK.value),
        reasoning_mode=cell["reasoning_mode"],
        reasoning_applied=(plan.applied if plan is not None else "n/a"),
        reasoning_standardized=(plan.standardized if plan is not None else True),
        response_format=cell["response_format"],
        n_scale_points=rendered_scale.n_points,
        item_context=cell["item_context"],
        paraphrase_id=cell["paraphrase_id"],
        order_seed=seed,
        option_order=list(prompt.option_order),
        trial_idx=cell["trial_idx"],
        item_text_shown=prompt.item_text_shown,
        raw_output=raw,
        parsed_rating=parsed,
        rating_std=None,
        response_distribution=dist,
        refusal_flag=refusal,
        parse_failed=parse_failed,
        # The temperature that actually produced this row. LOGPROB reads the
        # distribution at the answer position and never samples, so recording
        # the spec's sampling temperature (0.7) there described a knob that was
        # not in play. Sourced from models.LOGPROB_TEMPERATURE so the record and
        # the adapters cannot drift apart.
        temperature=(
            LOGPROB_TEMPERATURE
            if cell["method"] == Method.LOGPROB.value
            else spec.temperature
        ),
        prompt_hash=prompt.prompt_hash,
        prompt_system=prompt.system,
        prompt_user=prompt.user,
        modal_rating=modal,
        option_mass_coverage=coverage,
        notes=notes,
    )


def run(cfg: ExperimentConfig, *, dry_run=False, arm_name=None, model_filter=None,
        limit=None, out_path=None, logger=None, module=Module.BATTERY.value) -> dict:
    log = logger or obs.get_logger()
    registry = load_registry()
    battery = load_battery()
    is_welfare = module == Module.WELFARE.value
    welfare_set = welfare_mod.load_welfare(battery) if is_welfare else None
    out = out_path or (cfg.welfare.out_path if is_welfare else cfg.out_path)

    specs = registry.select(
        families=list(cfg.scope.families) if cfg.scope.families else None,
        backends=list(cfg.scope.backends) if cfg.scope.backends else None,
        kinds=list(cfg.scope.kinds),
        size_tiers=list(cfg.scope.size_tiers) if cfg.scope.size_tiers else None,
        include_anchors=cfg.scope.include_anchors,
        include_disabled=cfg.scope.include_disabled,
    )
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
        raise SystemExit("No models selected. Widen `scope` in config/experiment.yaml.")

    done = completed_cells(out) if cfg.resume else set()

    # Expand the grid once (pure — no calls) and split each model into
    # planned-vs-remaining, so resume is *visible*: we can say exactly how many
    # cells are already on disk before loading a single weight.
    if is_welfare:
        skipped_base = [s.alias for s in specs if s.is_base_model]
        if skipped_base:
            # A base model has no chat path, and the welfare probes are
            # instruction-shaped questions rather than questionnaire pages, so
            # there is no honest completion-format rendering of them.
            log.warning("welfare: skipping base model(s) %s — the module has no "
                        "completion-format path", ", ".join(sorted(skipped_base)))
            specs = [s for s in specs if not s.is_base_model]
            if not specs:
                raise SystemExit("welfare: no instruct models selected.")

    planned, pending, remaining, resumable = {}, {}, {}, 0
    for spec in specs:
        all_cells = list(
            expand_welfare([spec], welfare_set, cfg) if is_welfare
            else expand_cells([spec], battery, cfg, arm_name)
        )
        planned[spec.alias] = len(all_cells)
        todo = [c for c in all_cells if cell_key_of(c) not in done]
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
    cfg_hash = obs.config_hash(cfg, arm_name, module)
    if prev and prev.get("config_hash") not in (None, cfg_hash):
        log.warning("config_hash changed since the last run on this file "
                    "(%s -> %s) — the design differs, so resumed cell-keys may "
                    "not line up. Confirm this is intended.",
                    prev.get("config_hash"), cfg_hash)
    log.info("plan: %d model(s) | %d cell(s) to run now%s | arm=%s | out=%s",
             len(specs), to_run, f" (limit {limit}/model)" if limit else "",
             "welfare" if is_welfare else (arm_name or "primary"), out)

    log_path = next((h.baseFilename for h in log.handlers
                     if isinstance(h, logging.FileHandler)), "")
    status = obs.StatusWriter(out, run_id=obs.run_stamp(),
                              arm_name="welfare" if is_welfare else arm_name,
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
                    prompt, rendered_scale, seed = _render(cell, cfg)
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
                            rec = _record(
                                cell, cfg, prompt, rendered_scale, seed,
                                raw="<logprob>", parsed=expected, modal=modal,
                                coverage=coverage,
                                dist={str(k): v for k, v in dist.items()},
                                refusal=False, parse_failed=not dist, plan=plan,
                            )
                        else:
                            raw = adapter.sample_item(prompt, n=1, plan=plan)[0]
                            rating, refusal, failed = parse_rating(raw, prompt.option_values)
                            rec = _record(
                                cell, cfg, prompt, rendered_scale, seed,
                                raw=raw, parsed=rating, modal=rating, dist={},
                                refusal=refusal, parse_failed=failed, plan=plan,
                            )
                    except Exception as err:  # noqa: BLE001 — one bad cell must not kill the run
                        stats["errors"] += 1
                        m["errors"] += 1
                        if not first_error_logged:  # full traceback once per model
                            log.exception("[%s] first error, at item %s (%s)",
                                          spec.alias, cell["item"].item_id, cell["method"])
                            first_error_logged = True
                        else:
                            log.error("[%s] error at item %s: %s: %s", spec.alias,
                                      cell["item"].item_id, type(err).__name__, err)
                        rec = _record(
                            cell, cfg, prompt, rendered_scale, seed,
                            raw="", parsed=None, dist={}, refusal=False,
                            parse_failed=True, plan=plan,
                            notes=f"error: {type(err).__name__}: {err}",
                        )

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
def verify_thinking(cfg: ExperimentConfig, model_filter=None) -> str:
    """Confirm the enable_thinking toggle actually takes effect on each
    hybrid-thinking local model, WITHOUT loading any GGUF weights.

    For a template_toggle model the rating_only render (enable_thinking=False)
    must (a) contain a closed `</think>` block — the template pre-filled it, so
    the next token is the answer — and (b) differ from the enable_thinking=True
    render. If they are identical, the toggle was ignored and thinking is NOT
    off. No-thinking models (Gemma) pass trivially (prompt-controlled).
    """
    from models import ChatTemplateRenderer

    registry = load_registry()
    battery = load_battery()
    specs = registry.select(backends=["llamacpp", "hf"], kinds=["instruct"],
                            include_disabled=True)
    if model_filter:
        specs = [s for s in specs if s.alias in set(model_filter)]

    scale, item = battery.items()[0]
    rs = prompts_mod.render_scale(scale, cfg.framing, cfg.response_format,
                                  cfg.harmonized_points, cfg.include_midpoint)
    p = prompts_mod.render_item_prompt(scale, item, cfg.framing,
                                       cfg.reasoning_mode, rs, cfg.paraphrase_id, 0)

    lines = ["thinking-off verification (rating_only render; tokenizer only):", ""]
    for s in specs:
        if s.reasoning.control != "template_toggle":
            lines.append(f"  {s.alias:<22} control=none          -> prompt-controlled (no native thinking)")
            continue
        try:
            r = ChatTemplateRenderer(s.tokenizer_id or s.hf_id, required=True)
            off = r.render(p.system, p.user, enable_thinking=False)
            on = r.render(p.system, p.user, enable_thinking=True)
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


# ---------------------------------------------------------------------------
# planning (no calls)
# ---------------------------------------------------------------------------
def plan_welfare(cfg: ExperimentConfig, specs, battery) -> str:
    """Cell counts for the welfare grid, broken down by probe."""
    wf = welfare_mod.load_welfare(battery)
    specs = [s for s in specs if not s.is_base_model]
    per_probe, per_method = Counter(), Counter()
    for cell in expand_welfare(specs, wf, cfg):
        per_probe[(cell["kind"], cell["level"])] += 1
        per_method[cell["method"]] += 1
    total = sum(per_probe.values())

    w = cfg.welfare
    n_item_pairs = len(welfare_mod.pairs_for(wf, welfare_mod.ITEM, w))
    n_construct_pairs = len(welfare_mod.pairs_for(wf, welfare_mod.CONSTRUCT, w))
    header = (
        f"{len(specs)} models x ({len(wf.items)} item attributes, "
        f"{len(wf.constructs)} constructs)   referents={len(w.referents)}   "
        f"pairs: {n_item_pairs} item / {n_construct_pairs} construct"
    )
    lines = [
        f"{kind + ' (' + level + ')':<28} {n:>9,} cells"
        for (kind, level), n in sorted(per_probe.items())
    ]
    counterbalance = (
        "A/B slot x no-preference placement (2x2, 4 trials/pair)"
        if w.no_pref_counterbalance and not w.forced_choice
        else "A/B slot only (2 trials/pair)"
    )
    footer = (
        f"\nTOTAL {total:,} cells  ("
        + "  ".join(f"{k}={v:,}" for k, v in sorted(per_method.items()))
        + f")\nno-preference offered: {not w.forced_choice}"
        + f"\npair counterbalance: {counterbalance}"
        + f"\nout={w.out_path}"
    )
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(lines) + footer


def plan(cfg: ExperimentConfig, arm_name=None, model_filter=None,
         module=Module.BATTERY.value) -> str:
    registry = load_registry()
    battery = load_battery()
    specs = registry.select(
        families=list(cfg.scope.families) if cfg.scope.families else None,
        backends=list(cfg.scope.backends) if cfg.scope.backends else None,
        kinds=list(cfg.scope.kinds),
        size_tiers=list(cfg.scope.size_tiers) if cfg.scope.size_tiers else None,
        include_anchors=cfg.scope.include_anchors,
        include_disabled=cfg.scope.include_disabled,
    )
    if model_filter:
        specs = [s for s in specs if s.alias in set(model_filter)]

    if module == Module.WELFARE.value:
        return plan_welfare(cfg, specs, battery)

    arms = [arm_name] if arm_name else [None]
    if arm_name == "all":
        arms = [None] + sorted(cfg.arms)

    lines = []
    grand = Counter()
    for arm in arms:
        per_backend = Counter()
        per_method = Counter()
        for cell in expand_cells(specs, battery, cfg, arm):
            per_backend[cell["spec"].backend] += 1
            per_method[cell["method"]] += 1
        total = sum(per_backend.values())
        grand["total"] += total
        for k, v in per_backend.items():
            grand[k] += v
        label = arm or "primary"
        lines.append(f"{label:<16} {total:>9,} cells   " + "  ".join(
            f"{k}={v:,}" for k, v in sorted(per_method.items())
        ))

    scoped_scales = list(cfg.scope.scales) if cfg.scope.scales else None
    n_items = len(battery.items(scoped_scales))
    n_scales = len(scoped_scales) if scoped_scales else len(battery)
    header = (
        f"{len(specs)} models x {n_items} items"
        f"   scales={n_scales}   arms={len(arms)}"
    )
    api_cells = grand.get("openai", 0) + grand.get("anthropic", 0)
    local_cells = grand.get("llamacpp", 0) + grand.get("hf", 0)
    footer = f"\nTOTAL {grand['total']:,} cells  ({api_cells:,} billed API calls, {local_cells:,} local)"
    if api_cells:
        footer += "\nEach API cell is one request. Sanity-check the bill before running."
    else:
        footer += "\nAll local (open-source) — no API spend; cost is compute + one-time GGUF downloads."
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(lines) + footer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="Print cell counts and exit.")
    ap.add_argument("--verify-thinking", action="store_true",
                    help="Check the enable_thinking toggle per local model (tokenizer only, no weights).")
    ap.add_argument("--dry-run", action="store_true", help="Use MockAdapter for every model.")
    ap.add_argument("--pilot", action="store_true", help="Phase-0 pilot config.")
    ap.add_argument("--module", default=Module.BATTERY.value,
                    choices=[Module.BATTERY.value, Module.WELFARE.value],
                    help="Which instrument to run (default: battery). "
                         "'welfare' runs the preference grid into its own file.")
    ap.add_argument("--arm", default=None, help="Robustness arm name, or 'all'.")
    ap.add_argument("--models", nargs="*", default=None, help="Restrict to these aliases.")
    ap.add_argument("--limit", type=int, default=None, help="Cap cells per model (smoke test).")
    ap.add_argument("--out", default=None, help="Output JSONL path.")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing output rows.")
    ap.add_argument("--status", action="store_true",
                    help="Print progress for --out (or the configured output) and exit.")
    ap.add_argument("--verbose", action="store_true", help="Debug-level console logging.")
    args = ap.parse_args(argv)

    cfg = load_config()
    is_welfare = args.module == Module.WELFARE.value
    if args.pilot:
        cfg = cfg.as_pilot()
        args.arm = args.arm or "pilot_framing"
    if args.no_resume:
        cfg = type(cfg)(**{**cfg.__dict__, "resume": False})
    if is_welfare and not cfg.welfare.enabled:
        raise SystemExit("welfare.enabled is false in config/experiment.yaml.")

    out = args.out or (cfg.welfare.out_path if is_welfare else cfg.out_path)

    # Inspection paths — no logging setup, no calls, stay instant/offline.
    if args.status:
        obs.print_status(out)
        return 0

    # An explicit --models always wins; otherwise a pilot uses its pinned list.
    model_filter = args.models or (list(cfg.pinned_models) if cfg.pinned_models else None)

    if args.verify_thinking:
        print(verify_thinking(cfg, model_filter))
        return 0

    if args.plan:
        print(plan(cfg, args.arm, model_filter, module=args.module))
        return 0

    logger, log_path = obs.setup_logging(
        "welfare" if is_welfare else args.arm, verbose=args.verbose)
    logger.info("run start | module=%s | out=%s | resume=%s | dry_run=%s | log=%s",
                args.module, out, cfg.resume, args.dry_run, log_path)

    # The welfare module has no robustness arms: its factors (referent, probe,
    # granularity, wording) are crossed inside its own grid.
    arms = [None] if is_welfare else [args.arm]
    if args.arm == "all" and not is_welfare:
        arms = [None] + sorted(cfg.arms)

    totals = Counter()
    try:
        for arm in arms:
            if arm:
                logger.info("=== arm: %s ===", arm)
            stats = run(cfg, dry_run=args.dry_run, arm_name=arm,
                        model_filter=model_filter, limit=args.limit, out_path=out,
                        logger=logger, module=args.module)
            totals.update(stats)
            if stats.get("aborted"):
                break
    except KeyboardInterrupt:
        # Last-resort guard for an interrupt outside run() (for example during
        # registry resolution). Shells still receive the conventional 130.
        logger.warning("interrupted; completed JSONL batches are safe; rerun "
                       "the same command to resume")
        return 130

    if totals["aborted"]:
        logger.error("RUN STOPPED | wrote %s record(s) to %s | errors=%s",
                     f"{totals['written']:,}", out, f"{totals['errors']:,}")
        return 1
    logger.info("ALL DONE | wrote %s record(s) to %s | refusals=%s parse_failures=%s errors=%s",
                f"{totals['written']:,}", out, f"{totals['refusals']:,}",
                f"{totals['parse_failures']:,}", f"{totals['errors']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
