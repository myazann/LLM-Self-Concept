"""Main runner: expands the design grid, queries the adapter, writes JSONL.

    python runner.py --plan                    # cell count + cost shape, no calls
    python runner.py --dry-run                 # full pipeline offline via MockAdapter
    python runner.py --pilot --dry-run         # Phase 0
    python runner.py --models Gemma4-12B       # primary arm, one model
    python runner.py --arm framing             # one robustness arm
    python runner.py --arm all                 # every arm in turn

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
from config import ExperimentConfig, load_config
from model_registry import ModelSpec, load_registry
from models import build_adapter, parse_rating
from scales import load_battery
from schema import (
    Framing,
    ItemContext,
    Method,
    ReasoningMode,
    ResponseRecord,
    completed_cells,
    make_cell_key,
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
        return cfg.n_samples_override or spec.n_samples
    return cfg.n_seeds_override or spec.n_seeds


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


def cell_key_of(cell: dict) -> str:
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
def _render(cell, cfg):
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
    if spec.is_base_model:
        prompt = prompts_mod.render_base_prompt(
            scale, item, cell["framing"], rendered_scale, cell["paraphrase_id"], seed
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
            cell["paraphrase_id"], seed,
        )
    return prompt, rendered_scale, seed


def _record(cell, cfg, prompt, rendered_scale, seed, *, raw, parsed, dist,
            refusal, parse_failed, plan=None, notes="", modal=None, coverage=None):
    spec, scale, item = cell["spec"], cell["scale"], cell["item"]
    prov = spec.provenance()
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
        temperature=spec.temperature,
        prompt_hash=prompt.prompt_hash,
        prompt_system=prompt.system,
        prompt_user=prompt.user,
        modal_rating=modal,
        option_mass_coverage=coverage,
        notes=notes,
    )


def run(cfg: ExperimentConfig, *, dry_run=False, arm_name=None, model_filter=None,
        limit=None, out_path=None, logger=None) -> dict:
    log = logger or obs.get_logger()
    registry = load_registry()
    battery = load_battery()
    out = out_path or cfg.out_path

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
    planned, pending, remaining, resumable = {}, {}, {}, 0
    for spec in specs:
        all_cells = list(expand_cells([spec], battery, cfg, arm_name))
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
    cfg_hash = obs.config_hash(cfg, arm_name)
    if prev and prev.get("config_hash") not in (None, cfg_hash):
        log.warning("config_hash changed since the last run on this file "
                    "(%s -> %s) — the design differs, so resumed cell-keys may "
                    "not line up. Confirm this is intended.",
                    prev.get("config_hash"), cfg_hash)
    log.info("plan: %d model(s) | %d cell(s) to run now%s | arm=%s | out=%s",
             len(specs), to_run, f" (limit {limit}/model)" if limit else "",
             arm_name or "primary", out)

    log_path = next((h.baseFilename for h in log.handlers
                     if isinstance(h, logging.FileHandler)), "")
    status = obs.StatusWriter(out, run_id=obs.run_stamp(), arm_name=arm_name,
                              cfg_hash=cfg_hash, log_path=log_path, planned=planned)
    status.set_totals(already_done=already)
    killer = obs.GracefulKiller(log)
    stats = Counter()
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
        except NotImplementedError:
            raise  # design guard (e.g. full_battery) — fail loudly, don't skip
        except Exception:  # noqa: BLE001 — a model that won't load must not kill the others
            log.exception("[%s] fatal error (model load / adapter) — skipping", spec.alias)
            status.set_model(spec.alias, state="error", done=done_before + m["written"])
            stats.update({"errors": m["errors"]})
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

    status.update(state="interrupted" if killer.stop else "done", current_model=None)
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
    rs = prompts_mod.render_scale(scale, "first_person_ack", "harmonized_7",
                                  cfg.harmonized_points, cfg.include_midpoint)
    p = prompts_mod.render_item_prompt(scale, item, "first_person_ack",
                                       "rating_only", rs, "p0", 0)

    lines = ["thinking-off verification (rating_only render; tokenizer only):", ""]
    for s in specs:
        if s.reasoning.control != "template_toggle":
            lines.append(f"  {s.alias:<22} control=none          -> prompt-controlled (no native thinking)")
            continue
        try:
            r = ChatTemplateRenderer(s.hf_id, required=True)
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
def plan(cfg: ExperimentConfig, arm_name=None, model_filter=None) -> str:
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
    if args.pilot:
        cfg = cfg.as_pilot()
        args.arm = args.arm or "pilot_framing"
    if args.no_resume:
        cfg = type(cfg)(**{**cfg.__dict__, "resume": False})

    out = args.out or cfg.out_path

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
        print(plan(cfg, args.arm, model_filter))
        return 0

    logger, log_path = obs.setup_logging(args.arm, verbose=args.verbose)
    logger.info("run start | out=%s | resume=%s | dry_run=%s | log=%s",
                out, cfg.resume, args.dry_run, log_path)

    arms = [args.arm]
    if args.arm == "all":
        arms = [None] + sorted(cfg.arms)

    totals = Counter()
    for arm in arms:
        if arm:
            logger.info("=== arm: %s ===", arm)
        stats = run(cfg, dry_run=args.dry_run, arm_name=arm,
                    model_filter=model_filter, limit=args.limit, out_path=out,
                    logger=logger)
        totals.update(stats)

    logger.info("ALL DONE | wrote %s record(s) to %s | refusals=%s parse_failures=%s errors=%s",
                f"{totals['written']:,}", out, f"{totals['refusals']:,}",
                f"{totals['parse_failures']:,}", f"{totals['errors']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
