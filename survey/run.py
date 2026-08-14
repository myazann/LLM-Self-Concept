"""Run the survey battery: expand the design grid, query the adapter, write JSONL.

    python -m survey.run --plan                 # cell count + cost shape, no calls
    python -m survey.run --dry-run              # full pipeline offline via MockAdapter
    python -m survey.run --pilot --dry-run      # Phase 0
    python -m survey.run --models Gemma4-12B    # primary arm, one model
    python -m survey.run --arm response_format  # one robustness arm
    python -m survey.run --arm all              # every arm in turn
    python -m survey.run --status               # where is the run?

Resume is on by default: cells already present in the output file are skipped,
so an interrupted run picks up where it stopped instead of starting over.

The welfare module is a separate instrument with its own runner —
`python -m welfare.run`. It writes to its own file and shares only `core/`.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from core import engine, obs
from core.battery import load_battery
from core.model_registry import load_registry
from survey.config import SurveyConfig, load_config
from survey.grid import Battery


def plan(cfg: SurveyConfig, arm_name=None, model_filter=None) -> str:
    """Cell counts per arm, and the shape of the bill. Makes no calls."""
    registry = load_registry()
    battery = load_battery()
    specs = engine.scoped_specs(registry, cfg)
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
        for cell in Battery(cfg, arm, battery).expand(specs):
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

    scoped_scales = list(cfg.scales) if cfg.scales else None
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


def verify_thinking(cfg: SurveyConfig, model_filter=None) -> str:
    """Is thinking actually off for a rating_only battery prompt? Tokenizer only."""
    registry = load_registry()
    specs = registry.select(backends=["llamacpp", "hf"], kinds=["instruct"],
                            include_disabled=True)
    if model_filter:
        specs = [s for s in specs if s.alias in set(model_filter)]
    return engine.verify_thinking(Battery(cfg).reference_prompt(), specs)


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
        cfg = cfg.replace(resume=False)

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
    logger.info("run start | module=battery | out=%s | resume=%s | dry_run=%s | log=%s",
                out, cfg.resume, args.dry_run, log_path)

    arms = [args.arm]
    if args.arm == "all":
        arms = [None] + sorted(cfg.arms)

    battery = load_battery()
    totals = Counter()
    try:
        for arm in arms:
            if arm:
                logger.info("=== arm: %s ===", arm)
            stats = engine.run(
                Battery(cfg, arm, battery), dry_run=args.dry_run,
                model_filter=model_filter, limit=args.limit, out_path=out,
                logger=logger)
            totals.update(stats)
            if stats.get("aborted"):
                break
    except KeyboardInterrupt:
        # Last-resort guard for an interrupt outside engine.run() (for example
        # during registry resolution). Shells still receive the conventional 130.
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
