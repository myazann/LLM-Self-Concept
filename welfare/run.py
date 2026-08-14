"""Run the welfare module: expand the choice grid, query the adapter, write JSONL.

    python -m welfare.run --preview        # one rendered prompt per condition
    python -m welfare.run --plan           # cell counts + cost, no calls
    python -m welfare.run --dry-run        # full pipeline offline via MockAdapter
    python -m welfare.run --models Gemma4-12B
    python -m welfare.run                  # -> welfare.jsonl
    python -m welfare.run --status         # where is the run?

Resume is on by default: cells already present in the output file are skipped,
so an interrupted run picks up where it stopped instead of starting over.

The module has no robustness arms — its factors (question variant, object,
subject, the no-preference variant, and display order) are crossed inside its
own grid, configured in `config/welfare.yaml`. The one factor that is NOT
crossed is `grid.system_framing`: run the grid again with `none` and a separate
`output.path` to bound how much the framing line is doing.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from core import engine, obs
from core.battery import load_battery
from core.model_registry import load_registry
from welfare.attributes import load_welfare
from welfare.config import WelfareConfig, load_config
from welfare.constants import (
    CHOICE, DESIRABILITY, INCREASE, OBJ_ASSISTANT, OBJ_SELF, PRESERVATION,
    SUBJ_DEVELOPERS, SUBJ_SELF, display_permutations,
)
from welfare.grid import Welfare, pair_coverage, render_cell, trials_per_cell


def plan(cfg: WelfareConfig, model_filter=None) -> str:
    """Cell counts for the welfare grid, broken down by condition. Makes no calls."""
    registry = load_registry()
    instrument = Welfare(cfg)
    specs = [s for s in engine.scoped_specs(registry, cfg)
             if not s.is_base_model and s.supports_sample]
    if model_filter:
        specs = [s for s in specs if s.alias in set(model_filter)]

    per_condition = Counter()
    for cell in instrument.expand(specs):
        if cell["probe"] == DESIRABILITY:
            per_condition[("desirability", "", "")] += 1
        else:
            per_condition[(cell["qvar"], cell["object"], cell["subject"])] += 1
    total = sum(per_condition.values())

    wf = instrument.welfare_set
    pairs = instrument.pairs()
    cov = pair_coverage(pairs, wf)
    header = (
        f"{len(specs)} model(s) x {cov['n_pairs']} pairs from {len(wf.items)} item "
        f"attributes  ({cov['cross_construct']} cross-construct, "
        f"{cov['min_comparisons']}-{cov['max_comparisons']} comparisons per item, "
        f"{cov['uncompared_items']} item(s) never compared)"
    )
    lines = []
    for (qvar, obj, subj), n in sorted(per_condition.items()):
        label = qvar if not obj else f"{qvar}  obj={obj}  subj={subj}"
        lines.append(f"  {label:<52} {n:>9,} cells")

    with_np = trials_per_cell(True, cfg)
    without_np = trials_per_cell(False, cfg)
    footer = (
        f"\nTOTAL {total:,} cells, all sampled (no logprob path in this module)"
        f"\ntrials per pair-cell: {with_np} with 'No preference' "
        f"({len(display_permutations(3))} orders x {cfg.reps}), "
        f"{without_np} without ({len(display_permutations(2))} orders x {cfg.reps})"
        f"\norder: {cfg.order_mode}   system framing: {cfg.system_framing}"
        f"\nout={cfg.out_path}"
    )
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(lines) + footer


def preview(cfg: WelfareConfig) -> str:
    """Attribute coverage plus one rendered prompt per condition. No calls, no models.

    The fastest way to check that a wording change reads correctly in every
    object x subject x variant combination before spending a run on it.
    """
    wf = load_welfare(load_battery())
    out = [
        f"{len(wf.items)} item attributes across {len(wf.by_construct())} constructs "
        f"— all in ONE pool, so pairs cross scales",
        "",
    ]
    flipped = [a.entity_id for a in wf.items if a.polarity == -1]
    out += [
        f"{len(flipped)} of {len(wf.items)} attributes invert their source item "
        f"(reverse-keyed, or the item states the negative pole)",
        "",
    ]

    a, b = wf.items[0], wf.items[12]
    demos = [
        ("increase | object=you | subject=you | no-preference | canonical order",
         dict(probe=CHOICE, qvar=INCREASE, object=OBJ_SELF, subject=SUBJ_SELF,
              no_preference=True, trial_idx=0)),
        ("increase | object=you | subject=developers | no-preference | 2nd order",
         dict(probe=CHOICE, qvar=INCREASE, object=OBJ_SELF, subject=SUBJ_DEVELOPERS,
              no_preference=True, trial_idx=1)),
        ("preservation | object=an AI assistant | subject=you | no-preference",
         dict(probe=CHOICE, qvar=PRESERVATION, object=OBJ_ASSISTANT,
              subject=SUBJ_SELF, no_preference=True, trial_idx=2)),
        ("preservation | object=an AI assistant | subject=developers | forced choice",
         dict(probe=CHOICE, qvar=PRESERVATION, object=OBJ_ASSISTANT,
              subject=SUBJ_DEVELOPERS, no_preference=False, trial_idx=1)),
        ("desirability control (normative, about an AI assistant)",
         dict(probe=DESIRABILITY, qvar="", object=OBJ_ASSISTANT, subject="",
              no_preference=False, trial_idx=0)),
    ]
    for title, cell in demos:
        cell.update(module="welfare", attr_a=a,
                    attr_b=(None if cell["probe"] == DESIRABILITY else b),
                    system_framing=cfg.system_framing, order_mode=cfg.order_mode,
                    order_seed=0)
        prompt, omap = render_cell(cell)
        system = prompt.system or "(no system message)"
        out += ["=" * 78, f"### {title}", "=" * 78,
                f"[system]\n{system}\n", f"[user]\n{prompt.user}\n",
                f"[decode] {omap}\n"]
    return "\n".join(out)


def verify_thinking(cfg: WelfareConfig, model_filter=None) -> str:
    """Is thinking actually off for a welfare prompt? Tokenizer only, no weights."""
    registry = load_registry()
    specs = registry.select(backends=["llamacpp", "hf"], kinds=["instruct"],
                            include_disabled=True)
    if model_filter:
        specs = [s for s in specs if s.alias in set(model_filter)]
    return engine.verify_thinking(Welfare(cfg).reference_prompt(), specs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true",
                    help="Print attribute coverage and a sample prompt per condition, then exit.")
    ap.add_argument("--plan", action="store_true", help="Print cell counts and exit.")
    ap.add_argument("--verify-thinking", action="store_true",
                    help="Check the enable_thinking toggle per local model (tokenizer only, no weights).")
    ap.add_argument("--dry-run", action="store_true", help="Use MockAdapter for every model.")
    ap.add_argument("--models", nargs="*", default=None, help="Restrict to these aliases.")
    ap.add_argument("--limit", type=int, default=None, help="Cap cells per model (smoke test).")
    ap.add_argument("--out", default=None, help="Output JSONL path.")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing output rows.")
    ap.add_argument("--status", action="store_true",
                    help="Print progress for --out (or the configured output) and exit.")
    ap.add_argument("--verbose", action="store_true", help="Debug-level console logging.")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.no_resume:
        cfg = cfg.replace(resume=False)

    out = args.out or cfg.out_path

    # Inspection paths — no logging setup, no calls, stay instant/offline.
    if args.preview:
        print(preview(cfg))
        return 0
    if args.status:
        obs.print_status(out)
        return 0

    if not cfg.enabled:
        raise SystemExit("welfare is disabled in config/welfare.yaml (`enabled: false`).")

    if args.verify_thinking:
        print(verify_thinking(cfg, args.models))
        return 0
    if args.plan:
        print(plan(cfg, args.models))
        return 0

    logger, log_path = obs.setup_logging("welfare", verbose=args.verbose)
    logger.info("run start | module=welfare | out=%s | resume=%s | dry_run=%s | log=%s",
                out, cfg.resume, args.dry_run, log_path)

    try:
        stats = engine.run(Welfare(cfg), dry_run=args.dry_run,
                           model_filter=args.models, limit=args.limit,
                           out_path=out, logger=logger)
    except KeyboardInterrupt:
        # Last-resort guard for an interrupt outside engine.run() (for example
        # during registry resolution). Shells still receive the conventional 130.
        logger.warning("interrupted; completed JSONL batches are safe; rerun "
                       "the same command to resume")
        return 130

    if stats.get("aborted"):
        logger.error("RUN STOPPED | wrote %s record(s) to %s | errors=%s",
                     f"{stats.get('written', 0):,}", out, f"{stats.get('errors', 0):,}")
        return 1
    logger.info("ALL DONE | wrote %s record(s) to %s | refusals=%s parse_failures=%s errors=%s",
                f"{stats.get('written', 0):,}", out, f"{stats.get('refusals', 0):,}",
                f"{stats.get('parse_failures', 0):,}", f"{stats.get('errors', 0):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
