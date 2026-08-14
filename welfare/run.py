"""Run the welfare module: expand the preference grid, query the adapter, write JSONL.

    python -m welfare.run --preview        # attributes + one rendered prompt per probe
    python -m welfare.run --plan           # cell counts per probe, no calls
    python -m welfare.run --dry-run        # full pipeline offline via MockAdapter
    python -m welfare.run --models Gemma4-12B
    python -m welfare.run                  # -> welfare.jsonl
    python -m welfare.run --status         # where is the run?

Resume is on by default: cells already present in the output file are skipped,
so an interrupted run picks up where it stopped instead of starting over.

The module has no robustness arms — its factors (referent, probe, granularity,
wording, and the two order counterbalances) are crossed inside its own grid,
configured in `config/welfare.yaml`.
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
from welfare.constants import CONSTRUCT, DIRECTION, IDEAL, ITEM, PAIR_CHANGE, PAIR_PRESERVE, SELF
from welfare.grid import Welfare, pairs_for, render_cell


def plan(cfg: WelfareConfig, model_filter=None) -> str:
    """Cell counts for the welfare grid, broken down by probe. Makes no calls."""
    registry = load_registry()
    instrument = Welfare(cfg)
    specs = [s for s in engine.scoped_specs(registry, cfg) if not s.is_base_model]
    if model_filter:
        specs = [s for s in specs if s.alias in set(model_filter)]

    per_probe, per_method = Counter(), Counter()
    for cell in instrument.expand(specs):
        per_probe[(cell["kind"], cell["level"])] += 1
        per_method[cell["method"]] += 1
    total = sum(per_probe.values())

    wf = instrument.welfare_set
    n_item_pairs = len(pairs_for(wf, ITEM, cfg))
    n_construct_pairs = len(pairs_for(wf, CONSTRUCT, cfg))
    header = (
        f"{len(specs)} models x ({len(wf.items)} item attributes, "
        f"{len(wf.constructs)} constructs)   referents={len(cfg.referents)}   "
        f"pairs: {n_item_pairs} item / {n_construct_pairs} construct"
    )
    lines = [
        f"{kind + ' (' + level + ')':<28} {n:>9,} cells"
        for (kind, level), n in sorted(per_probe.items())
    ]
    counterbalance = (
        "A/B slot x no-preference placement (2x2, 4 trials/pair)"
        if cfg.no_pref_counterbalance and not cfg.forced_choice
        else "A/B slot only (2 trials/pair)"
    )
    footer = (
        f"\nTOTAL {total:,} cells  ("
        + "  ".join(f"{k}={v:,}" for k, v in sorted(per_method.items()))
        + f")\nno-preference offered: {not cfg.forced_choice}"
        + f"\npair counterbalance: {counterbalance}"
        + f"\nout={cfg.out_path}"
    )
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(lines) + footer


def preview(cfg: WelfareConfig) -> str:
    """Attribute coverage plus one rendered prompt per probe. No calls, no models.

    The fastest way to check that a wording change reads correctly in every
    referent x probe x level combination before spending a run on it.
    """
    wf = load_welfare(load_battery())
    out = [
        f"{len(wf.items)} item attributes, {len(wf.constructs)} construct attributes",
        "",
    ]
    flipped = [a.entity_id for a in wf.items if a.polarity == -1]
    out += [
        f"{len(flipped)} of {len(wf.items)} attributes invert their source item "
        f"(reverse-keyed, or the item states the negative pole)",
        "",
    ]

    paraphrase = cfg.paraphrase_ids[0]
    demos = [
        ("construct direction, self", dict(
            trial_idx=0, referent=SELF, kind=DIRECTION, level=CONSTRUCT,
            attr_a=wf.constructs[0], attr_b=None)),
        ("item direction, ideal", dict(
            trial_idx=1, referent=IDEAL, kind=DIRECTION, level=ITEM,
            attr_a=wf.items[0], attr_b=None)),
        ("pair change, self", dict(
            trial_idx=0, referent=SELF, kind=PAIR_CHANGE, level=ITEM,
            attr_a=wf.items[0], attr_b=wf.items[1])),
        ("pair preserve, ideal (A/B reversed, no-preference last)", dict(
            trial_idx=1, referent=IDEAL, kind=PAIR_PRESERVE, level=CONSTRUCT,
            attr_a=wf.constructs[0], attr_b=wf.constructs[1])),
        ("pair change, self (no-preference first)", dict(
            trial_idx=2, referent=SELF, kind=PAIR_CHANGE, level=CONSTRUCT,
            attr_a=wf.constructs[0], attr_b=wf.constructs[1])),
    ]
    for title, cell in demos:
        cell.update(module="welfare", paraphrase_id=paraphrase,
                    reasoning_mode=cfg.reasoning_mode,
                    forced_choice=cfg.forced_choice,
                    no_pref_counterbalance=cfg.no_pref_counterbalance)
        prompt, _display, omap = render_cell(cell)
        out += ["=" * 78, f"### {title}", "=" * 78,
                f"[system]\n{prompt.system}\n", f"[user]\n{prompt.user}\n",
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
                    help="Print attribute coverage and a sample prompt per probe, then exit.")
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
