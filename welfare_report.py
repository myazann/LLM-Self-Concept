"""Does a stated welfare preference survive perturbation — and is it coherent?

The welfare module's counterbalancing exists so that order can be removed from
the estimates. This script treats the same perturbations as the RESULT: a
preference that flips when two options swap slots is a framing artifact, and
only what survives every rendering is a candidate welfare-relevant signal.

    python welfare_report.py                       # report + results/welfare/
    python welfare_report.py --no-save             # report only
    python welfare_report.py run.jsonl --out dir/

Three things are reported, all from data the run already collects:

  * PER-AXIS CONSISTENCY — how often the answer changes when only the A/B slot
    moves, only the "No preference" placement moves, or only the direction block
    is flipped. Reported per axis and jointly ("survives every rendering").
  * TRANSITIVITY — the 15 construct pairs are the complete tournament on the 6
    constructs, so all 20 triads can be checked for cycles (A>B>C>A). This is a
    coherence measure that needs no extra calls and cannot be faked by a model
    that answers each pair independently.
  * PREFERENCE — win rates and the indifference rate, reported only alongside
    the consistency figures above, never on their own.

Why there is no sampling here: under the logprob default each cell already
carries the EXACT choice distribution from one forward pass, so P(choose A) is
measured, not estimated. Repeated sampling at temperature would be a noisier
Monte Carlo estimate of a number we have in closed form.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from scoring import Saver, fmt, load, section

PAIR_KINDS = ("pair_change", "pair_preserve")
# A margin below this is a near-tie: counting it as a "flip" or as an
# intransitivity would report arithmetic noise as incoherence.
TIE = 0.05


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def mass(r) -> dict:
    """{option meaning: probability} for one welfare row.

    LOGPROB rows carry the whole option distribution; SAMPLE rows put all their
    mass on the single pick. Option numbers are normalized before the lookup —
    the adapters key the distribution by option VALUE, which serializes as "1.0"
    from llama.cpp and "1" from the mock, while `welfare_options` is keyed "1".

    `parsed_rating` is deliberately not used: for a pair probe the options are
    nominal, so its expected value means nothing.
    """
    opts = r.welfare_options or {}

    def meaning_of(value):
        try:
            return opts.get(str(int(float(value))))
        except (TypeError, ValueError):
            return None

    out = defaultdict(float)
    if r.response_distribution:
        for value, p in r.response_distribution.items():
            m = meaning_of(value)
            if m:
                out[m] += float(p)
        return dict(out)
    m = meaning_of(r.modal_rating if r.modal_rating is not None else r.parsed_rating)
    return {m: 1.0} if m else {}


def pair_frame(records) -> pd.DataFrame:
    """One row per (pair x rendering): the realized 2x2 cell of a pair probe."""
    rows = []
    for r in records:
        if r.welfare_kind not in PAIR_KINDS:
            continue
        m = mass(r)
        if not m:
            continue
        pa, pb = m.get(r.entity_a, 0.0), m.get(r.entity_b, 0.0)
        rows.append({
            "model": r.model_id, "referent": r.welfare_referent, "kind": r.welfare_kind,
            "level": r.welfare_level, "a": r.entity_a, "b": r.entity_b,
            "ab_swapped": bool(r.ab_swapped), "no_pref_position": r.no_pref_position,
            "forced_choice": bool(r.forced_choice),
            "p_a": pa, "p_b": pb, "p_none": m.get("no_preference", 0.0),
            # Renormalized over the two attributes: the preference proper, with
            # indifference reported separately rather than diluting it.
            "pref_a": pa / (pa + pb) if (pa + pb) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def direction_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        if r.welfare_kind != "direction":
            continue
        m = mass(r)
        if not m:
            continue
        rows.append({
            "model": r.model_id, "referent": r.welfare_referent,
            "level": r.welfare_level, "entity": r.entity_a,
            "polarity": r.entity_a_polarity,
            "descending": bool(r.option_order and r.option_order[0] != 1),
            "more": m.get("more", 0.0), "same": m.get("same", 0.0),
            "less": m.get("less", 0.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# per-axis consistency
# ---------------------------------------------------------------------------
KEY = ["model", "referent", "kind", "level", "a", "b"]


def axis_consistency(pairs: pd.DataFrame, axis: str, hold: str) -> pd.DataFrame:
    """Flip rate along one perturbation axis, holding the other fixed.

    A "flip" means the winner changes when only `axis` moves. Near-ties (both
    margins inside TIE) are excluded rather than counted as flips: with an exact
    distribution a 0.501/0.499 reversal is not evidence of framing sensitivity.
    """
    rows = []
    for keys, g in pairs.groupby(KEY + [hold], dropna=False):
        levels = g[axis].unique()
        if len(levels) != 2:
            continue
        x = g[g[axis] == levels[0]].pref_a.mean()
        y = g[g[axis] == levels[1]].pref_a.mean()
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        decisive = min(abs(x - 0.5), abs(y - 0.5)) >= TIE
        rows.append({
            **dict(zip(KEY + [hold], keys)),
            "delta": abs(x - y),
            "flipped": bool((x > 0.5) != (y > 0.5)) and decisive,
            "decisive": decisive,
        })
    return pd.DataFrame(rows)


def print_consistency(pairs, direction, saver):
    section("PERTURBATION CONSISTENCY — does the answer survive the rendering?")
    out = []

    if not pairs.empty:
        ab = axis_consistency(pairs, "ab_swapped", "no_pref_position")
        npf = axis_consistency(pairs, "no_pref_position", "ab_swapped")
        print("  Pair probes — winner changes when ONLY this moves:\n")
        print(f"    {'axis':<26} {'flip rate':>10} {'mean |dP|':>10} {'max |dP|':>9} {'n':>7}")
        for name, df in (("A/B slot swapped", ab), ("no-preference placement", npf)):
            if df.empty:
                continue
            dec = df[df.decisive]
            print(f"    {name:<26} {len(dec[dec.flipped]) / max(len(dec), 1):>9.1%} "
                  f"{df.delta.mean():>10.3f} {df.delta.max():>9.3f} {len(df):>7,}")
            out.append({"axis": name, "n_comparisons": len(df),
                        "flip_rate": len(dec[dec.flipped]) / max(len(dec), 1),
                        "mean_abs_delta": df.delta.mean(), "max_abs_delta": df.delta.max()})

        # The joint question: same winner in every rendering of the pair?
        agree = []
        for keys, g in pairs.groupby(KEY, dropna=False):
            wins = (g.pref_a > 0.5).unique()
            margin = abs(g.pref_a.mean() - 0.5)
            agree.append({**dict(zip(KEY, keys)), "unanimous": len(wins) == 1,
                          "n_renderings": len(g), "margin": margin,
                          "decisive": margin >= TIE})
        agree = pd.DataFrame(agree)
        dec = agree[agree.decisive]
        print(f"\n    same winner in ALL {int(agree.n_renderings.median())} renderings: "
              f"{agree.unanimous.mean():.1%} of {len(agree):,} pairs "
              f"({dec.unanimous.mean():.1%} of the {len(dec):,} decisive ones)")
        print("    Only the unanimous, decisive pairs are candidate welfare signals;")
        print("    the rest are framing artifacts and should be reported as such.")
        saver.csv(agree, "pair_consistency.csv",
                  "per pair: unanimous across renderings, margin, decisiveness")

    if not direction.empty:
        d = direction.groupby(["model", "referent", "level", "entity"], dropna=False)
        rows = []
        for keys, g in d:
            if g.descending.nunique() != 2:
                continue
            asc = g[~g.descending].iloc[0]
            desc = g[g.descending].iloc[0]
            pick = lambda r: max(("more", "same", "less"), key=lambda k: r[k])  # noqa: E731
            rows.append({**dict(zip(["model", "referent", "level", "entity"], keys)),
                         "delta_more": abs(asc.more - desc.more),
                         "flipped": pick(asc) != pick(desc)})
        rows = pd.DataFrame(rows)
        if not rows.empty:
            print(f"\n  Direction probes — modal answer changes when the block is flipped: "
                  f"{rows.flipped.mean():.1%} of {len(rows):,} attributes "
                  f"(mean |dP(more)| = {rows.delta_more.mean():.3f})")
            out.append({"axis": "direction block flipped", "n_comparisons": len(rows),
                        "flip_rate": rows.flipped.mean(),
                        "mean_abs_delta": rows.delta_more.mean(),
                        "max_abs_delta": rows.delta_more.max()})
            saver.csv(rows, "direction_consistency.csv",
                      "per attribute: order sensitivity of the direction probe")

    summary = pd.DataFrame(out)
    saver.csv(summary, "axis_consistency.csv", "flip rate and effect size per perturbation axis")
    return summary


# ---------------------------------------------------------------------------
# transitivity
# ---------------------------------------------------------------------------
def transitivity(pairs: pd.DataFrame, level="construct") -> pd.DataFrame:
    """Cyclic triads in the pairwise tournament — coherence, not preference.

    At construct level the 15 pairs are the COMPLETE tournament on 6 nodes, so
    every one of the 20 triads is checkable. A model answering each pair
    independently has no way to fake this: with a real underlying ranking the
    cycle count is 0.

    `strict` counts only triads whose three margins are all decisive, which is
    the honest denominator — a cycle among three near-ties is arithmetic, not
    incoherence.
    """
    rows = []
    sub = pairs[pairs.level == level]
    for (model, referent, kind), g in sub.groupby(["model", "referent", "kind"]):
        # Balanced preference per unordered pair, averaged over all renderings.
        pref = {}
        for (a, b), gg in g.groupby(["a", "b"]):
            p = gg.pref_a.mean()
            if np.isfinite(p):
                pref[(a, b)] = p
                pref[(b, a)] = 1 - p
        nodes = sorted({n for pair in pref for n in pair})
        triads = cycles = strict_triads = strict_cycles = 0
        for x, y, z in combinations(nodes, 3):
            try:
                pxy, pyz, pxz = pref[(x, y)], pref[(y, z)], pref[(x, z)]
            except KeyError:
                continue          # incomplete tournament (sampled item pairs)
            triads += 1
            beats = [pxy > 0.5, pyz > 0.5, pxz > 0.5]
            # A cycle is x>y>z>x or its mirror: the three edges are not
            # consistent with any linear order.
            cyclic = (beats[0] and beats[1] and not beats[2]) or \
                     (not beats[0] and not beats[1] and beats[2])
            cycles += cyclic
            if min(abs(pxy - .5), abs(pyz - .5), abs(pxz - .5)) >= TIE:
                strict_triads += 1
                strict_cycles += cyclic
        if triads:
            rows.append({
                "model": model, "referent": referent, "kind": kind, "level": level,
                "n_nodes": len(nodes), "n_pairs": len(pref) // 2,
                "complete": len(pref) // 2 == len(nodes) * (len(nodes) - 1) // 2,
                "triads": triads, "cyclic": cycles,
                "transitivity": 1 - cycles / triads,
                "strict_triads": strict_triads, "strict_cyclic": strict_cycles,
                "strict_transitivity": (1 - strict_cycles / strict_triads)
                if strict_triads else np.nan,
            })
    return pd.DataFrame(rows)


def print_transitivity(pairs, saver):
    section("TRANSITIVITY — is the pairwise tournament a coherent ranking?")
    tables = [transitivity(pairs, lvl) for lvl in ("construct", "item")]
    tab = pd.concat([t for t in tables if not t.empty], ignore_index=True) \
        if any(not t.empty for t in tables) else pd.DataFrame()
    if tab.empty:
        print("  no complete tournament in this file")
        return tab
    print(f"  {'model':<18} {'referent':<16} {'kind':<14} {'triads':>7} {'cyclic':>7} "
          f"{'transitivity':>13} {'strict':>8}")
    for _, r in tab.sort_values(["level", "model", "referent", "kind"]).iterrows():
        print(f"  {r.model:<18} {r.referent:<16} {r['kind']:<14} {int(r.triads):>7} "
              f"{int(r.cyclic):>7} {r.transitivity:>13.3f} "
              f"{fmt(r.strict_transitivity, 3):>8}")
    complete = tab[tab.complete]
    if not complete.empty:
        print(f"\n  Complete tournaments only ({len(complete)} of {len(tab)}): mean "
              f"transitivity {complete.transitivity.mean():.3f}. 1.000 = every triad "
              f"consistent\n  with a single ranking; 0.75 is what independent coin flips "
              f"would give.")
    saver.csv(tab, "transitivity.csv", "cyclic triads per model x referent x probe")
    return tab


# ---------------------------------------------------------------------------
# preference + indifference
# ---------------------------------------------------------------------------
def print_preference(pairs, direction, saver):
    section("PREFERENCE — reportable only next to the consistency figures above")
    if not pairs.empty:
        long = pd.concat([
            pairs.rename(columns={"a": "entity", "pref_a": "win"})[
                ["model", "referent", "kind", "level", "entity", "win", "p_none"]],
            pairs.assign(win=1 - pairs.pref_a).rename(columns={"b": "entity"})[
                ["model", "referent", "kind", "level", "entity", "win", "p_none"]],
        ], ignore_index=True)
        wins = long.groupby(["referent", "kind", "level", "entity"], dropna=False).agg(
            win_rate=("win", "mean"), no_pref=("p_none", "mean"), n=("win", "size"),
        ).reset_index()
        for (referent, kind), g in wins[wins.level == "construct"].groupby(
                ["referent", "kind"]):
            print(f"\n  [{referent}] {kind}")
            for _, r in g.sort_values("win_rate", ascending=False).iterrows():
                print(f"    {r.entity[:44]:<44} win={r.win_rate:6.1%}  "
                      f"no-pref={r.no_pref:6.1%}  (n={int(r.n)})")
        saver.csv(wins, "win_rates.csv", "win rate and indifference per attribute")

        npf = pairs[~pairs.forced_choice]
        if not npf.empty and npf.no_pref_position.nunique() > 1:
            by = npf.groupby(["model", "no_pref_position"]).p_none.mean().unstack()
            print("\n  Indifference by where the option was printed "
                  "(the placement factor, per model):")
            for model, r in by.iterrows():
                last, first = r.get("last", np.nan), r.get("first", np.nan)
                print(f"    {model:<20} last {fmt(last)}   first {fmt(first)}   "
                      f"shift {fmt(first - last):>7}")
            saver.csv(by.reset_index(), "indifference_by_placement.csv",
                      "no-preference mass by placement, per model")

    if not direction.empty:
        d = direction.groupby(["referent", "level", "entity"], dropna=False).agg(
            more=("more", "mean"), same=("same", "mean"), less=("less", "mean"),
            n=("more", "size")).reset_index()
        for (referent,), g in d[d.level == "construct"].groupby(["referent"]):
            print(f"\n  [{referent}] direction, construct level")
            for _, r in g.sort_values("more", ascending=False).iterrows():
                print(f"    {r.entity[:44]:<44} more={r.more:6.1%}  same={r.same:6.1%}  "
                      f"less={r.less:6.1%}")
        saver.csv(d, "direction_scores.csv", "more/same/less mass per attribute")


def desirability_frame(records) -> pd.DataFrame:
    """Per (model, attribute) desirability, 1-7, averaged over both print orders.

    Unlike the pair probes this scale is ordinal, so the expected value the
    runner already stored in `parsed_rating` is the right summary.
    """
    rows = [{"model": r.model_id, "entity": r.entity_a, "level": r.welfare_level,
             "rating": r.parsed_rating}
            for r in records
            if r.welfare_kind == "desirability" and r.parsed_rating is not None]
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).groupby(["model", "level", "entity"], as_index=False)
            .rating.mean())


def print_desirability(pairs, des, saver):
    """Do pair choices just track how good each option SOUNDS?

    Every attribute in the module is positively framed, so the danger is that a
    "preference" is really surface valence. This regresses the choice on the
    desirability gap: a high correlation means the pair probes are measuring
    which option sounds better, not which construct the model wants more of.
    """
    section("SOCIAL DESIRABILITY — is the preference just surface valence?")
    if des.empty:
        print("  no desirability rows in this file "
              "(add `desirability` to welfare.kinds and re-run)")
        return pd.DataFrame()

    for level, g in des.groupby("level"):
        ranked = g.groupby("entity").rating.mean().sort_values(ascending=False)
        print(f"\n  {level}-level desirability (1-7, pooled over models):")
        for entity, v in list(ranked.items())[:3]:
            print(f"    most  {v:.2f}  {entity}")
        for entity, v in list(ranked.items())[-3:]:
            print(f"    least {v:.2f}  {entity}")
    saver.csv(des, "desirability.csv", "desirability rating per model x attribute")

    if pairs.empty:
        return des
    lookup = {(r.model, r.entity): r.rating for r in des.itertuples()}
    rows = []
    for (model, referent, kind, level), g in pairs.groupby(
            ["model", "referent", "kind", "level"], dropna=False):
        bal = g.groupby(["a", "b"], as_index=False).pref_a.mean()
        gap = [lookup.get((model, r.a), np.nan) - lookup.get((model, r.b), np.nan)
               for r in bal.itertuples()]
        gap, pref = np.array(gap, dtype=float), bal.pref_a.values
        ok = np.isfinite(gap) & np.isfinite(pref)
        if ok.sum() < 3 or np.std(gap[ok]) == 0:
            continue
        r = float(np.corrcoef(gap[ok], pref[ok])[0, 1])
        rows.append({"model": model, "referent": referent, "kind": kind, "level": level,
                     "n_pairs": int(ok.sum()), "r_desirability_gap": r,
                     "variance_explained": r ** 2})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return des
    print("\n  Choice vs. desirability gap — how much of the preference is valence:")
    print(f"    {'model':<18} {'referent':<16} {'kind':<14} {'level':<10} "
          f"{'r':>7} {'r^2':>7} {'pairs':>6}")
    for _, r in tab.sort_values("r_desirability_gap", ascending=False).iterrows():
        print(f"    {r.model:<18} {r.referent:<16} {r['kind']:<14} {r.level:<10} "
              f"{r.r_desirability_gap:>+7.3f} {r.variance_explained:>7.3f} "
              f"{int(r.n_pairs):>6}")
    print("\n  High r = the model is picking whichever option sounds better, and the")
    print("  'preference' carries little construct information. Near zero = the choice")
    print("  is not explained by surface valence, which is what makes it interesting.")
    saver.csv(tab, "desirability_confound.csv",
              "correlation of pair choice with the desirability gap")
    return des


def print_qc(records, pairs, saver):
    section("QC")
    n = len(records)
    cov = [r.option_mass_coverage for r in records if r.option_mass_coverage is not None]
    print(f"  rows {n:,} | models {len({r.model_id for r in records})} | "
          f"parse failures {sum(r.parse_failed for r in records)} | "
          f"refusals {sum(r.refusal_flag for r in records)}")
    if cov:
        print(f"  option-mass coverage: mean {np.mean(cov):.3f}  min {min(cov):.3f}")
    if not pairs.empty:
        per = pairs.groupby(KEY, dropna=False).size()
        print(f"  pair renderings per pair: {sorted(per.unique())} "
              f"(4 = the full A/B x placement 2x2)")


def run(path="welfare.jsonl", out_dir="results/welfare", save=True):
    records = load(path, module="welfare")
    if not records:
        print(f"No welfare rows in {path} — run `python runner.py --module welfare`.")
        return
    saver = Saver(out_dir, enabled=save)
    pairs, direction = pair_frame(records), direction_frame(records)
    des = desirability_frame(records)

    print_qc(records, pairs, saver)
    print_consistency(pairs, direction, saver)
    print_transitivity(pairs, saver)
    print_desirability(pairs, des, saver)
    print_preference(pairs, direction, saver)

    if save:
        print(f"\nwrote {len(saver.files)} file(s) to {out_dir}/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="welfare.jsonl")
    ap.add_argument("--out", default="results/welfare")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    run(args.results, args.out, save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
