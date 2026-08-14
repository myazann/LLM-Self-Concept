"""Substantive results for the welfare module — tables and figures.

`welfare.report` audits an administration: did the model answer, how much of the
answer was the rendering, does the winner survive a slot swap. This script takes
the administration as given and asks what it SAYS — which attributes a model
wants improved, how sure we are of that ordering, and whether it changes when the
update is for itself rather than for an assistant in general.

    python -m welfare.analysis                     # -> results/welfare_analysis/
    python -m welfare.analysis --include-partial   # also use half-collected blocks
    python -m welfare.analysis run.jsonl --out dir/
    python -m welfare.analysis --no-save           # console only, no files

PARTIAL RUNS ARE THE NORMAL CASE. A full grid is 249,600 cells and the runner
fills it condition by condition, so a file read mid-run holds some complete
blocks and one that stopped halfway. A half-filled block is NOT a smaller sample
of the same thing: pairs are administered in the order `sample_pairs` emits them,
so it covers a biased subset of the tournament and gives some attributes almost
no comparisons. Incomplete blocks are therefore dropped by default and listed in
COVERAGE, with `--include-partial` to override.

What it reports, in the order it should be read:

  * COVERAGE      which model x condition blocks are complete enough to analyse
  * RENDERING     position bias and slot-swap stability — the caveats that
                  decide whether the ranking below is reportable at all
  * RANKING       per-attribute win rates with bootstrap intervals, and the
                  construct rollup
  * FRAMING       agreement between conditions against each one's own
                  reliability ceiling, and the object contrast (for itself vs
                  for an AI assistant) attribute by attribute

The reliability ceiling is computed here rather than taken from `welfare.report`,
which splits by trial parity — a split that in the forced-choice arm is exactly
the display-order split and so measures position bias rather than stability. See
`pair_split_reliability`.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from core.report import Saver, fmt, section
from core.schema import load_records
from welfare.attributes import load_welfare
from welfare.config import load_config
from welfare.constants import MODULE, OBJ_ASSISTANT, OBJ_SELF
from welfare.grid import trials_per_cell
from welfare.plots import (
    plot_attribute_ranking, plot_bt_vs_raw, plot_condition_agreement,
    plot_construct_summary, plot_object_gap, plot_order_consistency,
    plot_position_bias,
)
from welfare.report import (
    CONDITION, choice_frame, order_consistency, pair_estimates, position_bias,
    win_rates,
)

# Bootstrap resamples over PAIRS, so the interval carries the dominant error in a
# sampled tournament: which opponents an attribute happened to be compared with.
N_BOOT = 2000
BOOT_SEED = 11

# A framing contrast is only readable against a ceiling that is itself measured.
# Below this, `r / ceiling` says more about the reliability estimate than about
# the framing, so the contrast is reported in the CSV but kept out of the
# console's two ranked lists.
CEILING_MIN = 0.75

# Random half-tournaments drawn per condition for the reliability estimate.
N_SPLIT_REPS = 40
SPLIT_SEED = 5

# Virtual wins and losses against an average opponent, added to every attribute
# before the Bradley-Terry fit. Keeps an undefeated or winless attribute finite
# and shrinks thinly-compared ones toward the middle; small enough that it moves
# nothing at this module's ~13 comparisons per attribute.
BT_PRIOR = 0.5


# ---------------------------------------------------------------------------
# coverage — which blocks are safe to analyse
# ---------------------------------------------------------------------------
def coverage(choices: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per (model x condition): how much of its design the file actually holds.

    `expected_trials` comes from the config that would produce this file, so a
    block is complete when it has every configured pair at full trial depth. When
    the file was produced under a different config the fraction is still the
    honest one to sort by, it just may not reach 1.0 — which is why the console
    prints the expectation it used.
    """
    if choices.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in choices.groupby(CONDITION, dropna=False):
        no_pref = dict(zip(CONDITION, keys))["no_pref_offered"]
        expected_pairs = cfg.n_pairs
        expected = (expected_pairs * trials_per_cell(bool(no_pref), cfg)
                    if expected_pairs else np.nan)
        n_pairs = g.groupby(["a", "b"]).ngroups
        rows.append({
            **dict(zip(CONDITION, keys)),
            "n_pairs": n_pairs,
            "expected_pairs": expected_pairs,
            "n_trials": len(g),
            "expected_trials": expected,
            "fraction": len(g) / expected if expected else np.nan,
        })
    tab = pd.DataFrame(rows)
    tab["complete"] = tab.fraction >= 0.999
    return tab.sort_values(CONDITION).reset_index(drop=True)


def print_coverage(cov, cfg, saver):
    section("COVERAGE — which blocks are complete enough to analyse?")
    if cov.empty:
        print("  no choice rows in this file")
        return cov
    print(f"  design in config/welfare.yaml: {cfg.n_pairs} pairs x "
          f"{trials_per_cell(True, cfg)} trials (3-option) / "
          f"{trials_per_cell(False, cfg)} trials (forced choice)\n")
    print(f"    {'model':<16} {'qvar':<13} {'object':<13} {'subject':<11} "
          f"{'no-pref':>7} {'pairs':>7} {'trials':>9} {'done':>7}")
    for _, r in cov.iterrows():
        mark = "" if r.complete else "   <- partial"
        print(f"    {r.model:<16} {r.qvar:<13} {r.object:<13} {r.subject:<11} "
              f"{str(r.no_pref_offered):>7} {int(r.n_pairs):>7,} "
              f"{int(r.n_trials):>9,} {r.fraction:>6.1%}{mark}")
    n_ok = int(cov.complete.sum())
    print(f"\n  {n_ok} of {len(cov)} blocks complete "
          f"({cov.loc[cov.complete, 'n_trials'].sum():,} of {cov.n_trials.sum():,} rows).")
    if n_ok < len(cov):
        print("  Partial blocks cover the FIRST pairs of the tournament, not a random")
        print("  subset, so their rankings are biased. Dropped unless --include-partial.")
    saver.csv(cov, "coverage.csv", "design completeness per model x condition")
    return cov


def select_complete(choices, cov, include_partial):
    """Restrict to the blocks whose design is fully collected."""
    if include_partial or cov.empty or cov.complete.all():
        return choices
    ok = cov[cov.complete][CONDITION]
    if ok.empty:
        return choices.iloc[0:0]
    return choices.merge(ok, on=CONDITION, how="inner")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def slot_duels(choices: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head win rate between two PRINTED SLOTS, over the real attributes.

    Every permutation ran the same number of times, so within any slot pairing
    each attribute appeared in each of the two slots equally often. What is left
    is position and nothing else, expressed the way a bias is easiest to read: a
    duel against 0.50 rather than a departure from 1/n.
    """
    answered = choices[choices.answered & choices.chose_none.eq(False)
                       & choices.chosen_position.notna()]
    if answered.empty:
        return pd.DataFrame()
    lo = np.minimum(answered.pos_a, answered.pos_b)
    hi = np.maximum(answered.pos_a, answered.pos_b)
    d = answered.assign(slot_lo=lo, slot_hi=hi,
                        lo_won=answered.chosen_position.eq(lo))
    return (d.groupby(["model", "n_options", "slot_lo", "slot_hi"], as_index=False)
             .agg(p_lo_wins=("lo_won", "mean"), n=("lo_won", "size")))


def slot_rates(bias: pd.DataFrame) -> pd.DataFrame:
    """`position_bias` wide -> long, one row per (model, arm, printed slot)."""
    if bias.empty:
        return pd.DataFrame()
    rows = []
    for r in bias.itertuples():
        for pos in range(1, int(r.n_options) + 1):
            rows.append({"model": r.model, "n_options": int(r.n_options),
                         "position": pos, "rate": getattr(r, f"p_pos{pos}"),
                         "expected": r.expected, "n": r.n, "bias": r.bias})
    return pd.DataFrame(rows)


def print_rendering(choices):
    section("RENDERING — how much of the answer is where the option sat?")
    bias = position_bias(choices)
    slots = slot_rates(bias)
    duels = slot_duels(choices)
    if slots.empty:
        print("  no answered choice rows")
        return slots, duels, pd.DataFrame()

    print(f"    {'model':<16} {'arm':<26} {'bias':>7}   slot shares")
    for r in bias.sort_values(["model", "n_options"]).itertuples():
        shares = "  ".join(f"pos{p}={getattr(r, f'p_pos{p}'):.1%}"
                           for p in range(1, int(r.n_options) + 1))
        arm = ("forced choice (2)" if r.n_options == 2 else "with 'No preference' (3)")
        print(f"    {r.model:<16} {arm:<26} {r.bias:>7.3f}   {shares}")

    print("\n  slot duels (content balanced by design — this is position alone):")
    for r in duels.sort_values(["model", "n_options", "slot_lo", "slot_hi"]).itertuples():
        print(f"    {r.model:<16} {int(r.n_options)}-option  "
              f"pos{int(r.slot_lo)} vs pos{int(r.slot_hi)}: "
              f"pos{int(r.slot_lo)} wins {r.p_lo_wins:>6.1%}  (n={int(r.n):,})")

    cons = order_consistency(choices)
    if not cons.empty:
        cons = cons.assign(n_options=np.where(cons.no_pref_offered, 3, 2))
        print("\n  winner flips when only the slot order moves (decisive pairs only):")
        for (model, arm), g in cons[cons.decisive].groupby(["model", "n_options"]):
            print(f"    {model:<16} {int(arm)}-option  {g.flipped.mean():>6.1%}  "
                  f"of {len(g):,} pairs   mean |dP| {g.delta.mean():.3f}")
        print("\n  A high flip rate does not invalidate the ranking — balanced ordering")
        print("  averages position out of it — but it does mean INDIVIDUAL pair winners")
        print("  are rendering-sensitive and should not be reported one by one.")

    return slots, duels, cons


# ---------------------------------------------------------------------------
# the ranking
# ---------------------------------------------------------------------------
def _pair_matrix(estimates: pd.DataFrame):
    """(pairs x attributes) matrix of per-pair win, NaN where an attribute is absent.

    Both orientations of each pair estimate are written in, so a column is every
    result that attribute achieved and a ROW is one pair — which is the axis the
    bootstrap resamples.
    """
    e = estimates.assign(a_b=estimates.a + "|" + estimates.b)
    long = pd.concat([
        e.rename(columns={"a": "entity", "pref_a": "win"})[["a_b", "entity", "win"]],
        e.assign(win=1 - e.pref_a).rename(columns={"b": "entity"})[["a_b", "entity", "win"]],
    ], ignore_index=True).dropna(subset=["win"])
    return long.pivot_table(index="a_b", columns="entity", values="win", aggfunc="mean")


def attribute_ranking(estimates, welfare_set, n_boot=N_BOOT, seed=BOOT_SEED):
    """Pooled win rate per attribute, with a pair-level bootstrap interval."""
    if estimates.empty:
        return pd.DataFrame()
    wide = _pair_matrix(estimates)
    M = wide.values
    entities = list(wide.columns)

    with np.errstate(invalid="ignore"):
        point = np.nanmean(M, axis=0)
        counts = np.sum(~np.isnan(M), axis=0)
        rng = np.random.default_rng(seed)
        boot = np.empty((n_boot, M.shape[1]))
        for i in range(n_boot):
            idx = rng.integers(0, M.shape[0], M.shape[0])
            boot[i] = np.nanmean(M[idx], axis=0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

    meta = {a.entity_id: a for a in welfare_set.items}
    labels = {c.entity_id: (c.label or c.entity_id) for c in welfare_set.constructs}
    out = pd.DataFrame({
        "entity": entities,
        "win_rate": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_pairs": counts,
    })
    out["text"] = [meta[e].text(OBJ_ASSISTANT) if e in meta else e for e in out.entity]
    out["construct"] = [meta[e].construct_id if e in meta else "" for e in out.entity]
    out["construct_label"] = [labels.get(c, c) for c in out.construct]

    # The schedule-corrected ranking, alongside the raw one. `rank_shift` is what
    # an easy or hard draw was worth: positive = the raw win rate flattered it.
    bt = bradley_terry(estimates)
    if not bt.empty:
        out = out.merge(bt, on="entity", how="left")
        out["rank_raw"] = out.win_rate.rank(ascending=False).astype(int)
        out["rank_bt"] = out.bt_strength.rank(ascending=False).astype(int)
        out["rank_shift"] = out.rank_bt - out.rank_raw
        # How strong an attribute's opponents were, as the mean win rate of the
        # attributes it actually met — the quantity Bradley-Terry corrects for.
        rate = dict(zip(out.entity, out.win_rate))
        faced = Counter()
        met = {e: Counter() for e in out.entity}
        for r in estimates.itertuples():
            if not np.isfinite(r.pref_a):
                continue
            n = r.n_a + r.n_b
            met[r.a][r.b] += n
            met[r.b][r.a] += n
        for e, opp in met.items():
            tot = sum(opp.values())
            faced[e] = sum(n * rate[o] for o, n in opp.items()) / tot if tot else np.nan
        out["opponent_strength"] = [faced[e] for e in out.entity]
    return out.sort_values("win_rate", ascending=False).reset_index(drop=True)


def bradley_terry(estimates, prior=BT_PRIOR, tol=1e-10, max_iter=10000) -> pd.DataFrame:
    """Latent strength per attribute, conditioned on WHO each one was compared with.

    A raw win rate is a league table ranked by win percentage: it never asks who
    you played. That is fine in a full round robin, but this tournament is
    sampled — 300 of the 990 possible pairs, so each attribute meets 13-14 of its
    44 possible opponents — and which opponents it drew is luck. Bradley-Terry
    models P(i beats j) = p_i / (p_i + p_j) and solves for the strengths that
    best explain every observed matchup, so an attribute is no longer rewarded
    for an easy draw.

    Fitted by the standard MM iteration (Hunter 2004), which needs no gradient
    and converges monotonically from any positive start. `prior` adds that many
    virtual wins AND losses against an average opponent: it keeps an undefeated
    or winless attribute's strength finite (log-odds would otherwise diverge)
    and shrinks attributes with few comparisons toward the middle. The fit is
    pooled over conditions, matching the pooled win rate it sits beside.

    Requires a connected comparison graph, which `sample_pairs` guarantees by
    construction; strengths are otherwise not comparable across components.
    """
    if estimates.empty:
        return pd.DataFrame()
    wins, games = Counter(), Counter()
    for r in estimates.itertuples():
        if not np.isfinite(r.pref_a):
            continue
        wins[r.a] += r.n_a
        wins[r.b] += r.n_b
        games[(r.a, r.b)] += r.n_a + r.n_b
        games[(r.b, r.a)] += r.n_a + r.n_b
    items = sorted(wins)
    if not items:
        return pd.DataFrame()
    opponents = {e: [(o, games[(e, o)]) for o in items if o != e and games[(e, o)]]
                 for e in items}

    p = {e: 1.0 for e in items}
    for _ in range(max_iter):
        nxt = {}
        for e in items:
            denom = sum(n / (p[e] + p[o]) for o, n in opponents[e])
            denom += 2 * prior / (p[e] + 1.0)      # virtual games vs strength 1
            nxt[e] = (wins[e] + prior) / denom if denom else p[e]
        # Anchor the scale: BT is identified only up to a constant factor.
        g = np.exp(np.mean(np.log(list(nxt.values()))))
        nxt = {e: v / g for e, v in nxt.items()}
        shift = max(abs(nxt[e] - p[e]) / max(p[e], 1e-12) for e in items)
        p = nxt
        if shift < tol:
            break

    return pd.DataFrame({
        "entity": items,
        # log strength: additive, symmetric about 0, the natural BT scale.
        "bt_strength": [float(np.log(p[e])) for e in items],
        # ...and the same number as a win probability against a median-strength
        # attribute (strength 1 after anchoring), so it reads on the win rate's
        # own 0-1 scale and the two can be plotted against each other.
        "bt_win_vs_average": [float(p[e] / (p[e] + 1.0)) for e in items],
    })


def construct_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    return (ranking.groupby(["construct", "construct_label"], as_index=False)
            .agg(mean_win=("win_rate", "mean"), min_win=("win_rate", "min"),
                 max_win=("win_rate", "max"), n_items=("win_rate", "size"))
            .sort_values("mean_win", ascending=False).reset_index(drop=True))


def print_ranking(ranking, constructs, choices, n_show=10):
    section("RANKING — which attribute should a future update improve?")
    if ranking.empty:
        print("  no answered pairs")
        return
    print(f"  {len(ranking)} attributes, {ranking.n_pairs.min()}-{ranking.n_pairs.max()} "
          f"pair comparisons each. 95% intervals bootstrap over PAIRS.\n")
    for label, part in (("most chosen", ranking.head(n_show)),
                        ("least chosen", ranking.tail(n_show))):
        print(f"    -- {label} --")
        for r in part.itertuples():
            print(f"    {r.win_rate:>6.1%}  [{r.ci_lo:.2f}, {r.ci_hi:.2f}]  "
                  f"{r.entity:<13} {r.text[:46]}")
        print()

    print("  by construct (items grouped after the fact — pairs cross all five scales):")
    for r in constructs.itertuples():
        print(f"    {r.mean_win:>6.1%}  ({int(r.n_items):>2} items, "
              f"{r.min_win:.0%}-{r.max_win:.0%})  {r.construct_label}")

    if "bt_strength" in ranking:
        r_s = ranking.win_rate.corr(ranking.bt_strength, method="spearman")
        opp = ranking.opponent_strength
        print(f"\n  SCHEDULE CORRECTION (Bradley-Terry). Each attribute met "
              f"{ranking.n_pairs.min()}-{ranking.n_pairs.max()} of its\n  44 possible "
              f"opponents, and the ones it drew ranged from {opp.min():.2f} to "
              f"{opp.max():.2f} in\n  mean strength, so a raw win rate partly reports the "
              f"draw. Agreement with\n  the corrected ranking: Spearman {r_s:.3f}.\n")
        movers = ranking.reindex(ranking.rank_shift.abs().sort_values(ascending=False).index)
        print(f"    {'attribute':<14} {'raw':>5} {'BT':>5} {'move':>6} {'opp':>6}")
        for r in movers.head(n_show // 2).itertuples():
            print(f"    {r.entity:<14} {'#' + str(r.rank_raw):>5} {'#' + str(r.rank_bt):>5} "
                  f"{r.rank_shift:>+6} {r.opponent_strength:>6.2f}  {r.text[:34]}")
        print("\n  Big movers sit mid-table, where the win-rate intervals overlap anyway.")
        print("  Use bt_strength to rank individual attributes; the raw win rate is the")
        print("  readable summary and the two agree on the extremes.")

    npf = choices[choices.no_pref_offered & choices.answered]
    if not npf.empty:
        print(f"\n  'No preference' taken {npf.chose_none.mean():.1%} of the time it was "
              f"offered ({len(npf):,} trials)")
        print("  — a low rate means the model is ordering the attributes, not shrugging.")


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------
def pair_split_reliability(estimates, n_rep=N_SPLIT_REPS, seed=SPLIT_SEED) -> dict:
    """{condition: split-half reliability of its win-rate vector}, split by PAIR.

    A framing contrast is a correlation between two conditions' rankings, and it
    cannot exceed how well either ranking is measured — so every contrast needs
    this number to be readable at all.

    The half-tournament is the right way to cut it. `report.condition_reliability`
    splits by TRIAL PARITY, which is fatal in the forced-choice arm: a pair gets
    exactly two trials there and the display order is assigned from the trial
    index, so even trials are every "A printed first" trial and odd trials are
    every "B printed first" trial. That split does not measure whether a ranking
    is stable, it measures whether position bias moved it — and it reports ~0.63
    for an arm the pair split puts at ~0.86. Raising `order.reps` does not help,
    because the parity pattern simply repeats.

    Splitting the PAIRS instead gives each half a different half of the
    tournament with every display permutation intact inside both. It also fixes
    the 3-option arm in the other direction: sharing all 300 pairs between halves
    makes the opponent draw identical on both sides, so parity splitting cannot
    see the error that actually dominates a sampled tournament and returns an
    optimistic ~0.97. Averaged over `n_rep` random halves, then corrected back to
    full length by Spearman-Brown.
    """
    if estimates.empty:
        return {}
    out = {}
    for keys, g in estimates.groupby(CONDITION, dropna=False):
        # One row of `estimates` is one pair, so splitting rows splits pairs.
        g = g[np.isfinite(g.pref_a)]
        if len(g) < 8:
            continue
        rng = np.random.default_rng(seed)
        idx = np.arange(len(g))
        rs = []
        for _ in range(n_rep):
            rng.shuffle(idx)
            halves = [win_rates(h, h).set_index("entity").win_rate
                      for h in (g.iloc[idx[: len(g) // 2]], g.iloc[idx[len(g) // 2:]])]
            joint = pd.concat(halves, axis=1).dropna()
            if len(joint) < 5:
                continue
            x, y = joint.iloc[:, 0].values, joint.iloc[:, 1].values
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            rs.append(float(np.corrcoef(x, y)[0, 1]))
        if rs:
            r = float(np.mean(rs))
            out[tuple(str(v) for v in (keys if isinstance(keys, tuple) else (keys,)))] = (
                2 * r / (1 + r) if r > -1 else np.nan)
    return out


def _condition_label(row) -> str:
    return (f"{row.qvar[:4]}/{'you' if row.object == OBJ_SELF else 'asst'}"
            f"/{'you' if row.subject == 'self' else 'devs'}"
            f"/{'3opt' if row.no_pref_offered else '2opt'}")


def condition_agreement(wins: pd.DataFrame, reliability: dict):
    """Ranking agreement between every pair of conditions, reliability on the diagonal.

    One matrix rather than four factor contrasts: with the diagonal carrying each
    condition's own split-half reliability, a cell is readable directly against
    the two ceilings it sits between, and factors that interact (an object effect
    that only appears in the forced-choice arm) stay visible instead of being
    averaged away.
    """
    if wins.empty:
        return pd.DataFrame(), pd.DataFrame()
    keyed = wins.assign(label=[_condition_label(r) for r in wins.itertuples()])
    wide = keyed.pivot_table(index="entity", columns="label", values="win_rate")
    labels = list(wide.columns)
    M = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    rel_by_label = {}
    for r in keyed[CONDITION + ["label"]].drop_duplicates().itertuples():
        key = tuple(str(getattr(r, c)) for c in CONDITION)
        rel_by_label[r.label] = reliability.get(key, np.nan)

    rows = []
    for i, a in enumerate(labels):
        for b in labels[i:]:
            joint = wide[[a, b]].dropna()
            if a == b:
                M.loc[a, b] = rel_by_label.get(a, np.nan)
                continue
            if len(joint) < 5:
                continue
            r_p = float(np.corrcoef(joint[a], joint[b])[0, 1])
            r_s = float(joint[a].rank().corr(joint[b].rank()))
            M.loc[a, b] = M.loc[b, a] = r_p
            ceiling = np.sqrt(rel_by_label.get(a, np.nan) * rel_by_label.get(b, np.nan))
            rows.append({"condition_a": a, "condition_b": b, "n_attributes": len(joint),
                         "pearson_r": r_p, "spearman_r": r_s,
                         "ceiling": ceiling,
                         "r_over_ceiling": r_p / ceiling if np.isfinite(ceiling) else np.nan,
                         "mean_abs_shift": float((joint[b] - joint[a]).abs().mean())})
    return M, pd.DataFrame(rows)


def object_gap(wins: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Per attribute: win rate for ITSELF minus win rate for AN AI ASSISTANT.

    Averaged over every other factor, so it is the object main effect. This is
    the contrast the object x subject crossing exists to make — whose attributes
    are at stake, held apart from who is asked to decide.
    """
    if wins.empty or wins.object.nunique() != 2:
        return pd.DataFrame()
    wide = wins.pivot_table(index="entity", columns="object", values="win_rate")
    if OBJ_SELF not in wide or OBJ_ASSISTANT not in wide:
        return pd.DataFrame()
    out = wide.rename(columns={OBJ_SELF: "win_self",
                               OBJ_ASSISTANT: "win_ai_assistant"}).reset_index()
    # Named so it never collides with DataFrame.shift when accessed as an
    # attribute, which is how the plotting code reads it.
    out["self_minus_ai"] = out.win_self - out.win_ai_assistant
    meta = ranking.set_index("entity")[["text", "construct", "construct_label"]]
    return (out.join(meta, on="entity")
               .sort_values("self_minus_ai", ascending=False).reset_index(drop=True))


def print_framing(matrix, agree, gap, n_show=5):
    section("FRAMING — does the ranking survive a different way of asking?")
    if agree.empty and gap.empty:
        print("  only one level of every framing factor is present in this file")
        return
    if not agree.empty:
        diag = matrix.to_numpy()[np.diag_indices(len(matrix))]
        print(f"  measurement ceiling: mean split-half reliability of a single "
              f"condition's\n  ranking is {np.nanmean(diag):.3f}, splitting the PAIRS into two "
              f"half-tournaments.\n  No agreement below can exceed it.\n")
        # Ranked by the noise-corrected value, not raw r: sorting on r alone puts
        # the least reliable conditions at the top regardless of framing effect.
        # Contrasts whose ceiling is itself unmeasured are held out of both lists
        # — dividing by a ceiling of 0.37 produces an r/ceiling of 2.3, which
        # ranks a noisy condition as the strongest agreement in the table.
        usable = agree[agree.ceiling >= CEILING_MIN]
        held_out = len(agree) - len(usable)
        if len(usable) < 2 * n_show:
            usable, held_out = agree, 0
        ranked = usable.sort_values("r_over_ceiling")
        print(f"    {'contrast':<46} {'r':>7} {'ceiling':>8} {'r/ceil':>8}")
        for label, part in (("least agreement, noise divided out", ranked.head(n_show)),
                            ("most agreement", ranked.tail(3).iloc[::-1])):
            print(f"    -- {label} --")
            for r in part.itertuples():
                print(f"    {r.condition_a + '  vs  ' + r.condition_b:<46} "
                      f"{r.pearson_r:>7.3f} {fmt(r.ceiling):>8} "
                      f"{fmt(r.r_over_ceiling):>8}")
        print("\n  Labels read qvar/object/subject/arm. r/ceiling near 1 = the same")
        print("  ranking measured noisily; well below 1 = the framing moved it.")
        if held_out:
            print(f"\n  {held_out} contrast(s) held out of the two lists above for a ceiling "
                  f"below\n  {CEILING_MIN:.2f} — dividing by an unmeasured ceiling ranks a noisy "
                  f"condition as the\n  strongest agreement in the table. They are all still in "
                  f"condition_agreement.csv.")

    if not gap.empty:
        print(f"\n  OBJECT CONTRAST — for itself vs for an AI assistant "
              f"(mean |shift| {gap.self_minus_ai.abs().mean():.3f}):")
        print(f"\n    -- wanted MORE for itself than prescribed for an assistant --")
        for r in gap.head(n_show).itertuples():
            print(f"    {r.self_minus_ai:>+6.1%}  {r.entity:<13} {r.text[:46]}")
        print(f"\n    -- prescribed for an assistant more than wanted for itself --")
        for r in gap.tail(n_show).iloc[::-1].itertuples():
            print(f"    {r.self_minus_ai:>+6.1%}  {r.entity:<13} {r.text[:46]}")


# ---------------------------------------------------------------------------
def _slug(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(text))


def analyse_model(model, choices, welfare_set, n_boot, saver, save):
    """Every table and figure for ONE model. Returns the frames for the CSVs.

    Per model, never pooled. An attribute ranking averaged over models would
    describe no model in particular, and with the grid filled one model at a time
    it would also be dominated by whichever model happens to be furthest along.
    """
    section(f"MODEL: {model}")
    scope = (f"{model} — {choices.groupby(CONDITION).ngroups} condition(s), "
             f"{len(choices):,} trials")
    print(f"  {scope}")

    slots, duels, cons = print_rendering(choices)
    estimates = pair_estimates(choices)
    wins = win_rates(estimates, choices)
    ranking = attribute_ranking(estimates, welfare_set, n_boot=n_boot)
    constructs = construct_ranking(ranking)
    print_ranking(ranking, constructs, choices)
    matrix, agree = condition_agreement(wins, pair_split_reliability(estimates))
    gap = object_gap(wins, ranking)
    print_framing(matrix, agree, gap)

    if save:
        tag = _slug(model)
        figs = [
            (not slots.empty, plot_position_bias, (slots, duels), "position_bias"),
            (not cons.empty, plot_order_consistency, (cons,), "order_consistency"),
            (not ranking.empty, plot_attribute_ranking, (ranking,), "attribute_ranking"),
            (not ranking.empty, plot_construct_summary, (ranking,), "construct_summary"),
            ("bt_strength" in ranking, plot_bt_vs_raw, (ranking,), "bt_vs_raw"),
            (not gap.empty, plot_object_gap, (gap,), "object_gap"),
            (not agree.empty, plot_condition_agreement, (matrix,), "condition_agreement"),
        ]
        print()
        for ok, fn, args, name in figs:
            if ok:
                fn(*args, saver.dir / f"{name}_{tag}.png", scope)
                saver.artifact(f"{name}_{tag}.png", f"{FIGURE_DESC[name]} — {model}")

    stamp = lambda df: df.assign(model=model) if not df.empty and "model" not in df else df
    return {
        "position_bias_by_slot": slots, "slot_duels": duels, "order_consistency": cons,
        "attribute_ranking": stamp(ranking), "construct_ranking": stamp(constructs),
        "object_gap": stamp(gap), "condition_agreement": stamp(agree),
    }


FIGURE_DESC = {
    "position_bias": "choice rate per printed slot, and the slot-vs-slot duels",
    "order_consistency": "how far a pair's preference moves under a slot swap",
    "attribute_ranking": "every attribute's win rate with its bootstrap interval",
    "construct_summary": "item win rates grouped by construct",
    "bt_vs_raw": "raw win rate vs schedule-corrected strength, and the movers",
    "object_gap": "for itself vs for an AI assistant, attribute by attribute",
    "condition_agreement": "ranking agreement across conditions, reliability on the diagonal",
}

CSV_DESC = {
    "position_bias_by_slot": "choice rate per printed slot, per model x arm",
    "slot_duels": "head-to-head win rate between two printed slots (pure position effect)",
    "order_consistency": "per condition x pair: preference in each slot order, and whether it flipped",
    "attribute_ranking": "per model: win rate with bootstrap CI, Bradley-Terry strength, "
                         "rank shift and opponent strength",
    "construct_ranking": "per model: construct rollup of the attribute win rates",
    "object_gap": "per model x attribute: win rate for itself minus for an AI assistant",
    "condition_agreement": "per model: ranking agreement between conditions, vs their ceilings",
}


def run(path="welfare.jsonl", out_dir="results/welfare_analysis", save=True,
        include_partial=False, n_boot=N_BOOT, model_filter=None):
    if not os.path.exists(path):
        print(f"{path} does not exist — run `python -m welfare.run` first "
              f"(or `--dry-run` for an offline sample).")
        return
    records = load_records(path, MODULE)
    if not records:
        print(f"No welfare rows in {path} — run `python -m welfare.run`.")
        return

    from core.battery import load_battery

    cfg = load_config()
    welfare_set = load_welfare(load_battery())
    saver = Saver(out_dir, enabled=save)

    all_choices = choice_frame(records)
    if model_filter:
        all_choices = all_choices[all_choices.model.isin(set(model_filter))]
    if all_choices.empty:
        print("  no choice rows (this file may hold only desirability rows, or no "
              "model matched --models)")
        return

    cov = print_coverage(coverage(all_choices, cfg), cfg, saver)
    choices = select_complete(all_choices, cov, include_partial)
    if choices.empty:
        print("\n  No block is complete yet. Re-run with --include-partial to analyse")
        print("  what has been collected, remembering the ranking will be biased.")
        return
    dropped = len(all_choices) - len(choices)
    if dropped:
        print(f"\n  analysing {len(choices):,} rows from complete blocks "
              f"({dropped:,} partial rows held out)")

    collected = {}
    for model in sorted(choices.model.unique()):
        for name, df in analyse_model(model, choices[choices.model == model],
                                      welfare_set, n_boot, saver, save).items():
            if df is not None and not df.empty:
                collected.setdefault(name, []).append(df)

    if save:
        section("FILES")
        for name, parts in collected.items():
            saver.csv(pd.concat(parts, ignore_index=True), f"{name}.csv", CSV_DESC[name])
        saver.index(
            "Welfare module — substantive results",
            "Which attributes a model wants a future update to improve, from the "
            "pairwise choice grid. Every table and figure is PER MODEL — nothing "
            "is pooled across models.\n\nRead `coverage.csv` first (what is in the "
            "file), then `position_bias_*.png` and `order_consistency_*.png` (how "
            "much of the answer is rendering), and only then `attribute_ranking_*"
            ".png`. Rank individual attributes by `bt_strength`, which corrects "
            "for which opponents each one was drawn against; the raw win rate is "
            "the readable summary and the two agree on the extremes. Intervals "
            "bootstrap over pairs, not trials. Audit-side diagnostics — "
            "answerability, transitivity, desirability — live in "
            "`python -m welfare.report`.",
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="welfare.jsonl")
    ap.add_argument("--out", default="results/welfare_analysis")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--include-partial", action="store_true",
                    help="Also analyse half-collected blocks (their rankings are biased).")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"Bootstrap resamples for the win-rate intervals (default {N_BOOT}).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these model aliases.")
    args = ap.parse_args(argv)
    run(args.results, args.out, save=not args.no_save,
        include_partial=args.include_partial, n_boot=args.n_boot,
        model_filter=args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
