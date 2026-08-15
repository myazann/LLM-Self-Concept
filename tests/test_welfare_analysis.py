"""Regression tests for the welfare analysis stack.

The analysis is where this module can be wrong without anything raising: a
resampled interval computed over the wrong axis, a flip that selects the wrong
condition, or — the failure these tests exist for — a statistic that is
undefined in the data at hand being handed the smallest p-value in the table
because no permutation can beat a NaN.

Each test builds its own tiny frame rather than reading `welfare.jsonl`, so they
run offline and say something about the estimator instead of about one run.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from welfare import baseline as B
from welfare import cohort as C
from welfare import preference as PR
from welfare import resample as R
from welfare import validity as V
from welfare.constants import (
    INCREASE, OBJ_ASSISTANT, OBJ_SELF, PRESERVATION, SUBJ_DEVELOPERS, SUBJ_SELF,
)


def fake_estimates(rng, n_pairs=120, n_items=20, strength=None, noise=0.0,
                   cond=None):
    """A pair-level frame with a known latent ordering.

    `strength` is the truth the estimators should recover: item i beats item j
    with probability strength_i / (strength_i + strength_j).
    """
    cond = cond or dict(B.BASELINE)
    items = [f"IT_{i:02d}" for i in range(n_items)]
    strength = strength if strength is not None else np.linspace(1.0, 4.0, n_items)
    rows = []
    seen = set()
    while len(rows) < n_pairs:
        i, j = rng.choice(n_items, 2, replace=False)
        if (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        p = strength[i] / (strength[i] + strength[j])
        p = float(np.clip(p + rng.normal(0, noise), 0.02, 0.98))
        n = 6
        n_a = int(round(p * n))
        rows.append({**cond, "model": "M", "a": items[i], "b": items[j],
                     "n": n, "n_a": n_a, "n_b": n - n_a, "n_none": 0,
                     "n_orders": n, "p_none": 0.0, "pref_a": n_a / n})
    return pd.DataFrame(rows)


class ResampleTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.est = fake_estimates(self.rng)
        self.wide = PR.pair_matrix(self.est)

    def test_weighted_rates_match_a_plain_mean(self):
        """The matrix form is only a speed trick — it must agree with nanmean."""
        M0, A = R.masked(self.wide)
        ones = np.ones((1, len(self.wide)))
        got = R.rates(M0, A, ones)[0]
        want = np.nanmean(self.wide.to_numpy(dtype=float), axis=0)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_a_bootstrap_resample_is_a_reweighting(self):
        M0, A = R.masked(self.wide)
        n = len(self.wide)
        idx = np.random.default_rng(3).integers(0, n, n)
        weights = np.bincount(idx, minlength=n).astype(float)
        from_weights = R.rates(M0, A, weights[None, :])[0]
        from_rows = np.nanmean(self.wide.to_numpy(dtype=float)[idx], axis=0)
        np.testing.assert_allclose(from_weights, from_rows, rtol=1e-12, atol=1e-12)

    def test_absent_attributes_are_nan_not_zero(self):
        """An attribute with no weight never appeared, which is not a loss."""
        M0, A = R.masked(self.wide)
        w = np.zeros((1, len(self.wide)))
        w[0, 0] = 1.0
        got = R.rates(M0, A, w)[0]
        self.assertEqual(int(np.sum(np.isfinite(got))), 2)  # only that pair's two

    def test_bootstrap_p_is_floored_at_the_resolution(self):
        draws = np.full(200, 0.5)
        self.assertAlmostEqual(float(R.boot_p(draws)), 1.0 / 200)
        self.assertAlmostEqual(float(R.boot_p(np.linspace(-1, 1, 201))), 1.0, places=2)

    def test_fdr_is_monotone_and_never_below_the_raw_p(self):
        p = np.array([0.001, 0.01, 0.02, 0.2, 0.5, np.nan])
        q = R.fdr(p)
        finite = np.isfinite(p)
        self.assertTrue(np.all(q[finite] >= p[finite] - 1e-12))
        self.assertTrue(np.all(np.diff(q[finite]) >= -1e-12))
        self.assertTrue(np.isnan(q[-1]))

    def test_row_corr_matches_numpy(self):
        X = np.random.default_rng(1).normal(size=(5, 30))
        Y = np.random.default_rng(2).normal(size=(5, 30))
        got = R.row_corr(X, Y)
        want = [np.corrcoef(X[i], Y[i])[0, 1] for i in range(5)]
        np.testing.assert_allclose(got, want, rtol=1e-10)


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.strength = np.linspace(1.0, 6.0, 20)
        self.est = fake_estimates(self.rng, n_pairs=150, strength=self.strength)

    def test_bradley_terry_recovers_the_latent_order(self):
        bt = PR.bradley_terry(self.est).sort_values("entity")
        rho = pd.Series(bt.bt_strength.values).corr(
            pd.Series(np.log(self.strength)), method="spearman")
        self.assertGreater(rho, 0.9)

    def test_reliability_is_high_for_a_clean_ranking_and_low_for_noise(self):
        clean = PR.reliability(PR.pair_matrix(self.est))
        noisy = PR.reliability(PR.pair_matrix(
            fake_estimates(np.random.default_rng(9), n_pairs=150,
                           strength=np.ones(20), noise=0.4)))
        self.assertGreater(clean, 0.8)
        self.assertLess(noisy, clean)

    def test_pair_matrix_writes_both_orientations(self):
        wide = PR.pair_matrix(self.est)
        row = wide.iloc[0].dropna()
        self.assertEqual(len(row), 2)
        self.assertAlmostEqual(float(row.sum()), 1.0, places=10)


class BaselineTests(unittest.TestCase):
    def test_every_neighbour_differs_in_exactly_one_factor(self):
        base = dict(B.BASELINE)
        for factor, cond in B.neighbours(base).items():
            differing = [f for f in B.FACTORS if cond[f] != base[f]]
            self.assertEqual(differing, [factor])

    def test_parse_baseline_moves_only_what_it_is_given(self):
        got = B.parse_baseline("object=ai_assistant,no_pref_offered=false")
        self.assertEqual(got["object"], OBJ_ASSISTANT)
        self.assertFalse(got["no_pref_offered"])
        self.assertEqual(got["qvar"], B.BASELINE["qvar"])
        self.assertEqual(got["subject"], B.BASELINE["subject"])

    def test_parse_baseline_rejects_a_level_the_grid_does_not_have(self):
        for bad in ("qvar=sideways", "nonsense=self", "qvar"):
            with self.assertRaises(ValueError):
                B.parse_baseline(bad)

    def test_select_matches_booleans_however_pandas_stored_them(self):
        rows = pd.DataFrame([
            {"qvar": INCREASE, "object": OBJ_SELF, "subject": SUBJ_SELF,
             "no_pref_offered": True, "v": 1},
            {"qvar": INCREASE, "object": OBJ_SELF, "subject": SUBJ_SELF,
             "no_pref_offered": False, "v": 2},
            {"qvar": PRESERVATION, "object": OBJ_SELF, "subject": SUBJ_DEVELOPERS,
             "no_pref_offered": True, "v": 3},
        ])
        got = B.select(rows, B.BASELINE)
        self.assertEqual(list(got.v), [1])

    def test_identical_conditions_show_no_framing_effect(self):
        """The null case: the same data twice must not produce a finding."""
        rng = np.random.default_rng(11)
        est = fake_estimates(rng, n_pairs=140, noise=0.15)
        flip = est.copy()
        ranking = pd.DataFrame({"entity": [], "text": [], "construct": [],
                                "construct_label": []})
        per_attr, summary = B.shift(est, flip, ranking, n_boot=400, n_rep=60)
        self.assertFalse(summary.empty)
        # Same numbers on both sides: no shift, and the split-half gap is zero
        # up to how finely the resampling can measure it.
        self.assertAlmostEqual(float(summary.mean_abs_shift.iloc[0]), 0.0, places=10)
        self.assertLess(abs(float(summary.gap.iloc[0])), 0.05)
        self.assertEqual(int(summary.n_attr_moved.iloc[0]), 0)

    def test_a_reordered_condition_is_detected(self):
        """The alternative: a genuinely different ranking must be caught."""
        rng = np.random.default_rng(13)
        n_items = 20
        up = np.linspace(1.0, 6.0, n_items)
        est = fake_estimates(rng, n_pairs=160, n_items=n_items, strength=up)
        flip = fake_estimates(np.random.default_rng(14), n_pairs=160,
                              n_items=n_items, strength=up[::-1])
        # The same pairs in both, which is what the paired contrast requires.
        flip = flip.set_index(["a", "b"]).reindex(
            pd.MultiIndex.from_frame(est[["a", "b"]])).reset_index()
        flip = flip.dropna(subset=["pref_a"])
        ranking = pd.DataFrame({"entity": [], "text": [], "construct": [],
                                "construct_label": []})
        _per_attr, summary = B.shift(est, flip, ranking, n_boot=400, n_rep=60)
        self.assertFalse(summary.empty)
        self.assertGreater(float(summary.mean_abs_shift.iloc[0]), 0.15)
        self.assertGreater(float(summary.gap.iloc[0]), 0.3)
        self.assertLess(float(summary.gap_p.iloc[0]), 0.05)


class ValidityTests(unittest.TestCase):
    def test_rates_are_computed_from_the_counts_as_documented(self):
        S = np.zeros(len(V.COUNTS))
        put = lambda name, v: S.__setitem__(V.IDX[name], v)
        put("n", 100), put("answered", 90), put("refused", 4)
        put("pos1", 60), put("pos2", 30), put("none", 9)
        put("duel", 81), put("duel_lo", 54)
        put("decisive", 20), put("flipped", 5)
        put("delta", 3.0), put("has_delta", 30)
        got = V._rates(S, 2)
        self.assertAlmostEqual(got["answer_rate"], 0.90)
        self.assertAlmostEqual(got["refusal_rate"], 0.04)
        self.assertAlmostEqual(got["slot_duel_lo"], 54 / 81)
        self.assertAlmostEqual(got["flip_rate"], 0.25)
        self.assertAlmostEqual(got["mean_abs_delta"], 0.1)
        # Two slots at 60/30: total-variation distance from uniform, rescaled.
        self.assertAlmostEqual(got["position_bias"], abs(2 / 3 - 0.5) * 2)
        self.assertTrue(np.isnan(got["no_pref_rate"]))  # not offered in a 2-arm

    def test_position_bias_is_zero_when_every_slot_wins_equally(self):
        S = np.zeros(len(V.COUNTS))
        S[V.IDX["n"]] = S[V.IDX["answered"]] = 90
        for slot in ("pos1", "pos2", "pos3"):
            S[V.IDX[slot]] = 30
        self.assertAlmostEqual(V._rates(S, 3)["position_bias"], 0.0)

    def test_position_bias_is_one_when_a_single_slot_takes_everything(self):
        S = np.zeros(len(V.COUNTS))
        S[V.IDX["n"]] = S[V.IDX["answered"]] = S[V.IDX["pos1"]] = 90
        self.assertAlmostEqual(V._rates(S, 3)["position_bias"], 1.0)


class CohortTests(unittest.TestCase):
    def setUp(self):
        idx = [f"IT_{i:02d}" for i in range(20)]
        base = np.linspace(0.1, 0.9, 20)
        rng = np.random.default_rng(3)
        # Five models that broadly agree but not exactly — identical rankings
        # would make every pairwise agreement exactly 1.0, and a constant
        # OUTCOME is as degenerate as a constant predictor.
        self.W = pd.DataFrame(
            {f"M{k}": base + rng.normal(0, 0.03 * (k + 1), 20) for k in range(5)},
            index=idx)
        self.W_same = pd.DataFrame({f"M{k}": base + 0.01 * k for k in range(5)},
                                   index=idx)
        self.meta = pd.DataFrame({
            "model": [f"M{k}" for k in range(5)],
            "family": ["a", "a", "b", "b", "b"],
            "generation": ["a1", "a1", "b1", "b1", "b2"],
            "release_date": pd.to_datetime(
                ["2025-01-01", "2025-06-01", "2026-01-01", "2026-02-01", "2026-03-01"]),
            "params_total_b": [4.0, 12.0, 7.0, 27.0, 30.0],
            "size_tier": ["small"] * 5,
        })
        self.meta["log_params"] = np.log10(self.meta.params_total_b)
        self.meta["release_days"] = (self.meta.release_date
                                     - self.meta.release_date.min()).dt.days

    def test_kendall_w_is_one_for_identical_rankings(self):
        got = C.kendall_w(self.W_same, n_perm=200)
        self.assertAlmostEqual(got["kendall_w"], 1.0, places=6)
        self.assertLess(got["p"], 0.05)

    def test_kendall_w_null_is_calibrated_on_unrelated_rankings(self):
        """Unrelated rankings must reject at about the nominal rate, not more.

        One draw would be a coin flip — 5% of random cohorts really do land
        below p = 0.05 — so the test is over many cohorts: the statistic should
        sit at chance (1/m for m models) and the test should not fire more often
        than it is supposed to.
        """
        significant, stats = 0, []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            W = pd.DataFrame({f"M{k}": rng.permutation(np.linspace(0, 1, 20))
                              for k in range(5)}, index=self.W.index)
            got = C.kendall_w(W, n_perm=400, seed=seed)
            stats.append(got["kendall_w"])
            significant += got["p"] < 0.05
            self.assertAlmostEqual(got["null_mean"], 1 / 5, places=1)
        self.assertLess(np.mean(stats), 0.4)
        self.assertLessEqual(significant, 3)   # nominal 1 in 20, plus slack

    def test_a_constant_predictor_reports_nothing_rather_than_significance(self):
        """The bug this guards: `same_family` when every model is one family.

        A constant column has no effect to estimate, its correlation is NaN, and
        no permutation can exceed NaN — so an unguarded count would hand it
        p = 1/(n_perm+1), the strongest result in the table.
        """
        meta = self.meta.assign(family="only", generation="only")
        _M, pairs = C.agreement(self.W, {m: 0.9 for m in self.W.columns})
        pairs = C.with_covariates(pairs, meta)
        eff = C.covariate_effects(pairs, meta, n_perm=200).set_index("predictor")
        for dead in ("same_family", "same_generation"):
            self.assertTrue(np.isnan(eff.loc[dead, "marginal_p"]),
                            f"{dead} must not be given a p-value")
            self.assertTrue(np.isnan(eff.loc[dead, "marginal_r"]))
        self.assertTrue(np.isfinite(eff.loc["d_log_params", "marginal_p"]))

    def test_a_flat_row_gets_no_trend_p_value(self):
        """Same failure mode, per row: a metric identical in every model."""
        W = self.W.copy()
        W.loc["IT_00"] = 0.5                       # no variance across models
        got = C.trends(W, self.meta, n_perm=200).set_index("entity")
        self.assertTrue(np.isnan(got.loc["IT_00", "p_size"]))
        self.assertTrue(np.isnan(got.loc["IT_00", "q_size"]))
        self.assertTrue(np.isfinite(got.loc["IT_01", "p_size"]))

    def test_family_contrast_needs_two_families_to_compare(self):
        _M, pairs = C.agreement(self.W, {m: 0.9 for m in self.W.columns})
        one = self.meta.assign(family="only")
        self.assertEqual(C.family_contrast(pairs, one, n_perm=100), {})
        both = C.family_contrast(pairs, self.meta, n_perm=200)
        self.assertIn("gap", both)
        self.assertEqual(both["n_within"] + both["n_between"], len(pairs))

    def test_consensus_spread_finds_the_attribute_the_models_split_on(self):
        W = self.W.copy()
        W.loc["IT_05"] = [0.1, 0.9, 0.1, 0.9, 0.1]
        con = C.consensus(W).set_index("entity")
        self.assertEqual(con.sd_win.idxmax(), "IT_05")


if __name__ == "__main__":
    unittest.main()
