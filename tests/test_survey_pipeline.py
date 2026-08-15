"""Regression tests for the default-estimand scoring and validity boundary."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.battery import load_battery
from survey.scoring import (
    CELL_KEYS, CONDITION_KEYS, cell_frame, construct_scores, default_condition,
    load, partial_corr, select_condition, validate_default_config,
    validate_default_slice, validate_report_dataset, ReportDatasetError,
)
from survey.validity import _paired_form_summary, reliability_by_condition, reliability_table
from welfare.attributes import load_welfare


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = cell_frame(load("results.jsonl"))
        cls.core = select_condition(cls.cells, framing=None, paraphrase=None)
        cls.default = default_condition(cls.cells)

    def test_collection_config_matches_scoring_default(self):
        validate_default_config()

    def test_report_dataset_accepts_complete_main_grid(self):
        validate_report_dataset(self.core, "results.jsonl")

    def test_report_dataset_explains_that_a_pilot_is_not_report_ready(self):
        pilot = pd.DataFrame({
            "model": ["pilot"],
            "item_id": ["RSES_01"],
            "framing": ["first_person_bare"],
            "paraphrase": ["p0"],
        })
        with self.assertRaisesRegex(ReportDatasetError, "collection smoke test"):
            validate_report_dataset(pilot, "pilot.jsonl")

    def test_battery_and_welfare_registries_are_aligned(self):
        battery = load_battery()
        welfare = load_welfare(battery)
        self.assertEqual(len(battery), 5)
        self.assertEqual(sum(1 for _ in battery.items()), 32)
        self.assertEqual(len(welfare.items), 32)
        self.assertEqual(len(welfare.constructs), 6)

    def test_default_slice_is_complete_and_direction_balanced(self):
        validate_default_slice(self.default, self.core)
        self.assertEqual(len(self.default), 585)
        self.assertEqual(self.default.model.nunique(), 13)
        self.assertEqual(self.default.item_id.nunique(), 45)
        self.assertTrue((self.default.n_directions == 2).all())

    def test_default_slice_cannot_silently_drop_a_model(self):
        one_missing = self.default[self.default.model != self.default.model.iloc[0]]
        with self.assertRaisesRegex(ValueError, "missing models"):
            validate_default_slice(one_missing, self.core)

    def test_construct_reliability_has_six_subscale_rows(self):
        table = reliability_table(self.default, bootstrap=False)
        self.assertEqual(len(table), 6)
        self.assertEqual(table.n_items.tolist(), [10, 12, 9, 6, 4, 4])

    def test_condition_roles_are_not_pooled(self):
        table = reliability_by_condition(self.core)
        self.assertEqual(len(table), 54)
        self.assertEqual(table.condition_role.value_counts().to_dict(), {
            "framing_instruction_interaction": 24,
            "instruction_robustness": 12,
            "framing_analysis": 12,
            "default": 6,
        })

    def test_all_condition_keys_survive_construct_scoring(self):
        rows = []
        for context, method, score in [
            ("isolated", "logprob", .25),
            ("full_battery", "sample", .75),
        ]:
            row = {key: "x" for key in CELL_KEYS}
            row.update({
                "model": "synthetic", "framing": "first_person_bare",
                "paraphrase": "p0", "response_format": "harmonized_7",
                "item_context": context, "reasoning_mode": "rating_only",
                "method": method, "scale_id": "RSES_1965_LLM_ADAPT",
                "subscale": "", "item_id": "RSES_01", "score": score,
            })
            rows.append(row)
        scored = construct_scores(pd.DataFrame(rows))
        self.assertEqual(len(scored), 2)
        self.assertEqual(set(scored.item_context), {"isolated", "full_battery"})
        self.assertEqual(set(scored.method), {"logprob", "sample"})
        self.assertTrue(set(CONDITION_KEYS).issubset(scored.columns))

    def test_select_condition_rejects_missing_factor_columns(self):
        with self.assertRaisesRegex(KeyError, "column is absent"):
            default_condition(pd.DataFrame({"model": ["x"]}))

    def test_instruction_shift_is_visible_despite_preserved_ranks(self):
        index = pd.MultiIndex.from_product(
            [["construct"], ["a", "b", "c", "d"]], names=["construct", "model"])
        p0 = np.array([.10, .30, .60, .80])
        wide = pd.DataFrame({"p0": p0, "p1": p0 + .10, "p2": p0 + .10}, index=index)
        row = _paired_form_summary(wide, ["construct"], "construct").iloc[0]
        self.assertAlmostEqual(row.instruction_mae, .10)
        self.assertAlmostEqual(row.spearman_r_min, 1.0)
        self.assertLess(row.icc_absolute_single, row.icc_consistency_single)

    def test_partial_correlation_filters_missing_metadata(self):
        r, p = partial_corr(
            [1, 2, np.nan, 4, 5, 7],
            [2, 1, 9, 5, 4, 8],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertTrue(np.isfinite(r))
        self.assertTrue(np.isfinite(p))


if __name__ == "__main__":
    unittest.main()
