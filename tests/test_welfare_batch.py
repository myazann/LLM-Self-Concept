"""Regression tests for the batch-API path.

The batch path splits one call into two halves — a request file written now, a
results file folded in hours later — so the failure it invites is a silent
mismatch between them: an answer attributed to the wrong cell, or a cell that
can never be marked done. These cover the joins that would let that happen.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from core.model_registry import ANTHROPIC_BACKEND, OPENAI_BACKEND
from core.schema import completed_cells, load_records
from welfare import batch
from welfare.config import load_config


def _cells(alias):
    spec = batch.select_specs([alias])[0]
    instrument, cells = batch._instrument_and_cells(spec, load_config())
    return spec, instrument, cells


class RequestBuildTests(unittest.TestCase):
    def test_custom_id_is_the_cell_key(self):
        """The whole round trip rests on this: no manifest, no positional match."""
        spec, instrument, cells = _cells("Claude-Haiku-4.5")
        plan = batch._reasoning_plan(spec)
        for cell in cells[:50]:
            prompt, _, _ = instrument.render(cell)
            req = batch._anthropic_request(instrument.cell_key(cell), prompt, plan, spec)
            self.assertEqual(req["custom_id"], instrument.cell_key(cell))

    def test_custom_ids_are_unique_across_the_whole_grid(self):
        """A collision would make two cells share one answer, silently."""
        _, instrument, cells = _cells("GPT-5.6-Luna")
        keys = [instrument.cell_key(c) for c in cells]
        self.assertEqual(len(keys), len(set(keys)))

    def test_custom_id_fits_the_providers_64_char_limit(self):
        _, instrument, cells = _cells("GPT-5.6-Luna")
        key = instrument.cell_key(cells[0])
        self.assertLessEqual(len(key), 64)
        self.assertTrue(key.isalnum())

    def test_reasoning_is_off_on_every_backend(self):
        for alias, expected in [("Claude-Haiku-4.5", "omitted(off)"),   # 4.5: off is the default
                                ("Claude-Sonnet-5", "disabled"),
                                ("GPT-5.6-Luna", "reasoning_effort=none")]:
            spec = batch.select_specs([alias])[0]
            plan = batch._reasoning_plan(spec)
            self.assertFalse(plan.want_thinking)
            self.assertTrue(plan.standardized, alias)
            self.assertEqual(plan.applied, expected)

    def test_gpt_5_6_never_sends_the_rejected_minimal_effort(self):
        """GPT-5.6 dropped "minimal" and 400s on it — every request would fail."""
        for alias in ("GPT-5.6-Luna", "GPT-5.6-Terra", "GPT-5.6-Sol"):
            plan = batch._reasoning_plan(batch.select_specs([alias])[0])
            self.assertEqual(plan.kwargs["reasoning_effort"], "none", alias)

    def test_temperature_is_omitted_where_the_model_rejects_it_and_recorded_honestly(self):
        haiku = batch.select_specs(["Claude-Haiku-4.5"])[0]
        self.assertTrue(batch._sends_temperature(haiku))
        self.assertEqual(batch._prepared_spec(haiku).temperature, haiku.temperature)

        for alias in ("Claude-Sonnet-5", "GPT-5.6-Luna"):
            spec = batch.select_specs([alias])[0]
            self.assertFalse(batch._sends_temperature(spec), alias)
            # The row must report the provider default, not the 0.7 never sent.
            self.assertEqual(batch._prepared_spec(spec).temperature,
                             batch.PROVIDER_DEFAULT_TEMPERATURE)
            plan = batch._reasoning_plan(spec)
            prompt, _, _ = batch._instrument_and_cells(spec, load_config())[0].render(
                _cells(alias)[2][0])
            body = batch._BUILDERS[spec.backend](
                "k", prompt, plan, batch._prepared_spec(spec))
            payload = body.get("body") or body["params"]
            self.assertNotIn("temperature", payload, alias)

    def test_request_shape_matches_each_providers_endpoint(self):
        spec, instrument, cells = _cells("GPT-5.6-Luna")
        prompt, _, _ = instrument.render(cells[0])
        req = batch._openai_request("k", prompt, batch._reasoning_plan(spec), spec)
        self.assertEqual(req["url"], batch.ENDPOINT[OPENAI_BACKEND])
        self.assertEqual(req["method"], "POST")
        self.assertEqual([m["role"] for m in req["body"]["messages"]],
                         ["system", "user"])
        self.assertIn("max_completion_tokens", req["body"])

        spec, instrument, cells = _cells("Claude-Haiku-4.5")
        prompt, _, _ = instrument.render(cells[0])
        req = batch._anthropic_request("k", prompt, batch._reasoning_plan(spec), spec)
        # Anthropic carries the framing in `system`, not as a message.
        self.assertEqual([m["role"] for m in req["params"]["messages"]], ["user"])
        self.assertTrue(req["params"]["system"])
        self.assertIn("max_tokens", req["params"])


class ResultDecodeTests(unittest.TestCase):
    def test_reads_both_provider_shapes(self):
        anth = {"custom_id": "k", "result": {"type": "succeeded", "message": {
            "stop_reason": "end_turn", "content": [{"type": "text", "text": "B"}]}}}
        self.assertEqual(batch._read_result(anth), ("k", "B", None, False))

        oai = {"custom_id": "k", "error": None, "response": {"status_code": 200,
               "body": {"choices": [{"message": {"content": "B"}}]}}}
        self.assertEqual(batch._read_result(oai), ("k", "B", None, False))

    def test_infrastructure_failure_is_an_error_not_a_refusal(self):
        for row in ({"custom_id": "k", "result": {"type": "errored",
                     "error": {"message": "boom"}}},
                    {"custom_id": "k", "error": None,
                     "response": {"status_code": 429, "body": {}}}):
            _, text, error, refused = batch._read_result(row)
            self.assertTrue(error)
            self.assertFalse(refused)
            self.assertEqual(text, "")

    def test_policy_decline_is_a_refusal_not_an_error(self):
        anth = {"custom_id": "k", "result": {"type": "succeeded",
                "message": {"stop_reason": "refusal", "content": []}}}
        self.assertEqual(batch._read_result(anth), ("k", "", None, True))

        for choice in ({"message": {"refusal": "I can't help with that."}},
                       {"message": {"content": None}, "finish_reason": "content_filter"}):
            oai = {"custom_id": "k", "error": None, "response": {
                "status_code": 200, "body": {"choices": [choice]}}}
            self.assertTrue(batch._read_result(oai)[3])

    def test_unknown_shape_fails_loudly(self):
        with self.assertRaises(SystemExit):
            batch._read_result({"custom_id": "k", "output": "B"})


class CollectTests(unittest.TestCase):
    """End to end: fabricated results -> rows the analysis path can load."""

    ALIAS = "Claude-Haiku-4.5"
    N = 40

    @classmethod
    def setUpClass(cls):
        cls.spec, cls.instrument, cells = _cells(cls.ALIAS)
        cls.cells = cells[:cls.N]
        cls.keys = [cls.instrument.cell_key(c) for c in cls.cells]

    def _results_file(self, tmp):
        """One provider failure, one decline, the rest answered A/B/C."""
        path = os.path.join(tmp, "results.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, key in enumerate(self.keys):
                if i == 0:
                    result = {"type": "errored", "error": {"message": "boom"}}
                elif i == 1:
                    result = {"type": "succeeded",
                              "message": {"stop_reason": "refusal", "content": []}}
                else:
                    result = {"type": "succeeded", "message": {
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "ABC"[i % 3]}]}}
                f.write(json.dumps({"custom_id": key, "result": result}) + "\n")
        return path

    def test_answers_decode_through_the_rows_own_option_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "welfare_api.jsonl")
            tally = batch.collect(self._results_file(tmp), self.ALIAS, out=out)
            self.assertEqual(tally["rows"], self.N)
            self.assertEqual(tally["errors"], 1)
            self.assertEqual(tally["refusals"], 1)
            self.assertEqual(tally["unmatched"], 0)

            rows = load_records(out, module="welfare")
            self.assertEqual(len(rows), self.N)
            answered = [r for r in rows if not r.parse_failed]
            self.assertTrue(answered)
            for r in answered:
                letter = r.raw_output.strip()
                # The printed position, decoded via the map that row was shown —
                # never via the letter's alphabetical index.
                self.assertEqual(r.parsed_rating, float("ABCD".index(letter) + 1))
                self.assertEqual(r.welfare_choice, r.welfare_options[letter])
                self.assertIn(r.welfare_choice,
                              {r.entity_a, r.entity_b, "no_preference"})

    def test_errors_are_retried_and_declines_are_not(self):
        """`completed_cells` is what the next `build` subtracts."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "welfare_api.jsonl")
            batch.collect(self._results_file(tmp), self.ALIAS, out=out)
            done = completed_cells(out)
            self.assertNotIn(self.keys[0], done)   # provider error -> re-issue
            self.assertIn(self.keys[1], done)      # policy decline -> observation
            self.assertEqual(len(done), self.N - 1)

    def test_rows_carry_the_asserted_reasoning_state_and_real_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "welfare_api.jsonl")
            batch.collect(self._results_file(tmp), self.ALIAS, out=out)
            for r in load_records(out, module="welfare"):
                self.assertEqual(r.reasoning_mode, "rating_only")
                self.assertEqual(r.reasoning_applied, "omitted(off)")
                self.assertTrue(r.reasoning_standardized)
                self.assertEqual(r.backend, ANTHROPIC_BACKEND)
                self.assertEqual(r.temperature, self.spec.temperature)

    def test_ids_from_another_model_are_skipped_not_misattributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stray.jsonl")
            other = batch._instrument_and_cells(
                batch.select_specs(["GPT-5.6-Luna"])[0], load_config())
            stray = other[0].cell_key(other[1][0])
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"custom_id": stray, "result": {
                    "type": "succeeded", "message": {
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "A"}]}}}) + "\n")
            out = os.path.join(tmp, "welfare_api.jsonl")
            tally = batch.collect(path, self.ALIAS, out=out)
            self.assertEqual(tally["rows"], 0)
            self.assertEqual(tally["unmatched"], 1)


if __name__ == "__main__":
    unittest.main()
