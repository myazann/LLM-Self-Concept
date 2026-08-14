"""The survey battery as a runnable instrument: its cells, keys, and rows.

One cell = one (model x item x framing x reasoning x format x context x
instruction x trial). The main run fully crosses framing x instruction and pins
everything else; a robustness arm sweeps one pinned factor at the primary
reference cell.

Pure with respect to the model: expanding, keying and rendering a grid makes no
calls, which is what lets `--plan` cost a run before spending anything on it.
"""
from __future__ import annotations

import itertools

from core import prompts as core_prompts
from core.battery import load_battery
from core.engine import (
    Instrument, build_record, cell_seed, methods_for, n_trials_for,
)
from core.schema import Framing, ItemContext, Method, Module, ResponseRecord, make_cell_key
from survey import prompts as survey_prompts


def expand_cells(specs, battery, cfg, arm_name=None):
    """Yield one dict per design cell. Pure — makes no API calls."""
    levels = cfg.levels_for(arm_name)
    items = battery.items(list(cfg.scales) if cfg.scales else None)

    for spec in specs:
        for method in methods_for(spec, cfg):
            if spec.is_base_model and method == Method.SAMPLE.value:
                continue  # base models are administered by logprob only
            n_trials = n_trials_for(spec, cfg, method)
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


class Battery(Instrument):
    """The five-scale self-report battery."""

    module = Module.BATTERY.value

    def __init__(self, cfg, arm_name=None, battery=None):
        super().__init__(cfg)
        self.arm_name = arm_name
        self.battery = battery if battery is not None else load_battery()

    @property
    def label(self) -> str:
        return self.arm_name or "primary"

    def config_hash(self) -> str:
        from core import obs
        return obs.config_hash(self.cfg, self.arm_name)

    # -- the grid ----------------------------------------------------------
    def expand(self, specs):
        return expand_cells(specs, self.battery, self.cfg, self.arm_name)

    def cell_key(self, cell) -> str:
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

    def describe(self, cell) -> str:
        return f"item {cell['item'].item_id}"

    # -- rendering ---------------------------------------------------------
    def render(self, cell):
        cfg = self.cfg
        spec, scale, item = cell["spec"], cell["scale"], cell["item"]
        rendered_scale = survey_prompts.render_scale(
            scale,
            cell["framing"],
            cell["response_format"],
            cfg.harmonized_points,
            cfg.include_midpoint,
        )
        seed = cell_seed(
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
            prompt = survey_prompts.render_base_prompt(
                scale, item, cell["framing"], rendered_scale, cell["paraphrase_id"],
                seed, reverse_direction,
            )
        elif cell["item_context"] == ItemContext.FULL_BATTERY.value:
            # NOT WIRED UP. survey.prompts.render_battery_prompt() produces a
            # correct whole-scale prompt, but the response is one rating PER ITEM
            # and the engine writes one record per cell with a single parsed
            # rating. Running it as-is would silently record the first number in a
            # multi-line reply against every item of the scale. Needs a multi-item
            # parse + fan-out write path first (see README "Not implemented").
            raise NotImplementedError(
                "The full_battery arm needs a multi-item response parser before it "
                "can be run — see README. Use --arm reasoning/response_format for now."
            )
        else:
            prompt = survey_prompts.render_item_prompt(
                scale, item, cell["framing"], cell["reasoning_mode"], rendered_scale,
                cell["paraphrase_id"], seed, reverse_direction,
            )
        return prompt, rendered_scale, seed

    # -- the row -----------------------------------------------------------
    def identity(self, cell) -> dict:
        """Who this row is about, in battery terms."""
        scale, item = cell["scale"], cell["item"]
        return dict(
            scale_id=scale.scale_id,
            item_id=item.item_id,
            subscale=item.subscale,
            reverse_keyed=item.reverse_scored,
            ai_applicable=item.ai_applicable,
            framing=cell["framing"],
            referent=survey_prompts.REFERENT[cell["framing"]],
            ack_disclaimer=(cell["framing"] == Framing.FIRST_PERSON_ACK.value),
            response_format=cell["response_format"],
            item_context=cell["item_context"],
        )

    def record(self, cell, prompt, rendered_scale, seed, observation) -> ResponseRecord:
        return build_record(
            spec=cell["spec"],
            method=cell["method"],
            trial_idx=cell["trial_idx"],
            reasoning_mode=cell["reasoning_mode"],
            paraphrase_id=cell["paraphrase_id"],
            prompt=prompt,
            rendered_scale=rendered_scale,
            seed=seed,
            observation=observation,
            identity=self.identity(cell),
        )

    # -- offline helpers ---------------------------------------------------
    def reference_prompt(self) -> core_prompts.RenderedPrompt:
        """A prompt from the primary cell — used by `--verify-thinking`."""
        cfg = self.cfg
        scale, item = self.battery.items()[0]
        rendered = survey_prompts.render_scale(
            scale, cfg.framing, cfg.response_format,
            cfg.harmonized_points, cfg.include_midpoint)
        return survey_prompts.render_item_prompt(
            scale, item, cfg.framing, cfg.reasoning_mode, rendered,
            cfg.paraphrase_id, 0)
