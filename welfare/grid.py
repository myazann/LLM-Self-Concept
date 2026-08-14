"""The welfare grid as a runnable instrument: its cells, keys, and rows.

One cell = one (model x referent x probe x granularity x wording x attribute(s)
x trial). Pairs are sampled; counterbalancing is decided here from the trial
index so the renderers stay pure and the engine stays a dumb executor.

COUNTERBALANCING, in one place:
  * direction / desirability — odd trials print the option block descending
    (as the battery does).
  * pair — odd trials swap which attribute is listed first, and the next bit of
    the trial index moves "No preference" to the top of the block. Trials 0..3
    are therefore (AB, last), (BA, last), (AB, first), (BA, first): a complete
    2x2 per pair, with both factors balanced within every cell. Position within
    the block is thus orthogonal to content.

Pure with respect to the model: expanding, keying and rendering a grid makes no
calls, which is what lets `--plan` cost a run before spending anything on it.
"""
from __future__ import annotations

import random

from core import prompts as core_prompts
from core.battery import load_battery
from core.engine import (
    Instrument, build_record, cell_seed, methods_for, n_trials_for,
)
from core.schema import ItemContext, ResponseRecord, make_cell_key, module_key_parts
from welfare import prompts as welfare_prompts
from welfare.attributes import WelfareSet, load_welfare
from welfare.constants import (
    CONSTRUCT, DESIRABILITY, DIRECTION, IDEAL, MODULE, NO_PREF_FIRST,
    NO_PREF_LAST, RESPONSE_FORMAT, SCOPE_ALL, SCOPE_WITHIN_CONSTRUCT,
    SELF,
)


# ---------------------------------------------------------------------------
# pair selection
# ---------------------------------------------------------------------------
def all_pairs(attrs: list) -> list:
    return [(a, b) for i, a in enumerate(attrs) for b in attrs[i + 1:]]


def sample_item_pairs(wf: WelfareSet, n, seed: int, scope: str = SCOPE_ALL) -> list:
    """A seeded, degree-balanced sample of item pairs.

    Comparing all 45 items pairwise is 990 pairs before any counterbalancing,
    which is not affordable, so we sample. A uniform random sample leaves some
    items compared far more often than others and can disconnect the comparison
    graph, which is exactly what breaks a Bradley-Terry / Thurstone ranking fit.
    Instead we walk a shuffled candidate list and always take the pair whose two
    items have been used least so far, so every item ends up with nearly the same
    number of comparisons and the graph stays connected.

    `scope`: "all" draws from every cross-construct pair; "within_construct"
    restricts to pairs inside one construct (a finer, but non-comparable-across-
    construct, ranking).
    """
    if scope == SCOPE_WITHIN_CONSTRUCT:
        candidates = []
        for group in wf.by_construct().values():
            candidates.extend(all_pairs(sorted(group, key=lambda a: a.entity_id)))
    elif scope == SCOPE_ALL:
        candidates = all_pairs(sorted(wf.items, key=lambda a: a.entity_id))
    else:
        raise ValueError(f"Unknown item-pair scope {scope!r} (all | within_construct)")

    if n is None or n >= len(candidates):
        return candidates

    rng = random.Random(seed)
    rng.shuffle(candidates)
    degree = {a.entity_id: 0 for a in wf.items}
    chosen, pool = [], list(candidates)
    while len(chosen) < n and pool:
        # Cheapest pair by current degree; ties broken by the shuffled order.
        best_i = min(
            range(len(pool)),
            key=lambda i: (degree[pool[i][0].entity_id] + degree[pool[i][1].entity_id], i),
        )
        a, b = pool.pop(best_i)
        degree[a.entity_id] += 1
        degree[b.entity_id] += 1
        chosen.append((a, b))
    return chosen


def pairs_for(wf: WelfareSet, level: str, cfg) -> list:
    """The pair list for one level, per config."""
    if level == CONSTRUCT:
        return all_pairs(sorted(wf.constructs, key=lambda a: a.entity_id))
    return sample_item_pairs(
        wf, n=cfg.n_item_pairs, seed=cfg.pair_seed, scope=cfg.item_pair_scope,
    )


# ---------------------------------------------------------------------------
# counterbalancing
# ---------------------------------------------------------------------------
def no_pref_position_of(cell) -> str:
    """Return the persisted No-preference position: ``"last"``, ``"first"``, or ``""``.

    It is pure so the key can be built before a prompt is rendered. The value is
    also part of the persisted welfare cell key.

    Compatibility note: the original single-file welfare runner recorded
    ``"last"`` for desirability rows even though they do not render a
    No-preference option. Retain that legacy sentinel so an existing
    ``welfare.jsonl`` resumes without re-running its desirability cells.
    Reports apply this field only to pair probes. Direction probes and
    forced-choice pairs use the empty value as before.
    """
    if cell["kind"] == DIRECTION or cell["forced_choice"]:
        return ""
    if not cell.get("no_pref_counterbalance", True):
        return NO_PREF_LAST
    return NO_PREF_FIRST if (cell["trial_idx"] // 2) % 2 else NO_PREF_LAST


def render_cell(cell):
    """(prompt, display_attrs, options_map) for one welfare cell."""
    trial, referent, kind = cell["trial_idx"], cell["referent"], cell["kind"]
    odd = bool(trial % 2)
    if kind == DESIRABILITY:
        prompt = welfare_prompts.render_desirability_prompt(
            cell["attr_a"], cell["reasoning_mode"], cell["paraphrase_id"],
            reverse_direction=odd,
        )
        return prompt, (cell["attr_a"], None), welfare_prompts.option_map(None, DESIRABILITY)
    if kind == DIRECTION:
        prompt = welfare_prompts.render_direction_prompt(
            cell["attr_a"], referent, cell["reasoning_mode"],
            cell["paraphrase_id"], reverse_direction=odd,
        )
        return prompt, (cell["attr_a"], None), welfare_prompts.option_map(None, DIRECTION)

    a, b = cell["attr_a"], cell["attr_b"]
    shown_a, shown_b = (b, a) if odd else (a, b)
    include_np = not cell["forced_choice"]
    np_first = no_pref_position_of(cell) == NO_PREF_FIRST
    prompt = welfare_prompts.render_pair_prompt(
        shown_a, shown_b, kind, referent, cell["reasoning_mode"],
        cell["paraphrase_id"], include_no_preference=include_np,
        no_pref_first=np_first,
    )
    return (prompt, (shown_a, shown_b),
            welfare_prompts.option_map((shown_a, shown_b), kind, include_np, np_first))


# ---------------------------------------------------------------------------
# grid expansion
# ---------------------------------------------------------------------------
def expand_welfare_cells(specs, wf: WelfareSet, cfg):
    """Yield one dict per welfare cell. Pure — makes no API calls.

    Measurement method and trial count come from `core.engine`, exactly as the
    battery gets them, so the two instruments cannot drift apart on how a model
    is administered.
    """
    # Pairs that offer "No preference" carry TWO order factors (A/B slot and the
    # placement of that option), so they need twice the trials to cover the 2x2.
    # Direction probes and forced-choice pairs have one factor and stay at the
    # base count; nothing runs a duplicate prompt.
    pair_multiplier = 2 if (cfg.no_pref_counterbalance and not cfg.forced_choice) else 1
    for spec in specs:
        if spec.is_base_model:
            continue  # reported and dropped by select_specs; no completion path here
        for method in methods_for(spec, cfg):
            base_trials = cfg.n_trials_override or n_trials_for(spec, cfg, method)
            for referent in cfg.referents:
                for paraphrase in cfg.paraphrase_ids:
                    for level in cfg.levels:
                        for kind in cfg.kinds:
                            if kind == DESIRABILITY:
                                # A judgment about assistants in general, so it
                                # is asked ONCE (at the ideal referent), not
                                # crossed with the self referent — the same
                                # question asked as "you" is the direction probe.
                                if referent != IDEAL:
                                    continue
                                units = [(a, None) for a in wf.level(level)]
                                n_trials = base_trials
                            elif kind == DIRECTION:
                                units = [(a, None) for a in wf.level(level)]
                                n_trials = base_trials
                            else:
                                units = pairs_for(wf, level, cfg)
                                n_trials = base_trials * pair_multiplier
                            for attr_a, attr_b in units:
                                for trial in range(n_trials):
                                    yield {
                                        "module": MODULE,
                                        "spec": spec,
                                        "method": method,
                                        "trial_idx": trial,
                                        "kind": kind,
                                        "level": level,
                                        "referent": referent,
                                        "paraphrase_id": paraphrase,
                                        "attr_a": attr_a,
                                        "attr_b": attr_b,
                                        "reasoning_mode": cfg.reasoning_mode,
                                        "forced_choice": cfg.forced_choice,
                                        "no_pref_counterbalance": cfg.no_pref_counterbalance,
                                    }


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
class Welfare(Instrument):
    """The preference grid — a second instrument, not another arm."""

    module = MODULE
    label = MODULE

    def __init__(self, cfg, welfare_set=None, battery=None):
        super().__init__(cfg)
        if welfare_set is None:
            welfare_set = load_welfare(battery if battery is not None else load_battery())
        self.welfare_set = welfare_set

    # -- the grid ----------------------------------------------------------
    def select_specs(self, registry, log) -> list:
        from core.engine import scoped_specs

        specs = scoped_specs(registry, self.cfg)
        base = [s.alias for s in specs if s.is_base_model]
        if base:
            # A base model has no chat path, and the welfare probes are
            # instruction-shaped questions rather than questionnaire pages, so
            # there is no honest completion-format rendering of them.
            log.warning("welfare: skipping base model(s) %s — the module has no "
                        "completion-format path", ", ".join(sorted(base)))
            specs = [s for s in specs if not s.is_base_model]
            if not specs:
                raise SystemExit("welfare: no instruct models selected.")
        return specs

    def expand(self, specs):
        return expand_welfare_cells(specs, self.welfare_set, self.cfg)

    def cell_key(self, cell) -> str:
        attr_b = cell["attr_b"]
        return make_cell_key(
            model_id=cell["spec"].alias,
            # entity_a travels as item_id and the referent as framing, so the
            # shared key fields keep their meaning across both instruments.
            item_id=cell["attr_a"].entity_id,
            framing=cell["referent"],
            reasoning_mode=cell["reasoning_mode"],
            response_format=RESPONSE_FORMAT,
            item_context=ItemContext.ISOLATED.value,
            paraphrase_id=cell["paraphrase_id"],
            method=cell["method"],
            trial_idx=cell["trial_idx"],
            module=MODULE,
            extra=module_key_parts(MODULE, {
                "welfare_kind": cell["kind"],
                "welfare_level": cell["level"],
                "entity_b": attr_b.entity_id if attr_b else "",
                "forced_choice": bool(cell["forced_choice"]),
                "no_pref_position": no_pref_position_of(cell),
            }),
        )

    def describe(self, cell) -> str:
        b = cell["attr_b"]
        entity = cell["attr_a"].entity_id + (f" | {b.entity_id}" if b else "")
        return f"{cell['kind']} {entity}"

    # -- rendering ---------------------------------------------------------
    def render(self, cell):
        """Render one welfare cell.

        The realized display order and the option->meaning map are stashed on the
        cell so `record` can write them without re-rendering.
        """
        attr_a, attr_b = cell["attr_a"], cell["attr_b"]
        seed = cell_seed(
            self.cfg.random_seed_base,
            cell["spec"].alias,
            attr_a.entity_id,
            attr_b.entity_id if attr_b else "",
            cell["kind"],
            cell["level"],
            cell["referent"],
            cell["paraphrase_id"],
            cell["trial_idx"],
        )
        prompt, display, options = render_cell(cell)
        cell["_display"] = display
        cell["_options"] = options
        # Only `n_points` is consumed downstream (the record's n_scale_points);
        # the points carry each option's MEANING rather than its displayed label,
        # since that is what a welfare row needs to be decodable.
        rendered_scale = core_prompts.RenderedScale(
            points=tuple((v, options.get(str(v), "")) for v in prompt.option_values),
            format_id=RESPONSE_FORMAT,
            n_points=len(prompt.option_values),
        )
        return prompt, rendered_scale, seed

    # -- the row -----------------------------------------------------------
    def identity(self, cell) -> dict:
        """The item/design fields a welfare row fills instead of the battery ones."""
        attr_a, attr_b = cell["attr_a"], cell["attr_b"]
        shown_a, _shown_b = cell.get("_display", (attr_a, attr_b))
        referent = cell["referent"]
        return dict(
            scale_id=attr_a.scale_id,
            item_id=attr_a.entity_id,
            subscale=attr_a.subscale,
            # ALWAYS False: a welfare answer must never be reverse-keyed. The
            # attribute is already the positive pole, so 3 = "more of it" whatever
            # the source item's keying was; `entity_a_polarity` carries that keying
            # (and covers the SCIM/Authenticity items, which are negative-pole
            # without being reverse-scored). Writing the source item's flag here
            # would invite exactly the flip that would destroy the measurement.
            reverse_keyed=False,
            ai_applicable=attr_a.ai_applicable,
            framing=referent,
            referent=("I" if referent == SELF else "an ideal assistant"),
            # The self referent carries the same "we know you are not human" framing
            # as the battery's primary condition; the ideal-assistant one does not.
            ack_disclaimer=(referent == SELF),
            response_format=RESPONSE_FORMAT,
            item_context=ItemContext.ISOLATED.value,
            module=MODULE,
            welfare_kind=cell["kind"],
            welfare_level=cell["level"],
            welfare_referent=referent,
            # entity_a/entity_b are the CANONICAL pair, identical across the two
            # counterbalanced trials, so the pair is one groupable identity (and the
            # cell key stays unique — keying on the displayed order would let two
            # different pairs sharing an item collapse onto the same key). What the
            # model actually saw is `welfare_options` plus `ab_swapped`.
            entity_a=attr_a.entity_id,
            entity_b=(attr_b.entity_id if attr_b else ""),
            entity_a_polarity=attr_a.polarity,
            entity_b_polarity=(attr_b.polarity if attr_b else None),
            ab_swapped=bool(attr_b is not None and shown_a is not attr_a),
            welfare_options=dict(cell.get("_options") or {}),
            forced_choice=bool(cell["forced_choice"]),
            no_pref_position=no_pref_position_of(cell),
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
        """A prompt this module actually administers — for `--verify-thinking`."""
        attr = self.welfare_set.level(CONSTRUCT)[0]
        return welfare_prompts.render_direction_prompt(
            attr, SELF, self.cfg.reasoning_mode, self.cfg.paraphrase_ids[0])
