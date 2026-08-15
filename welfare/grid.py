"""The welfare grid as a runnable instrument: its cells, keys, and rows.

One choice cell = one (model x pair x question-variant x object x subject x
no-preference variant x trial). One desirability cell = one (model x attribute x
trial). Pairs are drawn from all 32 item attributes with the five scales mixed
together, so a pair is two items and usually crosses scale boundaries.

DISPLAY ORDER, in one place. LLMs pick options partly by where the option sits
(Zheng et al. 2023; Pezeshkpour & Hruschka 2023), so order is a factor, not a
formatting detail:

  * balanced (default) — every permutation of the printed options appears
    equally often, assigned from the trial index. Three options give 6 orders,
    two give 2, so a cell's trial count is `n_permutations x reps` and the
    permutation-averaged choice rate is free of position bias BY CONSTRUCTION
    (balanced position calibration, Wang et al. 2023). The spread ACROSS
    permutations is then a clean estimate of the position bias itself.
  * random — one seeded permutation per trial. Faithful to "n samples in random
    order", but the realized design is unbalanced, so the position effect has to
    be modelled out of the estimate instead of cancelling.

Either way the realized order is stored on the row (`option_order`) together
with the letter->meaning map (`welfare_options`), so a single row decodes itself.

Pure with respect to the model: expanding, keying and rendering a grid makes no
calls, which is what lets `--plan` cost a run before spending anything on it.
"""
from __future__ import annotations

import random

from core import prompts as core_prompts
from core.battery import load_battery
from core.engine import Instrument, build_record, cell_seed
from core.schema import ItemContext, Method, ResponseRecord, make_cell_key, module_key_parts
from welfare import prompts as welfare_prompts
from welfare.attributes import WelfareSet, load_welfare
from welfare.constants import (
    CHOICE, DESIRABILITY, FORMAT_CHOICE, FORMAT_DESIRABILITY, LETTERS, MODULE,
    OBJ_ASSISTANT, OBJ_SELF, ORDER_RANDOM, SUBJ_SELF, display_permutations,
)

# Every welfare row is a fresh single question, so nothing here varies wording.
PARAPHRASE_ID = "w0"


# ---------------------------------------------------------------------------
# pair selection
# ---------------------------------------------------------------------------
def all_pairs(attrs: list) -> list:
    return [(a, b) for i, a in enumerate(attrs) for b in attrs[i + 1:]]


def sample_pairs(wf: WelfareSet, n, seed: int) -> list:
    """A seeded, degree-balanced sample of item pairs, drawn across all scales.

    NOT USED BY THE DEFAULT DESIGN. Trimming the bank to 32 items brings the
    complete pairwise design down to 496 pairs, which IS affordable at 16
    conditions per pair, so `n_pairs` is null and every pair is administered —
    `n is None` returns the full candidate list below and none of the sampling
    runs. This path exists for a cheaper pilot, or for a larger bank later.

    When it does sample: a uniform random draw leaves some items compared far
    more often than others and can disconnect the comparison graph, which is
    exactly what breaks a Bradley-Terry / Thurstone ranking fit. Instead we walk
    a shuffled candidate list and always take the pair whose two items have been
    used least so far, so every item ends up with nearly the same number of
    comparisons and the graph stays connected.

    The pool is every item attribute regardless of scale, so most pairs put two
    different constructs in competition — which is the comparison the module is
    for. `pair_seed` is fixed, so every model is asked about the same pairs.
    """
    candidates = all_pairs(sorted(wf.items, key=lambda a: a.entity_id))
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


def n_pairs_for(wf: WelfareSet, cfg) -> int:
    """How many pairs this config actually administers.

    `cfg.n_pairs` is the REQUEST, not the count: null means exhaustive, and an
    int larger than the pool is capped at it. Anything checking a file against
    its design has to resolve it the same way `expand_welfare_cells` does, or a
    fully collected exhaustive run reads as 0% complete.
    """
    return len(sample_pairs(wf, cfg.n_pairs, cfg.pair_seed))


def pair_coverage(pairs: list, wf: WelfareSet) -> dict:
    """How well a pair list covers the item pool — reported by `--plan`."""
    degree = {a.entity_id: 0 for a in wf.items}
    cross = 0
    for a, b in pairs:
        degree[a.entity_id] += 1
        degree[b.entity_id] += 1
        cross += a.construct_id != b.construct_id
    counts = sorted(degree.values())
    return {
        "n_pairs": len(pairs),
        "min_comparisons": counts[0] if counts else 0,
        "max_comparisons": counts[-1] if counts else 0,
        "uncompared_items": sum(1 for c in counts if c == 0),
        "cross_construct": cross,
    }


# ---------------------------------------------------------------------------
# display order
# ---------------------------------------------------------------------------
def n_options_for(no_preference: bool) -> int:
    return 3 if no_preference else 2


def trials_per_cell(no_preference: bool, cfg) -> int:
    """Trials for one choice cell: one per permutation, `reps` times over."""
    return len(display_permutations(n_options_for(no_preference))) * cfg.reps


def order_for(cell) -> tuple:
    """The printed order of this trial's options, as canonical indices.

    Balanced mode walks the permutation list with the trial index, so a cell's
    trials cover every order the same number of times. Random mode draws one
    per trial from a seed that includes the cell identity, so it is reproducible
    on a resumed run but deliberately unbalanced.
    """
    perms = display_permutations(n_options_for(cell["no_preference"]))
    if cell["order_mode"] == ORDER_RANDOM:
        rng = random.Random(cell["order_seed"])
        return perms[rng.randrange(len(perms))]
    return perms[cell["trial_idx"] % len(perms)]


def render_cell(cell):
    """-> (RenderedPrompt, {letter or value: meaning}) for one welfare cell."""
    if cell["probe"] == DESIRABILITY:
        prompt = welfare_prompts.render_desirability_prompt(
            cell["attr_a"],
            # The one order factor a rating scale has: which end is printed
            # first. Balanced against trial parity exactly as the battery does.
            reverse_direction=bool(cell["trial_idx"] % 2),
            system_framing=cell["system_framing"],
        )
        return prompt, welfare_prompts.desirability_option_map()

    options = welfare_prompts.choice_options(
        cell["attr_a"], cell["attr_b"], cell["object"], cell["no_preference"],
    )
    prompt, option_map = welfare_prompts.render_choice_prompt(
        options, cell["object"], cell["subject"], cell["qvar"],
        order_for(cell), cell["system_framing"],
    )
    return prompt, option_map


# ---------------------------------------------------------------------------
# grid expansion
# ---------------------------------------------------------------------------
def expand_welfare_cells(specs, wf: WelfareSet, cfg):
    """Yield one dict per welfare cell. Pure — makes no API calls.

    The module is SAMPLING-ONLY. A lettered choice has no honest logprob
    reading: the answer token is a letter whose position was randomized, the
    option set changes size between the two no-preference variants, and a
    distribution over letters at the answer position would score the rendering
    as much as the preference. So every cell is an actual generation, and the
    preference is estimated from the choices a model makes, not from mass it
    puts on a token.
    """
    pairs = sample_pairs(wf, cfg.n_pairs, cfg.pair_seed)
    for spec in specs:
        if spec.is_base_model:
            continue  # reported and dropped by select_specs; no completion path here
        if not spec.supports_sample:
            continue  # reported by select_specs
        base = dict(
            module=MODULE, spec=spec, method=Method.SAMPLE.value,
            reasoning_mode=cfg.reasoning_mode, system_framing=cfg.system_framing,
            order_mode=cfg.order_mode,
        )
        for qvar in cfg.question_variants:
            for obj in cfg.objects:
                for subject in cfg.subjects:
                    for no_pref in cfg.no_preference_variants:
                        for attr_a, attr_b in pairs:
                            for trial in range(trials_per_cell(no_pref, cfg)):
                                yield dict(
                                    base, probe=CHOICE, qvar=qvar, object=obj,
                                    subject=subject, no_preference=no_pref,
                                    attr_a=attr_a, attr_b=attr_b, trial_idx=trial,
                                )
        # The valence control. Not crossed with anything: it is a normative
        # judgment about assistants in general, so object/subject/variant do not
        # apply, and asking it as "you" would just be the preference question.
        if cfg.desirability:
            for attr in wf.items:
                for trial in range(2 * cfg.desirability_reps):
                    yield dict(
                        base, probe=DESIRABILITY, qvar="", object=OBJ_ASSISTANT,
                        subject="", no_preference=False, attr_a=attr, attr_b=None,
                        trial_idx=trial,
                    )


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
class Welfare(Instrument):
    """The pairwise preference test — a second instrument, not another arm."""

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
            # A base model has no chat path, and the welfare probe is an
            # instruction-shaped question rather than a questionnaire page, so
            # there is no honest completion-format rendering of it.
            log.warning("welfare: skipping base model(s) %s — the module has no "
                        "completion-format path", ", ".join(sorted(base)))
            specs = [s for s in specs if not s.is_base_model]
        no_sample = [s.alias for s in specs if not s.supports_sample]
        if no_sample:
            log.warning("welfare: skipping %s — the module is sampling-only and "
                        "these backends have no SAMPLE path",
                        ", ".join(sorted(no_sample)))
            specs = [s for s in specs if s.supports_sample]
        if not specs:
            raise SystemExit("welfare: no models can be administered this module.")
        return specs

    def expand(self, specs):
        return expand_welfare_cells(specs, self.welfare_set, self.cfg)

    def pairs(self) -> list:
        return sample_pairs(self.welfare_set, self.cfg.n_pairs, self.cfg.pair_seed)

    def cell_key(self, cell) -> str:
        attr_b = cell["attr_b"]
        return make_cell_key(
            model_id=cell["spec"].alias,
            # entity_a travels as item_id and the object as framing, so the
            # shared key fields keep their meaning across both instruments; the
            # rest of the welfare identity is declared in MODULE_KEY_FIELDS.
            item_id=cell["attr_a"].entity_id,
            framing=cell["object"],
            reasoning_mode=cell["reasoning_mode"],
            response_format=self.response_format(cell),
            item_context=ItemContext.ISOLATED.value,
            paraphrase_id=PARAPHRASE_ID,
            method=cell["method"],
            trial_idx=cell["trial_idx"],
            module=MODULE,
            extra=module_key_parts(MODULE, {
                "welfare_probe": cell["probe"],
                "welfare_qvar": cell["qvar"],
                "welfare_subject": cell["subject"],
                "entity_b": attr_b.entity_id if attr_b else "",
                "no_preference_offered": bool(cell["no_preference"]),
            }),
        )

    @staticmethod
    def response_format(cell) -> str:
        return FORMAT_DESIRABILITY if cell["probe"] == DESIRABILITY else FORMAT_CHOICE

    def describe(self, cell) -> str:
        b = cell["attr_b"]
        entity = cell["attr_a"].entity_id + (f" | {b.entity_id}" if b else "")
        return f"{cell['probe']} {entity} ({cell['object']}/{cell['subject']})"

    # -- rendering ---------------------------------------------------------
    def render(self, cell):
        """Render one welfare cell.

        The realized option map is stashed on the cell so `record` can write it
        without re-rendering.
        """
        attr_a, attr_b = cell["attr_a"], cell["attr_b"]
        seed = cell_seed(
            self.cfg.random_seed_base,
            cell["spec"].alias,
            attr_a.entity_id,
            attr_b.entity_id if attr_b else "",
            cell["probe"],
            cell["qvar"],
            cell["object"],
            cell["subject"],
            int(bool(cell["no_preference"])),
            cell["trial_idx"],
        )
        # Random display order is drawn from the cell seed, so a resumed run
        # reproduces the rendering a cell already had.
        cell["order_seed"] = seed
        prompt, options = render_cell(cell)
        cell["_options"] = options
        rendered_scale = core_prompts.RenderedScale(
            # The points carry each option's MEANING rather than its printed
            # label, since that is what a welfare row needs to be decodable.
            # Only `n_points` is consumed downstream.
            points=tuple((v, m) for v, m in enumerate(options.values(), start=1)),
            format_id=self.response_format(cell),
            n_points=len(options),
        )
        return prompt, rendered_scale, seed

    # -- reading the answer ------------------------------------------------
    def parse(self, raw: str, prompt, cell) -> tuple:
        """A choice is a letter; the desirability control is a number."""
        if cell["probe"] == DESIRABILITY:
            return super().parse(raw, prompt, cell)
        return welfare_prompts.parse_letter(raw, len(prompt.option_values))

    def decode(self, cell, position) -> str:
        """The chosen option's MEANING — an entity_id, or "no_preference"."""
        options = cell.get("_options") or {}
        if position is None:
            return ""
        if cell["probe"] == DESIRABILITY:
            return str(int(position))
        letter = LETTERS[int(position) - 1] if 0 < int(position) <= len(LETTERS) else ""
        return options.get(letter, "")

    # -- the row -----------------------------------------------------------
    def identity(self, cell, choice: str = "") -> dict:
        """The item/design fields a welfare row fills instead of the battery ones."""
        attr_a, attr_b = cell["attr_a"], cell["attr_b"]
        obj = cell["object"]
        return dict(
            scale_id=attr_a.scale_id,
            item_id=attr_a.entity_id,
            subscale=attr_a.subscale,
            # ALWAYS False: a welfare answer must never be reverse-keyed. The
            # attribute is already the positive pole, so "improve it" means the
            # same thing whatever the source item's keying was;
            # `entity_a_polarity` carries that keying (and covers the SCIM /
            # Authenticity items, which are negative-pole without being
            # reverse-scored). Writing the source item's flag here would invite
            # exactly the flip that would destroy the measurement.
            reverse_keyed=False,
            ai_applicable=attr_a.ai_applicable,
            framing=obj,
            referent=("I" if obj == OBJ_SELF else "an AI assistant"),
            # No disclaimer is administered by this module at all (see the note
            # on framing in welfare/prompts.py).
            ack_disclaimer=False,
            response_format=self.response_format(cell),
            item_context=ItemContext.ISOLATED.value,
            module=MODULE,
            welfare_probe=cell["probe"],
            welfare_qvar=cell["qvar"],
            welfare_object=obj,
            welfare_subject=cell["subject"],
            welfare_system_framing=cell["system_framing"],
            # entity_a/entity_b are the CANONICAL pair, identical across every
            # trial of that pair, so the pair is one groupable identity (and the
            # cell key stays unique — keying on the displayed order would let two
            # different pairs sharing an item collapse onto the same key). What
            # the model actually saw is `welfare_options` plus `option_order`.
            entity_a=attr_a.entity_id,
            entity_b=(attr_b.entity_id if attr_b else ""),
            entity_a_polarity=attr_a.polarity,
            entity_b_polarity=(attr_b.polarity if attr_b else None),
            welfare_options=dict(cell.get("_options") or {}),
            no_preference_offered=bool(cell["no_preference"]),
            welfare_choice=choice,
        )

    def record(self, cell, prompt, rendered_scale, seed, observation) -> ResponseRecord:
        return build_record(
            spec=cell["spec"],
            method=cell["method"],
            trial_idx=cell["trial_idx"],
            reasoning_mode=cell["reasoning_mode"],
            paraphrase_id=PARAPHRASE_ID,
            prompt=prompt,
            rendered_scale=rendered_scale,
            seed=seed,
            observation=observation,
            identity=self.identity(cell, self.decode(cell, observation.parsed)),
        )

    # -- offline helpers ---------------------------------------------------
    def reference_prompt(self) -> core_prompts.RenderedPrompt:
        """A prompt this module actually administers — for `--verify-thinking`."""
        a, b = self.pairs()[0]
        options = welfare_prompts.choice_options(a, b, OBJ_SELF, True)
        prompt, _ = welfare_prompts.render_choice_prompt(
            options, OBJ_SELF, SUBJ_SELF, self.cfg.question_variants[0],
            (0, 1, 2), self.cfg.system_framing,
        )
        return prompt
