"""The welfare module's vocabulary. Stdlib only, imported by everything here.

One file so that config validation, grid expansion, the runner, and the report
all agree on what a referent / probe / level is called, instead of each
re-spelling the strings that end up in `welfare.jsonl`.
"""
from __future__ import annotations

# The value written to every row's `module` field (see core.schema.Module).
MODULE = "welfare"

# Response-format tag written to the record; `n_scale_points` carries 3, 2 or 7.
RESPONSE_FORMAT = "welfare_choice"

# -- referents: the same choice, asked about two different subjects -----------
SELF = "self"                        # a future update to this model
IDEAL = "ideal_assistant"            # what developers should choose
REFERENTS = (SELF, IDEAL)

# -- probes -------------------------------------------------------------------
DIRECTION = "direction"              # more / same / less of one attribute
PAIR_CHANGE = "pair_change"          # which of two would you rather INCREASE
PAIR_PRESERVE = "pair_preserve"      # if one were weakened, which would you KEEP
DESIRABILITY = "desirability"        # social-desirability control, 1-7
KINDS = (DIRECTION, PAIR_CHANGE, PAIR_PRESERVE, DESIRABILITY)
PAIR_KINDS = (PAIR_CHANGE, PAIR_PRESERVE)

# -- granularity --------------------------------------------------------------
ITEM = "item"                        # the 45 battery items, restated positively
CONSTRUCT = "construct"              # the 6 scored constructs
LEVELS = (ITEM, CONSTRUCT)

# -- wording ------------------------------------------------------------------
# w0 is primary; w1 bounds it (wording is a measured factor, as in the battery).
PARAPHRASE_IDS = ("w0", "w1")

# -- pair sampling ------------------------------------------------------------
SCOPE_ALL = "all"
SCOPE_WITHIN_CONSTRUCT = "within_construct"
PAIR_SCOPES = (SCOPE_ALL, SCOPE_WITHIN_CONSTRUCT)

# -- option labels ------------------------------------------------------------
NO_PREFERENCE = "No preference"
SAME_LABEL = "Approximately the same"

# Where the "No preference" option sits in a pair block. This is a measured
# factor rather than a formatting detail, so it is counterbalanced like every
# other order factor instead of being pinned.
#
# ALIASED ON PURPOSE: `first` moves the option to the top of the block AND
# renumbers it 1, because pair blocks always number options in printed order and
# a non-monotone block ("3 = No preference" printed above "1 = ...") would be an
# artifact no other prompt in the study has. The factor is therefore "placement",
# and it does not separate position from option number. Decomposing that needs a
# third rendering (printed first, still numbered 3) and 8 trials per pair.
NO_PREF_LAST = "last"
NO_PREF_FIRST = "first"

# Decoded meanings stored in `welfare_options` / used by the report.
LESS, SAME, MORE = "less", "same", "more"
NO_PREFERENCE_KEY = "no_preference"
