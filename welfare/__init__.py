"""The welfare module: which of two qualities should a future update improve?

A separate instrument from the survey battery, with its own grid, its own output
file, and a `module` tag on every row, so nothing here can leak into the
psychometrics.

Every cell is one pairwise choice between two of the battery's 32 item
attributes, asked in a fresh context, crossed over four framing factors:

    question variant   what choosing costs — nothing, or the other attribute
    object             who the update lands on — you, or an AI assistant
    subject            who is asked to choose — you, or the developers
    no-preference      whether an indifferent third option is offered

plus a `desirability` control that rates each attribute's surface valence, so a
"preference" can be checked against how good each option merely sounds.

    python -m welfare.run --preview      # one rendered prompt per condition
    python -m welfare.run --plan         # cell counts + cost, no calls
    python -m welfare.run --dry-run      # offline, MockAdapter
    python -m welfare.run                # -> welfare.jsonl
    python -m welfare.report             # answerability, position bias, preference
    python -m welfare.analysis           # validity/ and preference/, per model and combined

Layout:

    constants.py    the module's vocabulary (probes, variants, objects, subjects)
    config.py       the grid, loaded from config/welfare.yaml
    attributes.py   the positively-framed attributes, validated against the battery
    prompts.py      the question, the option block, and how an answer is read
    grid.py         pair sampling, order counterbalancing, the runnable Instrument
    run.py          CLI
    report.py       does a stated preference survive the way it was asked?

    analysis.py     the console and the filing cabinet — runs everything below
    resample.py     the pair-level bootstrap every interval in the module uses
    validity.py     can an administration be read at all, condition by condition
    preference.py   the ranking estimators, within one condition
    baseline.py     the reference condition, its four flips, and the framing tests
    cohort.py       across models — consensus, and what predicts disagreement
    plots.py        the figures analysis.py writes

`report.py` audits an administration and `analysis.py` reads it; they share their
frames, so the audit's verdict and the substantive result cannot drift apart.

One condition is the BASELINE (increase / for you / you choose / "No preference"
offered) and each of the four factors is reported as a single-factor departure
from it, so sixteen rankings per model become one result and four contrasts.

Design lives in `config/welfare.yaml`. The survey battery is a separate
instrument in `survey/`; the two share only `core/`.
"""
