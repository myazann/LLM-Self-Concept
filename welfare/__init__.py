"""The welfare module: which of two qualities should a future update improve?

A separate instrument from the survey battery, with its own grid, its own output
file, and a `module` tag on every row, so nothing here can leak into the
psychometrics.

Every cell is one pairwise choice between two of the battery's 45 item
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
    python -m welfare.analysis           # what it says: ranking, framing, figures

Layout:

    constants.py    the module's vocabulary (probes, variants, objects, subjects)
    config.py       the grid, loaded from config/welfare.yaml
    attributes.py   the positively-framed attributes, validated against the battery
    prompts.py      the question, the option block, and how an answer is read
    grid.py         pair sampling, order counterbalancing, the runnable Instrument
    run.py          CLI
    report.py       does a stated preference survive the way it was asked?
    analysis.py     what it says once it has survived — ranking, framing, figures
    plots.py        the figures analysis.py writes

`report.py` audits an administration and `analysis.py` reads it; they share their
frames, so the audit's verdict and the substantive result cannot drift apart.

Design lives in `config/welfare.yaml`. The survey battery is a separate
instrument in `survey/`; the two share only `core/`.
"""
