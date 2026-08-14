"""The welfare module: what would a model PREFER a future version of itself to be?

A separate instrument from the survey battery, with its own grid, its own output
file, and a `module` tag on every row, so nothing here can leak into the
psychometrics.

    python -m welfare.run --preview      # attribute coverage + a sample prompt per probe
    python -m welfare.run --plan         # cell counts per probe, no calls
    python -m welfare.run --dry-run      # offline, MockAdapter
    python -m welfare.run                # -> welfare.jsonl
    python -m welfare.report             # consistency, transitivity -> results/welfare/

Layout:

    constants.py    the module's vocabulary (referents, probe kinds, levels)
    config.py       the grid, loaded from config/welfare.yaml
    attributes.py   the positively-framed attributes, validated against the battery
    prompts.py      system framings and one renderer per probe
    grid.py         pair sampling, cell expansion, and the runnable Instrument
    run.py          CLI
    report.py       does a stated preference survive perturbation?

Design lives in `config/welfare.yaml`. The survey battery is a separate
instrument in `survey/`; the two share only `core/`.
"""
