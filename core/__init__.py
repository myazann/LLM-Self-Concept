"""Shared infrastructure for both instruments.

Nothing in this package knows the *design* of either study. It provides the
things a study is built out of:

    schema.py           the row written to JSONL, the cell key, resume
    config.py           the run-level knobs both instruments share
    battery.py          the 32-item bank (welfare attributes are derived from it)
    prompts.py          rendering primitives — options block, prompt hashing
    model_registry.py   which models exist and how each is administered
    models.py           the backend adapters
    obs.py              logging, status file, graceful shutdown
    engine.py           the run loop, driven by an Instrument
    stats.py            correlation / reliability maths
    report.py           Saver and console formatting for the report scripts

The design lives in `survey/` (what a model says it IS) and `welfare/` (what it
says it would PREFER TO BE). Neither imports the other; both import this.
"""
