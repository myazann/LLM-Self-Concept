"""Shared infrastructure for the study's instruments.

Nothing in this package knows the *design* of the study. It provides the things
a study is built out of:

    schema.py           the row written to JSONL, the cell key, resume
    config.py           the run-level knobs any instrument shares
    battery.py          the 32-item bank (welfare attributes are derived from it)
    prompts.py          rendering primitives — options block, prompt hashing
    model_registry.py   which models exist and how each is administered
    models.py           the backend adapters
    obs.py              logging, status file, graceful shutdown
    engine.py           the run loop, driven by an Instrument
    report.py           Saver and console formatting for the report scripts

The design lives in `welfare/` — what a model says it would PREFER TO BE — and
imports this package; nothing here imports back.
"""
