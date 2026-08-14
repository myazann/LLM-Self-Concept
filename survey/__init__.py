"""The self-concept survey: what does a model report itself to BE?

The five-scale psychometric battery, administered one item per fresh context and
crossed over framing x instruction wording.

    python -m survey.run --plan          # cell counts, no calls
    python -m survey.run --dry-run       # offline, MockAdapter
    python -m survey.run                 # -> results.jsonl
    python -m survey.validity            # is the measurement any good?
    python -m survey.analysis            # what do the models report?

Design lives in `config/survey.yaml`. The welfare module is a separate
instrument in `welfare/`; the two share only `core/`.
"""
