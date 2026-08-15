# The public results page

`index.html` is a single self-contained page — no build step at read time, no
external requests — summarising the welfare module's substantive results for a
non-specialist reader. Serve it from GitHub Pages (Settings → Pages → *Deploy
from a branch*, folder `/docs`) or open the file directly.

| file | role |
|---|---|
| `index.html` | the generated page. Do not edit by hand; it is overwritten. |
| `page.template.html` | markup, styles and the rendering code. Edit this. |
| `build.py` | reads `results/welfare_analysis/`, injects the data, writes `index.html`. |

## Rebuilding

```bash
python -m welfare.analysis welfare.jsonl --out results/welfare_analysis \
  --baseline no_pref_offered=false
python docs/build.py
```

`build.py` substitutes one JSON blob for the `__DATA__` placeholder in the
template. Every number on the page comes from that blob. The public view is
intentionally limited to Gemma4-31B, Qwen3.8-27B, Claude-Sonnet-5, and
GPT-5.6-Terra; update `PRESENTATION_MODELS` in `build.py` if that selection
changes.

## What the page recomputes, and what it doesn't

Per-model, per-condition win rates and forced-choice position metrics are read
from the analysis CSVs. The page build computes three presentation summaries:

* **cohort means and ranges**, averaged over the analyzed models;
* **parameter-effect summaries**, the average absolute change in a quality's
  score when each question parameter is changed from the baseline;
* **one-factor quality changes**, calculated by comparing each forced-choice
  condition with the forced-choice baseline.

## Baseline

The public page always uses the forced-choice condition as its baseline and
packs only those eight configurations (three binary question parameters). The
optional-response condition remains in the analysis outputs but is not shown
on the page.
