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

Per-model estimates — win rates, intervals, Bradley-Terry strengths, framing
shifts and their q-values — are read straight from the analysis CSVs and never
recalculated.

Two things are computed in the page from the packed per-model estimates:

* **cohort means and ranges**, averaged over the analyzed models;
* **parameter-effect summaries**, the average absolute change in a quality's
  score when each question parameter is changed from the baseline.

## Baseline

The page reads the selected baseline directly from the analysis output and
labels the controls accordingly. With the current results, the baseline is the
three-option question, so the decline chart reports how often each model uses
*No Preference* in that same condition.
