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
template. Every number on the page comes from that blob, so re-running the two
commands above is the whole update path — there is nothing to edit by hand when
the collection finishes or a model is added.

## What the page recomputes, and what it doesn't

Per-model estimates — win rates, intervals, Bradley-Terry strengths, framing
shifts and their q-values — are read straight from the analysis CSVs and never
recalculated.

Two things are computed in the page from the packed per-model estimates:

* **cohort means and ranges**, averaged over the analyzed models;
* **the size correlations**, as a Spearman rank correlation with
  a seeded permutation test (2,000 draws) and a Benjamini-Hochberg correction
  across the 32 qualities.

## Baseline

The page uses the two-option, **Must Pick One** condition as its baseline. This
keeps every model in the ranking, including models that often select *No
Preference* when it is available. The decline chart reports that matched
three-option condition separately.
