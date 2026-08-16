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
python -m welfare.analysis --out results/welfare_analysis        # any baseline
python -m welfare.analysis --out results/welfare_analysis_forced \
  --baseline no_pref_offered=false \
  --models Gemma4-31B Qwen3.8-27B Claude-Sonnet-5 GPT-5.6-Terra
python docs/build.py
```

`build.py` substitutes one JSON blob for the `__DATA__` placeholder in the
template. Every number on the page comes from that blob. The public view is
intentionally limited to Gemma4-31B, Qwen3.8-27B, Claude-Sonnet-5, and
GPT-5.6-Terra; update `PRESENTATION_MODELS` in `build.py` if that selection
changes.

Two runs, because the page reads two different kinds of number. Rankings are
read per condition, so the first run's own baseline does not matter. The
significance marks are one-factor tests *against a baseline*, so they are only
meaningful from a run whose baseline is the page's forced-choice condition —
`build.py` checks each candidate directory's recorded baseline and fails with
the command above rather than marking a contrast it did not test. A single run
at `--baseline no_pref_offered=false` also satisfies both; `SHIFT_DIRS` prefers
`results/welfare_analysis_forced` and falls back to `results/welfare_analysis`
when that one is already at the forced-choice baseline.

## Model colors

Blue (`--s1`) is the cohort average. Each model owns one hue — `--m1`…`--m4`,
slots 2/3/5/6 of the categorical palette — assigned by its position in
`PRESENTATION_MODELS`, so a color belongs to a model rather than to its rank.

**The ranking chart shows one model hue at a time on purpose.** Five differently
colored marks on one row cannot be told apart from this palette: the best
five-way set reaches OKLab ΔE 11.9, under the floor of 15, and the four-hue sets
that clear it put a model within ΔE 1.9 of the blue average under simulated
color-blindness. So the cohort view keeps its dots neutral and colors only the
model being pointed at, and the per-model view draws a single ball. Red is also
excluded: `--neg` / `--warn` already mean "down" and "problem" elsewhere.

Changing these, or adding a fifth model, means re-running the validator rather
than picking a nice-looking hex — `build.py` raises instead of wrapping around
the four:

```bash
python .../dataviz/scripts/validate_palette.py \
  "#2a78d6,#eb6834,#1baf7a,#e87ba4,#008300" --mode light \
  --surface "#fbfbfc" --pairs all      # then --mode dark --surface "#14171b"
```

The prompt card is generated from `welfare.prompts` and
`config/scales/welfare_attributes.json` — the question template, the wording
each parameter level contributes, and the two example options are all read out
of the instrument, so the card cannot drift from what was administered.

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
