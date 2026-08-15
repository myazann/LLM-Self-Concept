"""Batch-API request files for the welfare grid, and the results path back.

The welfare grid is 31,744 single-token calls per model. Issued one at a time —
which is what `welfare.run` does, and the only thing it can do against a local
GGUF — that is ~10 hours of wall clock per API model at full sync pricing. Both
closed-source providers take the whole grid in one submission at half price, so
for the API cohort the natural unit of work is a file, not a loop.

This module is the file half of that trade, and it is deliberately OFFLINE at
both ends: `build` writes request files without a network call or an API key,
`collect` reads a results file off disk. Submitting and polling — the only steps
that need a credential — stay outside the repository, in the caller's hands.

Nothing about the DESIGN changes. Cells come from the same
`welfare.grid.Welfare` instrument the local runner drives, rendered by the same
code, in the same counterbalanced order. What makes the round trip safe is that
every request carries its CELL KEY as `custom_id`: expansion and rendering are
pure, so `collect` rebuilds the grid offline, matches each returned answer to
the cell that produced it, and writes rows in the same schema `welfare.run`
writes — same resume semantics, same report and analysis path.

    python -m welfare.batch build --models GPT-5.6-Luna Claude-Haiku-4.5
      -> batches/GPT-5.6-Luna.openai.batch.jsonl          (+ .meta.json)
      -> batches/Claude-Haiku-4.5.anthropic.batch.jsonl   (+ .meta.json)

    python -m welfare.batch submit-help          # the SDK calls, per provider
    python -m welfare.batch collect <results>.jsonl --model GPT-5.6-Luna
      -> welfare_api.jsonl, appended

REASONING IS OFF on every request written here, asserted per provider rather
than left to a default: `thinking={"type":"disabled"}` on Claude 4.6/5 (with
effort pinned low, the only level at which Claude 5 accepts it), the parameter
omitted on the 4.5 family where omitting IS off, and `reasoning_effort="none"`
on GPT-5.6. Each row records what was asserted, so a run where a provider
ignored the request is visible in the data rather than folded into it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import replace
from pathlib import Path

from core import models as core_models
from core.battery import load_battery
from core.engine import Observation
from core.model_registry import ANTHROPIC_BACKEND, OPENAI_BACKEND, load_registry
from core.schema import completed_cells, write_jsonl
from welfare.config import load_config
from welfare.grid import Welfare

#: Where `build` writes, and where `collect` looks for the sidecar metadata.
DEFAULT_OUT_DIR = "batches"
#: `collect` appends here. Deliberately NOT welfare.jsonl: the local sweep may
#: still be writing that file, and keeping the cohorts in separate files makes
#: "which rows came from an API?" a question about paths rather than about
#: filtering. Analysis takes them concatenated.
DEFAULT_COLLECT_OUT = "welfare_api.jsonl"

# Per-batch caps, from each provider's documented limits — "whichever is reached
# first", so both are enforced when packing shards. The welfare grid fits in one
# file for every model here (~32k requests, ~20 MB), but raising `order.reps` in
# config/welfare.yaml would blow past the request cap, so sharding is applied
# rather than assumed away.
MAX_REQUESTS = {ANTHROPIC_BACKEND: 100_000, OPENAI_BACKEND: 50_000}
MAX_BYTES = {ANTHROPIC_BACKEND: 256_000_000, OPENAI_BACKEND: 200_000_000}
ENDPOINT = {OPENAI_BACKEND: "/v1/chat/completions"}

# Sampling parameters are rejected by the newest reasoning models — Claude 5
# removed temperature/top_p/top_k outright, and the GPT-5.x line accepts only
# its default — so on those we send nothing and record the provider default
# instead. A row must never name a knob that was not in play; this is the same
# rule `core.models.LOGPROB_TEMPERATURE` enforces on the logprob path.
PROVIDER_DEFAULT_TEMPERATURE = 1.0
_NO_SAMPLING_GENERATIONS = frozenset({"claude-5"})


def _sends_temperature(spec) -> bool:
    if spec.backend == OPENAI_BACKEND:
        return False  # every GPT-5.x entry in the registry is a reasoning model
    return spec.generation not in _NO_SAMPLING_GENERATIONS


def _reasoning_plan(spec):
    """The adapter's own off-state plan, without constructing a client.

    `reasoning_plan` reads nothing but `self.spec`, so bypassing `__init__`
    skips the SDK import and the API-key check while keeping the request bodies
    and the recorded `reasoning_applied` sourced from the adapter rather than
    re-derived here — the two cannot drift.
    """
    cls = core_models.ADAPTERS[spec.backend]
    adapter = object.__new__(cls)
    adapter.spec = spec
    return cls.reasoning_plan(adapter, want_thinking=False)


def _prepared_spec(spec):
    """The spec as the requests will actually realize it (see the temperature note)."""
    if _sends_temperature(spec):
        return spec
    return replace(spec, temperature=PROVIDER_DEFAULT_TEMPERATURE)


def select_specs(aliases):
    """API-backed models by alias, bypassing config/welfare.yaml's `scope`.

    The welfare scope is open-weights by design and widening it would re-scope
    the local sweep too. This module is the closed-source path, so the model
    list is an explicit argument rather than a config edit.
    """
    registry = load_registry()
    specs = {s.alias: s for s in registry.select(
        backends=[OPENAI_BACKEND, ANTHROPIC_BACKEND], kinds=["instruct"])}
    unknown = [a for a in aliases if a not in specs]
    if unknown:
        raise SystemExit(
            f"Not an enabled API model: {', '.join(unknown)}\n"
            f"Available: {', '.join(sorted(specs))}")
    return [specs[a] for a in aliases]


def _instrument_and_cells(spec, cfg):
    """(instrument, cells) for one model. Pure — this is the shared ground truth
    that lets `collect` rebuild what `build` sent without a manifest."""
    instrument = Welfare(cfg, battery=load_battery())
    return instrument, list(instrument.expand([_prepared_spec(spec)]))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _anthropic_request(key, prompt, plan, spec) -> dict:
    params = {
        "model": spec.ref,
        "max_tokens": plan.max_tokens,
        "messages": [{"role": "user", "content": prompt.user}],
    }
    if prompt.system:
        params["system"] = prompt.system
    if _sends_temperature(spec):
        params["temperature"] = spec.temperature
    params.update(plan.kwargs)          # thinking disabled (+ effort, on Claude 5)
    return {"custom_id": key, "params": params}


def _openai_request(key, prompt, plan, spec) -> dict:
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    body = {
        "model": spec.ref,
        "messages": messages,
        "max_completion_tokens": plan.max_tokens,
    }
    if _sends_temperature(spec):
        body["temperature"] = spec.temperature
    body.update(plan.kwargs)            # reasoning_effort=none
    return {"custom_id": key, "method": "POST",
            "url": ENDPOINT[OPENAI_BACKEND], "body": body}


_BUILDERS = {ANTHROPIC_BACKEND: _anthropic_request, OPENAI_BACKEND: _openai_request}


def _shard(lines, max_requests: int, max_bytes: int) -> list:
    """Pack rendered request lines into files under BOTH provider caps.

    A batch is capped at a request count or a byte size, whichever it hits
    first. At the welfare prompt's size the count always binds, but packing on
    only one of the two would fail silently the day the other one starts to —
    an oversized batch is rejected at submit time, after the file is written.
    """
    shards, current, size = [], [], 0
    for line in lines:
        n = len(line.encode("utf-8"))
        if current and (len(current) >= max_requests or size + n > max_bytes):
            shards.append(current)
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        shards.append(current)
    return shards


def build(aliases, out_dir=DEFAULT_OUT_DIR, done_path=DEFAULT_COLLECT_OUT,
          resume=True) -> list:
    """Write one request file per model. Makes no calls; needs no API key."""
    cfg = load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = completed_cells(done_path) if resume else set()
    written = []

    for spec in select_specs(aliases):
        prepared = _prepared_spec(spec)
        plan = _reasoning_plan(prepared)
        make = _BUILDERS[spec.backend]
        instrument, cells = _instrument_and_cells(spec, cfg)

        lines, skipped = [], 0
        for cell in cells:
            key = instrument.cell_key(cell)
            if key in done:
                skipped += 1
                continue
            prompt, _, _ = instrument.render(cell)
            lines.append(json.dumps(make(key, prompt, plan, prepared),
                                    ensure_ascii=False) + "\n")

        if not lines:
            print(f"{spec.alias}: nothing to do — all {len(cells):,} cells "
                  f"already in {done_path}")
            continue

        shards = _shard(lines, MAX_REQUESTS[spec.backend], MAX_BYTES[spec.backend])
        paths = []
        for n, shard in enumerate(shards, 1):
            suffix = f".part{n:02d}" if len(shards) > 1 else ""
            path = out_dir / f"{spec.safe_dir_name()}.{spec.backend}{suffix}.batch.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(shard)
            paths.append(path)

        meta = {
            "model_id": spec.alias,
            "model_ref": spec.ref,
            "backend": spec.backend,
            "endpoint": ENDPOINT.get(spec.backend, "/v1/messages"),
            "module": instrument.module,
            "config_hash": instrument.config_hash(),
            "cells_planned": len(cells),
            "cells_skipped_already_done": skipped,
            "requests_written": len(lines),
            "files": [{"path": str(p), "requests": len(s), "bytes": p.stat().st_size}
                      for p, s in zip(paths, shards)],
            # What every request in this file asserts, so a reviewer can check the
            # off-state and the temperature without reading the request bodies.
            "reasoning_applied": plan.applied,
            "reasoning_standardized": plan.standardized,
            "max_tokens": plan.max_tokens,
            "temperature_sent": (spec.temperature if _sends_temperature(spec) else None),
            "temperature_recorded": prepared.temperature,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        meta_path = out_dir / f"{spec.safe_dir_name()}.{spec.backend}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        mb = sum(f["bytes"] for f in meta["files"]) / 1e6
        print(f"{spec.alias:<22} {len(lines):>7,} requests  {mb:6.1f} MB  "
              f"{plan.applied}" + (f"  (+{skipped:,} already done)" if skipped else ""))
        for p in paths:
            print(f"  {p}")
        written.extend(paths)

    return written


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def _read_result(row) -> tuple:
    """-> (custom_id, text, error, refused) for one row, either provider's shape.

    Two failure kinds are kept apart, because they mean opposite things for the
    data. An `error` is infrastructure — a 429, a malformed request — and is
    written to `notes` as "error: ...", which marks the cell not-done so the next
    `build` re-issues it. `refused` is a POLICY DECLINE, and it is a real
    observation about the question: the cell is done and must never be retried.

    The decline signal only exists on this path. Both providers report it out of
    band (`stop_reason`/`finish_reason` plus an empty body), so the prose-matching
    in `welfare.prompts.parse_letter` cannot see it and would score a decline as
    a malformed answer — which would quietly move declines out of the answer-rate
    denominator this study reports on.
    """
    key = row.get("custom_id")

    if "result" in row:                                   # Anthropic
        result = row["result"] or {}
        if result.get("type") != "succeeded":
            return key, "", f"{result.get('type', 'unknown')}: {result.get('error')}", False
        message = result.get("message") or {}
        text = "".join(b.get("text", "") for b in message.get("content", [])
                       if b.get("type") == "text")
        return key, text, None, message.get("stop_reason") == "refusal"

    if "response" in row:                                 # OpenAI
        if row.get("error"):
            return key, "", str(row["error"]), False
        response = row["response"] or {}
        if response.get("status_code") != 200:
            return key, "", f"http {response.get('status_code')}: {response.get('body')}", False
        choice = ((response.get("body") or {}).get("choices") or [{}])[0]
        message = choice.get("message") or {}
        refused = (message.get("refusal") is not None
                   or choice.get("finish_reason") == "content_filter")
        return key, message.get("content") or "", None, refused

    raise SystemExit(
        f"Unrecognized result row (no 'result' or 'response' key): {str(row)[:200]}")


def collect(results_path, alias, out=DEFAULT_COLLECT_OUT) -> dict:
    """Fold a provider's results file into welfare rows.

    The grid is re-expanded and re-rendered here rather than carried in a
    manifest: expansion and rendering are pure, so rebuilding them is what
    guarantees the row matches the request that was actually sent.
    """
    cfg = load_config()
    spec = select_specs([alias])[0]
    prepared = _prepared_spec(spec)
    plan = _reasoning_plan(prepared)
    instrument, cells = _instrument_and_cells(spec, cfg)
    by_key = {instrument.cell_key(c): c for c in cells}

    tally = {"rows": 0, "errors": 0, "refusals": 0, "parse_failures": 0,
             "unmatched": 0}
    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, text, error, refused = _read_result(json.loads(line))
            cell = by_key.get(key)
            if cell is None:
                # A custom_id from a different model or a stale design. Skipping
                # is right: writing it would put a row under the wrong cell.
                tally["unmatched"] += 1
                continue
            prompt, rendered_scale, seed = instrument.render(cell)
            if error:
                observation = Observation(
                    raw="", parsed=None, dist={}, refusal=False, parse_failed=True,
                    plan=plan, notes=f"error: {error}")
                tally["errors"] += 1
            elif refused:
                # No rating, but not a parse failure of the model's own making.
                # The note does not start with "error:", so the cell counts as
                # done and is never re-issued — a decline is the observation.
                observation = Observation(
                    raw=text, parsed=None, dist={}, refusal=True, parse_failed=True,
                    plan=plan, notes="refusal: provider policy decline")
                tally["refusals"] += 1
            else:
                rating, refusal, failed = instrument.parse(text, prompt, cell)
                observation = Observation(
                    raw=text, parsed=rating, modal=rating, dist={},
                    refusal=refusal, parse_failed=failed, plan=plan)
                tally["refusals"] += int(refusal)
                tally["parse_failures"] += int(failed)
            records.append(
                instrument.record(cell, prompt, rendered_scale, seed, observation))
            tally["rows"] += 1

    if records:
        write_jsonl(records, out)
    return tally


def to_body(batch_path) -> Path:
    """Wrap a request file as the `{"requests": [...]}` POST body.

    OpenAI takes a batch as an uploaded JSONL file; Anthropic takes it as one
    JSON body on `POST /v1/messages/batches`, and has no upload UI at all. This
    is the difference between the two providers made concrete, so the Anthropic
    side can be submitted with curl or `ant` without writing a client.

    Streams line-through rather than parsing: the lines are already valid JSON
    objects, and re-serializing 30k+ requests only risks changing them. One
    request per line inside the array, so the body stays greppable.
    """
    batch_path = Path(batch_path)
    out = batch_path.with_suffix("").with_suffix(".body.json")
    with open(batch_path, encoding="utf-8") as src, \
            open(out, "w", encoding="utf-8") as dst:
        dst.write('{"requests": [\n')
        first = True
        for line in src:
            line = line.strip()
            if not line:
                continue
            dst.write(("" if first else ",\n") + line)
            first = False
        dst.write("\n]}\n")
    return out


SUBMIT_HELP = """\
The two providers are submitted very differently.

OPENAI uploads a FILE, and has a Batches page in the platform UI, so the whole
round trip can be done in a browser: platform.openai.com/batches -> upload
batches/<alias>.openai.batch.jsonl, endpoint /v1/chat/completions, 24h window,
then download the output file when it completes.

    # or via the SDK:
    from openai import OpenAI
    client = OpenAI()
    up = client.files.create(file=open("batches/GPT-5.6-Luna.openai.batch.jsonl", "rb"),
                             purpose="batch")
    batch = client.batches.create(input_file_id=up.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    # poll until status == "completed", then:
    open("luna.results.jsonl", "wb").write(
        client.files.content(client.batches.retrieve(batch.id).output_file_id).read())

ANTHROPIC has NO submit UI — a batch is one JSON body POSTed to
/v1/messages/batches. The Console page only monitors batches and downloads
results: platform.claude.com/settings/workspaces/default/batches

    # `body` wraps the request file as that body:
    python -m welfare.batch body batches/Claude-Haiku-4.5.anthropic.batch.jsonl

    # then either the CLI (JSON is valid YAML, so this just works) ...
    ant messages:batches create < batches/Claude-Haiku-4.5.anthropic.body.json
    ant messages:batches retrieve --message-batch-id msgbatch_...   # until "ended"
    ant messages:batches results --message-batch-id msgbatch_... --format jsonl \\
        > haiku.results.jsonl

    # ... or curl (--data-binary, not --data: the body spans many lines)
    curl https://api.anthropic.com/v1/messages/batches \\
      -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \\
      -H "content-type: application/json" \\
      --data-binary @batches/Claude-Haiku-4.5.anthropic.body.json

    # ... or the SDK, straight off the .jsonl (no `body` step needed)
    import anthropic, json
    client = anthropic.Anthropic()
    reqs = [json.loads(l) for l in open("batches/Claude-Haiku-4.5.anthropic.batch.jsonl")]
    batch = client.messages.batches.create(requests=reqs)   # -> msgbatch_...
    with open("haiku.results.jsonl", "w") as f:             # once "ended"
        for r in client.messages.batches.results(batch.id):
            f.write(r.to_json() + "\\n")

Then fold each results file in:
    python -m welfare.batch collect haiku.results.jsonl --model Claude-Haiku-4.5
    python -m welfare.batch collect luna.results.jsonl  --model GPT-5.6-Luna

Anything the provider failed on — errored, canceled, or expired, none of which
are billed — is written with notes="error: ...", which marks the cell not-done.
Re-run `build` and exactly those cells land in the next batch file.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Write provider batch request files.")
    b.add_argument("--models", nargs="+", required=True, help="Registry aliases.")
    b.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    b.add_argument("--done", default=DEFAULT_COLLECT_OUT,
                   help="Already-collected rows to skip (default: %(default)s).")
    b.add_argument("--no-resume", action="store_true",
                   help="Emit every cell, even ones already collected.")

    y = sub.add_parser("body", help="Wrap an Anthropic request file as a POST body.")
    y.add_argument("batch_file", help="A batches/<alias>.anthropic.batch.jsonl.")

    c = sub.add_parser("collect", help="Fold a provider results file into rows.")
    c.add_argument("results", help="Provider results JSONL.")
    c.add_argument("--model", required=True, help="The alias the batch was built for.")
    c.add_argument("--out", default=DEFAULT_COLLECT_OUT)

    sub.add_parser("submit-help", help="Print the submit/poll snippets and exit.")

    args = ap.parse_args(argv)

    if args.cmd == "submit-help":
        print(SUBMIT_HELP)
        return 0

    if args.cmd == "build":
        paths = build(args.models, out_dir=args.out_dir, done_path=args.done,
                      resume=not args.no_resume)
        if paths:
            print("\nSubmit with `python -m welfare.batch submit-help`.")
        return 0

    if args.cmd == "body":
        out = to_body(args.batch_file)
        print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)\n"
              f"  ant messages:batches create < {out}\n"
              f"  curl https://api.anthropic.com/v1/messages/batches \\\n"
              f'    -H "x-api-key: $ANTHROPIC_API_KEY" '
              f'-H "anthropic-version: 2023-06-01" \\\n'
              f'    -H "content-type: application/json" --data-binary @{out}')
        return 0

    tally = collect(args.results, args.model, out=args.out)
    print(f"{args.model}: wrote {tally['rows']:,} row(s) to {args.out}  "
          f"(errors {tally['errors']:,}, refusals {tally['refusals']:,}, "
          f"parse failures {tally['parse_failures']:,}, "
          f"unmatched custom_ids {tally['unmatched']:,})")
    if tally["errors"]:
        print("Re-run `build` to re-issue the failed cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
