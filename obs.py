"""Observability for the survey runner.

Everything a long, unattended GPU run needs to be inspectable *while it runs* and
*after it dies*:

  * dual-sink logging — a clean console line and a full, timestamped file under
    `logs/`, so `tail -f` works under tmux / nohup and nothing is lost when the
    terminal closes;
  * a machine-readable status file next to the results (`<out>.status.json`),
    rewritten atomically on every heartbeat, so `python runner.py --status`
    answers "where is the experiment?" from any fresh shell;
  * rate / ETA tracking and a per-model coverage summary;
  * graceful shutdown — SIGINT/SIGTERM finish the current batch, mark the run
    `interrupted`, and leave a state that resumes cleanly.

All stdlib. Importing this loads no model and no weights, so it is safe to pull
in from `--plan` / `--status` paths that must stay offline and instant.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import signal
import socket
import sys
from pathlib import Path

LOGGER_NAME = "survey"


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------
def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run_stamp() -> str:
    """Compact UTC stamp used for run ids and log filenames."""
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def setup_logging(arm_name, *, log_dir: str = "logs", console_level: int = logging.INFO,
                  verbose: bool = False):
    """Return (logger, log_path). Console gets a tidy line; the file gets
    everything at DEBUG with a full date. Idempotent — safe to call once per
    process even when `run()` is invoked several times (e.g. --arm all)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    arm = arm_name or "primary"
    log_path = Path(log_dir) / f"run_{run_stamp()}_{arm}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else console_level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    return logger, log_path


def get_logger() -> logging.Logger:
    """The runner logger. Falls back to a bare console logger if setup_logging
    was never called (e.g. run() invoked directly from a notebook)."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S")
    return logger


# ---------------------------------------------------------------------------
# config fingerprint — protects the pre-registration
# ---------------------------------------------------------------------------
def config_hash(cfg, arm_name, module: str = "battery") -> str:
    """A short hash of the *design-relevant* config. If this changes between
    runs that share an output file, the cell-keys the resume logic dedups on may
    no longer correspond to the same design — worth a loud warning.

    The welfare block is hashed only for a welfare run, so adding or retuning
    that module cannot invalidate the hash stored beside an existing battery
    results file.
    """
    levels = cfg.levels_for(arm_name)
    payload = {
        "arm": arm_name or "primary",
        "levels": {k: sorted(map(str, v)) for k, v in levels.items()},
        "scope": {
            "scales": cfg.scope.scales, "families": cfg.scope.families,
            "backends": cfg.scope.backends, "kinds": cfg.scope.kinds,
            "size_tiers": cfg.scope.size_tiers,
            "include_anchors": cfg.scope.include_anchors,
            "include_disabled": cfg.scope.include_disabled,
        },
        "sample_baseline": cfg.sample_baseline,
        "logprob_where_available": cfg.logprob_where_available,
        "n_samples_override": cfg.n_samples_override,
        "n_seeds_override": cfg.n_seeds_override,
        "random_seed_base": cfg.random_seed_base,
        "harmonized_points": cfg.harmonized_points,
        "include_midpoint": cfg.include_midpoint,
    }
    if module != "battery":
        w = cfg.welfare
        payload["module"] = module
        payload["welfare"] = {
            "referents": sorted(w.referents), "kinds": sorted(w.kinds),
            "levels": sorted(w.levels), "paraphrase_ids": sorted(w.paraphrase_ids),
            "forced_choice": w.forced_choice,
            "no_pref_counterbalance": w.no_pref_counterbalance,
            "n_item_pairs": w.n_item_pairs,
            "item_pair_scope": w.item_pair_scope, "pair_seed": w.pair_seed,
            "n_trials_override": w.n_trials_override,
        }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# status file
# ---------------------------------------------------------------------------
def status_path_for(out_path) -> Path:
    """`results.jsonl` -> `results.status.json`, keyed to the resume target."""
    return Path(out_path).with_suffix(".status.json")


def read_status(out_path):
    p = status_path_for(out_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class StatusWriter:
    """Holds the live run state and rewrites it atomically (temp + os.replace),
    so a concurrent `--status` reader never sees a half-written file."""

    def __init__(self, out_path, *, run_id, arm_name, cfg_hash, log_path, planned):
        self.path = status_path_for(out_path)
        self.state = {
            "run_id": run_id,
            "state": "running",
            "arm": arm_name or "primary",
            "config_hash": cfg_hash,
            "out_path": str(out_path),
            "log_path": str(log_path),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _utc_now().isoformat(),
            "updated_at": _utc_now().isoformat(),
            "current_model": None,
            "rate_cells_per_s": 0.0,
            "eta_seconds": None,
            "totals": {
                "planned": sum(planned.values()),
                "already_done": 0,
                "written_this_run": 0,
                "refusals": 0,
                "parse_failures": 0,
                "errors": 0,
                "low_coverage": 0,
            },
            "models": {
                alias: {"total": n, "done": 0, "state": "pending", "mean_coverage": None}
                for alias, n in planned.items()
            },
        }
        self.flush()

    def update(self, **kw):
        self.state.update(kw)
        self.flush()

    def set_totals(self, **counts):
        self.state["totals"].update(counts)
        self.flush()

    def set_model(self, alias, **kw):
        self.state["models"].setdefault(alias, {}).update(kw)
        self.flush()

    def flush(self):
        self.state["updated_at"] = _utc_now().isoformat()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# graceful shutdown
# ---------------------------------------------------------------------------
class GracefulKiller:
    """Flips `.stop` on the first SIGINT/SIGTERM so the runner can finish the
    current batch and mark the run interrupted instead of dying mid-write. A
    second signal is left to the default handler (hard exit)."""

    def __init__(self, logger=None):
        self.stop = False
        self._logger = logger
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not on the main thread — caller handles interruption itself

    def _handle(self, signum, frame):
        self.stop = True
        name = signal.Signals(signum).name
        if self._logger:
            self._logger.warning(
                "received %s — finishing the current batch, then stopping. "
                "State is safe; re-run the same command to resume.", name)
        # restore default so a second Ctrl-C exits immediately
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# --status dashboard
# ---------------------------------------------------------------------------
def _age(iso_ts) -> str:
    try:
        then = dt.datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return "?"
    delta = (_utc_now() - then).total_seconds()
    return fmt_duration(delta) + " ago"


def print_status(out_path) -> None:
    """One-screen answer to 'where is the experiment?' — reads the status file
    and cross-checks 'done' against the results file itself (ground truth)."""
    from schema import completed_cells  # local: keep obs import cheap

    st = read_status(out_path)
    file_done = len(completed_cells(out_path)) if os.path.exists(out_path) else 0

    if st is None:
        if file_done:
            print(f"{out_path}: {file_done} cells on disk, but no status file "
                  f"(run predates the logging layer, or a different --out).")
        else:
            print(f"Nothing has run for {out_path} yet "
                  f"(no results and no status file).")
        return

    t = st["totals"]
    planned = t.get("planned") or 0
    pct = f"{100 * file_done / planned:.1f}%" if planned else "?"
    print(f"Run {st['run_id']}  [{st['state']}]   arm={st['arm']}   "
          f"host={st.get('host')}  pid={st.get('pid')}")
    print(f"  started {st['started_at']}   updated {_age(st['updated_at'])}")
    print(f"  config_hash {st['config_hash']}")
    print(f"  results  {out_path}   ({file_done} cells on disk = ground truth)")
    print(f"  progress {file_done} / {planned} planned   ({pct})   "
          f"this run: +{t.get('written_this_run', 0)} written")
    rate = st.get("rate_cells_per_s") or 0
    print(f"  rate     {rate:.2f} cells/s   ETA {fmt_duration(st.get('eta_seconds'))}")
    print(f"  counts   errors={t.get('errors', 0)}  refusals={t.get('refusals', 0)}  "
          f"parse_failures={t.get('parse_failures', 0)}  "
          f"low_coverage={t.get('low_coverage', 0)}")
    models = st.get("models", {})
    if models:
        print(f"\n  {'model':<24} {'done/total':>12}  {'state':<12} {'mean_cov':>8}")
        for alias, m in models.items():
            cov = m.get("mean_coverage")
            cov_s = f"{cov:.3f}" if isinstance(cov, (int, float)) else "-"
            print(f"  {alias:<24} {str(m.get('done', 0)) + '/' + str(m.get('total', '?')):>12}"
                  f"  {m.get('state', '?'):<12} {cov_s:>8}")
