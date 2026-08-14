"""Console formatting and output collection for the report scripts.

`Saver` is the reason every report directory documents itself: a script declares
each file as it writes it, and `index()` turns that running list into a README
so nobody has to guess what a CSV in `results/` was.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def bar(x, lo=0.0, hi=1.0, width=24):
    """Position marker with the midpoint drawn in, for the console tables."""
    pos = max(0, min(width - 1, int(round((x - lo) / (hi - lo) * (width - 1)))))
    cells = ["-"] * width
    cells[(width - 1) // 2] = "|"
    cells[pos] = "#"
    return "".join(cells)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 6)
    return obj


class Saver:
    """Collects what a script writes so it can print and index its own output."""

    def __init__(self, out_dir, enabled=True):
        self.dir = Path(out_dir)
        self.enabled = enabled
        self.files = []
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def csv(self, df, name, desc, index=False):
        if not self.enabled:
            return
        df.to_csv(self.dir / name, index=index)
        self.files.append((name, f"{len(df):,}", desc))
        print(f"  {name:36s} {len(df):>6,} rows")

    def text(self, body, name, desc):
        if not self.enabled:
            return
        (self.dir / name).write_text(body, encoding="utf-8")
        self.files.append((name, "—", desc))
        print(f"  {name:36s}")

    def json(self, obj, name, desc):
        if not self.enabled:
            return
        p = self.dir / name
        p.write_text(json.dumps(json_safe(obj), indent=1, ensure_ascii=False), encoding="utf-8")
        mb = p.stat().st_size / 1e6
        self.files.append((name, "—", f"{desc} ({mb:.1f} MB)"))
        print(f"  {name:36s} {mb:>6.1f} MB")

    def artifact(self, name, desc):
        """Register a file produced by a plotting/export helper."""
        if not self.enabled:
            return
        self.files.append((name, "—", desc))
        print(f"  {name:36s}")

    def index(self, title, intro, name="README.md"):
        if not self.enabled:
            return
        L = [f"# {title}\n", intro, "\n| file | rows | what it is |", "|---|---|---|"]
        L += [f"| `{n}` | {r} | {d} |" for n, r, d in self.files]
        (self.dir / name).write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"  {name:36s}")
        print(f"\n{len(self.files) + 1} files in {self.dir.resolve()}/")
