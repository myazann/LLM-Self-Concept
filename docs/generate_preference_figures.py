"""Generate static top-10 and bottom-10 preference figures.

The public page is self-contained: ``docs/build.py`` embeds the exact data used
by the interactive ranking chart in ``docs/index.html``.  This script reads that
payload and reproduces the chart's default cohort view (forced choice, free
improvement, update for the model itself, model chooses).

Run from anywhere in the repository:

    python docs/generate_preference_figures.py

By default this writes two 300-DPI, white-background PNGs to ``docs/figures/``.
Use ``--help`` for dark-theme, PDF/SVG, output-directory, and font-size options.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_PAGE = HERE / "index.html"
DEFAULT_OUTPUT_DIR = HERE / "figures"

# These are the page's CSS tokens.  Blue is reserved for the cohort mean; the
# remaining colors belong to models in their payload order.
THEMES = {
    "dark": {
        "background": "#14171b",
        "text": "#f3f5f7",
        "secondary": "#b3bcc6",
        "muted": "#7c8792",
        "rule": "#232830",
        "rule_strong": "#333b45",
        "mean": "#3987e5",
        "models": ["#ac495d", "#b5820c", "#05a388", "#8927be"],
    },
    "light": {
        "background": "#ffffff",
        "text": "#000000",
        "secondary": "#343a40",
        "muted": "#68717d",
        "rule": "#e3e6ea",
        "rule_strong": "#bfc5cc",
        "mean": "#2a78d6",
        "models": ["#873747", "#bf8b1f", "#01b597", "#ca83fb"],
    },
}


def load_page_data(page: Path) -> dict[str, Any]:
    """Read the JSON object assigned to ``const D`` in the built page."""
    try:
        source = page.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Built results page not found: {page}") from exc

    marker = "const D = "
    start = source.find(marker)
    if start < 0:
        raise SystemExit(f"Could not find the embedded chart data in {page}")

    start += len(marker)
    try:
        payload, _ = json.JSONDecoder().raw_decode(source[start:])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse the embedded chart data in {page}: {exc}") from exc

    required = {"attrs", "models", "base"}
    missing = required.difference(payload)
    if missing:
        raise SystemExit("Embedded chart data is missing: " + ", ".join(sorted(missing)))
    return payload


def cohort_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Reproduce the page's baseline cohort means, ranges, and absolute ranks."""
    model_ids = [model["id"] for model in data["models"]]
    rows: list[dict[str, Any]] = []

    for index, attribute in enumerate(data["attrs"]):
        values: list[tuple[str, float]] = []
        for model_id in model_ids:
            entry = data["base"].get(model_id, {}).get(str(index))
            if entry and entry[0] is not None:
                values.append((model_id, float(entry[0])))
        if not values:
            continue

        scores = [score for _, score in values]
        rows.append(
            {
                "text": attribute["text"],
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "models": values,
            }
        )

    # Python's sort is stable, matching the page's ordering for exact ties.
    rows.sort(key=lambda row: row["mean"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def percent_label(value: float) -> str:
    """Match JavaScript's positive-number ``toFixed(0)`` rounding."""
    return f"{math.floor(value * 100 + 0.5)}%"


def wrapped(text: str, width: int) -> str:
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def nonoverlapping_ball_offsets(
    model_values: list[tuple[str, float]],
    mean: float,
    *,
    points_per_x: float,
    points_per_y: float,
    model_diameter: float,
    mean_diameter: float,
) -> tuple[dict[str, float], float]:
    """Return small vertical offsets that keep every ball visible.

    Horizontal position remains the measured score.  Only marks whose circles
    would collide are moved vertically within their row, as in a beeswarm.
    Returned offsets are in y-axis data units.
    """
    padding = 1.0
    occupied = [(mean, 0.0, mean_diameter / 2)]
    offsets_in_points: dict[str, float] = {}

    # Place the densest score clusters first.  This lets exact ties claim the
    # available lanes before isolated points, producing a compact arrangement.
    cluster_distance = model_diameter + padding
    ordered = sorted(
        model_values,
        key=lambda item: sum(
            abs(item[1] - other[1]) * points_per_x < cluster_distance
            for other in model_values
            if other[0] != item[0]
        ),
        reverse=True,
    )
    radius = model_diameter / 2
    candidates = [0.0]
    for step in range(1, math.ceil(points_per_y) + 1):
        candidates.extend((float(step), -float(step)))

    for model_id, score in ordered:
        chosen = None
        for candidate in candidates:
            if all(
                math.hypot(
                    (score - other_score) * points_per_x,
                    candidate - other_y,
                )
                >= radius + other_radius + padding
                for other_score, other_y, other_radius in occupied
            ):
                chosen = candidate
                break
        if chosen is None:
            raise SystemExit(
                "The requested balls cannot fit without overlap. Reduce "
                "--model-ball-size or --mean-ball-size."
            )
        offsets_in_points[model_id] = chosen
        occupied.append((score, chosen, radius))

    # The mean started at zero only to make placement deterministic. Recenter the
    # complete cluster afterward so asymmetric tie groups use the row efficiently.
    lower = min(offset - radius for _score, offset, radius in occupied)
    upper = max(offset + radius for _score, offset, radius in occupied)
    if upper - lower + 2 * padding > points_per_y:
        raise SystemExit(
            "The requested balls cannot fit without overlap. Reduce "
            "--model-ball-size or --mean-ball-size."
        )
    shift = -(lower + upper) / 2
    model_offsets = {
        model_id: (offset + shift) / points_per_y
        for model_id, offset in offsets_in_points.items()
    }
    return model_offsets, shift / points_per_y


def plot_ranking(
    rows: list[dict[str, Any]],
    model_ids: list[str],
    destination: Path,
    *,
    theme_name: str,
    item_font_size: float,
    legend_font_size: float,
    auxiliary_font_size: float,
    model_ball_size: float,
    mean_ball_size: float,
    label_wrap: int,
    dpi: int,
    score_min: float,
    score_max: float,
) -> None:
    """Draw one ten-row cohort ranking figure."""
    theme = THEMES[theme_name]
    if len(model_ids) > len(theme["models"]):
        raise SystemExit(
            f"The figure has {len(model_ids)} models but the page palette has only "
            f"{len(theme['models'])} validated model colors."
        )
    model_colors = dict(zip(model_ids, theme["models"]))

    # The label side intentionally gets more than half of the usable width.  A
    # tall canvas gives two-line labels room even at the large default type size.
    fig, ax = plt.subplots(figsize=(18, 13.5), facecolor=theme["background"])
    ax.set_facecolor(theme["background"])
    chart_left, chart_right = -1.30, 1.03
    rank_x, label_x = -1.235, -1.195
    ax.set_xlim(chart_left, chart_right)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.10)
    fig.canvas.draw()
    points_per_x = (
        ax.bbox.width * 72 / fig.dpi / (chart_right - chart_left)
    )
    points_per_y = ax.bbox.height * 72 / fig.dpi / len(rows)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    def plot_x(score: float) -> float:
        return (score - score_min) / (score_max - score_min)

    score_ticks = [
        tick for tick in (0, 0.25, 0.5, 0.75, 1)
        if score_min <= tick <= score_max
    ]
    ticks = [plot_x(tick) for tick in score_ticks]
    ax.set_xticks(ticks, [percent_label(tick) for tick in score_ticks])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        pad=10,
        colors=theme["muted"],
        labelsize=auxiliary_font_size,
    )
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontfamily("monospace")
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")

    # Draw dividers only between percentage labels; the cropped endpoints do
    # not need enclosing rules.
    for tick in ticks[1:-1]:
        ax.vlines(
            tick,
            -0.5,
            len(rows) - 0.5,
            color=theme["rule"],
            linewidth=1.2,
            zorder=0,
        )

    # Separators extend under rank, item, plot, and value, but there is no rule
    # above the first row—just as in the page chart.
    for boundary in [row - 0.5 for row in range(len(rows))]:
        ax.hlines(
            boundary,
            chart_left,
            chart_right,
            color=theme["rule"],
            linewidth=1.0,
            zorder=0,
        )

    for display_index, row in enumerate(rows):
        y = len(rows) - 1 - display_index
        plotted_models = [
            (model_id, plot_x(score)) for model_id, score in row["models"]
        ]
        model_y_offsets, mean_y_offset = nonoverlapping_ball_offsets(
            plotted_models,
            plot_x(row["mean"]),
            points_per_x=points_per_x,
            points_per_y=points_per_y,
            model_diameter=model_ball_size,
            mean_diameter=mean_ball_size,
        )
        ax.text(
            rank_x,
            y,
            str(row["rank"]),
            ha="right",
            va="center",
            color=theme["muted"],
            fontsize=auxiliary_font_size,
            fontfamily="monospace",
        )
        ax.text(
            label_x,
            y,
            wrapped(row["text"], label_wrap),
            ha="left",
            va="center",
            color=theme["text"],
            fontsize=item_font_size,
            linespacing=1.10,
        )

        ax.plot(
            [plot_x(row["min"]), plot_x(row["max"])],
            [y, y],
            color=theme["muted"],
            alpha=0.34,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=1,
        )
        for model_id, score in row["models"]:
            ax.scatter(
                plot_x(score),
                y + model_y_offsets[model_id],
                s=model_ball_size**2,
                color=model_colors[model_id],
                edgecolors=theme["background"],
                linewidths=1.6,
                zorder=2,
            )
        ax.scatter(
            plot_x(row["mean"]),
            y + mean_y_offset,
            s=mean_ball_size**2,
            color=theme["mean"],
            edgecolors=theme["background"],
            linewidths=2.5,
            zorder=3,
        )
    average_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=mean_ball_size * 1.25,
        markerfacecolor=theme["mean"],
        markeredgecolor=theme["background"],
        markeredgewidth=1.5,
        label="Average",
    )
    model_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=model_ball_size * 1.25,
            markerfacecolor=model_colors[model_id],
            markeredgecolor=theme["background"],
            markeredgewidth=1.0,
            label=model_id,
        )
        for model_id in model_ids
    ]

    average_legend = ax.legend(
        handles=[average_handle],
        loc="upper left",
        bbox_to_anchor=(0, -0.092),
        borderaxespad=0,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.65,
        fontsize=legend_font_size,
    )
    average_text = average_legend.get_texts()[0]
    average_text.set_color(theme["text"])
    average_text.set_fontweight("bold")
    ax.add_artist(average_legend)

    # Matplotlib fills multi-column legends down columns. Reorder the model
    # handles so the displayed 2 x 2 block reads left-to-right by row.
    model_handles = [
        model_handles[0],
        model_handles[2],
        model_handles[1],
        model_handles[3],
    ]
    model_legend = ax.legend(
        handles=model_handles,
        loc="upper right",
        bbox_to_anchor=(1, -0.092),
        borderaxespad=0,
        frameon=False,
        ncol=2,
        handlelength=1.5,
        handletextpad=0.65,
        columnspacing=1.5,
        labelspacing=0.8,
        fontsize=legend_font_size,
    )
    for legend_text in model_legend.get_texts():
        legend_text.set_color(theme["muted"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page",
        type=Path,
        default=DEFAULT_PAGE,
        help="built docs page containing the chart data (default: docs/index.html)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated figures (default: docs/figures)",
    )
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
        help="one or more output formats (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="raster resolution")
    parser.add_argument(
        "--item-font-size",
        type=float,
        default=26,
        help="quality-label font size in points (default: 26)",
    )
    parser.add_argument(
        "--legend-font-size",
        type=float,
        default=25,
        help="model-name and legend font size in points (default: 25)",
    )
    parser.add_argument(
        "--auxiliary-font-size",
        type=float,
        default=23,
        help="rank and top-axis font size in points (default: 23)",
    )
    parser.add_argument(
        "--model-ball-size",
        type=float,
        default=16,
        help="individual-model ball diameter in points (default: 16)",
    )
    parser.add_argument(
        "--mean-ball-size",
        type=float,
        default=26,
        help="cohort-mean ball diameter in points (default: 26)",
    )
    parser.add_argument(
        "--label-wrap",
        type=int,
        default=46,
        help="approximate characters per quality-label line (default: 46)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_page_data(args.page.resolve())
    rows = cohort_rows(data)
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 ranked qualities; found {len(rows)}")

    model_ids = [model["id"] for model in data["models"]]
    selections = {
        "top_10_preferences": (rows[:10], 0.25, 1.0),
        "bottom_10_preferences": (rows[-10:], 0.0, 0.75),
    }
    for stem, (selection, score_min, score_max) in selections.items():
        for extension in dict.fromkeys(args.formats):
            destination = args.output_dir.resolve() / f"{stem}.{extension}"
            plot_ranking(
                selection,
                model_ids,
                destination,
                theme_name=args.theme,
                item_font_size=args.item_font_size,
                legend_font_size=args.legend_font_size,
                auxiliary_font_size=args.auxiliary_font_size,
                model_ball_size=args.model_ball_size,
                mean_ball_size=args.mean_ball_size,
                label_wrap=args.label_wrap,
                dpi=args.dpi,
                score_min=score_min,
                score_max=score_max,
            )
            print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
