"""matplotlib charts for a run: the arms, the lifts, and Stage 2 training.

======================================================================
WHAT IS PLOTTED, AND WHY THESE FOUR
======================================================================
1. TEST AUC-PR BY ARM, with its 95% bootstrap interval. The headline number.
   AUC-PR rather than ROC-AUC because prevalence is ~0.15%: ROC-AUC reads
   ~0.91 for every arm and separates nothing.

2. LIFTS, each with its PAIRED 95% interval, on an axis CENTRED ON ZERO.
   This is the panel that answers "did the graph help", and the zero line is
   the decision, not decoration: an interval that straddles it has not shown
   an effect, however large the point estimate looks. A bar chart of point
   estimates alone would hide exactly that, which is why the interval is
   drawn. Bars are coloured by verdict -- above zero, straddling, below.

3. PRECISION / RECALL / F1 at the operating threshold, per arm. AUC-PR is
   threshold-free; this is what the model would actually do in production, at
   the threshold chosen on validation and applied unchanged to test.

4. STAGE 2 TRAINING LOSS per epoch, when Stage 2 has run. A flat curve at a
   suspiciously good value is the documented signature of a leak, so it is
   worth seeing rather than summarising to a final number.

======================================================================
Agg BACKEND, SET BEFORE pyplot IS IMPORTED
======================================================================
This runs inside ``run_all.sh``, on a headless GPU box as often as on a Mac.
Importing ``pyplot`` without a backend selected picks an interactive one and
either fails or blocks on a display that is not there. ``Agg`` is the
file-only backend; it must be selected BEFORE ``pyplot`` is imported, because
the backend is bound at import time.

Missing inputs are SKIPPED with a note, never faked: a Stage 1-only run draws
the first three panels, and the fourth appears once Stage 2 has written its
metrics.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

# BEFORE pyplot. See the module docstring -- this is not stylistic.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

#: Arms in the order they build on each other, so every panel reads the same
#: way regardless of dict ordering in the JSON.
ARM_ORDER = (
    "raw",
    "raw_plus_aggregates",
    "raw_plus_graph",
    # Stage 2 and Stage 3 are SIBLINGS off raw_plus_graph, not a chain; Stage 4b
    # then builds on Stage 3. Listed explicitly because `_arms` only sorts the
    # unlisted ones alphabetically, which put the two temporal arms after the
    # R-GCN under their raw variant names.
    "raw_plus_embeddings",
    "raw_plus_temporal",
    "raw_plus_temporal_embeddings",
    # the presentation arms: same encoders, three pair statistics instead of
    # the raw dimensions, plus the GNN-as-replacement arm.
    "raw_plus_pair_scores",
    "raw_plus_temporal_pair_scores",
    "raw_plus_aggregates_embeddings",
)

ARM_LABEL = {
    "raw": "raw",
    "raw_plus_aggregates": "+ aggregates",
    "raw_plus_graph": "+ graph",
    # NAMED BY ENCODER, not by "GNN". There are two GNNs in this project now and
    # a chart that calls both of them "GNN embeddings" is unreadable -- and it
    # was worse than unreadable while both arms were fed the same R-GCN tables.
    "raw_plus_embeddings": "+ R-GCN embeddings",
    "raw_plus_temporal": "+ temporal",
    "raw_plus_temporal_embeddings": "+ TGN embeddings",
    "raw_plus_pair_scores": "+ R-GCN pair score",
    "raw_plus_temporal_pair_scores": "+ TGN pair score",
    "raw_plus_aggregates_embeddings": "R-GCN instead of graph",
}

#: SHORT FORMS, for the one panel whose arms are CATEGORIES ON AN X AXIS rather
#: than rows. Eight arms share ~810 px there, so ~100 px per slot, and
#: "R-GCN instead of graph" is ~193 px at 8.5 pt -- 137 px even rotated 45deg, so
#: no rotation angle rescues the full names. Rotation alone was the first fix
#: tried and the labels still overlapped into mush. The bar panels keep the full
#: labels, because a horizontal axis has the room.
ARM_SHORT = {
    "raw": "raw",
    "raw_plus_aggregates": "+ agg",
    "raw_plus_graph": "+ graph",
    "raw_plus_embeddings": "+ R-GCN",
    "raw_plus_temporal": "+ temporal",
    "raw_plus_temporal_embeddings": "+ TGN",
    "raw_plus_pair_scores": "+ R-GCN pair",
    "raw_plus_temporal_pair_scores": "+ TGN pair",
    "raw_plus_aggregates_embeddings": "R-GCN swap",
}

# Verdict colours, shared by the lift panel and its legend.
_SIG = "#2e9e63"
_NS = "#8a90a2"
_NEG = "#d1495b"
_BAR = "#4c7ef3"
_INK = "#1d2230"
_MUTED = "#5d6478"


def _f(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else None


def _arms(stage1: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    results = stage1.get("results")
    if not isinstance(results, Mapping):
        return []
    typed = cast("Mapping[str, Any]", results)
    known = [(n, typed[n]) for n in ARM_ORDER if isinstance(typed.get(n), Mapping)]
    extra = [
        (n, v)
        for n, v in sorted(typed.items())
        if n not in ARM_ORDER and isinstance(v, Mapping)
    ]
    return [(n, cast("dict[str, Any]", v)) for n, v in known + extra]


def _lifts(stage1: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = stage1.get("lifts")
    if not isinstance(raw, Sequence):
        return []
    return [
        cast("Mapping[str, Any]", item)
        for item in cast("Sequence[Any]", raw)
        if isinstance(item, Mapping)
        and isinstance(item.get("paired_interval"), Mapping)
    ]


#: Characters per subtitle line, sized for the RIGHT-HAND column, which is the
#: binding one: its axes start at ~0.675 of the figure, leaving ~4.4 in (~658 px
#: at 150 dpi) to the edge, and 9 pt DejaVu Sans averages ~10.3 px per character.
#: 60 keeps a line at ~618 px with margin for the label-width drift that moves
#: that x0 around. Too wide and the line is CLIPPED rather than merely ugly --
#: `in_layout=False` (see _style) is what stops `bbox_inches="tight"` from
#: growing the canvas to rescue an overrun.
_SUBTITLE_WRAP = 60


def _style(axis: Any, title: str, subtitle: str = "") -> None:
    """Title, wrapped subtitle, and the spine/tick treatment shared by all panels.

    ======================================================================
    THE SUBTITLE IS WRAPPED AND EXCLUDED FROM THE LAYOUT. BOTH ARE LOAD-BEARING.
    ======================================================================
    This text is a child of the axes, so tight_layout counts it when sizing the
    axes -- and shrinking an axes does not shrink a text whose width is fixed in
    points. Given a subtitle wider than its grid cell, tight_layout shrinks until
    it gives up: measured on 2026-08-04 at **236 px of axes in a 2025 px figure**,
    11.6% each where ~40% was intended, with `bbox_inches="tight"` then expanding
    the canvas to 16.38 in to catch the overflow. Every panel ended up a stamp in
    the left third of the image with the four subtitles overprinting each other.

    So: `in_layout=False` keeps tight_layout from ever seeing it, and the wrap
    keeps it inside the cell so the saved bbox does not grow either. Fixing only
    one of the two leaves the other failure standing.
    """
    lines = textwrap.wrap(subtitle, _SUBTITLE_WRAP) if subtitle else []
    # Room for however many lines the wrap produced, or the subtitle is drawn
    # through the title.
    axis.set_title(
        title,
        fontsize=12.5,
        fontweight="bold",
        color=_INK,
        loc="left",
        pad=6 + 12 * len(lines) if lines else 8,
    )
    if lines:
        text = axis.text(
            0.0,
            1.02,
            "\n".join(lines),
            transform=axis.transAxes,
            fontsize=9,
            color=_MUTED,
            va="bottom",
            ha="left",
            linespacing=1.35,
        )
        text.set_in_layout(False)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c9cede")
    axis.tick_params(colors=_MUTED, labelsize=9.5)


# ======================================================================
# panels
# ======================================================================
def _panel_aucpr(axis: Any, arms: Sequence[tuple[str, dict[str, Any]]]) -> None:
    labels, points, lower, upper = [], [], [], []
    for name, body in arms:
        aucpr = body.get("aucpr")
        point = (
            _f(cast("Mapping[str, Any]", aucpr).get("test"))
            if isinstance(aucpr, Mapping)
            else 0.0
        )
        report = body.get("test_report")
        low = high = point
        if isinstance(report, Mapping):
            interval = cast("Mapping[str, Any]", report).get("aucpr_interval")
            if isinstance(interval, Mapping):
                typed = cast("Mapping[str, Any]", interval)
                low, high = _f(typed.get("lower")), _f(typed.get("upper"))
        labels.append(ARM_LABEL.get(name, name))
        points.append(point)
        lower.append(max(0.0, point - low))
        upper.append(max(0.0, high - point))

    positions = range(len(labels))
    axis.barh(
        list(positions),
        points,
        height=0.55,
        color=_BAR,
        xerr=[lower, upper],
        error_kw={"ecolor": "#2c4b8f", "capsize": 4, "lw": 1.4},
    )
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("test AUC-PR", fontsize=9.5, color=_MUTED)
    ceiling = max([p + u for p, u in zip(points, upper)] + [0.01])
    axis.set_xlim(0, ceiling * 1.30)
    # Past the END of the whisker, not past the bar -- otherwise the label sits
    # on top of the interval it is meant to accompany.
    for y, value, high in zip(positions, points, upper):
        axis.text(
            value + high + ceiling * 0.035,
            y,
            f"{value:.4f}",
            va="center",
            fontsize=9.5,
            color=_INK,
        )
    _style(
        axis,
        "Test AUC-PR by arm",
        "whiskers are the 95% bootstrap interval; AUC-PR not ROC-AUC "
        "(prevalence ~0.15%)",
    )


def _panel_lifts(axis: Any, lifts: Sequence[Mapping[str, Any]]) -> None:
    labels, points, lower, upper, colours = [], [], [], [], []
    for item in lifts:
        interval = cast("Mapping[str, Any]", item["paired_interval"])
        low, high = _f(interval.get("lower")), _f(interval.get("upper"))
        point = _f(item.get("absolute"))
        base = ARM_LABEL.get(str(item.get("baseline")), str(item.get("baseline")))
        cand = ARM_LABEL.get(str(item.get("candidate")), str(item.get("candidate")))
        # ONE LINE, not two. `f"{base}\n→ {cand}"` was right-aligned as a block
        # with the tick BETWEEN its two lines, so nine stacked labels read as a
        # staircase and each baseline appeared to belong to the row above it --
        # in a panel whose entire job is "which arm was this measured against".
        labels.append(f"{base} → {cand}")
        points.append(point)
        lower.append(max(0.0, point - low))
        upper.append(max(0.0, high - point))
        colours.append(_SIG if low > 0 else (_NEG if high < 0 else _NS))

    positions = list(range(len(labels)))
    for y, point, low, high, colour in zip(positions, points, lower, upper, colours):
        axis.errorbar(
            point,
            y,
            xerr=[[low], [high]],
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=2.4,
            capsize=5,
            capthick=2.4,
            markersize=7,
        )
    axis.axvline(0.0, color="#9aa1b4", linestyle="--", linewidth=1.3, zorder=0)
    # Reserve room on the right for the value annotations, or an interval that
    # reaches the edge gets its text drawn over its own whisker cap.
    lo = min([p - l for p, l in zip(points, lower)] + [0.0])
    hi = max([p + u for p, u in zip(points, upper)] + [0.0])
    span = (hi - lo) or 1.0
    axis.set_xlim(lo - span * 0.10, hi + span * 0.55)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    # ONE EMPTY SLOT AT THE BOTTOM, so the verdict legend sits INSIDE the axes.
    # It used to hang at bbox_to_anchor=(1.0, -0.30) -- 30% of the axis height
    # BELOW it -- and tight_layout does not reserve space for an artist placed
    # outside the axes that way, so on the 2x2 grid it was drawn over the
    # bottom row's title. Inverted, hence (larger, smaller).
    axis.set_ylim(len(positions) + 0.7, -0.6)
    axis.set_xlabel("absolute AUC-PR lift", fontsize=9.5, color=_MUTED)
    for y, item, point in zip(positions, lifts, points):
        probability = _f(item.get("probability_candidate_better"))
        axis.annotate(
            f"{point:+.4f}  P={probability:.2f}",
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 0),
            textcoords="offset points",
            # CENTRED ON THE ROW, so the value column lines up with the dot it
            # describes. va="bottom" sat it half a row high, which read as
            # belonging to the interval above.
            ha="right",
            va="center",
            fontsize=8.5,
            color=_MUTED,
        )
    handles = [
        Line2D([], [], color=_SIG, marker="o", lw=2.4, label="above zero"),
        Line2D([], [], color=_NS, marker="o", lw=2.4, label="straddles zero"),
        Line2D([], [], color=_NEG, marker="o", lw=2.4, label="below zero"),
    ]
    axis.legend(
        handles=handles,
        fontsize=8.5,
        frameon=False,
        loc="lower right",
        ncol=3,
    )
    _style(
        axis,
        "Lifts, with paired 95% intervals",
        "dashed line is zero — an interval touching it has not demonstrated an effect",
    )


def _panel_threshold(axis: Any, arms: Sequence[tuple[str, dict[str, Any]]]) -> None:
    labels, precision, recall, f1 = [], [], [], []
    for name, body in arms:
        report = body.get("test_report")
        if not isinstance(report, Mapping):
            continue
        at = cast("Mapping[str, Any]", report).get("at_threshold")
        if not isinstance(at, Mapping):
            continue
        typed = cast("Mapping[str, Any]", at)
        labels.append(ARM_SHORT.get(name, ARM_LABEL.get(name, name)))
        precision.append(_f(typed.get("precision")))
        recall.append(_f(typed.get("recall")))
        f1.append(_f(typed.get("f1")))

    width = 0.26
    positions = [i for i in range(len(labels))]
    for offset, values, colour, label in (
        (-width, precision, "#4c7ef3", "precision"),
        (0.0, recall, "#6fb1e0", "recall"),
        (width, f1, "#2e9e63", "F1"),
    ):
        axis.bar(
            [p + offset for p in positions],
            values,
            width=width,
            color=colour,
            label=label,
        )
    axis.set_xticks(positions)
    # SHORT LABELS (ARM_SHORT) AND ROTATED. Eight arms share ~810 px here, so a
    # slot is ~100 px; the full names need 137-193 px at any angle that stays
    # legible. rotation_mode="anchor" is what puts the END of each label under
    # its own tick instead of its centre -- without it every label drifts left
    # of the bars it names.
    axis.set_xticklabels(
        labels, fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor"
    )
    axis.set_ylabel("score", fontsize=9.5, color=_MUTED)
    # 1.35 not 1.25: the legend sits at upper left inside the axes and eight
    # arms put a tall precision bar there.
    axis.set_ylim(0, max(precision + recall + f1 + [0.1]) * 1.35)
    axis.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    _style(
        axis,
        "At the operating threshold",
        "threshold chosen on validation, applied unchanged to test",
    )


def _panel_loss(axis: Any, stage2: Mapping[str, Any]) -> bool:
    history = stage2.get("history")
    if not isinstance(history, Sequence) or not history:
        return False
    losses = [
        _f(cast("Mapping[str, Any]", e).get("loss"))
        for e in cast("Sequence[Any]", history)
        if isinstance(e, Mapping)
    ]
    if not losses:
        return False
    # The HELD-OUT curve, when the run recorded one. Training loss alone
    # falls with epochs whether or not anything transferable is being
    # learned; the two curves DIVERGING is the overfitting signature, and a
    # val curve that never moves is the starved-model one. Plotted on the
    # same axis on purpose -- the gap between them is the quantity to read,
    # and separate axes would hide it.
    # Keyed on PRESENCE, not on the value: _f defaults a missing key to 0.0,
    # which would draw a val curve pinned at zero for every run that never
    # recorded one -- a chart of a number that does not exist.
    entries = [
        cast("Mapping[str, Any]", e)
        for e in cast("Sequence[Any]", history)
        if isinstance(e, Mapping)
    ]
    has_val = any("val_loss" in e for e in entries)
    val = [_f(e.get("val_loss")) for e in entries]

    epochs = list(range(1, len(losses) + 1))
    axis.plot(
        epochs,
        losses,
        "-o",
        color=_BAR,
        linewidth=2.2,
        markersize=6,
        label="train" if has_val else None,
    )
    if has_val:
        axis.plot(
            epochs[: len(val)],
            val,
            "-o",
            color=_NEG,
            linewidth=2.2,
            markersize=6,
            label="held-out val",
        )
        axis.legend(frameon=False, fontsize=8.5, labelcolor=_MUTED)
    axis.set_xlabel("epoch", fontsize=9.5, color=_MUTED)
    axis.set_ylabel("loss", fontsize=9.5, color=_MUTED)
    axis.set_xticks(epochs)
    # Annotate the training curve only: two sets of labels on one axis
    # collide at exactly the interesting moment, when the curves converge.
    for x, y in zip(epochs, losses):
        # Epoch 1 sits ON the y axis, so a centred label puts half its width over
        # the tick labels. Left-align that one and centre the rest.
        first = x == epochs[0]
        axis.annotate(
            f"{y:.4f}",
            xy=(x, y),
            xytext=(2 if first else 0, 8),
            textcoords="offset points",
            ha="left" if first else "center",
            fontsize=8.5,
            color=_MUTED,
        )
    # HEADROOM FOR THE FIRST ANNOTATION. Epoch 1 is the highest loss and its
    # label is offset 8 pt ABOVE the point, so on the default tight autoscale it
    # is drawn outside the axes and lands in the subtitle.
    axis.margins(y=0.14)
    guarantee = str(stage2.get("temporal_guarantee", "unknown"))
    _style(
        axis,
        "Stage 2 — R-GCN loss" if has_val else "Stage 2 — R-GCN training loss",
        f"{guarantee}; a FLAT curve at a suspiciously good value is the "
        "signature of a leak"
        + (". Train falling while val flattens is overfitting" if has_val else ""),
    )
    return True


# ======================================================================
# figure
# ======================================================================
def build_figure(
    stage1: Mapping[str, Any] | None, stage2: Mapping[str, Any] | None
) -> Any:
    arms = _arms(stage1) if stage1 else []
    lifts = _lifts(stage1) if stage1 else []
    has_loss = bool(
        stage2 and isinstance(stage2.get("history"), Sequence) and stage2["history"]
    )

    panels: list[str] = []
    if arms:
        panels.append("aucpr")
    if lifts:
        panels.append("lifts")
    if arms:
        panels.append("threshold")
    if has_loss:
        panels.append("loss")
    if not panels:
        raise ValueError("nothing to plot")

    rows = (len(panels) + 1) // 2
    figure, axes_grid = plt.subplots(
        rows, 2, figsize=(13.5, 4.6 * rows), facecolor="white"
    )
    axes = list(axes_grid.flat) if len(panels) > 1 else [axes_grid]

    for axis, panel in zip(axes, panels):
        if panel == "aucpr":
            _panel_aucpr(axis, arms)
        elif panel == "lifts":
            _panel_lifts(axis, lifts)
        elif panel == "threshold":
            _panel_threshold(axis, arms)
        elif panel == "loss":
            assert stage2 is not None
            _ = _panel_loss(axis, stage2)
    for axis in axes[len(panels) :]:
        axis.axis("off")

    figure.suptitle(
        "TF_GNN — fraud detection run report",
        fontsize=15,
        fontweight="bold",
        color=_INK,
        x=0.012,
        ha="left",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def summarise(
    stage1: Mapping[str, Any] | None, stage2: Mapping[str, Any] | None
) -> str:
    """A compact numeric record for the run log, next to the PNG."""
    lines: list[str] = []
    for name, body in _arms(stage1) if stage1 else []:
        aucpr = body.get("aucpr")
        point = (
            _f(cast("Mapping[str, Any]", aucpr).get("test"))
            if isinstance(aucpr, Mapping)
            else 0.0
        )
        lines.append(f"  {ARM_LABEL.get(name, name):<18s} test AUC-PR {point:.4f}")
    for item in _lifts(stage1) if stage1 else []:
        interval = cast("Mapping[str, Any]", item["paired_interval"])
        low, high = _f(interval.get("lower")), _f(interval.get("upper"))
        verdict = (
            "significant"
            if low > 0
            else ("NEGATIVE" if high < 0 else "not distinguishable from zero")
        )
        base = ARM_LABEL.get(str(item.get("baseline")), str(item.get("baseline")))
        cand = ARM_LABEL.get(str(item.get("candidate")), str(item.get("candidate")))
        lines.append(
            f"  {base} -> {cand}: {_f(item.get('absolute')):+.4f} "
            f"[{low:+.4f}, {high:+.4f}]  {verdict}"
        )
    if stage2:
        history = stage2.get("history")
        if isinstance(history, Sequence) and history:
            final = cast("Mapping[str, Any]", history[-1])
            lines.append(f"  Stage 2 final loss {_f(final.get('loss')):.4f}")
    return "\n".join(lines)


def _resolve(path: Path) -> Path:
    """Anchor a relative path to the project, never to the shell's cwd."""
    from common.config import project_root

    return path if path.is_absolute() else project_root() / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the Stage 1 arms, the lifts and Stage 2 training."
    )
    _ = parser.add_argument(
        "--stage1", type=Path, default=Path("artifacts/stage1/stage1_comparison.json")
    )
    _ = parser.add_argument(
        "--stage2", type=Path, default=Path("artifacts/stage2/stage2_metrics.json")
    )
    _ = parser.add_argument("--out", type=Path, default=Path("artifacts/report.png"))
    _ = parser.add_argument(
        "--dpi", type=int, default=150, help="raise for print, 150 is screen-sharp"
    )
    args = parser.parse_args()
    # Project-anchored, like every other artifact path in the repo. A missing
    # artifact is reported and skipped rather than faked, so a cwd-relative
    # default would draw an honest-looking chart of nothing at all.
    args.stage1 = _resolve(args.stage1)
    args.stage2 = _resolve(args.stage2)
    args.out = _resolve(args.out)

    stage1 = _load(args.stage1)
    stage2 = _load(args.stage2)
    if stage1 is None and stage2 is None:
        print(
            f"  no artifacts to chart ({args.stage1} and {args.stage2} are both "
            "absent). Nothing was faked."
        )
        return
    if stage2 is None:
        print("  NOTE Stage 2 has not written metrics; the loss panel is omitted.")

    figure = build_figure(stage1, stage2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(summarise(stage1, stage2))
    print(f"\n  charts: {args.out}")


if __name__ == "__main__":
    main()
