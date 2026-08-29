"""Tiny dependency-free SVG line-chart helper.

Deliberately avoids a JS charting library (Chart.js, etc.) so the app keeps working
when opened from an offline HPC node with no internet access.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]

WIDTH = 640
HEIGHT = 220
PAD = 36


def series_to_points(
    values: Sequence[float],
    y_range: Optional[Tuple[float, float]] = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    pad: int = PAD,
) -> Optional[str]:
    """Return an SVG <polyline points="..."> value, or None if there's nothing to plot."""
    values = [v for v in values]
    if not values:
        return None
    n = len(values)
    ymin, ymax = y_range if y_range else (min(values), max(values))
    if ymin == ymax:
        ymin, ymax = ymin - 1, ymax + 1

    def scale_x(i: int) -> float:
        if n == 1:
            return width / 2
        return pad + (i / (n - 1)) * (width - 2 * pad)

    def scale_y(v: float) -> float:
        return height - pad - ((v - ymin) / (ymax - ymin)) * (height - 2 * pad)

    return " ".join(f"{scale_x(i):.1f},{scale_y(v):.1f}" for i, v in enumerate(values))
