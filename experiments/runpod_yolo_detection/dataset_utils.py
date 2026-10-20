"""Validation and conversion helpers for the isolated RunPod detector experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence


def parse_segmentation_row(line: str, source: str = "<memory>") -> list[tuple[float, float]]:
    fields = line.split()
    if len(fields) < 7 or fields[0] != "0" or (len(fields) - 1) % 2:
        raise ValueError(f"Expected one class-0 polygon with at least 3 points in {source}")
    values = [float(value) for value in fields[1:]]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Non-finite or out-of-range normalized coordinates in {source}")
    points = list(zip(values[::2], values[1::2]))
    validate_polygon(points, source)
    return points


def validate_polygon(points: Sequence[tuple[float, float]], source: str = "<memory>") -> None:
    if len(points) < 3 or len(set(points)) < 3:
        raise ValueError(f"Degenerate polygon in {source}")
    area2 = sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )
    if abs(area2) <= 1e-12:
        raise ValueError(f"Zero-area polygon in {source}")

    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i or j == (i + 1) % count or (j + 1) % count == i:
                continue
            c, d = points[j], points[(j + 1) % count]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"Self-intersecting polygon in {source}")


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-12
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    eps = 1e-12
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if ((ab_c > eps and ab_d < -eps) or (ab_c < -eps and ab_d > eps)) and (
        (cd_a > eps and cd_b < -eps) or (cd_a < -eps and cd_b > eps)
    ):
        return True
    return (
        (abs(ab_c) <= eps and _on_segment(a, b, c))
        or (abs(ab_d) <= eps and _on_segment(a, b, d))
        or (abs(cd_a) <= eps and _on_segment(c, d, a))
        or (abs(cd_b) <= eps and _on_segment(c, d, b))
    )


def canonical_polygon(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    rounded = [(round(x, 10), round(y, 10)) for x, y in points]
    variants = []
    for sequence in (rounded, list(reversed(rounded))):
        variants.extend(tuple(sequence[i:] + sequence[:i]) for i in range(len(sequence)))
    return min(variants)


def polygon_to_detection(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Convert normalized polygon vertices to YOLO cx, cy, width, height."""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x1, x2 = min(x_values), max(x_values)
    y1, y2 = min(y_values), max(y_values)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Polygon has a zero-width or zero-height detection box")
    return (x1 + x2) / 2, (y1 + y2) / 2, width, height


def convert_segmentation_text(text: str, source: str = "<memory>") -> tuple[str, int]:
    output: list[str] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        location = f"{source}:{line_no}"
        points = parse_segmentation_row(line, location)
        canonical = canonical_polygon(points)
        if canonical in seen:
            raise ValueError(f"Exact duplicate product polygon in {location}")
        seen.add(canonical)
        box = polygon_to_detection(points)
        output.append("0 " + " ".join(f"{value:.8f}" for value in box))
    return "\n".join(output) + ("\n" if output else ""), len(output)
