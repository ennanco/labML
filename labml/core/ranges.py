"""Utilities to expand scalar, list and range parameter specs."""

from __future__ import annotations

from itertools import product
from typing import Any


def _parse_range(spec: str) -> list[float]:
    """Parse an inclusive range formatted as 'start:step:stop'."""
    tokens = [chunk.strip() for chunk in spec.split(":")]
    if len(tokens) != 3:
        raise ValueError(f"Invalid range '{spec}', expected start:step:stop")
    start, step, stop = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
    if step == 0:
        raise ValueError(f"Invalid range '{spec}', step cannot be zero")

    values: list[float] = []
    tolerance = abs(step) * 1e-9 + 1e-12
    current = start

    if step > 0:
        while current <= stop + tolerance:
            values.append(round(current, 12))
            current += step
    else:
        while current >= stop - tolerance:
            values.append(round(current, 12))
            current += step

    return values


def expand_value(value: Any) -> list[Any]:
    """Expand one parameter value into a list of possible values."""
    if isinstance(value, str) and value.count(":") == 2:
        return _parse_range(value)
    if isinstance(value, list):
        expanded: list[Any] = []
        for item in value:
            if isinstance(item, str) and item.count(":") == 2:
                expanded.extend(_parse_range(item))
            else:
                expanded.append(item)
        return expanded
    return [value]


def expand_param_grid(params: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Expand a dictionary of params into cartesian combinations."""
    if not params:
        return [{}]

    keys = list(params.keys())
    all_values = [expand_value(params[key]) for key in keys]

    combinations: list[dict[str, Any]] = []
    for values in product(*all_values):
        current = dict(zip(keys, values, strict=True))
        for key, value in current.items():
            if isinstance(value, float) and value.is_integer():
                current[key] = int(value)
        combinations.append(current)
    return combinations
