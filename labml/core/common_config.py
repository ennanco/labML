"""Shared configuration helpers used across workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .partitioning import PartitionConfig


def resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_partition(section: dict[str, Any]) -> PartitionConfig:
    return PartitionConfig(
        mode=str(section.get("mode", "random")),
        n_splits=int(section.get("n_splits", 10)),
        shuffle=bool(section.get("shuffle", True)),
        random_state=section.get("random_state", 42),
        group_column=section.get("group_column"),
    )
