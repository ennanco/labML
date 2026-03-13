"""Configuration parsing and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import tomllib


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return a dictionary."""
    if not path.is_file():
        raise typer.BadParameter(f"Config file not found: {path}")
    with path.open("rb") as handle:
        try:
            return tomllib.load(handle)
        except Exception as exc:  # pragma: no cover - parser emits dynamic errors
            raise typer.BadParameter(f"Invalid TOML in {path}: {exc}") from exc


def require_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract a required section from a loaded configuration."""
    section = config.get(key)
    if not isinstance(section, dict):
        raise typer.BadParameter(f"Missing required section [{key}]")
    return section


def require_value(section: dict[str, Any], key: str, section_name: str) -> Any:
    """Extract a required value from a section."""
    value = section.get(key)
    if value is None:
        raise typer.BadParameter(f"Missing required key '{key}' in [{section_name}]")
    return value
