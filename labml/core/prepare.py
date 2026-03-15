"""Prepare workflow: clean data and persist reusable artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from .config import read_toml, require_section
from .common_config import parse_partition, resolve_path
from .io import load_table, load_transform_hook, save_artifacts
from .partitioning import build_fold_ids


def run_prepare(config_path: Path) -> None:
    """Execute the prepare command according to a TOML config."""
    config = read_toml(config_path)
    base_dir = config_path.parent.resolve()
    input_cfg = require_section(config, "input")
    output_cfg = require_section(config, "output")
    partition_cfg = require_section(config, "partition")

    raw_data_path = input_cfg.get("path")
    if raw_data_path is None or not str(raw_data_path).strip():
        raise typer.BadParameter("Missing [input].path")
    data_path = resolve_path(base_dir, str(raw_data_path))

    sheet = input_cfg.get("sheet")
    data = load_table(data_path, sheet=sheet)

    hook_cfg = config.get("hook")
    if isinstance(hook_cfg, dict) and hook_cfg.get("path"):
        hook_path = resolve_path(base_dir, str(hook_cfg["path"]))
        hook_fn = str(hook_cfg.get("function", "transform"))
        hook_params = hook_cfg.get("params", {})
        transform = load_transform_hook(hook_path, hook_fn)
        transformed = transform(data, hook_params)
        if not isinstance(transformed, pd.DataFrame):
            raise typer.BadParameter("Transform hook must return a pandas DataFrame")
        data = transformed

    dataset_cfg = config.get("dataset", {})
    target = dataset_cfg.get("target")
    if not target or target not in data.columns:
        raise typer.BadParameter(
            "[dataset].target is required in prepare config and must exist in data"
        )

    task = str(dataset_cfg.get("task", "regression"))
    if task not in {"regression", "classification"}:
        raise typer.BadParameter(
            "[dataset].task must be 'regression' or 'classification'"
        )

    partition = parse_partition(partition_cfg)
    fold_ids = build_fold_ids(data, y=data[target], task=task, partition=partition)

    output_dir = resolve_path(
        base_dir, str(output_cfg.get("dir", "_artifacts_/prepared/default"))
    )
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_data": str(data_path),
        "target": target,
        "task": task,
        "partition": {
            "mode": partition.mode,
            "n_splits": partition.n_splits,
            "shuffle": partition.shuffle,
            "random_state": partition.random_state,
            "group_column": partition.group_column,
        },
        "rows": len(data),
        "columns": list(data.columns),
    }

    data_file, folds_file, metadata_file = save_artifacts(
        output_dir, data, fold_ids, metadata
    )
    typer.echo(f"Prepared data saved: {data_file}")
    typer.echo(f"Prepared folds saved: {folds_file}")
    typer.echo(f"Prepared metadata saved: {metadata_file}")
