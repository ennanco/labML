"""Context and search-space builders for benchmark workflows."""

from __future__ import annotations

from itertools import product
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from .benchmark_models import BenchmarkRuntimeContext, StepVariant
from .common_config import parse_partition, resolve_path
from .config import read_toml, require_section
from .io import load_folds, load_metadata, load_table
from .partitioning import build_fold_ids, fold_ids_to_splits
from .ranges import expand_param_grid
from .registry import (
    CLASSIFICATION_FILTER_REGISTRY,
    CLASSIFICATION_MODEL_REGISTRY,
    DEFAULT_CLASSIFICATION_METRICS,
    DEFAULT_REGRESSION_METRICS,
    FILTER_REGISTRY,
    REDUCTION_REGISTRY,
    REGRESSION_MODEL_REGISTRY,
    SCALE_REGISTRY,
)


def _resolve_metrics(task: str, config: dict[str, Any]) -> tuple[list[str], str]:
    """Resolve metrics list and guarantee that primary metric is included."""
    evaluation = config.get("evaluation", {})
    metrics = evaluation.get("metrics")
    if not metrics:
        metrics = (
            DEFAULT_REGRESSION_METRICS
            if task == "regression"
            else DEFAULT_CLASSIFICATION_METRICS
        )
    if not isinstance(metrics, list):
        raise typer.BadParameter("[evaluation].metrics must be a list")

    primary_metric = evaluation.get("primary_metric")
    if primary_metric is None:
        primary_metric = (
            "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
        )
    if primary_metric not in metrics:
        metrics = [primary_metric, *metrics]
    return metrics, primary_metric


def _build_variants(
    enabled: list[str],
    params_cfg: dict[str, Any],
    registry: dict[str, Any],
    label: str,
) -> list[StepVariant]:
    """Expand enabled techniques and parameter grids into concrete variants."""
    variants: list[StepVariant] = []
    for key in enabled:
        technique = registry.get(key)
        if technique is None:
            raise typer.BadParameter(f"Unknown {label} technique: {key}")
        param_grid = expand_param_grid(params_cfg.get(key, {}))
        for params in param_grid:
            estimator = technique.builder(**params)
            variants.append(StepVariant(key=key, params=params, estimator=estimator))
    return variants


def _build_pipeline(
    scale: StepVariant, filt: StepVariant, red: StepVariant, model: StepVariant
) -> Pipeline:
    """Build a sklearn pipeline from a concrete combination of step variants."""
    steps: list[tuple[str, Any]] = []
    if scale.estimator is not None:
        steps.append(("scale", scale.estimator))
    if filt.estimator is not None:
        steps.append(("filter", filt.estimator))
    if red.estimator is not None:
        steps.append(("reduction", red.estimator))
    steps.append(("model", model.estimator))
    return Pipeline(steps=steps)


def _prepare_input(
    config: dict[str, Any],
    task: str,
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], dict[str, Any]]:
    """Load dataset/folds/metadata and validate target/features contract."""
    input_cfg = require_section(config, "input")
    source = str(input_cfg.get("source", "prepared"))

    data_path = resolve_path(base_dir, str(input_cfg.get("data_path", "")))
    if not data_path:
        raise typer.BadParameter("Missing [input].data_path")
    data = cast(pd.DataFrame, load_table(data_path))

    metadata_path = input_cfg.get("metadata_path")
    metadata = (
        load_metadata(resolve_path(base_dir, str(metadata_path)))
        if metadata_path
        else {}
    )

    raw_target = input_cfg.get("target") or metadata.get("target")
    target = str(raw_target) if raw_target is not None else ""
    if not target or target not in data.columns:
        raise typer.BadParameter(
            "Target must be set in [input].target or available in metadata"
        )

    feature_list_raw = input_cfg.get("features")
    if feature_list_raw is None:
        feature_list = [str(col) for col in data.columns if col != target]
    elif isinstance(feature_list_raw, list):
        feature_list = [str(col) for col in feature_list_raw]
    else:
        feature_list = []
    if not feature_list:
        raise typer.BadParameter(
            "[input].features must be a non-empty list when provided"
        )

    missing = [col for col in feature_list if col not in data.columns]
    if missing:
        raise typer.BadParameter(f"Missing feature columns in data: {missing}")

    y: pd.Series = cast(pd.Series, data[target])
    X: pd.DataFrame = cast(pd.DataFrame, data[feature_list].copy())

    folds_path = input_cfg.get("folds_path")
    if folds_path:
        fold_ids = load_folds(
            resolve_path(base_dir, str(folds_path)), expected_rows=len(data)
        )
    else:
        partition_cfg = parse_partition(config.get("partition", {}))
        fold_ids = build_fold_ids(
            data,
            y=cast(pd.Series, y),
            task=task,
            partition=partition_cfg,
        )

    if source not in {"prepared", "external"}:
        raise typer.BadParameter("[input].source must be 'prepared' or 'external'")

    return X, y, fold_ids, feature_list, metadata


def _expand_search(
    task: str, config: dict[str, Any]
) -> tuple[list[StepVariant], list[StepVariant], list[StepVariant], list[StepVariant]]:
    """Read `[search]` section and return expanded variants per pipeline stage."""
    search = require_section(config, "search")

    scale_cfg = require_section(search, "scale")
    filter_cfg = require_section(search, "filter")
    reduction_cfg = require_section(search, "reduction")
    model_cfg = require_section(search, "model")

    scales = _build_variants(
        enabled=scale_cfg.get("enabled", ["none"]),
        params_cfg=scale_cfg.get("params", {}),
        registry=SCALE_REGISTRY,
        label="scale",
    )
    filters = _build_variants(
        enabled=filter_cfg.get("enabled", ["none"]),
        params_cfg=filter_cfg.get("params", {}),
        registry=FILTER_REGISTRY
        if task == "regression"
        else CLASSIFICATION_FILTER_REGISTRY,
        label="filter",
    )
    reductions = _build_variants(
        enabled=reduction_cfg.get("enabled", ["none"]),
        params_cfg=reduction_cfg.get("params", {}),
        registry=REDUCTION_REGISTRY,
        label="reduction",
    )
    models = _build_variants(
        enabled=model_cfg.get("enabled", []),
        params_cfg=model_cfg.get("params", {}),
        registry=(
            REGRESSION_MODEL_REGISTRY
            if task == "regression"
            else CLASSIFICATION_MODEL_REGISTRY
        ),
        label="model",
    )

    if not models:
        raise typer.BadParameter("At least one model must be enabled in [search.model]")
    return scales, filters, reductions, models


def _can_run_nmf(
    scale: StepVariant, reduction: StepVariant, X: pd.DataFrame
) -> tuple[bool, str]:
    """Check NMF compatibility for the current combination and data."""
    if reduction.key != "nmf":
        return True, ""
    if scale.key == "std":
        return False, "NMF skipped: standardization can create negative values"
    if (X.to_numpy() < 0).any():
        return False, "NMF skipped: input data contains negative values"
    return True, ""


def _resolve_n_jobs(configured_n_jobs: int) -> int:
    """Normalize outer parallelism value (`-1` -> available CPU cores)."""
    if configured_n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if configured_n_jobs < -1 or configured_n_jobs == 0:
        raise typer.BadParameter("[evaluation].n_jobs must be -1 or a positive integer")
    return configured_n_jobs


def _build_runtime_context(config_path: Path, task: str) -> BenchmarkRuntimeContext:
    """Build the full runtime context consumed by the execution engine."""
    config = read_toml(config_path)
    base_dir = config_path.parent.resolve()
    X, y, fold_ids, _feature_list, _metadata = _prepare_input(
        config, task=task, base_dir=base_dir
    )
    metrics, primary_metric = _resolve_metrics(task, config)
    scales, filters, reductions, models = _expand_search(task, config)

    configured_n_jobs = int(config.get("evaluation", {}).get("n_jobs", -1))
    effective_n_jobs = _resolve_n_jobs(configured_n_jobs)
    splits = list(fold_ids_to_splits(fold_ids))
    scorers = {metric: get_scorer(metric) for metric in metrics}
    all_combinations = list(product(scales, filters, reductions, models))

    output_cfg = require_section(config, "output")
    output_file = resolve_path(
        base_dir, str(output_cfg.get("file", f"benchmark_{task}.xlsx"))
    )
    latex_enabled = bool(output_cfg.get("latex", False))
    latex_dir_cfg = output_cfg.get("latex_dir")
    latex_dir = (
        resolve_path(base_dir, str(latex_dir_cfg))
        if latex_dir_cfg
        else output_file.parent
    )

    return BenchmarkRuntimeContext(
        config_path=config_path,
        task=task,
        metrics=metrics,
        primary_metric=primary_metric,
        configured_n_jobs=configured_n_jobs,
        effective_n_jobs=effective_n_jobs,
        inner_parallelism_guard=effective_n_jobs > 1,
        X=X,
        y=y,
        splits=splits,
        scorers=scorers,
        all_combinations=all_combinations,
        output_file=output_file,
        latex_enabled=latex_enabled,
        latex_dir=latex_dir,
    )
