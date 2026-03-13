"""Shared data models for benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class MachineProfile:
    cpu_count: int
    cpu_model: str
    mem_total_gb: float
    mem_available_gb: float


@dataclass
class BenchmarkPlan:
    task: str
    data_rows: int
    data_features: int
    n_folds: int
    n_variants_scale: int
    n_variants_filter: int
    n_variants_reduction: int
    n_variants_model: int
    combinations_total: int
    combinations_skipped: int
    combinations_runnable: int
    estimated_fold_evaluations: int
    n_jobs_configured: int
    n_jobs_effective: int
    inner_parallelism_guard: bool
    output_file: Path
    latex_enabled: bool
    latex_dir: Path
    estimated_seconds_low: float
    estimated_seconds_mid: float
    estimated_seconds_high: float


@dataclass
class StepVariant:
    """One configured variant for a pipeline step."""

    key: str
    params: dict[str, Any]
    estimator: Any


@dataclass
class BenchmarkRuntimeContext:
    config_path: Path
    task: str
    metrics: list[str]
    primary_metric: str
    configured_n_jobs: int
    effective_n_jobs: int
    inner_parallelism_guard: bool
    X: pd.DataFrame
    y: pd.Series
    splits: list[tuple[Any, Any]]
    scorers: dict[str, Any]
    all_combinations: list[tuple[StepVariant, StepVariant, StepVariant, StepVariant]]
    output_file: Path
    latex_enabled: bool
    latex_dir: Path


@dataclass
class BenchmarkExecutionResult:
    detailed_rows: list[dict[str, Any]]
    failed_rows: list[dict[str, Any]]
    experiment_id: int
    inner_jobs_forced_to_one: int
