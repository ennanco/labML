"""Result shaping and output payload builders for benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer

from .benchmark_models import BenchmarkExecutionResult, BenchmarkRuntimeContext
from .output_handlers import (
    BenchmarkOutputPayload,
    CompositeOutputHandler,
    ExcelOutputHandler,
    LatexOutputHandler,
    OutputHandler,
)


def _build_default_output_handler(
    output_file: Path, latex_enabled: bool, latex_dir: Path
) -> OutputHandler:
    """Build default output strategy (always Excel, optional LaTeX)."""
    handlers: list[OutputHandler] = [ExcelOutputHandler()]
    if latex_enabled:
        handlers.append(LatexOutputHandler(latex_dir=latex_dir))
    return CompositeOutputHandler(handlers)


def _build_result_dataframes(
    detailed_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    metrics: list[str],
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Convert row lists into detailed/summary/ranking dataframes."""
    detailed_df = pd.DataFrame(detailed_rows)
    failed_df = pd.DataFrame(failed_rows)

    if detailed_df.empty:
        raise typer.BadParameter(
            "No successful experiments. Check the failed sheet for details."
        )

    group_cols = [
        "experiment_id",
        "scale",
        "scale_params",
        "filter",
        "filter_params",
        "reduction",
        "reduction_params",
        "model",
        "model_params",
    ]
    summary_grouped = detailed_df.groupby(group_cols)[metrics].agg(["mean", "std"])
    summary = cast(pd.DataFrame, cast(Any, summary_grouped).reset_index())
    summary.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    ranking_df = cast(
        pd.DataFrame,
        detailed_df.groupby(group_cols, as_index=False)[primary_metric].mean(),
    )
    ranking = cast(
        pd.DataFrame,
        ranking_df.sort_values(by=primary_metric, ascending=False),
    )
    return detailed_df, failed_df, summary, ranking, group_cols


def _build_metadata_df(
    task: str,
    config_path: Path,
    primary_metric: str,
    metrics: list[str],
    configured_n_jobs: int,
    effective_n_jobs: int,
    inner_parallelism_guard: bool,
    inner_jobs_forced_to_one: int,
    latex_enabled: bool,
    experiment_id: int,
    detailed_df: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build metadata dataframe persisted in the report workbook."""
    metadata_rows = [
        {"key": "task", "value": task},
        {"key": "config_path", "value": str(config_path)},
        {"key": "primary_metric", "value": primary_metric},
        {"key": "metrics", "value": ",".join(metrics)},
        {"key": "n_jobs_configured", "value": str(configured_n_jobs)},
        {"key": "n_jobs_effective", "value": str(effective_n_jobs)},
        {"key": "inner_parallelism_guard", "value": str(inner_parallelism_guard)},
        {"key": "inner_jobs_forced_to_one", "value": str(inner_jobs_forced_to_one)},
        {"key": "latex_enabled", "value": str(latex_enabled)},
        {"key": "n_experiments", "value": str(experiment_id)},
        {"key": "n_success", "value": str(detailed_df["experiment_id"].nunique())},
        {"key": "n_failed_or_skipped", "value": str(len(failed_df))},
    ]
    return pd.DataFrame(metadata_rows)


def _build_output_payload(
    context: BenchmarkRuntimeContext,
    execution_result: BenchmarkExecutionResult,
) -> BenchmarkOutputPayload:
    """Assemble all final report tables into output handler payload."""
    detailed_df, failed_df, summary, ranking, group_cols = _build_result_dataframes(
        detailed_rows=execution_result.detailed_rows,
        failed_rows=execution_result.failed_rows,
        metrics=context.metrics,
        primary_metric=context.primary_metric,
    )

    metadata_df = _build_metadata_df(
        task=context.task,
        config_path=context.config_path,
        primary_metric=context.primary_metric,
        metrics=context.metrics,
        configured_n_jobs=context.configured_n_jobs,
        effective_n_jobs=context.effective_n_jobs,
        inner_parallelism_guard=context.inner_parallelism_guard,
        inner_jobs_forced_to_one=execution_result.inner_jobs_forced_to_one,
        latex_enabled=context.latex_enabled,
        experiment_id=execution_result.experiment_id,
        detailed_df=detailed_df,
        failed_df=failed_df,
    )

    return BenchmarkOutputPayload(
        detailed_df=detailed_df,
        summary_df=summary,
        ranking_df=ranking,
        failed_df=failed_df,
        metadata_df=metadata_df,
        output_file=context.output_file,
        primary_metric=context.primary_metric,
        metrics=context.metrics,
        group_cols=group_cols,
    )
