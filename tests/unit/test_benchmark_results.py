from pathlib import Path

import pandas as pd
import pytest
import typer

from labml.core.benchmark_models import (
    BenchmarkExecutionResult,
    BenchmarkRuntimeContext,
)
from labml.core.benchmark_results import (
    _build_default_output_handler,
    _build_metadata_df,
    _normalize_failed_rows,
    _build_output_payload,
    _build_result_dataframes,
)
from labml.core.output_handlers import CompositeOutputHandler, LatexOutputHandler


def _sample_detailed_rows() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": 1,
            "scale": "none",
            "scale_params": "{}",
            "filter": "none",
            "filter_params": "{}",
            "reduction": "none",
            "reduction_params": "{}",
            "model": "pls",
            "model_params": "{'n_components': 1}",
            "fold": 0,
            "r2": 0.90,
        },
        {
            "experiment_id": 1,
            "scale": "none",
            "scale_params": "{}",
            "filter": "none",
            "filter_params": "{}",
            "reduction": "none",
            "reduction_params": "{}",
            "model": "pls",
            "model_params": "{'n_components': 1}",
            "fold": 1,
            "r2": 0.80,
        },
        {
            "experiment_id": 2,
            "scale": "none",
            "scale_params": "{}",
            "filter": "none",
            "filter_params": "{}",
            "reduction": "none",
            "reduction_params": "{}",
            "model": "rf",
            "model_params": "{'n_estimators': 10}",
            "fold": 0,
            "r2": 0.10,
        },
        {
            "experiment_id": 2,
            "scale": "none",
            "scale_params": "{}",
            "filter": "none",
            "filter_params": "{}",
            "reduction": "none",
            "reduction_params": "{}",
            "model": "rf",
            "model_params": "{'n_estimators': 10}",
            "fold": 1,
            "r2": 0.20,
        },
    ]


def test_build_default_output_handler_includes_latex_when_enabled() -> None:
    handler = _build_default_output_handler(
        output_file=Path("results.xlsx"),
        latex_enabled=True,
        latex_dir=Path("latex"),
    )
    assert isinstance(handler, CompositeOutputHandler)
    assert len(handler.handlers) == 2
    assert isinstance(handler.handlers[1], LatexOutputHandler)


def test_build_default_output_handler_excludes_latex_when_disabled() -> None:
    handler = _build_default_output_handler(
        output_file=Path("results.xlsx"),
        latex_enabled=False,
        latex_dir=Path("latex"),
    )
    assert isinstance(handler, CompositeOutputHandler)
    assert len(handler.handlers) == 1


def test_build_result_dataframes_returns_ordered_ranking() -> None:
    detailed_df, failed_df, summary_df, ranking_df, group_cols = (
        _build_result_dataframes(
            detailed_rows=_sample_detailed_rows(),
            failed_rows=[
                {
                    "experiment_id": 3,
                    "status": "failed",
                    "error_type": "model_execution",
                    "reason": "boom",
                }
            ],
            metrics=["r2"],
            primary_metric="r2",
        )
    )

    assert not detailed_df.empty
    assert len(failed_df) == 1
    assert "r2_mean" in summary_df.columns
    assert "r2_std" in summary_df.columns
    assert ranking_df.iloc[0]["experiment_id"] == 1
    assert ranking_df.iloc[1]["experiment_id"] == 2
    assert group_cols[0] == "experiment_id"


def test_build_result_dataframes_requires_at_least_one_successful_experiment() -> None:
    with pytest.raises(typer.BadParameter, match="No successful experiments"):
        _build_result_dataframes(
            detailed_rows=[],
            failed_rows=[{"status": "failed", "reason": "boom"}],
            metrics=["r2"],
            primary_metric="r2",
        )


def test_build_metadata_df_contains_key_runtime_metrics() -> None:
    detailed_df = pd.DataFrame({"experiment_id": [1, 1, 2]})
    failed_df = pd.DataFrame([{"status": "failed"}, {"status": "skipped"}])

    metadata_df = _build_metadata_df(
        task="regression",
        config_path=Path("cfg.toml"),
        primary_metric="r2",
        metrics=["r2"],
        configured_n_jobs=-1,
        effective_n_jobs=8,
        inner_parallelism_guard=True,
        inner_jobs_forced_to_one=3,
        latex_enabled=True,
        experiment_id=5,
        detailed_df=detailed_df,
        failed_df=failed_df,
    )

    meta = dict(zip(metadata_df["key"], metadata_df["value"], strict=True))
    assert meta["n_success"] == "2"
    assert meta["n_failed_or_skipped"] == "2"
    assert meta["n_jobs_effective"] == "8"
    assert meta["latex_enabled"] == "True"


def test_normalize_failed_rows_infers_missing_error_type() -> None:
    normalized = _normalize_failed_rows(
        [
            {"status": "skipped", "reason": "nmf"},
            {"status": "failed", "reason": "boom"},
            {"status": "unknown", "reason": "x"},
        ]
    )
    assert normalized[0]["error_type"] == "incompatible_combo"
    assert normalized[1]["error_type"] == "model_execution"
    assert normalized[2]["error_type"] == "internal"


def test_build_output_payload_assembles_all_sections(tmp_path: Path) -> None:
    context = BenchmarkRuntimeContext(
        config_path=tmp_path / "benchmark.toml",
        task="regression",
        metrics=["r2"],
        primary_metric="r2",
        configured_n_jobs=1,
        effective_n_jobs=1,
        inner_parallelism_guard=False,
        X=pd.DataFrame({"f1": [1.0, 2.0]}),
        y=pd.Series([3.0, 5.0]),
        splits=[],
        scorers={},
        all_combinations=[],
        output_file=tmp_path / "results.xlsx",
        latex_enabled=False,
        latex_dir=tmp_path,
    )
    execution_result = BenchmarkExecutionResult(
        detailed_rows=_sample_detailed_rows(),
        failed_rows=[
            {"status": "failed", "error_type": "model_execution", "reason": "boom"}
        ],
        experiment_id=2,
        inner_jobs_forced_to_one=0,
    )

    payload = _build_output_payload(context=context, execution_result=execution_result)
    assert payload.output_file == tmp_path / "results.xlsx"
    assert not payload.detailed_df.empty
    assert not payload.summary_df.empty
    assert not payload.ranking_df.empty
    assert "n_jobs_effective" in set(payload.metadata_df["key"])
