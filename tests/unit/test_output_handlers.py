from pathlib import Path

import pandas as pd

from labml.core.output_handlers import (
    BenchmarkOutputPayload,
    CompositeOutputHandler,
    ExcelOutputHandler,
    LatexOutputHandler,
)


def _build_payload(tmp_path: Path, output_name: str = "results.xlsx") -> BenchmarkOutputPayload:
    detailed_df = pd.DataFrame(
        {
            "experiment_id": [1, 1],
            "scale": ["none", "none"],
            "scale_params": ["{}", "{}"],
            "filter": ["none", "none"],
            "filter_params": ["{}", "{}"],
            "reduction": ["none", "none"],
            "reduction_params": ["{}", "{}"],
            "model": ["pls", "pls"],
            "model_params": ["{'n_components': 1}", "{'n_components': 1}"],
            "fold": [0, 1],
            "r2": [0.90, 0.80],
        }
    )
    summary_df = pd.DataFrame(
        {
            "experiment_id": [1],
            "scale": ["none"],
            "scale_params": ["{}"],
            "filter": ["none"],
            "filter_params": ["{}"],
            "reduction": ["none"],
            "reduction_params": ["{}"],
            "model": ["pls"],
            "model_params": ["{'n_components': 1}"],
            "r2_mean": [0.85],
            "r2_std": [0.070710678],
        }
    )
    ranking_df = pd.DataFrame(
        {
            "experiment_id": [1],
            "scale": ["none"],
            "scale_params": ["{}"],
            "filter": ["none"],
            "filter_params": ["{}"],
            "reduction": ["none"],
            "reduction_params": ["{}"],
            "model": ["pls"],
            "model_params": ["{'n_components': 1}"],
            "r2": [0.85],
        }
    )
    failed_df = pd.DataFrame(
        {
            "experiment_id": [2],
            "status": ["failed"],
            "error_type": ["model_execution"],
            "reason": ["boom"],
        }
    )
    metadata_df = pd.DataFrame(
        {
            "key": ["task", "n_jobs_effective"],
            "value": ["regression", "1"],
        }
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

    return BenchmarkOutputPayload(
        detailed_df=detailed_df,
        summary_df=summary_df,
        ranking_df=ranking_df,
        failed_df=failed_df,
        metadata_df=metadata_df,
        output_file=tmp_path / output_name,
        primary_metric="r2",
        metrics=["r2"],
        group_cols=group_cols,
    )


def test_excel_output_handler_writes_expected_sheets(tmp_path: Path) -> None:
    payload = _build_payload(tmp_path, output_name="excel_results.xlsx")

    handler = ExcelOutputHandler()
    written = handler.handle(payload)

    assert written == [payload.output_file]
    assert payload.output_file.is_file()

    workbook = pd.ExcelFile(payload.output_file)
    assert set(workbook.sheet_names) == {
        "detailed",
        "summary",
        "ranking",
        "failed",
        "metadata",
    }


def test_latex_output_handler_writes_ranking_and_summary_tables(tmp_path: Path) -> None:
    payload = _build_payload(tmp_path, output_name="latex_results.xlsx")
    latex_dir = tmp_path / "latex_tables"

    handler = LatexOutputHandler(latex_dir=latex_dir)
    written = handler.handle(payload)

    ranking_tex = latex_dir / "latex_results_ranking.tex"
    summary_tex = latex_dir / "latex_results_summary.tex"
    assert written == [ranking_tex, summary_tex]
    assert ranking_tex.is_file()
    assert summary_tex.is_file()

    ranking_content = ranking_tex.read_text(encoding="utf-8")
    summary_content = summary_tex.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in ranking_content
    assert "\\begin{tabular}" in summary_content
    assert "0.8500" in ranking_content
    assert "0.8500 +- 0.0707" in summary_content


def test_composite_output_handler_concatenates_written_paths(tmp_path: Path) -> None:
    payload = _build_payload(tmp_path, output_name="composite_results.xlsx")
    latex_dir = tmp_path / "latex_tables"

    handler = CompositeOutputHandler([ExcelOutputHandler(), LatexOutputHandler(latex_dir)])
    written = handler.handle(payload)

    assert written == [
        tmp_path / "composite_results.xlsx",
        latex_dir / "composite_results_ranking.tex",
        latex_dir / "composite_results_summary.tex",
    ]
