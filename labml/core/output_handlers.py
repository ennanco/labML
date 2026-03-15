"""Output handlers for benchmark results (DI-friendly strategies)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


LATEX_DECIMALS = 4


@dataclass
class BenchmarkOutputPayload:
    detailed_df: pd.DataFrame
    summary_df: pd.DataFrame
    ranking_df: pd.DataFrame
    failed_df: pd.DataFrame
    metadata_df: pd.DataFrame
    output_file: Path
    primary_metric: str
    metrics: list[str]
    group_cols: list[str]


class OutputHandler(Protocol):
    def handle(self, payload: BenchmarkOutputPayload) -> list[Path]:
        """Persist benchmark output and return written paths."""


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.{LATEX_DECIMALS}f} +- {std_value:.{LATEX_DECIMALS}f}"


def _build_summary_latex_table(
    summary: pd.DataFrame, group_cols: list[str], metrics: list[str]
) -> pd.DataFrame:
    table = summary[group_cols].copy()
    for metric in metrics:
        table[metric] = summary.apply(
            lambda row: _format_mean_std(
                float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])
            ),
            axis=1,
        )
    return table


def _write_latex_tabular(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.style.hide(axis="index").to_latex()
    output_path.write_text(latex, encoding="utf-8")


class ExcelOutputHandler:
    def handle(self, payload: BenchmarkOutputPayload) -> list[Path]:
        payload.output_file.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(payload.output_file) as writer:
            payload.detailed_df.to_excel(writer, sheet_name="detailed", index=False)
            payload.summary_df.to_excel(writer, sheet_name="summary", index=False)
            payload.ranking_df.to_excel(writer, sheet_name="ranking", index=False)
            payload.failed_df.to_excel(writer, sheet_name="failed", index=False)
            payload.metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        return [payload.output_file]


class LatexOutputHandler:
    def __init__(self, latex_dir: Path):
        self.latex_dir = latex_dir

    def handle(self, payload: BenchmarkOutputPayload) -> list[Path]:
        ranking_tex_file = self.latex_dir / f"{payload.output_file.stem}_ranking.tex"
        summary_tex_file = self.latex_dir / f"{payload.output_file.stem}_summary.tex"

        ranking_latex_df = payload.ranking_df.copy()
        if payload.primary_metric in ranking_latex_df.columns:
            ranking_latex_df[payload.primary_metric] = ranking_latex_df[
                payload.primary_metric
            ].map(lambda value: f"{float(value):.{LATEX_DECIMALS}f}")

        summary_latex_df = _build_summary_latex_table(
            payload.summary_df, payload.group_cols, payload.metrics
        )

        _write_latex_tabular(ranking_latex_df, ranking_tex_file)
        _write_latex_tabular(summary_latex_df, summary_tex_file)
        return [ranking_tex_file, summary_tex_file]


class CompositeOutputHandler:
    def __init__(self, handlers: list[OutputHandler]):
        self.handlers = handlers

    def handle(self, payload: BenchmarkOutputPayload) -> list[Path]:
        written: list[Path] = []
        for handler in self.handlers:
            written.extend(handler.handle(payload))
        return written
