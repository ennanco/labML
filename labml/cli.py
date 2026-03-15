"""Main CLI entrypoint for labML."""

from pathlib import Path

import typer

from .core.benchmark import inspect_benchmark, run_benchmark
from .core.prepare import run_prepare

app = typer.Typer(help="Config-driven ML experimentation tool.")


@app.command(name="prepare")
def cmd_prepare(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to prepare TOML config"
    ),
) -> None:
    """Run the data preparation stage and create reusable artifacts."""
    run_prepare(config)


@app.command(name="benchmark-regression")
def cmd_benchmark_regression(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to benchmark TOML config"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show execution plan without training",
    ),
) -> None:
    """Run configured regression benchmark combinations."""
    run_benchmark(config, task="regression", dry_run=dry_run)


@app.command(name="benchmark-classification")
def cmd_benchmark_classification(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to benchmark TOML config"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show execution plan without training",
    ),
) -> None:
    """Run configured classification benchmark combinations."""
    run_benchmark(config, task="classification", dry_run=dry_run)


@app.command(name="inspect-config")
def cmd_inspect_config(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to benchmark TOML config"
    ),
    task: str = typer.Option(
        ..., "--task", "-t", help="Benchmark task: regression or classification"
    ),
) -> None:
    """Inspect benchmark config and estimate execution cost."""
    normalized_task = task.strip().lower()
    if normalized_task not in {"regression", "classification"}:
        raise typer.BadParameter("--task must be 'regression' or 'classification'")
    inspect_benchmark(config, task=normalized_task)


if __name__ == "__main__":
    app()
