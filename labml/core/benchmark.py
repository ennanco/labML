"""Public benchmark API.

Only `run_benchmark` and `inspect_benchmark` are part of the stable interface.
Implementation details live in `benchmark_*` internal modules.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .benchmark_context import _build_runtime_context
from .benchmark_engine import _execute_combinations_with_reporter
from .benchmark_plan import (
    _collect_benchmark_plan,
    _echo_benchmark_plan,
    inspect_benchmark,
)
from .benchmark_progress import ProgressReporter, RichProgressReporter
from .benchmark_results import _build_default_output_handler, _build_output_payload
from .output_handlers import OutputHandler


def run_benchmark(
    config_path: Path,
    task: str,
    dry_run: bool = False,
    output_handler: OutputHandler | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> None:
    """Run benchmark pipeline end-to-end for one task and one config file.

    The function orchestrates context creation, execution, result shaping, and
    output persistence while keeping UI reporting and output handlers injectable.
    """
    if dry_run:
        plan, machine = _collect_benchmark_plan(config_path, task)
        _echo_benchmark_plan(plan, machine, title="Dry run summary")
        return

    context = _build_runtime_context(config_path=config_path, task=task)
    reporter = (
        progress_reporter if progress_reporter is not None else RichProgressReporter()
    )

    execution_result = _execute_combinations_with_reporter(
        X=context.X,
        y=context.y,
        splits=context.splits,
        all_combinations=context.all_combinations,
        scorers=context.scorers,
        effective_n_jobs=context.effective_n_jobs,
        reporter=reporter,
    )

    payload = _build_output_payload(context=context, execution_result=execution_result)

    handler = (
        output_handler
        if output_handler is not None
        else _build_default_output_handler(
            context.output_file, context.latex_enabled, context.latex_dir
        )
    )
    written_paths = handler.handle(payload)

    typer.echo("Benchmark finished.")
    for path in written_paths:
        typer.echo(f"Output saved: {path}")


__all__ = ["inspect_benchmark", "run_benchmark"]
