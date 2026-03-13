"""Progress reporter implementations for benchmark execution."""

from __future__ import annotations

from typing import Any, Protocol

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class ProgressReporter(Protocol):
    """Observer protocol for execution progress notifications."""

    # Lifecycle hooks
    def __enter__(self) -> "ProgressReporter": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None: ...

    # Run-level events
    def start(self, total_experiments: int, total_folds: int) -> None: ...

    # Combination-level events
    def start_combination(self, combo_label: str, total_folds: int) -> None: ...

    def on_fold_running(
        self, combo_label: str, fold_idx: int, total_folds: int
    ) -> None: ...

    def on_parallel_start(self, combo_label: str, workers: int) -> None: ...

    def on_fold_progress(self, completed: int, total_folds: int) -> None: ...

    def on_combination_skipped(self, combo_label: str) -> None: ...

    def on_combination_completed(self, combo_label: str) -> None: ...

    def on_combination_failed(self, combo_label: str) -> None: ...

    def finish_combination(self, completed_experiments: int) -> None: ...


class NullProgressReporter:
    """No-op reporter used for tests and headless execution."""

    def __enter__(self) -> "NullProgressReporter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None:
        return None

    def start(self, total_experiments: int, total_folds: int) -> None:
        _ = (total_experiments, total_folds)

    def start_combination(self, combo_label: str, total_folds: int) -> None:
        _ = (combo_label, total_folds)

    def on_fold_running(
        self, combo_label: str, fold_idx: int, total_folds: int
    ) -> None:
        _ = (combo_label, fold_idx, total_folds)

    def on_parallel_start(self, combo_label: str, workers: int) -> None:
        _ = (combo_label, workers)

    def on_fold_progress(self, completed: int, total_folds: int) -> None:
        _ = (completed, total_folds)

    def on_combination_skipped(self, combo_label: str) -> None:
        _ = combo_label

    def on_combination_completed(self, combo_label: str) -> None:
        _ = combo_label

    def on_combination_failed(self, combo_label: str) -> None:
        _ = combo_label

    def finish_combination(self, completed_experiments: int) -> None:
        _ = completed_experiments


def _append_fifo(items: list[Any], item: Any, limit: int) -> Any | None:
    """Append item and evict oldest one when list exceeds fixed limit."""
    items.append(item)
    if len(items) > limit:
        return items.pop(0)
    return None


class RichProgressReporter:
    """Rich terminal reporter with Overall/Current/Status/Recent panels."""

    def __init__(self) -> None:
        self.overall_progress = Progress(
            TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}")
        )
        self.fold_progress = Progress(
            TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}")
        )
        self.current_progress = Progress(
            TextColumn("{task.description}"), SpinnerColumn("dots")
        )
        self.recent_progress = Progress(TextColumn("{task.description}"))
        self.group = Group(
            Panel(self.overall_progress, title="Overall"),
            Panel(self.fold_progress, title="Current Combination"),
            Panel(self.current_progress, title="Status"),
            Panel(self.recent_progress, title="Recent Results"),
        )
        self.live = Live(self.group)
        self.overall_task: Any | None = None
        self.current_task: Any | None = None
        self.fold_task: Any | None = None
        self.current_total_folds = 0
        self.recent_task_ids: list[Any] = []
        self.recent_limit = 10

    def __enter__(self) -> "RichProgressReporter":
        """Open live rendering session for progress panels."""
        self.live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None:
        """Close live rendering session and restore terminal state."""
        return self.live.__exit__(exc_type, exc, tb)

    def start(self, total_experiments: int, total_folds: int) -> None:
        """Initialize top-level progress bar for total combinations."""
        _ = total_folds
        self.overall_task = self.overall_progress.add_task(
            "0 combinations completed", total=total_experiments
        )

    def start_combination(self, combo_label: str, total_folds: int) -> None:
        """Create status and fold tasks for the active combination."""
        self.current_total_folds = total_folds
        self.current_task = self.current_progress.add_task(f"Running: {combo_label}")
        self.fold_task = self.fold_progress.add_task(
            f"0/{total_folds} folds complete", total=total_folds
        )

    def on_fold_running(
        self, combo_label: str, fold_idx: int, total_folds: int
    ) -> None:
        """Update textual status while each fold starts in sequential mode."""
        if self.current_task is not None:
            self.current_progress.update(
                self.current_task,
                description=f"Running: {combo_label} - fold {fold_idx + 1}/{total_folds}",
            )
        if self.fold_task is not None:
            self.fold_progress.update(
                self.fold_task,
                total=total_folds,
                description=f"{fold_idx}/{total_folds} folds complete",
            )

    def on_parallel_start(self, combo_label: str, workers: int) -> None:
        """Switch status message to parallel-fold execution mode."""
        if self.current_task is not None:
            self.current_progress.update(
                self.current_task,
                description=(
                    f"Running: {combo_label} - parallel folds (workers={workers})"
                ),
            )

    def on_fold_progress(self, completed: int, total_folds: int) -> None:
        """Advance fold progress for either sequential or parallel execution."""
        if self.fold_task is not None:
            self.fold_progress.update(
                self.fold_task,
                completed=completed,
                total=total_folds,
                description=f"{completed}/{total_folds} folds complete",
            )

    def _add_recent_result(self, description: str) -> None:
        """Track latest results as bounded FIFO list shown in UI."""
        task_id = self.recent_progress.add_task(description)
        oldest_task = _append_fifo(self.recent_task_ids, task_id, self.recent_limit)
        if oldest_task is not None:
            self.recent_progress.remove_task(oldest_task)

    def on_combination_skipped(self, combo_label: str) -> None:
        """Render skipped status and register warning entry in recent history."""
        if self.fold_task is not None:
            self.fold_progress.update(
                self.fold_task,
                description="Skipped (NMF compatibility)",
                completed=self.current_total_folds,
            )
        if self.current_task is not None:
            self.current_progress.update(
                self.current_task, description=f"Skipped: {combo_label}"
            )
        self._add_recent_result(f"[yellow]⚠ {combo_label}[/yellow]")

    def on_combination_completed(self, combo_label: str) -> None:
        """Render completed status and register success entry in history."""
        if self.current_task is not None:
            self.current_progress.update(
                self.current_task, description=f"Completed: {combo_label}"
            )
        self._add_recent_result(f"[green]✅ {combo_label}[/green]")

    def on_combination_failed(self, combo_label: str) -> None:
        """Render failed status and register error entry in history."""
        if self.current_task is not None:
            self.current_progress.update(
                self.current_task, description=f"Failed: {combo_label}"
            )
        self._add_recent_result(f"[red]✖ {combo_label}[/red]")

    def finish_combination(self, completed_experiments: int) -> None:
        """Close active tasks and advance overall progress by one combination."""
        if self.overall_task is not None:
            self.overall_progress.update(
                self.overall_task,
                advance=1,
                description=f"{completed_experiments} combinations completed",
            )
        if self.fold_task is not None:
            self.fold_progress.remove_task(self.fold_task)
            self.fold_task = None
            self.current_total_folds = 0
        if self.current_task is not None:
            self.current_progress.remove_task(self.current_task)
            self.current_task = None
