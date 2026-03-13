"""Execution engine for benchmark combinations and folds."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .benchmark_context import _build_pipeline, _can_run_nmf
from .benchmark_models import BenchmarkExecutionResult, StepVariant
from .benchmark_progress import NullProgressReporter, ProgressReporter


def _evaluate_fold(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: Any,
    test_idx: Any,
    fold_idx: int,
    scorers: dict[str, Any],
    combo: dict[str, Any],
) -> dict[str, Any]:
    """Train/evaluate one fold and return a single detailed result row."""
    estimator = cast(Any, clone(pipeline))
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    estimator.fit(X_train, y_train)

    row = {**combo, "fold": fold_idx}
    for metric, scorer in scorers.items():
        value = float(scorer(estimator, X_test, y_test))
        row[metric] = value
    return row


def _guard_variant_inner_n_jobs(
    variant: StepVariant, outer_n_jobs: int
) -> tuple[StepVariant, int]:
    """Avoid nested oversubscription by forcing estimator `n_jobs=1` when needed."""
    if outer_n_jobs <= 1 or variant.estimator is None:
        return variant, 0
    if "n_jobs" in variant.params:
        return variant, 0

    estimator = variant.estimator
    if not hasattr(estimator, "get_params"):
        return variant, 0

    params = estimator.get_params(deep=False)
    if "n_jobs" not in params:
        return variant, 0

    cloned_estimator = cast(Any, clone(estimator))
    if not hasattr(cloned_estimator, "set_params"):
        return variant, 0
    guarded_estimator = cloned_estimator.set_params(n_jobs=1)
    return (
        StepVariant(
            key=variant.key, params=dict(variant.params), estimator=guarded_estimator
        ),
        1,
    )


def _apply_inner_parallelism_guard(
    scale: StepVariant,
    filt: StepVariant,
    reduction: StepVariant,
    model: StepVariant,
    outer_n_jobs: int,
) -> tuple[StepVariant, StepVariant, StepVariant, StepVariant, int]:
    """Apply inner parallelism guard to all stages in one combination."""
    guarded_scale, scale_count = _guard_variant_inner_n_jobs(scale, outer_n_jobs)
    guarded_filter, filter_count = _guard_variant_inner_n_jobs(filt, outer_n_jobs)
    guarded_reduction, reduction_count = _guard_variant_inner_n_jobs(
        reduction, outer_n_jobs
    )
    guarded_model, model_count = _guard_variant_inner_n_jobs(model, outer_n_jobs)

    return (
        guarded_scale,
        guarded_filter,
        guarded_reduction,
        guarded_model,
        scale_count + filter_count + reduction_count + model_count,
    )


def _execute_combinations(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[Any, Any]],
    all_combinations: list[tuple[StepVariant, StepVariant, StepVariant, StepVariant]],
    scorers: dict[str, Any],
    effective_n_jobs: int,
) -> BenchmarkExecutionResult:
    """Run engine without UI updates (useful for tests and non-interactive paths)."""
    return _execute_combinations_with_reporter(
        X=X,
        y=y,
        splits=splits,
        all_combinations=all_combinations,
        scorers=scorers,
        effective_n_jobs=effective_n_jobs,
        reporter=NullProgressReporter(),
    )


def _execute_combinations_with_reporter(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[Any, Any]],
    all_combinations: list[tuple[StepVariant, StepVariant, StepVariant, StepVariant]],
    scorers: dict[str, Any],
    effective_n_jobs: int,
    reporter: ProgressReporter,
) -> BenchmarkExecutionResult:
    """Run all combinations/folds and emit progress events through a reporter."""
    detailed_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    total_experiments = len(all_combinations)
    experiment_id = 0
    inner_jobs_forced_to_one = 0

    with reporter:
        reporter.start(total_experiments=total_experiments, total_folds=len(splits))
        for scale, filt, reduction, model in all_combinations:
            experiment_id += 1
            (
                guarded_scale,
                guarded_filter,
                guarded_reduction,
                guarded_model,
                forced_count,
            ) = _apply_inner_parallelism_guard(
                scale,
                filt,
                reduction,
                model,
                effective_n_jobs,
            )
            inner_jobs_forced_to_one += forced_count

            can_run, reason = _can_run_nmf(guarded_scale, guarded_reduction, X)
            combo = {
                "experiment_id": experiment_id,
                "scale": guarded_scale.key,
                "scale_params": str(guarded_scale.params),
                "filter": guarded_filter.key,
                "filter_params": str(guarded_filter.params),
                "reduction": guarded_reduction.key,
                "reduction_params": str(guarded_reduction.params),
                "model": guarded_model.key,
                "model_params": str(guarded_model.params),
            }

            combo_label = (
                f"{guarded_scale.key}->{guarded_filter.key}->{guarded_reduction.key}->{guarded_model.key} "
                f"({experiment_id}/{total_experiments})"
            )
            reporter.start_combination(combo_label=combo_label, total_folds=len(splits))

            if not can_run:
                failed_rows.append({**combo, "status": "skipped", "reason": reason})
                reporter.on_combination_skipped(combo_label)
                reporter.finish_combination(completed_experiments=experiment_id)
                continue

            pipeline = _build_pipeline(
                guarded_scale, guarded_filter, guarded_reduction, guarded_model
            )
            try:
                if effective_n_jobs == 1:
                    for fold_idx, (train_idx, test_idx) in enumerate(splits):
                        reporter.on_fold_running(
                            combo_label=combo_label,
                            fold_idx=fold_idx,
                            total_folds=len(splits),
                        )
                        row = _evaluate_fold(
                            pipeline=pipeline,
                            X=X,
                            y=y,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            fold_idx=fold_idx,
                            scorers=scorers,
                            combo=combo,
                        )
                        detailed_rows.append(row)
                        reporter.on_fold_progress(
                            completed=fold_idx + 1,
                            total_folds=len(splits),
                        )
                else:
                    completed_folds = 0
                    reporter.on_parallel_start(
                        combo_label=combo_label,
                        workers=effective_n_jobs,
                    )
                    with ThreadPoolExecutor(max_workers=effective_n_jobs) as executor:
                        futures = [
                            executor.submit(
                                _evaluate_fold,
                                pipeline,
                                X,
                                y,
                                train_idx,
                                test_idx,
                                fold_idx,
                                scorers,
                                combo,
                            )
                            for fold_idx, (train_idx, test_idx) in enumerate(splits)
                        ]

                        for future in as_completed(futures):
                            row = future.result()
                            detailed_rows.append(row)
                            completed_folds += 1
                            reporter.on_fold_progress(
                                completed=completed_folds,
                                total_folds=len(splits),
                            )

                reporter.on_combination_completed(combo_label)
            except Exception as exc:
                failed_rows.append({**combo, "status": "failed", "reason": str(exc)})
                reporter.on_combination_failed(combo_label)

            reporter.finish_combination(completed_experiments=experiment_id)

    return BenchmarkExecutionResult(
        detailed_rows=detailed_rows,
        failed_rows=failed_rows,
        experiment_id=experiment_id,
        inner_jobs_forced_to_one=inner_jobs_forced_to_one,
    )
