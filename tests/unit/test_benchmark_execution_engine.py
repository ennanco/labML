from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import get_scorer

from labml.core.benchmark_engine import _execute_combinations_with_reporter
from labml.core.benchmark_models import StepVariant
from labml.core.benchmark_progress import ProgressReporter


class FailingRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FailingRegressor":
        _ = (X, y)
        raise RuntimeError("boom")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _ = X
        return np.zeros(1)


@dataclass
class SpyReporter(ProgressReporter):
    starts: list[tuple[int, int]] = field(default_factory=list)
    started_combinations: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)

    def __enter__(self) -> "SpyReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        _ = (exc_type, exc, tb)
        return None

    def start(self, total_experiments: int, total_folds: int) -> None:
        self.starts.append((total_experiments, total_folds))

    def start_combination(self, combo_label: str, total_folds: int) -> None:
        _ = total_folds
        self.started_combinations.append(combo_label)

    def on_fold_running(
        self, combo_label: str, fold_idx: int, total_folds: int
    ) -> None:
        _ = (combo_label, fold_idx, total_folds)

    def on_parallel_start(self, combo_label: str, workers: int) -> None:
        _ = (combo_label, workers)

    def on_fold_progress(self, completed: int, total_folds: int) -> None:
        _ = (completed, total_folds)

    def on_combination_skipped(self, combo_label: str) -> None:
        self.skipped.append(combo_label)

    def on_combination_completed(self, combo_label: str) -> None:
        self.completed.append(combo_label)

    def on_combination_failed(self, combo_label: str) -> None:
        self.failed.append(combo_label)

    def finish_combination(self, completed_experiments: int) -> None:
        self.finished.append(completed_experiments)


def test_execute_combinations_engine_handles_completed_skipped_and_failed() -> None:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([3.0, 5.0, 7.0, 9.0])
    splits = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([2, 3]), np.array([0, 1])),
    ]

    scale_none = StepVariant(key="none", params={}, estimator=None)
    filter_none = StepVariant(key="none", params={}, estimator=None)
    reduction_none = StepVariant(key="none", params={}, estimator=None)
    reduction_nmf = StepVariant(
        key="nmf", params={"n_components": 1}, estimator=object()
    )

    model_ok = StepVariant(
        key="pls",
        params={"n_components": 1},
        estimator=PLSRegression(n_components=1),
    )
    model_fail = StepVariant(key="fail", params={}, estimator=FailingRegressor())

    all_combinations = [
        (scale_none, filter_none, reduction_none, model_ok),
        (
            StepVariant(key="std", params={}, estimator=object()),
            filter_none,
            reduction_nmf,
            model_ok,
        ),
        (scale_none, filter_none, reduction_none, model_fail),
    ]
    scorers = {"r2": get_scorer("r2")}
    reporter = SpyReporter()

    result = _execute_combinations_with_reporter(
        X=X,
        y=y,
        splits=splits,
        all_combinations=all_combinations,
        scorers=scorers,
        effective_n_jobs=1,
        reporter=reporter,
    )

    assert result.experiment_id == 3
    assert len(result.detailed_rows) == len(splits)
    assert len(result.failed_rows) == 2
    assert any(row["status"] == "skipped" for row in result.failed_rows)
    assert any(row["status"] == "failed" for row in result.failed_rows)
    assert any(row["error_type"] == "incompatible_combo" for row in result.failed_rows)
    assert any(row["error_type"] == "model_execution" for row in result.failed_rows)
    assert reporter.starts == [(3, 2)]
    assert len(reporter.completed) == 1
    assert len(reporter.skipped) == 1
    assert len(reporter.failed) == 1
    assert reporter.finished == [1, 2, 3]


def test_execute_combinations_engine_reraises_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([3.0, 5.0, 7.0, 9.0])
    splits = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([2, 3]), np.array([0, 1])),
    ]

    scale_none = StepVariant(key="none", params={}, estimator=None)
    filter_none = StepVariant(key="none", params={}, estimator=None)
    reduction_none = StepVariant(key="none", params={}, estimator=None)
    model_ok = StepVariant(
        key="pls",
        params={"n_components": 1},
        estimator=PLSRegression(n_components=1),
    )

    all_combinations = [(scale_none, filter_none, reduction_none, model_ok)]
    scorers = {"r2": get_scorer("r2")}
    reporter = SpyReporter()

    def _boom(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise RuntimeError("internal boom")

    monkeypatch.setattr("labml.core.benchmark_engine._build_pipeline", _boom)

    with pytest.raises(RuntimeError, match="internal boom"):
        _execute_combinations_with_reporter(
            X=X,
            y=y,
            splits=splits,
            all_combinations=all_combinations,
            scorers=scorers,
            effective_n_jobs=1,
            reporter=reporter,
        )
