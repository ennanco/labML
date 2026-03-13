import pytest
import typer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

from labml.core.benchmark_context import _resolve_n_jobs
from labml.core.benchmark_engine import _guard_variant_inner_n_jobs
from labml.core.benchmark_models import StepVariant
from labml.core.benchmark_plan import _parallel_speedup, _seconds_to_hms


def test_resolve_n_jobs_accepts_all_cores() -> None:
    assert _resolve_n_jobs(-1) >= 1


def test_resolve_n_jobs_accepts_positive_values() -> None:
    assert _resolve_n_jobs(1) == 1
    assert _resolve_n_jobs(3) == 3


@pytest.mark.parametrize("value", [0, -2, -5])
def test_resolve_n_jobs_rejects_invalid_values(value: int) -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_n_jobs(value)


def test_inner_parallelism_guard_forces_n_jobs_to_one_when_outer_parallel() -> None:
    variant = StepVariant(
        key="rf",
        params={},
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    )
    guarded, forced = _guard_variant_inner_n_jobs(variant, outer_n_jobs=2)
    assert forced == 1
    assert guarded.estimator.get_params()["n_jobs"] == 1


def test_inner_parallelism_guard_respects_explicit_model_n_jobs() -> None:
    variant = StepVariant(
        key="rf",
        params={"n_jobs": 4},
        estimator=RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=4),
    )
    guarded, forced = _guard_variant_inner_n_jobs(variant, outer_n_jobs=2)
    assert forced == 0
    assert guarded.estimator.get_params()["n_jobs"] == 4


def test_inner_parallelism_guard_does_not_touch_models_without_n_jobs() -> None:
    variant = StepVariant(
        key="pls",
        params={},
        estimator=PLSRegression(),
    )
    guarded, forced = _guard_variant_inner_n_jobs(variant, outer_n_jobs=2)
    assert forced == 0
    assert guarded.estimator.get_params()["max_iter"] == 500


def test_inner_parallelism_guard_disabled_when_outer_is_single_thread() -> None:
    variant = StepVariant(
        key="rf",
        params={},
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    )
    guarded, forced = _guard_variant_inner_n_jobs(variant, outer_n_jobs=1)
    assert forced == 0
    assert guarded.estimator.get_params()["n_jobs"] is None


def test_seconds_to_hms_formats_expected_values() -> None:
    assert _seconds_to_hms(0) == "00:00:00"
    assert _seconds_to_hms(5.2) == "00:00:05"
    assert _seconds_to_hms(3661) == "01:01:01"


def test_parallel_speedup_is_non_decreasing() -> None:
    assert _parallel_speedup(1) == 1.0
    assert _parallel_speedup(2) > _parallel_speedup(1)
    assert _parallel_speedup(8) > _parallel_speedup(2)
