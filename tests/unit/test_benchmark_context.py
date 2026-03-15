from pathlib import Path

import pandas as pd
import pytest
import typer

from labml.core.benchmark_context import (
    _build_pipeline,
    _build_runtime_context,
    _build_variants,
    _can_run_nmf,
    _expand_search,
    _prepare_input,
    _resolve_metrics,
)
from labml.core.benchmark_models import StepVariant
from labml.core.registry import REDUCTION_REGISTRY, SCALE_REGISTRY


def _write_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    data_path = tmp_path / name
    pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    ).to_csv(data_path, index=False)
    return data_path


def _base_config(data_path: Path) -> dict[str, object]:
    return {
        "input": {
            "source": "external",
            "data_path": data_path.name,
            "target": "target",
            "features": ["f1", "f2"],
        },
        "partition": {
            "mode": "random",
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
        },
        "evaluation": {
            "metrics": ["r2"],
            "primary_metric": "r2",
            "n_jobs": 1,
        },
        "search": {
            "scale": {"enabled": ["none"]},
            "filter": {"enabled": ["none"]},
            "reduction": {"enabled": ["none"]},
            "model": {"enabled": ["pls"], "params": {"pls": {"n_components": [1]}}},
        },
        "output": {"file": "results.xlsx"},
    }


def test_resolve_metrics_uses_defaults_and_includes_primary() -> None:
    metrics, primary = _resolve_metrics("regression", config={"evaluation": {}})
    assert primary == "neg_root_mean_squared_error"
    assert primary in metrics

    metrics, primary = _resolve_metrics(
        "regression",
        config={"evaluation": {"metrics": ["r2"], "primary_metric": "neg_mean_squared_error"}},
    )
    assert primary == "neg_mean_squared_error"
    assert metrics[0] == "neg_mean_squared_error"


def test_resolve_metrics_rejects_non_list_metrics() -> None:
    with pytest.raises(typer.BadParameter, match=r"\[evaluation\]\.metrics must be a list"):
        _resolve_metrics("regression", config={"evaluation": {"metrics": "r2"}})


def test_build_variants_rejects_unknown_technique() -> None:
    with pytest.raises(typer.BadParameter, match="Unknown scale technique"):
        _build_variants(
            enabled=["missing"],
            params_cfg={},
            registry=SCALE_REGISTRY,
            label="scale",
        )


def test_build_variants_expands_parameter_grid() -> None:
    variants = _build_variants(
        enabled=["pca"],
        params_cfg={"pca": {"n_components": [1, 2]}},
        registry=REDUCTION_REGISTRY,
        label="reduction",
    )
    assert len(variants) == 2
    assert {variant.params["n_components"] for variant in variants} == {1, 2}


def test_build_pipeline_skips_none_steps() -> None:
    scale = StepVariant(key="none", params={}, estimator=None)
    filt = StepVariant(key="none", params={}, estimator=None)
    reduction = StepVariant(key="none", params={}, estimator=None)
    model = StepVariant(key="none", params={}, estimator=object())
    pipeline = _build_pipeline(scale, filt, reduction, model)
    assert list(pipeline.named_steps.keys()) == ["model"]


def test_prepare_input_supports_implicit_feature_list(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path)
    config = _base_config(data_path)
    input_cfg = config["input"]
    assert isinstance(input_cfg, dict)
    input_cfg.pop("features")

    X, y, fold_ids, feature_list, metadata = _prepare_input(
        config=config,
        task="regression",
        base_dir=tmp_path,
    )

    assert list(X.columns) == ["f1", "f2"]
    assert y.name == "target"
    assert len(fold_ids) == len(X)
    assert feature_list == ["f1", "f2"]
    assert metadata == {}


def test_prepare_input_rejects_invalid_source_and_missing_columns(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path)
    config = _base_config(data_path)

    input_cfg = config["input"]
    assert isinstance(input_cfg, dict)
    input_cfg["source"] = "invalid"
    with pytest.raises(typer.BadParameter, match=r"\[input\]\.source"):
        _prepare_input(config=config, task="regression", base_dir=tmp_path)

    input_cfg["source"] = "external"
    input_cfg["target"] = "missing"
    with pytest.raises(typer.BadParameter, match="Target must be set"):
        _prepare_input(config=config, task="regression", base_dir=tmp_path)


def test_prepare_input_rejects_invalid_features_type(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path)
    config = _base_config(data_path)
    input_cfg = config["input"]
    assert isinstance(input_cfg, dict)
    input_cfg["features"] = "f1"

    with pytest.raises(
        typer.BadParameter,
        match=r"\[input\]\.features must be a non-empty list",
    ):
        _prepare_input(config=config, task="regression", base_dir=tmp_path)


def test_expand_search_requires_model_enabled(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path)
    config = _base_config(data_path)
    search_cfg = config["search"]
    assert isinstance(search_cfg, dict)
    model_cfg = search_cfg["model"]
    assert isinstance(model_cfg, dict)
    model_cfg["enabled"] = []

    with pytest.raises(typer.BadParameter, match="At least one model"):
        _expand_search("regression", config)


def test_can_run_nmf_validates_scale_and_data_sign() -> None:
    std_scale = StepVariant(key="std", params={}, estimator=object())
    no_scale = StepVariant(key="none", params={}, estimator=None)
    nmf_reduction = StepVariant(key="nmf", params={}, estimator=object())
    no_reduction = StepVariant(key="none", params={}, estimator=None)

    ok, reason = _can_run_nmf(std_scale, nmf_reduction, pd.DataFrame({"x": [1.0]}))
    assert ok is False
    assert "standardization" in reason

    ok, reason = _can_run_nmf(
        no_scale,
        nmf_reduction,
        pd.DataFrame({"x": [-1.0, 2.0]}),
    )
    assert ok is False
    assert "negative values" in reason

    ok, reason = _can_run_nmf(no_scale, no_reduction, pd.DataFrame({"x": [1.0]}))
    assert ok is True
    assert reason == ""


def test_build_runtime_context_smoke(tmp_path: Path) -> None:
    data_path = _write_csv(tmp_path)
    cfg_path = tmp_path / "benchmark.toml"
    cfg_path.write_text(
        "[input]\n"
        'source = "external"\n'
        f'data_path = "{data_path.name}"\n'
        'target = "target"\n'
        'features = ["f1", "f2"]\n\n'
        "[partition]\n"
        'mode = "random"\n'
        "n_splits = 3\n"
        "shuffle = true\n"
        "random_state = 42\n\n"
        "[evaluation]\n"
        'metrics = ["r2"]\n'
        'primary_metric = "r2"\n'
        "n_jobs = 1\n\n"
        "[search.scale]\n"
        'enabled = ["none"]\n\n'
        "[search.filter]\n"
        'enabled = ["none"]\n\n'
        "[search.reduction]\n"
        'enabled = ["none"]\n\n'
        "[search.model]\n"
        'enabled = ["pls"]\n\n'
        "[search.model.params.pls]\n"
        "n_components = [1]\n\n"
        "[output]\n"
        'file = "results.xlsx"\n',
        encoding="utf-8",
    )

    context = _build_runtime_context(cfg_path, task="regression")
    assert context.task == "regression"
    assert context.primary_metric == "r2"
    assert context.effective_n_jobs == 1
    assert len(context.splits) == 3
    assert context.output_file == (tmp_path / "results.xlsx").resolve()
