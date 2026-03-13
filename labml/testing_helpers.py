"""Shared helper builders for integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_regression_data_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / name
    data.to_csv(data_path, index=False)
    return data_path


def write_group_regression_data_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "individual_id": ["a", "a", "b", "b", "c", "c"],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / name
    data.to_csv(data_path, index=False)
    return data_path


def base_benchmark_config_text(data_path: Path) -> str:
    return (
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
        'metrics = ["neg_root_mean_squared_error", "r2"]\n'
        'primary_metric = "neg_root_mean_squared_error"\n'
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
        'file = "invalid_test_results.xlsx"\n'
    )


def base_prepare_config_text(data_path: Path) -> str:
    return (
        "[input]\n"
        f'path = "{data_path.name}"\n\n'
        "[dataset]\n"
        'task = "regression"\n'
        'target = "target"\n\n'
        "[partition]\n"
        'mode = "group"\n'
        "n_splits = 3\n"
        'group_column = "individual_id"\n\n'
        "[output]\n"
        'dir = "prepared"\n'
    )


def write_config(tmp_path: Path, text: str, name: str) -> Path:
    cfg_path = tmp_path / name
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path
