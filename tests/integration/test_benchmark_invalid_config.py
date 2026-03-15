from pathlib import Path

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from labml.core.benchmark import run_benchmark
from labml.cli import app
from tests.helpers.testing_helpers import (
    base_benchmark_config_text,
    write_config,
    write_regression_data_csv,
)


def _write_data_and_config(tmp_path: Path, config_text: str) -> Path:
    write_regression_data_csv(tmp_path, name="data.csv")
    return write_config(tmp_path, config_text, name="benchmark_invalid.toml")


def test_benchmark_rejects_invalid_n_jobs(tmp_path: Path) -> None:
    base = base_benchmark_config_text(tmp_path / "data.csv").replace(
        "n_jobs = 1", "n_jobs = 0"
    )
    cfg_path = _write_data_and_config(tmp_path, base)

    with pytest.raises(typer.BadParameter, match="n_jobs"):
        run_benchmark(cfg_path, task="regression")


def test_benchmark_rejects_unknown_model(tmp_path: Path) -> None:
    base = base_benchmark_config_text(tmp_path / "data.csv").replace(
        'enabled = ["pls"]', 'enabled = ["does_not_exist"]'
    )
    cfg_path = _write_data_and_config(tmp_path, base)

    with pytest.raises(typer.BadParameter, match="Unknown model technique"):
        run_benchmark(cfg_path, task="regression")


def test_benchmark_rejects_missing_input_data_path(tmp_path: Path) -> None:
    base = base_benchmark_config_text(tmp_path / "data.csv").replace(
        'data_path = "data.csv"\n', ""
    )
    cfg_path = _write_data_and_config(tmp_path, base)

    with pytest.raises(typer.BadParameter, match=r"Missing \[input\]\.data_path"):
        run_benchmark(cfg_path, task="regression")


@pytest.mark.parametrize("path_value", ["", "   "])
def test_benchmark_rejects_blank_input_data_path(
    tmp_path: Path, path_value: str
) -> None:
    base = base_benchmark_config_text(tmp_path / "data.csv").replace(
        'data_path = "data.csv"', f'data_path = "{path_value}"'
    )
    cfg_path = _write_data_and_config(tmp_path, base)

    with pytest.raises(typer.BadParameter, match=r"Missing \[input\]\.data_path"):
        run_benchmark(cfg_path, task="regression")


def test_benchmark_rejects_folds_length_mismatch(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 3.0, 4.0, 5.0],
            "target": [3.0, 5.0, 7.0, 9.0],
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    folds_path = tmp_path / "folds.csv"
    pd.DataFrame({"fold_id": [0, 1]}).to_csv(folds_path, index=False)

    cfg_path = tmp_path / "benchmark_invalid_folds.toml"
    cfg_path.write_text(
        (
            base_benchmark_config_text(data_path)
            .replace('source = "external"', 'source = "prepared"')
            .replace('target = "target"', 'target = "target"\nfolds_path = "folds.csv"')
        ),
        encoding="utf-8",
    )

    with pytest.raises(typer.BadParameter, match="Folds length"):
        run_benchmark(cfg_path, task="regression")


def test_benchmark_rejects_missing_output_section(tmp_path: Path) -> None:
    base = base_benchmark_config_text(tmp_path / "data.csv")
    base_without_output = base.split("[output]\n", maxsplit=1)[0]
    cfg_path = _write_data_and_config(tmp_path, base_without_output)

    with pytest.raises(
        typer.BadParameter, match=r"Missing required section \[output\]"
    ):
        run_benchmark(cfg_path, task="regression")


def test_benchmark_rejects_missing_search_section(tmp_path: Path) -> None:
    base = base_benchmark_config_text(tmp_path / "data.csv")
    base_without_search = base.replace(
        "[search.scale]\n"
        'enabled = ["none"]\n\n'
        "[search.filter]\n"
        'enabled = ["none"]\n\n'
        "[search.reduction]\n"
        'enabled = ["none"]\n\n'
        "[search.model]\n"
        'enabled = ["pls"]\n\n'
        "[search.model.params.pls]\n"
        "n_components = [1]\n\n",
        "",
    )
    cfg_path = _write_data_and_config(tmp_path, base_without_search)

    with pytest.raises(
        typer.BadParameter, match=r"Missing required section \[search\]"
    ):
        run_benchmark(cfg_path, task="regression")


def test_inspect_config_rejects_invalid_task_value() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["inspect-config", "--config", "any.toml", "--task", "invalid"],
    )

    assert result.exit_code != 0
    assert "--task must be 'regression' or 'classification'" in result.stdout


def test_benchmark_command_reports_missing_config_file() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["benchmark-regression", "--config", "this_file_does_not_exist.toml"],
    )

    assert result.exit_code != 0
    assert "Config file not found" in result.stdout
