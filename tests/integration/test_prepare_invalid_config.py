from pathlib import Path

import pytest
import typer

from labml.core.prepare import run_prepare
from labml.testing_helpers import (
    base_prepare_config_text,
    write_config,
    write_group_regression_data_csv,
)


def _write_data(tmp_path: Path) -> Path:
    return write_group_regression_data_csv(tmp_path, name="data.csv")


def _write_prepare_cfg(tmp_path: Path, text: str, name: str = "prepare_invalid.toml") -> Path:
    return write_config(tmp_path, text, name=name)


def test_prepare_rejects_invalid_toml(tmp_path: Path) -> None:
    cfg_path = _write_prepare_cfg(tmp_path, "[input\npath='x'")

    with pytest.raises(typer.BadParameter, match="Invalid TOML"):
        run_prepare(cfg_path)


def test_prepare_rejects_missing_input_section(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace("[input]\n", "")
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match=r"Missing required section \[input\]"):
        run_prepare(cfg_path)


def test_prepare_rejects_missing_output_section(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).split("[output]\n", maxsplit=1)[0]
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(
        typer.BadParameter, match=r"Missing required section \[output\]"
    ):
        run_prepare(cfg_path)


def test_prepare_rejects_missing_partition_section(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace(
        "[partition]\n"
        'mode = "group"\n'
        "n_splits = 3\n"
        'group_column = "individual_id"\n\n',
        "",
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(
        typer.BadParameter, match=r"Missing required section \[partition\]"
    ):
        run_prepare(cfg_path)


def test_prepare_rejects_invalid_task_value(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace(
        'task = "regression"', 'task = "foo"'
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match=r"\[dataset\]\.task"):
        run_prepare(cfg_path)


def test_prepare_rejects_missing_target(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace('target = "target"\n', "")
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match=r"\[dataset\]\.target"):
        run_prepare(cfg_path)


def test_prepare_rejects_target_not_in_data(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace(
        'target = "target"', 'target = "missing"'
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match=r"\[dataset\]\.target"):
        run_prepare(cfg_path)


def test_prepare_rejects_group_mode_without_group_column(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace(
        'group_column = "individual_id"\n', ""
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match="Partition mode 'group'"):
        run_prepare(cfg_path)


def test_prepare_rejects_unknown_group_column(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    text = base_prepare_config_text(data_path).replace(
        'group_column = "individual_id"', 'group_column = "missing_id"'
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match="Partition mode 'group'"):
        run_prepare(cfg_path)


def test_prepare_rejects_hook_with_missing_function(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    hook_path = tmp_path / "hook.py"
    hook_path.write_text(
        "def transform(df, params):\n    return df\n", encoding="utf-8"
    )

    text = base_prepare_config_text(data_path) + (
        "\n[hook]\n" f'path = "{hook_path.name}"\n' 'function = "not_found"\n'
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match="Hook function"):
        run_prepare(cfg_path)


def test_prepare_rejects_hook_returning_non_dataframe(tmp_path: Path) -> None:
    data_path = _write_data(tmp_path)
    hook_path = tmp_path / "hook_bad_return.py"
    hook_path.write_text(
        "def transform(df, params):\n    return 123\n", encoding="utf-8"
    )

    text = base_prepare_config_text(data_path) + (
        "\n[hook]\n" f'path = "{hook_path.name}"\n' 'function = "transform"\n'
    )
    cfg_path = _write_prepare_cfg(tmp_path, text)

    with pytest.raises(typer.BadParameter, match="must return a pandas DataFrame"):
        run_prepare(cfg_path)
