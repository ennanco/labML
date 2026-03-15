from pathlib import Path

import pytest
import typer

from labml.core.common_config import parse_partition, resolve_path
from labml.core.config import read_toml, require_section, require_value


def test_read_toml_reads_valid_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[input]\nsource='external'\n", encoding="utf-8")

    cfg = read_toml(cfg_path)
    assert cfg["input"]["source"] == "external"


def test_read_toml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="Config file not found"):
        read_toml(tmp_path / "missing.toml")


def test_read_toml_rejects_invalid_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[input\nsource='external'", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="Invalid TOML"):
        read_toml(cfg_path)


def test_require_section_returns_dict_and_rejects_missing() -> None:
    cfg = {"input": {"source": "external"}}
    assert require_section(cfg, "input") == {"source": "external"}

    with pytest.raises(typer.BadParameter, match=r"Missing required section \[output\]"):
        require_section(cfg, "output")


def test_require_value_returns_value_and_rejects_missing() -> None:
    section = {"target": "y"}
    assert require_value(section, "target", "dataset") == "y"

    with pytest.raises(typer.BadParameter, match="Missing required key 'task'"):
        require_value(section, "task", "dataset")


def test_resolve_path_handles_relative_and_absolute(tmp_path: Path) -> None:
    rel = resolve_path(tmp_path, "data.csv")
    assert rel == (tmp_path / "data.csv").resolve()

    abs_path = Path("/tmp/somewhere/file.csv")
    assert resolve_path(tmp_path, str(abs_path)) == abs_path


def test_parse_partition_applies_defaults_and_overrides() -> None:
    default_cfg = parse_partition({})
    assert default_cfg.mode == "random"
    assert default_cfg.n_splits == 10
    assert default_cfg.shuffle is True
    assert default_cfg.random_state == 42
    assert default_cfg.group_column is None

    custom_cfg = parse_partition(
        {
            "mode": "group",
            "n_splits": 3,
            "shuffle": False,
            "random_state": 7,
            "group_column": "individual_id",
        }
    )
    assert custom_cfg.mode == "group"
    assert custom_cfg.n_splits == 3
    assert custom_cfg.shuffle is False
    assert custom_cfg.random_state == 7
    assert custom_cfg.group_column == "individual_id"
