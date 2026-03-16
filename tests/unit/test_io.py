from pathlib import Path

import pandas as pd
import pytest
import typer

from labml.core.io import (
    load_folds,
    load_metadata,
    load_table,
    load_transform_hook,
    save_artifacts,
)


def test_load_table_reads_csv(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(data_path, index=False)

    table = load_table(data_path)
    assert list(table.columns) == ["a", "b"]
    assert len(table) == 2


def test_load_table_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="Data file not found"):
        load_table(tmp_path / "missing.csv")


def test_load_table_rejects_unsupported_extension(tmp_path: Path) -> None:
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("hello", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="Unsupported data format"):
        load_table(txt_path)


def test_save_artifacts_writes_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepared"
    data = pd.DataFrame({"x": [1.0, 2.0], "target": [3.0, 5.0]})
    folds = pd.Series([0, 1])
    metadata = {"task": "regression", "target": "target"}

    data_path, folds_path, metadata_path = save_artifacts(
        output_dir, data, folds, metadata
    )

    assert data_path.is_file()
    assert folds_path.is_file()
    assert metadata_path.is_file()

    folds_df = pd.read_csv(folds_path)
    assert list(folds_df.columns) == ["row_id", "fold_id"]
    assert folds_df["fold_id"].tolist() == [0, 1]


def test_load_folds_supports_single_fold_id_column(tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame({"fold_id": [2, 1, 0]}).to_csv(folds_path, index=False)

    fold_ids = load_folds(folds_path, expected_rows=3)
    assert fold_ids.tolist() == [2, 1, 0]


def test_load_folds_sorts_by_row_id_when_present(tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame({"row_id": [2, 0, 1], "fold_id": [9, 7, 8]}).to_csv(
        folds_path, index=False
    )

    fold_ids = load_folds(folds_path, expected_rows=3)
    assert fold_ids.tolist() == [7, 8, 9]


def test_load_folds_rejects_invalid_columns(tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(folds_path, index=False)
    with pytest.raises(typer.BadParameter, match="must include 'fold_id' column"):
        load_folds(folds_path, expected_rows=1)


def test_load_folds_rejects_length_mismatch(tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame({"fold_id": [0, 1]}).to_csv(folds_path, index=False)
    with pytest.raises(typer.BadParameter, match="Folds length"):
        load_folds(folds_path, expected_rows=3)


def test_load_metadata_behaviour(tmp_path: Path) -> None:
    assert load_metadata(None) == {}

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"target": "y"}', encoding="utf-8")
    assert load_metadata(metadata_path)["target"] == "y"

    with pytest.raises(typer.BadParameter, match="Metadata file not found"):
        load_metadata(tmp_path / "missing.json")


def test_load_transform_hook_happy_path(tmp_path: Path) -> None:
    hook_path = tmp_path / "hook.py"
    hook_path.write_text(
        "def transform(df, params):\n    return df.assign(extra=params.get('x', 0))\n",
        encoding="utf-8",
    )

    transform = load_transform_hook(hook_path, "transform")
    out = transform(pd.DataFrame({"a": [1]}), {"x": 3})
    assert out["extra"].tolist() == [3]


def test_load_transform_hook_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="Hook file not found"):
        load_transform_hook(tmp_path / "missing.py", "transform")


def test_load_transform_hook_rejects_missing_function(tmp_path: Path) -> None:
    hook_path = tmp_path / "hook.py"
    hook_path.write_text("def another(df, params):\n    return df\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="Hook function 'transform' not found"):
        load_transform_hook(hook_path, "transform")
