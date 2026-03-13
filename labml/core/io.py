"""Input/output helpers for datasets, folds, metadata and hook loading."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import typer


def load_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Load tabular data from csv, xlsx or parquet."""
    suffix = path.suffix.lower()
    if not path.is_file():
        raise typer.BadParameter(f"Data file not found: {path}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise typer.BadParameter(f"Unsupported data format: {path.suffix}")


def save_artifacts(
    output_dir: Path,
    data: pd.DataFrame,
    folds: pd.Series,
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Persist prepared artifacts in a stable machine-portable format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.parquet"
    folds_path = output_dir / "folds.csv"
    metadata_path = output_dir / "metadata.json"

    data.to_parquet(data_path, index=False)
    fold_df = pd.DataFrame({"row_id": range(len(folds)), "fold_id": folds.to_numpy()})
    fold_df.to_csv(folds_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_path, folds_path, metadata_path


def load_folds(path: Path, expected_rows: int) -> pd.Series:
    """Load fold assignments from CSV with row_id/fold_id or a fold_id column."""
    if not path.is_file():
        raise typer.BadParameter(f"Folds file not found: {path}")
    folds_df = pd.read_csv(path)
    if "fold_id" not in folds_df.columns:
        if folds_df.shape[1] == 1:
            folds_df.columns = ["fold_id"]
        else:
            raise typer.BadParameter("Folds CSV must include 'fold_id' column")

    if "row_id" in folds_df.columns:
        folds_df = folds_df.sort_values("row_id")

    fold_series = folds_df["fold_id"].astype(int).reset_index(drop=True)
    if len(fold_series) != expected_rows:
        raise typer.BadParameter(
            f"Folds length ({len(fold_series)}) does not match data rows ({expected_rows})"
        )
    return fold_series


def load_metadata(path: Path | None) -> dict[str, Any]:
    """Load optional metadata JSON, returning an empty dictionary if missing."""
    if path is None:
        return {}
    if not path.is_file():
        raise typer.BadParameter(f"Metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_transform_hook(
    module_path: Path, function_name: str
) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    """Load a user preprocessing function from a Python file."""
    if not module_path.is_file():
        raise typer.BadParameter(f"Hook file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("labml_user_hook", module_path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Unable to import hook module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    transform = getattr(module, function_name, None)
    if transform is None or not callable(transform):
        raise typer.BadParameter(
            f"Hook function '{function_name}' not found in {module_path}"
        )
    return transform
