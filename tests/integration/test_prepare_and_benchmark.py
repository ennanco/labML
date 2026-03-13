from pathlib import Path

import pandas as pd
import pytest

from labml.core.benchmark import run_benchmark
from labml.core.prepare import run_prepare


def test_prepare_and_regression_benchmark(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "individual_id": ["a", "a", "b", "b", "c", "c"],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    hook_path = tmp_path / "hook.py"
    hook_path.write_text(
        "import pandas as pd\n"
        "def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:\n"
        "    return df.reset_index(drop=True)\n",
        encoding="utf-8",
    )

    prepare_cfg = tmp_path / "prepare.toml"
    prepare_cfg.write_text(
        "[input]\n"
        f'path = "{data_path.name}"\n\n'
        "[hook]\n"
        f'path = "{hook_path.name}"\n'
        'function = "transform"\n\n'
        "[dataset]\n"
        'task = "regression"\n'
        'target = "target"\n\n'
        "[partition]\n"
        'mode = "group"\n'
        "n_splits = 3\n"
        'group_column = "individual_id"\n\n'
        "[output]\n"
        'dir = "prepared"\n',
        encoding="utf-8",
    )

    run_prepare(prepare_cfg)

    benchmark_cfg = tmp_path / "benchmark_reg.toml"
    benchmark_cfg.write_text(
        "[input]\n"
        'source = "prepared"\n'
        'data_path = "prepared/data.parquet"\n'
        'folds_path = "prepared/folds.csv"\n'
        'metadata_path = "prepared/metadata.json"\n'
        'target = "target"\n'
        'features = ["f1", "f2"]\n\n'
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
        'file = "results.xlsx"\n',
        encoding="utf-8",
    )

    run_benchmark(benchmark_cfg, task="regression")
    assert (tmp_path / "results.xlsx").is_file()


def test_regression_benchmark_dry_run_does_not_write_output(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    benchmark_cfg = tmp_path / "benchmark_reg_dry_run.toml"
    benchmark_cfg.write_text(
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
        "n_jobs = 2\n\n"
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
        'file = "dry_run_results.xlsx"\n',
        encoding="utf-8",
    )

    run_benchmark(benchmark_cfg, task="regression", dry_run=True)
    assert not (tmp_path / "dry_run_results.xlsx").exists()


def test_regression_benchmark_parallel_writes_output(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "parallel_data.csv"
    data.to_csv(data_path, index=False)

    benchmark_cfg = tmp_path / "benchmark_reg_parallel.toml"
    benchmark_cfg.write_text(
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
        "n_jobs = 2\n\n"
        "[search.scale]\n"
        'enabled = ["none"]\n\n'
        "[search.filter]\n"
        'enabled = ["none"]\n\n'
        "[search.reduction]\n"
        'enabled = ["none"]\n\n'
        "[search.model]\n"
        'enabled = ["rf"]\n\n'
        "[search.model.params.rf]\n"
        "n_estimators = [10]\n"
        "max_depth = [3]\n\n"
        "[output]\n"
        'file = "parallel_results.xlsx"\n',
        encoding="utf-8",
    )

    run_benchmark(benchmark_cfg, task="regression")
    output_file = tmp_path / "parallel_results.xlsx"
    assert output_file.is_file()

    metadata_df = pd.read_excel(output_file, sheet_name="metadata")
    metadata_map = dict(zip(metadata_df["key"], metadata_df["value"], strict=True))
    assert metadata_map["inner_parallelism_guard"] == "True"
    assert int(metadata_map["inner_jobs_forced_to_one"]) > 0


@pytest.mark.filterwarnings(
    "ignore:Maximum number of iterations.*:sklearn.exceptions.ConvergenceWarning"
)
def test_regression_nmf_incompatible_combo_is_reported_as_skipped(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "nmf_data.csv"
    data.to_csv(data_path, index=False)

    benchmark_cfg = tmp_path / "benchmark_reg_nmf.toml"
    benchmark_cfg.write_text(
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
        'enabled = ["none", "std"]\n\n'
        "[search.filter]\n"
        'enabled = ["none"]\n\n'
        "[search.reduction]\n"
        'enabled = ["none", "nmf"]\n\n'
        "[search.reduction.params.nmf]\n"
        "n_components = [1]\n"
        'init = ["nndsvda"]\n'
        "max_iter = [50]\n\n"
        "[search.model]\n"
        'enabled = ["pls"]\n\n'
        "[search.model.params.pls]\n"
        "n_components = [1]\n\n"
        "[output]\n"
        'file = "nmf_results.xlsx"\n',
        encoding="utf-8",
    )

    run_benchmark(benchmark_cfg, task="regression")
    failed_df = pd.read_excel(tmp_path / "nmf_results.xlsx", sheet_name="failed")
    assert (failed_df["status"] == "skipped").any()
    assert failed_df["reason"].str.contains("NMF skipped", regex=False).any()


@pytest.mark.filterwarnings(
    "ignore:np.find_common_type is deprecated.*:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:In future versions `DataFrame.to_latex` is expected.*:FutureWarning"
)
def test_regression_benchmark_generates_latex_tabular_files(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "latex_data.csv"
    data.to_csv(data_path, index=False)

    benchmark_cfg = tmp_path / "benchmark_reg_latex.toml"
    benchmark_cfg.write_text(
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
        'file = "latex_results.xlsx"\n'
        "latex = true\n"
        'latex_dir = "latex_tables"\n',
        encoding="utf-8",
    )

    run_benchmark(benchmark_cfg, task="regression")

    ranking_tex = tmp_path / "latex_tables" / "latex_results_ranking.tex"
    summary_tex = tmp_path / "latex_tables" / "latex_results_summary.tex"
    assert ranking_tex.is_file()
    assert summary_tex.is_file()

    ranking_content = ranking_tex.read_text(encoding="utf-8")
    summary_content = summary_tex.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in ranking_content
    assert "\\begin{tabular}" in summary_content
    assert "+-" in summary_content
