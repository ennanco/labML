from pathlib import Path
import re

import pandas as pd
from typer.testing import CliRunner

from labml.cli import app


def test_inspect_config_outputs_time_estimate(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "inspect_data.csv"
    data.to_csv(data_path, index=False)

    cfg_path = tmp_path / "inspect.toml"
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
        'file = "inspect_results.xlsx"\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect-config",
            "--config",
            str(cfg_path),
            "--task",
            "regression",
        ],
    )

    assert result.exit_code == 0
    assert "Execution Estimate" in result.stdout
    assert re.search(r"\b\d{2}:\d{2}:\d{2}\b", result.stdout)
