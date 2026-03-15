from pathlib import Path

import pandas as pd

from labml.core.benchmark import run_benchmark


def test_classification_benchmark_with_group_split(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 1.1, 2.0, 2.2, 3.0, 3.2, 4.0, 4.2],
            "f2": [0.2, 0.3, 0.8, 0.9, 1.5, 1.6, 2.1, 2.2],
            "individual_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    data_path = tmp_path / "cls.csv"
    data.to_csv(data_path, index=False)

    config_path = tmp_path / "benchmark_cls.toml"
    config_path.write_text(
        "[input]\n"
        'source = "external"\n'
        f'data_path = "{data_path.name}"\n'
        'target = "label"\n'
        'features = ["f1", "f2"]\n\n'
        "[partition]\n"
        'mode = "group"\n'
        "n_splits = 4\n"
        'group_column = "individual_id"\n\n'
        "[evaluation]\n"
        'metrics = ["f1_macro", "accuracy"]\n'
        'primary_metric = "f1_macro"\n\n'
        "[search.scale]\n"
        'enabled = ["none"]\n\n'
        "[search.filter]\n"
        'enabled = ["none"]\n\n'
        "[search.reduction]\n"
        'enabled = ["none"]\n\n'
        "[search.model]\n"
        'enabled = ["logreg"]\n\n'
        "[search.model.params.logreg]\n"
        "max_iter = [500]\n\n"
        "[output]\n"
        'file = "cls_results.xlsx"\n',
        encoding="utf-8",
    )

    run_benchmark(config_path, task="classification")
    assert (tmp_path / "cls_results.xlsx").is_file()
