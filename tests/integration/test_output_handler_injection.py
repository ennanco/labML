from pathlib import Path

import pandas as pd

from labml.core.benchmark import run_benchmark
from labml.core.output_handlers import BenchmarkOutputPayload, OutputHandler


class DummyOutputHandler(OutputHandler):
    def __init__(self) -> None:
        self.calls = 0
        self.last_payload: BenchmarkOutputPayload | None = None

    def handle(self, payload: BenchmarkOutputPayload) -> list[Path]:
        self.calls += 1
        self.last_payload = payload
        return []


def test_run_benchmark_accepts_injected_output_handler(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    benchmark_cfg = tmp_path / "benchmark_reg_custom_handler.toml"
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
        'file = "results_not_written.xlsx"\n'
        "latex = true\n",
        encoding="utf-8",
    )

    handler = DummyOutputHandler()
    run_benchmark(benchmark_cfg, task="regression", output_handler=handler)

    assert handler.calls == 1
    assert handler.last_payload is not None
    assert not (tmp_path / "results_not_written.xlsx").exists()
    assert "n_jobs_effective" in set(handler.last_payload.metadata_df["key"])
