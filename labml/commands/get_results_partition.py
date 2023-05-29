#! /usr/bin/env python3
# General system
import os
import itertools
import typer
from datetime import datetime
from pathlib import Path

# General Research libbraries
import numpy as np
import pandas as pd

# Presentation options
from rich import print
from rich.console import Group
from rich.panel import Panel
from rich.live import Live, Console
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


# Machine Learning libraries
from sklearn.base import clone
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.feature_selection import (
    SelectPercentile,
    f_regression,
    mutual_info_regression,
)

# TODO incluir Transformin & Regressorin de sklearn para poder crear las clases y usarlas en el pipeline


# Import of own libreries
from .util.cli_report import screen_header, report, report_arguments, report_output


# Define the progress bars
overall_progress = Progress(
    TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}")
)

experiment_progress = Progress(TimeElapsedColumn(), TextColumn("{task.description}"))

training_progress = Progress(
    TextColumn("{task.description}"), SpinnerColumn("dots"), BarColumn()
)
partition_progress = Progress(TimeElapsedColumn(), TextColumn("{task.description}"))

group_progress = Group(
    Panel(partition_progress, width=80, title="Partitions..."),
    Panel(
        Group(experiment_progress, training_progress), width=80, title="Experiments..."
    ),
    overall_progress,
)


DQO_all_experiments = [
    (
        "Nothing",
        "PCA",
        "RandomForest",
        make_pipeline(None, PCA(), RandomForestRegressor()),
    ),
    (
        "Normalizer",
        "PCA",
        "RandomForest",
        make_pipeline(Normalizer(), PCA(), RandomForestRegressor()),
    ),
    (
        "Normalizer",
        "PCA",
        "Boosting",
        make_pipeline(Normalizer(), PCA(), GradientBoostingRegressor()),
    ),
    (
        "StandardScaler",
        "PCA",
        "MLP",
        make_pipeline(
            StandardScaler(),
            PCA(),
            MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True),
        ),
    ),
]

DQO_RAW_experiments = [
    (
        "Nothing",
        "PCA",
        "RandomForest",
        make_pipeline(None, PCA(), RandomForestRegressor()),
    ),
    (
        "Normalizer",
        "PCA",
        "RandomForest",
        make_pipeline(Normalizer(), PCA(), RandomForestRegressor()),
    ),
    (
        "StandardScaler",
        "PCA",
        "RandomForest",
        make_pipeline(StandardScaler(), PCA(), RandomForestRegressor()),
    ),
]

DQO_Efluent_experiments = [
    (
        "Nothing",
        "PCA",
        "RandomForest",
        make_pipeline(None, PCA(), RandomForestRegressor()),
    ),
    (
        "Normalizer",
        "PCA",
        "RandomForest",
        make_pipeline(Normalizer(), PCA(), RandomForestRegressor()),
    ),
    (
        "Normalizer",
        "Nothing",
        "Bagging",
        make_pipeline(Normalizer(), None, BaggingRegressor()),
    ),
    (
        "Normalizer",
        "PCA",
        "Boosting",
        make_pipeline(Normalizer(), PCA(), GradientBoostingRegressor()),
    ),
]

SST_all_experiments = [
    (
        "Nothing",
        "Nothing",
        "MLP",
        make_pipeline(
            None,
            None,
            MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True),
        ),
    ),
    (
        "Nothing",
        "PCA",
        "MLP",
        make_pipeline(
            None,
            PCA(),
            MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True),
        ),
    ),
    (
        "StandardScaler",
        "PCA",
        "MLP",
        make_pipeline(
            StandardScaler(),
            PCA(),
            MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True),
        ),
    ),
    (
        "StandardScaler",
        "Nothing",
        "SGD",
        make_pipeline(StandardScaler(), None, SGDRegressor()),
    ),
]

SST_RAW_experiments = [
    ("Nothing", "Nothing", "PLS", make_pipeline(None, None, PLSRegression())),
    (
        "Normalizer",
        "Nothing",
        "PLS",
        make_pipeline(Normalizer(), None, PLSRegression()),
    ),
    (
        "StandardScaler",
        "Nothing",
        "PLS",
        make_pipeline(StandardScaler(), None, PLSRegression()),
    ),
    (
        "StandardScaler",
        "Nothing",
        "SGD",
        make_pipeline(StandardScaler(), None, SGDRegressor()),
    ),
]

SST_Efluent_experiments = [
    (
        "Normalizer",
        "Nothing",
        "RandomForest",
        make_pipeline(Normalizer(), None, RandomForestRegressor()),
    ),
    (
        "Normalizer",
        "MI",
        "RandomForest",
        make_pipeline(
            Normalizer(),
            SelectPercentile(mutual_info_regression, percentile=90),
            RandomForestRegressor(),
        ),
    ),
    (
        "Normalizer",
        "ANOVA",
        "Boosting",
        make_pipeline(
            Normalizer(),
            SelectPercentile(f_regression, percentile=90),
            GradientBoostingRegressor(),
        ),
    ),
    (
        "Normalizer",
        "MI",
        "PLSRegression",
        make_pipeline(
            Normalizer(),
            SelectPercentile(mutual_info_regression, percentile=90),
            PLSRegression(),
        ),
    ),
    (
        "Normalizer",
        "PCA",
        "Boosting",
        make_pipeline(Normalizer(), PCA(), GradientBoostingRegressor()),
    ),
]


@report_arguments(label=None)
def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the data from an specific file
    """
    if filepath.is_file():
        data = pd.read_excel(str(filepath), index_col=0)
        return data
    else:
        raise "No file found in the specific location"


@report_arguments(label=None)
def prepare_data(data: pd.DataFrame, split: int, seed=None, index_file=None):
    # Split the data
    data_input = data.iloc[:, 6:]
    data_output = data.iloc[:, 5].ravel()
    data_input = data_input.astype("float64")

    if index_file:
        indexes = np.loadtxt(index_file).astype("int")
        data = data.reset_index()
        train_indexes = indexes != split
        test_indexes = indexes == split

        train_input, train_output = (
            data_input[train_indexes],
            data_output[train_indexes],
        )
        test_input, test_output = data_input[test_indexes], data_output[test_indexes]
    else:
        train_input, test_input, train_output, test_output = train_test_split(
            data_input, data_output, test_size=0.1, random_state=seed
        )

    return train_input, train_output, test_input, test_output


def run_experiments(water, train_input, train_output, test_input, test_output):
    # Definir las posibles combinaciones
    experiments = {
        "all": SST_all_experiments,
        "Raw": SST_RAW_experiments,
        "Efluent": SST_Efluent_experiments,
    }

    test_number = len(experiments[water])

    results = pd.DataFrame(
        columns=["scale", "preprocess", "regressor", "Lab measure", "Prediction"]
    )

    with Live(group_progress):
        id_overall = overall_progress.add_task("", total=test_number)
        id_train_progress = training_progress.add_task(
            "[red]Training[/red]", total=None
        )
        completed_experiments = list()

        for scale, preprocess, regressor, experiment in experiments[water]:
            # Update the general progressbar with the completed experiments
            overall_progress.update(
                id_overall,
                description=f"{len(completed_experiments)} of {test_number} Experiments Completed",
            )
            # Generate the id to show the experiment that is running
            id_experiment = experiment_progress.add_task(
                f"[red]({scale}->{preprocess}->{regressor})[/red]"
            )

            # Train the experiment
            experiment.fit(train_input, train_output)
            # Test the experiment
            predictions = experiment.predict(test_input)
            results.loc[len(results.index)] = [
                scale,
                preprocess,
                regressor,
                test_output,
                predictions,
            ]
            overall_progress.update(id_overall, advance=1)
            experiment_progress.stop_task(id_experiment)
            experiment_progress.update(
                id_experiment,
                description=f"[green]({scale}->{preprocess}->{regressor}) ✅ [/green]",
            )
            completed_experiments.append(id_experiment)
            if len(completed_experiments) > 10:
                experiment_progress.update(completed_experiments.pop(0), visible=False)
            # Clear remaining lines in the panel before another run
            for id_experiment in completed_experiments:
                experiment_progress.update(id_experiment, visible=False)

        training_progress.update(id_train_progress, visible=False)
        overall_progress.update(id_overall, visible=False)

        # Plain the results making a row for each test, it takes care to pair both colunms
        results = results.apply(pd.Series.explode)

    return results


def get_results_partition(
    datapath: str, seed: int, partition: int = 0, output_filename: str = None
):
    screen_header("Setting up the Laboratory")
    filepath = Path(datapath)
    problem_name = filepath.stem
    partitions_dir = Path(filepath.parent) / "_partitions_"
    try:
        data = load_data(filepath)
    except:
        print(
            f"[bold red]ERROR[/bold red] Unable to load file [yellow] {filepath}[/yellow]"
        )
        return

    screen_header("Starting Experiments")
    results_filename = (
        output_filename
        if output_filename
        else f'{datetime.today().strftime("%Y%m%d")}_{problem_name}_results.xlsx'
    )
    print(f"Results file name {results_filename}")

    origins = np.append(data["Origen"].unique(), None)
    waters = np.append(data["Tipo de agua"].unique(), None)
    with pd.ExcelWriter(results_filename) as writer:
        for origin, water in itertools.product(origins, waters):
            if origin is None:
                origin = "all"
                partition = data
            else:
                partition = data[data.Origen == origin]

            if water is None:
                water = "all"
            else:
                partition = partition[partition["Tipo de agua"] == water]

            partition_name = f"{origin}_{water}"
            id_partition = partition_progress.add_task(f"[red]{partition_name}[/red]")
            if partition.shape[0] > 0:
                partition_file_name = (
                    partitions_dir / f"{problem_name}_{partition_name}.csv"
                )
                train_input, train_output, test_input, test_output = prepare_data(
                    partition,
                    split=0,
                    seed=seed,
                    index_file=str(partition_file_name),
                )
                results = run_experiments(
                    water, train_input, train_output, test_input, test_output
                )
                results.to_excel(writer, sheet_name=f"{partition_name}")
            partition_progress.stop_task(id_partition)
            partition_progress.update(
                id_partition, description=f"[green]{partition_name}[/green]"
            )
    screen_header("Writing the report")
    report("Printing output to", results_filename)


if __name__ == "__main__":
    typer.run(regression)
