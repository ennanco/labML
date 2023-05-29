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
from sklearn.model_selection import KFold, cross_validate
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

# Basic treatment of the data
scales = {
    "Nothing": None,
    "Normalization": Normalizer(),
    "StandardScaler": StandardScaler(),
}

# Define posible preprocessing paths
preprocesses = {
    "Nothing": None,
    "PCA": PCA(),
    "F-score": SelectPercentile(f_regression, percentile=90),
    "MI": SelectPercentile(mutual_info_regression, percentile=90)
    # TODO posibilidad de implementar un autoencoder para reducir la dimanesionalidad
}

# Define the regressors to be tested
regressors = {
    "PLS": PLSRegression(),
    "SGDRegressor": SGDRegressor(),
    "SVM": SVR(),
    "Bagging": BaggingRegressor(),
    "Boosting": GradientBoostingRegressor(),
    "RandomForest": RandomForestRegressor(),
    "MLP": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True)
    # TODO implementar clase para una red convolucional sencilla
}


@report_arguments(label=None)
def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the data from an specific file
    """
    if filepath.is_file():
        data = pd.read_excel(filepath, index_col=0)
        return data
    else:
        raise "No file found in the specific location"


@report_arguments(label=None)
def prepare_data(data: pd.DataFrame, n_splits: int, seed=None, index_file=None):
    X = data.iloc[:, 6:]
    y = data.iloc[:, 5].ravel()
    X = X.astype("float64")
    if index_file:
        indexes = np.loadtxt(index_file).astype("int")
        data = data.reset_index()
        splits = (
            (data[indexes != i].index, data[indexes == i].index)
            for i in np.unique(indexes)
        )
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = cv.split(X, y)

    return splits, (X, y)


def run_experiments(data, splits):
    # Definir las posibles combinaciones
    experiments = itertools.product(
        scales.keys(), preprocesses.keys(), regressors.keys()
    )

    results = pd.DataFrame(columns=["scale", "preprocess", "regressor", "R2", "RMSE"])

    experiments, test_number = itertools.tee(experiments)
    test_number = len(list(test_number))

    X = data[0]
    y = data[1]
    with Live(group_progress):
        id_overall = overall_progress.add_task("", total=test_number)
        id_train_progress = training_progress.add_task(
            "[red]Training[/red]", total=None
        )
        completed_experiments = list()

        for index, (scale, preprocess, regressor) in enumerate(experiments):
            overall_progress.update(
                id_overall,
                description=f"{index} of {test_number} Experiments Completed",
            )
            id_experiment = experiment_progress.add_task(
                f"[red]({scale}->{preprocess}->{regressor})[/red]"
            )
            pipeline = make_pipeline(
                scales[scale], preprocesses[preprocess], regressors[regressor]
            )
            pipeline = clone(pipeline)
            splits, cv = itertools.tee(splits)
            # TODO cambiar por el RandomSearchCV
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=("r2", "neg_root_mean_squared_error"),
                return_train_score=True,
                n_jobs=-1,
            )
            results.loc[len(results.index)] = [
                scale,
                preprocess,
                regressor,
                scores["test_r2"],
                scores["test_neg_root_mean_squared_error"],
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


def regression(
    datapath: str, seed: int, n_splits: int = 10, output_filename: str = None
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
                splits, partition = prepare_data(
                    partition,
                    n_splits=n_splits,
                    seed=seed,
                    index_file=str(partition_file_name),
                )
                results = run_experiments(partition, splits)
                results.to_excel(writer, f"{partition_name}_tests")
                results.groupby(["scale", "preprocess", "regressor"]).agg(
                    ["mean", "std"]
                ).to_excel(writer, f"{partition_name}")
            partition_progress.stop_task(id_partition)
            partition_progress.update(
                id_partition, description=f"[green]{partition_name}[/green]"
            )
    screen_header("Writing the report")
    report("Printing output to", results_filename)


if __name__ == "__main__":
    typer.run(regression)
