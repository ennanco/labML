# General system
import os
import typer
import itertools
from datetime import datetime
from pathlib import Path

# Presentation options
from rich import print

# General Research libbraries
import numpy as np
import pandas as pd

# Machine Learning libraries
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer

# TODO incluir Transformin & Regressorin de sklearn para poder crear las clases y usarlas en el pipeline


# Import of own libreries
from .util.cli_report import screen_header, report, report_arguments, report_output


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
def split_data(data: pd.DataFrame, n_splits: int, seed=None):
    X = data.iloc[:, 6:-1]
    y = data.iloc[:, 5].ravel()
    X = X.astype("float64")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = cv.split(X, y)

    return splits, (X, y)


def prepare_data(
    file: str,
    n_splits: int = 10,
    seed: int = typer.Option(None, help="set the random seed for any split"),
    output_folder: str = typer.Option("_partition_", help="Output for the partitions"),
):
    filepath = Path(file)
    PROBLEM = filepath.stem

    try:
        data = load_data(filepath)
    except:
        print(
            f"[bold red]ERROR[/bold red] Unable to load file [yellow] {filepath}[/yellow]"
        )
        return

    origins = np.append(data["Origen"].unique(), None)
    waters = np.append(data["Tipo de agua"].unique(), None)

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

        partition_name = f"{PROBLEM}_{origin}_{water}"

        if partition.shape[0] > 0:
            splits, partition = split_data(partition, n_splits=n_splits, seed=seed)
            partitions = np.zeros(len(partition[1]), dtype=int)
            for id_partition, (train, test) in enumerate(splits):
                partitions[list(test)] = id_partition

            np.savetxt(
                f"{output_folder}/{partition_name}.csv", partitions, delimiter=","
            )


if __name__ == "__main__":
    typer.run(prepare_data)
