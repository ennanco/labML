# General system
import os
import argparse
import itertools
from datetime import datetime
from pathlib import Path

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
    TimeElapsedColumn)

# General Research libbraries
import numpy as np
import pandas as pd

# Machine Learning libraries
from sklearn.base import clone
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (BaggingRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor
                             )
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression

# TODO incluir Transformin & Regressorin de sklearn para poder crear las clases y usarlas en el pipeline


#Import of own libreries
from util.cli_report import (screen_header, report,
                        report_arguments, report_output)

#Define the progress bars
overall_progress = Progress(TimeElapsedColumn(),
                            BarColumn(),
                            TextColumn("{task.description}")
                           )

experiment_progress = Progress(TimeElapsedColumn(),
                        TextColumn("{task.description}")
                       )

training_progress = Progress(TextColumn("{task.description}"),
                             SpinnerColumn("dots"),
                             BarColumn()
                            )

group_progress = Group(
    Panel(Group(experiment_progress, training_progress), width=80),
    overall_progress)

#Basic treatment of the data
scales = {
    'Nothing': None,
    'Normalization': Normalizer(),
    'StandardScaler': StandardScaler()
}

# Define posible preprocessing paths
preprocesses = {
    'Nothing':None,
    'PCA': PCA(),
    'F-score': SelectPercentile(f_regression, percentile=90),
    'MI': SelectPercentile(mutual_info_regression, percentile=90)
    # TODO posibilidad de implementar un autoencoder para reducir la dimanesionalidad
}

# Define the regressors to be tested
regressors = {
    'PLS': PLSRegression(),
    'SGDRegressor': SGDRegressor(),
    'SVM': SVR(),
    'Bagging': BaggingRegressor(),
    'Boosting': GradientBoostingRegressor(),
    'RandomForest': RandomForestRegressor(),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True)
    # TODO implementar clase para una red convolucional sencilla
}


@report_arguments("Set seed")
def set_seed(seed) -> None:
    """"
        This function loads the seed or generate it, if required
        Return: None
        Post: Sets all the random seed for generators to the same value
    """
    to_seed = int(seed) if seed else int.from_bytes(os.urandom(4), byteorder='big')
    np.random.seed(to_seed)

@report_arguments(label=None)
def load_data(filepath:Path)->pd.DataFrame:
    """
        Load the data from an specific file
    """
    if filepath.is_file():
        data = pd.read_excel(filepath, index_col=0)
        return data
    else:
        raise "No file found in the specific location"


@report_arguments(label=None)
def prepare_data(data:pd.DataFrame, n_splits:int, seed=None):
    X = data.iloc[:,6:-1]
    y = data.iloc[:,5].ravel()
    X = X.replace(np.inf, np.nan)
    X = X.replace(r'INF', np.nan, regex=True)
    X = X.replace(r' ', np.nan, regex=True)
    X = X.astype('float64')
    imputer = KNNImputer(missing_values=np.nan)
    X = imputer.fit_transform(X)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = cv.split(X,y)

    return splits,(X,y)



def main():
    seed=47
    set_seed(seed=seed)
    filepath = Path('_data_/DQO.xlsx')
    n_splits = 10
    try:
        data = load_data(filepath)
    except:
        print(f"[bold red]ERROR[/bold red] Unable to load file [yellow] {filepath}[/yellow]")
        return


    origins = np.append(data['Origen'].unique(), None)
    waters = np.append(data['Tipo de agua'].unique(), None)
    for origin, water in itertools.product(origins, waters):
        if origin is None:
            origin = 'all'
            partition = data
        else:
            partition = data[data.Origen==origin]

        if water is None:
            water = 'all'
        else:
            partition = partition[partition['Tipo de agua']==water]

        partition_name = f"{origin}_{water}"
        if partition.shape[0] > 0:
            splits, partition = prepare_data(partition, n_splits=n_splits, seed=seed)
            partitions = np.zeros(len(partition[1]), dtype=int)
            for (id_partition, (train, test)) in enumerate(splits):
                partitions[list(test)] = id_partition

            np.savetxt(f'_partitions_/{partition_name}.csv', partitions, delimiter=',')
            if origin == 'all' and water == 'all':
                data.iloc[:,6:-1] = partition[0]
                data.to_excel(f'_partitions_/DQO_corrected.xlsx')



if  __name__ == '__main__':
    main()

