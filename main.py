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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import (BaggingRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor
                             )
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline

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
    'ICA': FastICA()
}

# Define the regressors to be tested
regressors = {
    'PLS': PLSRegression(),
#    'LogisticRegression': LogisticRegression(),
    'SVM': SVR(),
    'Bagging': BaggingRegressor(),
    'Boosting': GradientBoostingRegressor(),
    'RandomForest': RandomForestRegressor(),
    'MLP': MLPRegressor()
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
def prepare_data(filepath:Path, splits:int):
    if filepath.is_file():
        data = pd.read_csv(filepath,sep=',', index_col=0)
    else:
        raise "No file found in the specific location"
    #split using the sklearn
    cv = KFold(n_splits=splits)

    splits = cv.split(data)

    return splits, (data.iloc[:,25:], data.iloc[:,1].ravel())


def run_experiments(data, splits):
    # Definir las posibles combinaciones
    experiments = itertools.product(scales.keys(),
                                          preprocesses.keys(),
                                          regressors.keys())

    results = pd.DataFrame(columns=['scale', 'preprocess', 'regressor', 'R2', 'RMSE'])

    experiments, test_number = itertools.tee(experiments)
    test_number = len(list(test_number))

    X = data[0]
    y = data[1]
    with Live(group_progress):
        id_overall = overall_progress.add_task("", total=test_number)
        id_train_progress = training_progress.add_task("[red]Training[/red]", total=None)
        for index, (scale, preprocess, regressor) in enumerate(experiments):
            overall_progress.update(id_overall, description=f"{index} of {test_number} Experiments Completed")
            id_experiment = experiment_progress.add_task(
                f"[red]({scale}->{preprocess}->{regressor})[/red]")
            pipeline = make_pipeline(scales[scale],
                                 preprocesses[preprocess],
                                 regressors[regressor])
            pipeline = clone(pipeline)
            splits, cv = itertools.tee(splits)
            scores = cross_validate(pipeline, X, y, cv=cv,
                                scoring =('r2', 'neg_mean_squared_error'),
                                return_train_score=True)
            results.loc[len(results.index)] = [scale, preprocess,
                                               regressor,scores['test_r2'],
                                               scores['test_neg_mean_squared_error']]
            overall_progress.update(id_overall, advance=1)
            experiment_progress.stop_task(id_experiment)
            experiment_progress.update(id_experiment,
                                       description=f"[green]({scale}->{preprocess}->{regressor}) ✅ [/green]")
        training_progress.update(id_train_progress, visible= False)
        overall_progress.update(id_overall, description="All Experiments Compleated!")

    return results



def main():
    parser = argparse.ArgumentParser(description='Plain Tester')
    parser.add_argument('datapath', type=str, nargs=1,
                        help='path to the files with the dataset')
    parser.add_argument('--seed',
                        help='Fixing the seed to this value to train and to split the dataset')
    parser.add_argument('--splits',
                        help='Number of splits to be made in the cross validation (default:10)')
     args = parser.parse_args()

    screen_header("Setting up the Laboratory")
    set_seed(seed=args.seed)
    filepath = Path(args.datapath[0])
    splits = int(args.splits) if args.splits else 10
    try:
        splits, data = prepare_data(filepath=filepath, splits=splits)
    except:
        print(f"[bold red]ERROR[/bold red] Unable to load file [yellow] {filepath}[/yellow]")
        return

    screen_header("Starting Experiments")
    results = run_experiments(data, splits)

    screen_header("Writing the report")
    results_filename =f'{datetime.today().strftime("%Y%m%d_%H%M%S")}_results.csv'
    report("Printing output to", results_filename)
    results.to_csv(results_filename)


if  __name__ == '__main__':
    main()

