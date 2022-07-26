#!/usr/bin/env python3

import typer
from regression import regression, set_seed
from prepare_data import prepare_data
from util.common import set_seed

app = typer.Typer()


@app.command(name='regression')
def cmd_regression(datapath:str,
                   seed:int = typer.Option(None,
                                 help='Fixing the seed to this value to train and to split the dataset',
                                 callback=set_seed),
                   n_splits:int = typer.Option(10, help='Number of splits to be made in the cross validation')
                  ):
    regression(datapath, seed, n_splits)


@app.command(name='prepare_data')
def cmd_prepare_data(file:str,
                    n_splits:int=10,
                    seed:int=typer.Option(None,
                                       callback=set_seed,
                                       help='set the random seed for any split'),
                    output_folder:str = typer.Option('_partition_',
                                     help='Output for the partitions')
                ):
    prepare_data(file, n_splits, seed, output_folder)


@app.callback
def callback():
    """
        This command allows to operate over a problem to prepate them or execute some preliminary test
    """

if __name__ == "__main__":
    app()
