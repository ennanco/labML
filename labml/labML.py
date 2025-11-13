#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is a Facade patten for the different commands of the app.
Any new command would implicate to add a new point in this class.
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import typer
from commands.regression import regression
from commands.prepare_data import prepare_data
from commands.util.common import set_seed

# ---------------------------------------------------------------------------
# Meta-information
# ---------------------------------------------------------------------------
__author__ = "Enrique Fernandez-Blanco"
__copyright__ = "Copyright 2022, Enrique Fernandez-Blanco"
__credits__ = ["Enrique Fernandez Blanco"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Enrique Fernandez-Blanco"
__email__ = "enrique.fernandez@udc.es"
__status__ = "Prototype"

# ---------------------------------------------------------------------------
# Create the App
# ---------------------------------------------------------------------------
app = typer.Typer()
""" This is the app based on typer for CLI applications"""


@app.command(name="regression")
def cmd_regression(
    datapath: str,
    seed: int = typer.Option(
        None,
        help="Fixing the seed to this value to train and to split the dataset",
        callback=set_seed,
    ),
    n_splits: int = typer.Option(
        10, help="Number of splits to be made in the cross validation"
    ),
    output_filename: str = typer.Option(None, help=" Name to give to the results file"),
):
    regression(datapath, seed, n_splits, output_filename)


@app.command(name="prepare_data")
def cmd_prepare_data(
    file: str,
    n_splits: int = 10,
    seed: int = typer.Option(
        None, callback=set_seed, help="set the random seed for any split"
    ),
    output_folder: str = typer.Option("_partition_", help="Output for the partitions"),
):
    prepare_data(file, n_splits, seed, output_folder)


if __name__ == "__main__":
    app()
