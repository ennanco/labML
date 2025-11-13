![GitHub](https://img.shields.io/github/license/ennanco/MIA_ML1?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.9.0-blue?logo=Python)
# Laboratory of Machine Learning

This command-line app is designed to perform a battery of regression tests on a specific research dataset. It is not a general-purpose library, but rather a case-study or template for a specific type of analysis.

The application has two main commands:

-   `prepare-data`: This command generates and saves CSV files containing integer indices for cross-validation folds. It does not save preprocessed data. The logic is tightly coupled to a specific dataset structure, expecting an Excel file where features start at column 6, and the target variable is in column 5.
-   `regression`: This command runs an exhaustive grid search over a hardcoded set of scalers, dimensionality reduction techniques, and regression models. It can either generate cross-validation splits on the fly or use the index files created by the `prepare-data` command.

The result is a general idea of how well these techniques work with different standardization techniques and some dimension reduction on the different machine learning approaches.

After that, the researcher is encouraged to explore a fine-tuning of the more prominent techniques to obtain the best model.

There is also an unregistered, standalone script `get_results_partition.py` for running a manually-defined list of specific pipelines on a single train-test split.

## TODO list:

*   [ ] Include a RandomSearch in the process to select also the more prominent approaches
*   [ ] Add classification techniques
*   [ ] Make the split and treatment of the problem agnostic of the file.
*   [ ] Allow the inclusion of other approaches in a more organic way (for example, a config file)

## Dependencies (python)

*   pandas
*   numpy
*   scikit-learn
*   pathlib
*   rich
*   typer
*   scipy
*   openpyxl

## Installation and Usage

1.  Clone the repository to a local folder.
2.  It is recommended to use a virtual environment.
3.  Install the dependencies using `uv`:
    ```bash
    uv pip install -e .
    ```
4.  Run the application using `python -m labml.labML` or `python labml/labML.py`.

    For example:
    ```bash
    python -m labml.labML regression --help
    ```
