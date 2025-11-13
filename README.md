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

1.  Clone the repository.
2.  Install the dependencies and the command-line tool:
    ```bash
    uv sync
    ```
    > [!NOTE]
    > After modifying `pyproject.toml` or `setup.py`, you may need to run this command again to make new scripts available.

3.  Run the application:
    ```bash
    uv run labml --help
    ```

    For example, to run the regression command:
    ```bash
    uv run labml regression --help
    ```
