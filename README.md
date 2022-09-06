# Laboratory of Machine Learning

This commandline app is dedsign to perform some easy test on a problem by applying several machine learning techniques. Nowadays the app has only two command:
- *prepare-data*. This extracts the different matrices and performs a cross-validation to store the data on an specific folder for the following command. Actually, it is bound to the problem, so, it could be required to be implemented for each problem.
- *regression*. executes the regression on the specified problem. This could come from a folder structure (previous command) or from an split made on the go.

The result is a general idea of how well these techniques work with different standarized techniques and some dimension reduction on the different machine learning approaches.

After that, the researcher is encourage to explore a fine-tuning of the more prominent techniques to obtain the best model.

## TODO list:
* [ ] Include a RandomSearch in the process to select also the more prominent approaches
* [ ] Add classition techniques
* [ ] Make the split an treatment of the problem agnostic of the file.
* [ ] Allow the inclussion of other aproaches in a more organical way (for example, config file)

## Dependencies (python)
* pandas
* numpy
* scikit-learn
* pathlib
* rich
* typer
* poetry (for package manament, so it is optional but highly recomended)
