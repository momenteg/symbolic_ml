# Symbolic Machine Learning Modeling


## Description
This framework represents a wrapper around [CatBoost](https://github.com/catboost/catboost)[^fn1] for the modeling and [Shap](https://github.com/slundberg/shap)[^fn2]
to explain the model output. 


## Requirements

__Python-2.7__, __Numpy-1.15.2__ , __Scipy-1.1.0_, Sklearn-0.20.0, Pandas-0.23.4 , CatBoost-0.11.1 ,_Shap-0.25.2.


## Files
* __syml_train.py__  main script to train a model, it has dependency on __syml_class.py__;
* __syml_class.py__  class handling the training of a model, it has dependency only on __syml_metrics.py__;
* __syml_config.yml__  training config file;
* __syml_metrics.py__  class to compute all necessary metrics to monitor the training process;
* __syml_roc_plot.py__  script to create separate and composite ROC plots from a .json history file (one of the training outputs);
* __syml_shap_analysis.py__ script to create shap analysis plots, as outlined in [here](https://github.com/slundberg/shap);


## Help
All main python scripts handle input arguments by means of *argparse*. To see the description of the input arguments
and some command line examples type:
```
python < main script > -h
```


## About the training
Training command line:
```
python syml_train.py   
  -i1 < CSV file >                # Input CSV file with "," as separator   
  -i2 < YAML config >             # Input YAML config file 
  -o  < Output path >             # Path where to save all training outputs
  -v  --verbose                   # Enable verbose mode
  -p  --print_vars                # Print column names nad unique values of each column 
```
The routine does not handle missing values and returns an error if at least one NaN occurs
in the input CSV.

The config file covers several functions and allows to specify:
- whether we are dealing with a classification or a regression problem (key --> __model:algorithm__);
- which metric to use to monitor the training process and save the best model (key --> __model:metric__);
- what is the outcome variable (key --> __model:col_out__);
- which features to select by providing a list of string or a common substring surrounded by "*", "*< substring >*" (key --> __model:select__);
- which features to exclude by providing a list of string or a common substring surrounded by "*", "*< substring >*" (key --> __model:exclude__);
- whether k-fold or random-resampling cross-validation should be enabled (key --> __validation:cv_type__);
- whether a column has to be used to constrain the splitting between training and testing (key --> __validation:col_constr__);
- some of the CatBoost parameters;

The config file allows to train the model with multiple values for the same parameter. For example, if the number of iterations
has to get values [3,4,5,6,7], type:
```
iterations:'3,4,5,6,7'
``` 
If the number of iterations has to span from 2 to 500 with step=2m type:
```
iterations:'2:500:2'
```
The routine will train the model for all possible combinations of the CatBoost parameter values provided by the config file.


## About the outputs
The outputs related to a certain training job are all hashed with the same code. The set of outputs includes:
- a CSV logger, updated every time after a parameter grid point has been used for training; it contains the hash, the monitoring metric (over the cross-validation),
  the monitoring metric for each fold/split, the parameter values corresponding to the parameter grid point;
- a JSON history file, updated only when the model has improved in terms of the monitoring metric; it specifies what was the input CSV file, the type of problem, various metrics
  and contains the arrays of ground truth and predictions for the test set of each fold/split; these arrays can be used to construct the separate and composite ROC plot with __syml_roc_plot.py__;
- .BIN model files, updated only when the model has improved in terms of the monitoring metric; it contains the CatBoost model(s); the number of models depends on the number of
  folds/splits for cross-validation;
- a .NPY shap file, updated only when the model has improved in terms of the monitoring metric; it contains the floating shap arrays for all models (as many as the number of folds/splits);
  this file can be used as input for __syml_shap_analysis.py__ to create the shap analysis plots;
- CSV files storing the training/testing sets of each cross-validation fold/split. 


## Create separate and composite ROC plots
```
python syml_roc_plot.py   
  -i1 < JSON history file >       # JSON history file outputted by __syml_train.py__   
```


## Create shap analysis plots
```
python syml_shap_analysis.py   
  -i1 < NPY shap file >           # Input NPY file containing the shap values outputted by __syml_train.py__   
  -i2 < CSV file >                # Input CSV file containing the data points to analyze
  -i3 < list of keys >            # List of columns of the CSV to exclude
```


## Author
* **Filippo Arcadu** - November 2018


## References
[^fn1]: Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin, "CatBoost: gradient boosting with categorical features support". Workshop on ML Systems at NIPS 2017. 
[^fn2]: Scott M. Lundberg and Su-In Lee, "A Unified Approach to Interpreting Model Predictions", Advances in Neural Information Processing Systems 30 at NIPS 2017.
