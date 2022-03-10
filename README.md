# MaLPPoSyRe
machine learning for predicting power system reliability

## Contributing
The code is formated according to PEP-8 black and nox is used for test automation. Nox can be installed using pip: `pip install nox`. Please run `nox -rs black` to format the code before pushing. Please also run `nox` before pushing to ensure there are no errors.

## Usage

`create dataset.py` - creates dataset used as input for model
`div_methods` - methods that are used in multiple files.

`gcn_regression_multidim.py` and `gcn_classification.py` used to run GCN-models. Set up so Jupytext can be used.

In order to visualise while running the code, assigned notebooks can be used, `Notebook_regression_multidim.ipynb` and `Notebook_classification.ipynb`.


`NNConv_regression.py` and `NNConv_classification.py`, set up using NNConv, which allows including edge attributes.

`gcn_regression.py` - used for single output regression. 

