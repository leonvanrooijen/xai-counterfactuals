# FlexCEy -- FAQ
Increasing flexiblility of CE algorithms in order to promote mass adoption.  

 ## Repo structure
    ├── README.md
    ├── ce-generation.ipynb #
    ├── cf_methods # WIP
    │   └── clean
    │       └── ... # Clean datasets
    ├── data
    │   ├── boilerplate.py
    │   └── clean
    │       └── ... # Clean datasets
    │   ├── cleaning.ipynb
    │   ├── datasets.py
    │   └── raw
    │       └── ...
    ├── model_training.ipynb
    ├── trained_models
    │   ├── ... # Different versions
    └── training_settings.json

## Running the project
*Tested on Python 3.11.8*
1. Download or clone the project
2. Run the following command
```
pip install -r requirements.txt
```
3. IF necessary: adjust datasets, black-box classifiers. See below
4. Run model training notebook.
5. Run CF generation notebook.

## Used datasets (Kaggle)

### 1. Student Drug Addiction Dataset
[Student drug addiction dataset, source](https://www.kaggle.com/datasets/atifmasih/students-drugs-addiction-dataset)

### 2. Credit Score Classification dataset
100.000 datapoints
[Data cleaning by M. El Haddad](https://www.kaggle.com/code/mohamedahmed10000/credit-score-eda-prediction-multi-class/notebook#Plotting-The-Target)

### 3. Thyroid dataset
Multiple datasets within this dataset: not sure which one will be used.
[Source](https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set)

## Training the models using different hyperparameters
*Note that pre-trained models can be found in the trained_models folder.*
1. Adjust training_settings.json in order to tweak the GridSearch parameters
2. Adjust model_training.ipynb where necessary to add more models
3. Run the notebook!

## Adding datasets
1. Put the raw dataset in data/raw
2. Put the cleaned dataset in data/clean
3. Add a class to tweak the settings in data/datasets.py
4. Add the dataset to training_settings.json to specify the hyperparameters. 

## Tweaking CE generation methods.
*To be added later*
*CE implementation still WIP*

