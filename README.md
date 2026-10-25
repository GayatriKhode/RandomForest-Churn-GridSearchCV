# Random Forest GridSearchCV for Customer Churn Prediction

This project tunes a RandomForestClassifier with GridSearchCV and 3-fold cross-validation to predict telecom customer churn.

## Setup

1. Put your telecom churn CSV in `data/telecom_churn.csv` (or specify a different CSV with `--data`).
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Run: `python grid_search_churn.py`

The dataset must have a `Churn` (or `Exited`) column containing Yes/No or 0/1. Common Telco Customer Churn datasets are supported, including `TotalCharges` conversion.

## Output

- `outputs/results.txt`: best hyperparameters, cross-validation accuracy, test accuracy and classification report
- `outputs/grid_search_results.csv`: results of all parameter combinations
- `outputs/confusion_matrix.png`: test-set confusion matrix

## Approach

The dataset is split into stratified training (80%) and test (20%) sets. Missing values are imputed and categorical features one-hot encoded inside a pipeline, avoiding cross-validation leakage. GridSearchCV tries combinations of `n_estimators`, `max_depth` and `max_features` with 3-fold cross-validation. The best model is fitted on the full training set and evaluated on held-out test data.
