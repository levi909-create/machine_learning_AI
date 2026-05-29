# Machine Learning Practice

A collection of standalone machine-learning exercises covering the core
supervised and unsupervised learning techniques, built with
[scikit-learn](https://scikit-learn.org/), pandas, NumPy, and matplotlib.

Each script is self-contained and focuses on a single concept — from
implementing gradient descent by hand to building full preprocessing
pipelines with grid search.

## Repository structure

| Directory | Topics covered |
| --- | --- |
| [`regression/`](regression/) | Linear and Lasso regression, gradient descent from scratch, loss functions, multiple linear regression |
| [`classification/`](classification/) | Decision trees, k-nearest neighbors, support vector machines and decision boundaries |
| [`clustering/`](clustering/) | K-Means clustering and pre-clustering visualization |
| [`dimensionality_reduction/`](dimensionality_reduction/) | Principal Component Analysis (PCA) |
| [`preprocessing/`](preprocessing/) | Standardization, centering, scaling, log transforms, categorical encoding |
| [`pipelines/`](pipelines/) | End-to-end scikit-learn `Pipeline` + `ColumnTransformer` + `GridSearchCV` workflows |

## Getting started

```bash
# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Run any script directly:

```bash
python regression/lasso_regression.py
python classification/decision_tree_car.py
```

## Data sources

Scripts obtain their data in one of three ways:

- **Remote datasets** — fetched at runtime from the
  [UCI Machine Learning Repository](https://archive.ics.uci.edu/) or via
  `sklearn.datasets` (e.g. Iris, Olivetti faces). These require an internet
  connection but no setup.
- **Local CSV files** — a few scripts read a CSV from the working directory
  (e.g. `student_math.csv`, `starbucks_customers.csv`, `housing_data.csv`,
  `reviews.csv`). These data files are not included in the repository; place
  the corresponding CSV in the directory before running.
- **Companion modules** — a small number of exercises originally shipped with a
  helper module (e.g. `graph`, `gradient_descent_funcs`, `plot`) that is not
  included here. Those scripts illustrate the technique but need the helper
  module to run end to end.

## Topics

- **Regression** — line of best fit, gradient descent (step-by-step and full
  loop), mean-squared-error loss, multiple linear regression, L1 (Lasso)
  regularization.
- **Classification** — decision trees (training, depth tuning, visualization),
  k-nearest neighbors, support vector machines (linear and polynomial kernels,
  decision boundaries, outlier sensitivity).
- **Clustering** — K-Means on the Iris dataset and on usage data.
- **Dimensionality reduction** — PCA on facial image data.
- **Preprocessing** — centering, standard scaling, log transforms, one-hot and
  ordinal encoding.
- **Pipelines** — combining imputation, encoding, scaling, and model selection
  into reproducible scikit-learn pipelines with cross-validated grid search.
