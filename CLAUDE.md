# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A personal collection of standalone machine-learning practice scripts (mostly
from Codecademy's ML course). Each `.py` file is an independent, self-contained
exercise focused on a single concept — there is no application, package, build
system, or test suite. Scripts are organized into topic folders; see
`README.md` for the full layout.

## Directory layout

- `regression/` — linear/Lasso regression, gradient descent from scratch, loss
  functions, multiple linear regression
- `classification/` — decision trees, KNN, SVM kernels and decision boundaries
- `clustering/` — K-Means and pre-clustering visualization
- `dimensionality_reduction/` — PCA
- `preprocessing/` — centering, scaling, log transforms, categorical encoding
- `pipelines/` — `Pipeline` + `ColumnTransformer` + `GridSearchCV` workflows

## Running scripts

There is no entry point or test runner. Set up and run individual scripts:

```bash
pip install -r requirements.txt   # numpy, pandas, scikit-learn, matplotlib, seaborn
python <dir>/<script>.py          # e.g. python regression/lasso_regression.py
```

## Important: external dependencies that are NOT in this repo

Many scripts will not run as-is in a fresh clone. Before assuming a script is
broken, check which of these it needs:

- **Companion exercise modules** that shipped with the original Codecademy
  lesson but are absent here — e.g. `graph` (provides `points`, `labels`, `ax`,
  `x_1`...), `gradient_descent_funcs` (provides `gradient_descent`), `plot`
  (provides `plot_clusters`). Scripts importing these cannot run standalone
  without recreating those modules.
- **Local CSV files** that are not committed (and are `.gitignore`d) — e.g.
  `reviews.csv`, `student_math.csv`, `heights.csv`, `starbucks_customers.csv`,
  `housing_data.csv`, `media_usage.csv`, `cars.csv`. Scripts reading these via
  `pd.read_csv('...')` need the data file present in the working directory.
- **Remote datasets** fetched over HTTP — several scripts pull from UCI
  (`archive.ics.uci.edu/.../car.data`, `.../abalone.data`) or via
  `sklearn.datasets` (`fetch_olivetti_faces`, `load_iris`, `make_regression`).
  These require network access at runtime.

Note: the Codecademy-only display modules (`codecademylib3`,
`codecademylib3_seaborn`) have been removed from all scripts; do not reintroduce
them.

## Conventions observed across scripts

- Numbered, comment-driven structure: steps are marked with `## 1.`, `## 2.`,
  etc., mirroring the original exercise prompts. Preserve this style when
  editing an existing exercise.
- Results are reported with `print(...)` and/or `plt.show()` — there are no
  assertions or return-value checks.
- scikit-learn idioms used throughout: `train_test_split(..., random_state=0)`
  for reproducibility, `Pipeline`/`ColumnTransformer` for preprocessing,
  `pd.get_dummies` / `OneHotEncoder` for categorical encoding, `StandardScaler`
  for scaling, and `GridSearchCV` with a `search_space` list of estimator dicts
  for model selection.

## Git workflow

Commit messages are short and descriptive in the imperative/topic style of the
existing history (e.g. "Add decision tree classifier implementation"). Keep
related changes scoped to a single concern per commit.
