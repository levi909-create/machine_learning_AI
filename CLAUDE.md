# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A personal collection of standalone machine-learning practice scripts, mostly
from Codecademy's ML course. Each file at the repo root is an independent,
self-contained exercise — there is no application, package, build system, test
suite, or dependency manifest. Files are named by topic (e.g. `pipeline_Ml`,
`carDATA_ml`, `faces_PCA_images`) rather than by `.py` extension, but they are
all Python 3.

## Running scripts

There is no entry point or test runner. Run an individual script directly:

```bash
python <filename>          # e.g. python carDATA_ml
```

Dependencies are not declared anywhere. Scripts assume a scientific-Python
environment with: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

## Important: external dependencies that are NOT in this repo

Many scripts will not run as-is in a fresh clone. Before assuming a script is
broken, check which of these it needs:

- **Codecademy-only modules** — `codecademylib3`, `codecademylib3_seaborn`,
  `helpers`. These exist only inside Codecademy's learning environment and are
  not pip-installable. The `import` lines can be removed safely; they only set
  up display/styling and have no effect on the ML logic.
- **Companion exercise modules** that ship with the original Codecademy lesson
  but are absent here — e.g. `graph` (provides `points`, `labels`, `ax`,
  `x_1`...), `gradient_descent_funcs` (provides `gradient_descent`). Scripts
  importing these cannot run standalone without recreating those modules.
- **Local CSV files** that are not committed — e.g. `reviews.csv`,
  `student_math.csv`, `heights.csv`, `photo_id_times.csv`. Scripts reading these
  via `pd.read_csv('...')` need the data file present in the working directory.
- **Remote datasets** fetched over HTTP — several scripts pull from UCI
  (`archive.ics.uci.edu/.../car.data`, `.../abalone.data`) or via
  `sklearn.datasets` (`fetch_olivetti_faces`, `load_iris`, `make_regression`).
  These require network access at runtime.

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
- Topics covered: linear/lasso/ridge regression and gradient descent, decision
  trees, SVM kernels and decision boundaries, KNN, PCA, and K-Means.

## Git workflow

Commit messages are short and descriptive in the imperative/topic style of the
existing history (e.g. "Add decision tree classifier implementation",
"Implement Lasso regression model for student grades"). One script per commit is
the established pattern.
