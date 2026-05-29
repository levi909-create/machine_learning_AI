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

## Data and dependencies

All scripts run out of the box. Data is sourced two ways:

- **Bundled CSVs in `data/`** — scripts that read a CSV resolve it relative to
  their own location via `os.path.join(os.path.dirname(__file__), "..",
  "data")`, so they work from any working directory. The committed CSVs are
  small synthetic samples; `.gitignore` ignores `*.csv` everywhere *except*
  `data/`, so a user's own data files stay untracked.
- **Remote datasets** — `pipelines/abalone_*` and `classification/decision_tree_*`
  fetch from UCI over HTTP; `dimensionality_reduction/pca_faces.py` calls
  `fetch_olivetti_faces`; several use bundled `sklearn.datasets` (`load_iris`,
  `make_regression`). The HTTP/`fetch_*` ones need network access at runtime.

**Companion helper modules live next to the scripts that import them** (and rely
on the script's own directory being on `sys.path` when run directly):
`classification/graph.py`, `regression/gradient_descent_funcs.py`,
`regression/plot_loss.py`, `clustering/plot.py`. Keep helpers in the same folder
as their consumer.

Notes:
- The Codecademy-only display modules (`codecademylib3`,
  `codecademylib3_seaborn`) have been removed; do not reintroduce them.
- Scripts target current library versions (numpy 2.x, pandas 3.x,
  scikit-learn 1.8.x). Use `OneHotEncoder(sparse_output=...)` (not `sparse=`),
  `float`/`np.float64` (not the removed `np.float`), and the modern
  `fig.add_subplot(projection="3d")` 3-D API.

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
