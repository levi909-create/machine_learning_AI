"""Shared 2D dataset and plotting helpers for the SVM exercises.

Imported by the ``svm_*`` scripts. Provides:

- ``points`` / ``labels`` — a small two-class, 2D dataset for classifiers.
- ``x_1, y_1`` (class 0) and ``x_2, y_2`` (class 1) — the same points split by
  coordinate, plus a pre-built top axes ``ax`` showing them.
- ``draw_points`` / ``draw_margin`` — helpers to plot points and an SVM's
  decision boundary with its margins.

Importing this module draws the top subplot as a side effect, matching the
behavior the original exercises relied on.
"""
import matplotlib.pyplot as plt
import numpy as np

# Two well-separated clusters: class 0 (upper-left), class 1 (lower-right).
x_1 = [1.0, 1.5, 2.0, 2.5, 1.2, 1.8]
y_1 = [7.0, 8.0, 7.5, 8.5, 6.8, 7.2]
x_2 = [6.0, 6.5, 7.0, 7.5, 6.2, 6.8]
y_2 = [2.0, 3.0, 2.5, 1.5, 2.2, 2.8]

# Combined point list and labels for the SVM classifiers.
points = [[x, y] for x, y in zip(x_1, y_1)] + [[x, y] for x, y in zip(x_2, y_2)]
labels = [0] * len(x_1) + [1] * len(x_2)

# Top axes used by the decision-boundary exercises.
ax = plt.subplot(2, 1, 1)
plt.title("Decision Boundary")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.scatter(x_1, y_1, color="b")
plt.scatter(x_2, y_2, color="r")


def draw_points(points, labels):
    """Scatter ``points`` colored by class ``labels``."""
    points = np.array(points)
    labels = np.array(labels)
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="bwr", edgecolor="k")


def draw_margin(classifier):
    """Plot the decision boundary and margins of a fitted linear SVM."""
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1],
               linestyles=["--", "-", "--"])
