"""Scatter-plot helper for visualizing K-Means cluster assignments.

Imported by ``kmeans_media_usage.py``. Plots the first two feature columns of
the data, colored by the predicted cluster label.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(data, labels):
    """Scatter the first two columns of ``data``, colored by ``labels``."""
    values = data.values if hasattr(data, "values") else np.asarray(data)
    columns = list(data.columns) if hasattr(data, "columns") else ["feature 1", "feature 2"]

    plt.scatter(values[:, 0], values[:, 1], c=labels, cmap="viridis", edgecolor="k")
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title("K-Means clusters")
    plt.show()
