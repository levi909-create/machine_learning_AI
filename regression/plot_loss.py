"""Contour-plot helper for visualizing a two-parameter MSE loss surface.

Imported by ``loss_function.py``. Given ranges of candidate coefficients
``b1`` and ``b2`` and the data ``(x1, x2) -> y``, it evaluates the mean-squared
error over the grid and draws a filled contour plot.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_function(b1, b2, y, x1, x2):
    """Draw and return a filled contour plot of MSE loss over (b1, b2)."""
    B1, B2 = np.meshgrid(b1, b2)
    squared_error = np.zeros(B1.shape)
    for k in range(len(y)):
        error = y[k] - B1 * x1[k] - B2 * x2[k]
        squared_error += error ** 2
    loss = squared_error / len(y)

    contour = plt.contourf(B1, B2, loss, levels=30, cmap="viridis")
    plt.colorbar(contour, label="Mean squared error")
    plt.xlabel("b1")
    plt.ylabel("b2")
    plt.title("Loss surface")
    return contour
