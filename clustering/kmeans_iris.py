"""Visualize the Iris dataset in 3D with its ground-truth species labels."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

# Plot the ground truth in 3D.
fig = plt.figure(1, figsize=(4, 3))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=48, azim=134)

for name, label in [("Robots", 0), ("Cyborgs", 1), ("Humans", 2)]:
    ax.text(x[y == label, 3].mean(),
            x[y == label, 0].mean(),
            x[y == label, 2].mean() + 2, name,
            horizontalalignment="center",
            bbox=dict(alpha=.2, edgecolor="w", facecolor="w"))

# Reorder the labels so the colors match a typical clustering result.
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(x[:, 3], x[:, 0], x[:, 2], c=y, edgecolor="k")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel("Time to Heal")
ax.set_ylabel("Reading Speed")
ax.set_zlabel("EQ")
ax.set_title("")

plt.show()
