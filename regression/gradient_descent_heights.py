"""Fit a line to height/weight data using the gradient descent in gradient_descent_funcs.py."""

import os

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

from gradient_descent_funcs import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(_DATA, "heights.csv"))

X = df["height"]
y = df["weight"]

plt.plot(X, y, 'o')

b, m = gradient_descent(X, y, num_iterations=1000, learning_rate=0.0001)
y_predictions = [m*x + b for x in X]

plt.plot(X, y_predictions)
