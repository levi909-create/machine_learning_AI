"""Compare raw vs. log-transformed car selling prices with histograms."""

import os

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

## add code below
## read in csv file
cars = pd.read_csv(os.path.join(_DATA, "cars.csv"))

## set you price variable
prices = cars['sellingprice']

## plot a histogram of prices
plt.hist(prices, bins = 150)
plt.show();
## log transform price
log_prices = np.log(prices)

## plot a histogram of log_prices
plt.hist(log_prices, bins = 150)
plt.title('my Graph')

plt.show();
