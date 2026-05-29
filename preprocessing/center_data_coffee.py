"""Center customer ages by subtracting the mean and plot the distribution."""

import os

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

coffee = pd.read_csv(os.path.join(_DATA, "starbucks_customers.csv"))


ages = coffee['age']


min_age = np.min(ages)
print(min_age)


max_age = np.max(ages)
print(max_age)


print(max_age - min_age)


mean_age = np.mean(ages)
print(mean_age)


centered_ages = ages - mean_age
print(centered_ages)


plt.hist(centered_ages)
plt.show();
