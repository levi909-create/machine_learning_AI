"""Standardize customer ages with StandardScaler and check the resulting mean and std."""

import os

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 

coffee = pd.read_csv(os.path.join(_DATA, "starbucks_customers.csv"))
ages = coffee['age']

## add code below
## import standardscaler
scaler = StandardScaler()

## reshape
ages_reshaped =  np.array(ages).reshape(-1,1)

## fit_transform
ages_scaled = scaler.fit_transform(ages_reshaped)

## print mean and standard deviation 
print(np.mean(ages_scaled))
print(np.std(ages_scaled))
