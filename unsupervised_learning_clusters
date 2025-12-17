import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import codecademylib3
from plot import plot_clusters

# Load the data
media_usage = pd.read_csv('media_usage.csv')

# Create the model
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(media_usage)

labels = kmeans.predict(media_usage)

# Plot the clusters
plot_clusters(media_usage, labels)
