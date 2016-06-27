'''
Monodimensional C-means (inspired by fuzzy.py with 2 variables)
'''

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from matplotlib.ticker import FormatStrFormatter

# define 3 centroids (to generate random points around)
centroids = [47,
             49,
             60]

# define the standard deviations (sigmas) for each centroid
sigmas =  [0.3,
           1.5,
           1.0]

np.random.seed(433) # seed for reproducibility

# initialize empty array for (normal) randomly generated points
x_pts = np.array([])

for i, (mu, sigma) in enumerate(zip(centroids, sigmas)):
    # pass in tuple, initialize 200 vals in x_pts
    x_pts = np.hstack((x_pts, np.random.standard_normal(200) * sigma + mu))

# creating histogram

binwidth = 1
clusters = 3
m = 2 # 1 <= m < inf

counts, bins, patches = plt.hist(x_pts, bins=range(int(np.amin(x_pts)),
                                                   int(np.amax(x_pts)) + 1, 1))

# apply learning algorithm, ε = 0.005
# data param is a [S, N] matrix
data = x_pts.reshape(1, x_pts.size)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, clusters, m,
                                                 error=0.005, maxiter=1000)

# used to track
bin_counts = np.zeros((clusters, bins.size))

# returns array with cluster index j for highest "weighted" DOM
degree_of_membership = np.argmax(u, axis=0)

for j in range(clusters):
    indices = np.array([])

    # separate all points that belong to the cluster j
    for pt in x_pts[degree_of_membership == j]:
        # append the index of the bin with highest DOM
        indices = np.append(indices, np.digitize(pt, bins) - 1)

    print(indices)
    print("\n")

    for index in indices:
        bin_counts[j][index] += 1 # tally bin counts

print(bin_counts)

colors = ['r', 'g', 'b']

for pt in cntr:
    print(pt)

# plt.show()
