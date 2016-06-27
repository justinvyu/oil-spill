'''
2D FCM Example using scikit-fuzzy
See: http://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
'''

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define 3 centroids (to generate random points around)
centroids = [[3, 1],
           [5, 6],
           [1, 8]]

# Define the standard deviations (sigmas) for each centroid
# [x std. dev., y std. dev.]
sigmas =  [[0.6, 0.9],
           [0.5, 0.8],
           [1.0, 1.2]]

np.random.seed(433) # seed for reproducibility

# Initialize empty arrays
x_pts = np.zeros(1)
y_pts = np.zeros(1)
labels = np.zeros(1)

for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centroids, sigmas)):
    x_pts = np.hstack((x_pts, np.random.standard_normal(200) * xsigma + xmu)) # pass in tuple, initialize 200 vals in x_pts
    y_pts = np.hstack((y_pts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i)) # creates [0, 0, ..., 1, 1, ..., 2, 2]

# print(x_pts, y_pts, labels)

# Graph

# fig, ax = plt.subplots()
# for label in range(3):
#     plt.plot(x_pts[labels == label], y_pts[labels == label], colors[label])

data = np.vstack((x_pts, y_pts)) # vertical concatenation of x and y points as columns

print(data, data.shape)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fits = [] # how well each model fits

for nc, ax in enumerate(axes.reshape(-1), 2): # -1 --> infers to be a vector, 2 --> start at 2
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, nc, 2, error=0.005, maxiter=1000, init=None)
    fits.append(fpc)

    cluster_membership = np.argmax(u, axis=0)
    for j in range(nc):
        ax.plot(x_pts[cluster_membership == j],
                y_pts[cluster_membership == j], 'o', color=colors[j]) # which cluster does it belong to?

    for pt in cntr:
        print(pt)
        ax.plot(pt[0], pt[1], 'rs')
    print("\n")

    ax.set_title('nc=' + str(nc) + ' fpc=' + str(fpc))

ncs = range(2, 11)
print(np.vstack((fits, ncs)))

fig.tight_layout()
plt.show()
