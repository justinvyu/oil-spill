'''
Monodimensional FCM (inspired by fuzzy.py with 2 variables)
'''

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from matplotlib.ticker import FormatStrFormatter

def visualize_fuzzy(_data, cluster_data):
    # creating histogram
    fig, ax = plt.subplots()

    binwidth = 1
    counts, bins, patches = ax.hist(_data, bins=range(int(np.amin(_data)),
                                                      int(np.amax(_data)) + binwidth, binwidth))

    # ax.annotate("c(" + str(j) + ")=%.2f" % cntr[j], xy=(cntr[j], 0),
    #             xytext=(0, -30), textcoords='offset points', va='top', ha='center')

    bin_counts = np.zeros((1, bins.size))

    # index = (curr - min) / bin_count
    indices = np.floor((cluster_data - np.amin(cluster_data)) / binwidth)

    for index in indices:
        bin_counts[0][index] += 1

    # visualize cluster separation
    colors = ['red', 'green', 'blue']

    for i, cluster_count in enumerate(bin_counts):
        for patch, _bin in zip(patches, cluster_count):
            if int(_bin) is not 0:
                patch.set_facecolor(colors[i])

    plt.subplots_adjust(bottom=0.15)
    plt.show()

def isolate_cluster(_data, nc=2):
    # apply learning algorithm, e = 0.005
    # data param is a [S, N] matrix
    data = _data.reshape(1, _data.size)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, nc, 2,
                                                     error=0.005, maxiter=1000)

    # used to track
    # returns array with cluster index j for highest "weighted" DOM
    degree_of_membership = np.argmax(u, axis=0)

    dark_cluster = np.argmin(cntr)
    points = np.array(data[0][degree_of_membership == dark_cluster])

    return dark_cluster, points

# def threshold():
    # print("std=" + str(np.std(points)), np.amin(cntr) - np.std(points))

if __name__ == "__main__":
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
        # pass in tuple, initialize 200 vals in _data
        x_pts = np.hstack((x_pts, np.random.standard_normal(200) * sigma + mu))

    cluster, cluster_data = isolate_cluster(x_pts, 3)
    visualize_fuzzy(x_pts, cluster_data)
