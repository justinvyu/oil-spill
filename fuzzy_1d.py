'''
Monodimensional FCM (inspired by fuzzy.py with 2 variables)
'''

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

from pixel import Pixel

def visualize_fuzzy(_data, cluster_data, img_path):
    # creating histogram
    fig, ax = plt.subplots()

    sigma_data_all = np.array([i.sigma0 for i in _data[0]])
    sigma_data_cluster = np.array([i.sigma0 for i in cluster_data])

    binwidth = 1
    counts, bins, patches = ax.hist(sigma_data_all, bins=range(int(np.amin(sigma_data_all)),
                                    int(np.amax(sigma_data_all)) + binwidth, binwidth))

    # ax.annotate("c(" + str(j) + ")=%.2f" % cntr[j], xy=(cntr[j], 0),
    #             xytext=(0, -30), textcoords='offset points', va='top', ha='center')

    bin_counts = np.zeros((1, bins.size))

    # index = (curr - min) / bin_count
    indices = np.floor((sigma_data_cluster - np.amin(sigma_data_cluster)) / binwidth)

    for index in indices:
        bin_counts[0][index] += 1

    # visualize cluster separation
    for cluster_count in bin_counts:
        for patch, _bin in zip(patches, cluster_count):
            if int(_bin) is not 0:
                patch.set_facecolor('red')
            else:
                patch.set_facecolor('blue')

    plt.subplots_adjust(bottom=0.15)

    img = Image.open(img_path)

    p = img.load()

    x_pts = [i.x for i in cluster_data]
    y_pts = [i.y for i in cluster_data]

    for x, y in zip(x_pts, y_pts):
        if x < img.size[0] and y < img.size[1]:
            p[x, y] = (255, 0, 0)

    img.show()
    plt.show()

def isolate_cluster(data, nc=2):
    pixel_data = data.reshape(1, data.size)

    # apply learning algorithm, e = 0.005
    # data param is a [S, N] matrix
    sigma_data = np.array([i.sigma0 for i in data[0]])
    _data = sigma_data.reshape(1, sigma_data.size)

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(_data, nc, 2,
                                                     error=0.005, maxiter=1000)

    # returns array with cluster index j for highest "weighted" DOM
    degree_of_membership = np.argmax(u, axis=0)

    dark_cluster = np.argmin(cntr)
    dark_data = np.array(pixel_data[0][degree_of_membership == dark_cluster])

    return np.amin(cntr), dark_data

def threshold(mu, cluster_data):
    sigma_data = np.array([i.sigma0 for i in cluster_data])

    std = np.std(sigma_data)
    threshold = mu - 1.5 * std # 1 std. away from mu

    print(threshold)

    return threshold, [i for i in cluster_data if i.sigma0 <= threshold]

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

    mu, cluster, cluster_data = isolate_cluster(x_pts, 3)
    visualize_fuzzy(x_pts, cluster_data)
