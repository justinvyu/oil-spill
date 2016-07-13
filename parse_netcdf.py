'''
Oceanic dark-object extraction using FCM algorithm

WIP
'''

import os
import sys
import numbers
import time

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from pixel import Pixel
import fuzzy_1d

# See: http://schubert.atmos.colostate.edu/~cslocum/code/netcdf_example.py
def ncdump(nc_fid, verb=True):
    nc_attrs = nc_fid.ncattrs()
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    return nc_attrs, nc_dims, nc_vars

def parse(filename, jpg):
    nc_dataset = Dataset(filename, "r") # read-only
    nc_attrs, nc_dims, nc_vars = ncdump(nc_dataset)

    data = nc_dataset.variables["Sigma0_VV_dB"][:].T
    data_list = []

    # data_list = [i for i in data if isinstance(i, numbers.Number) and i > 10**-3]

    # for i in xrange(0, len(data) - 2, 2):
    #     for j in xrange(0, len(data[i]) - 2, 2):
    #
    #         tl = data[i][j]
    #         tr = data[i][j + 1]
    #         bl = data[i + 1][j]
    #         br = data[i + 1][j + 1]
    #
    #         pixels = []
    #
    #         if isinstance(tl, numbers.Number) and tl > 10**-3:
    #             pixels.append(tl)
    #         if isinstance(tr, numbers.Number) and tr > 10**-3:
    #             pixels.append(tr)
    #         if isinstance(bl, numbers.Number) and bl > 10**-3:
    #             pixels.append(bl)
    #         if isinstance(br, numbers.Number) and br > 10**-3:
    #             pixels.append(br)
    #
    #         if len(pixels) == 0:
    #             continue
    #
    #         avg = sum(pixels) / len(pixels) # 2x2 average
    #
    #         data_list.append(avg)

    # mode = 50.5-52.5

    for x in xrange(0, len(data)):
        for y in xrange(0, len(data[x])):
            pixel = Pixel(x, y, data[x][y])
            if isinstance(pixel.sigma0, numbers.Number) and pixel.sigma0 > 10**-3:
                data_list.append(pixel)

    # imgplot = plt.imshow(data)

    np_data = np.asarray(data_list).reshape(1, len(data_list))

    # np.asarray(data_list).reshape(len(data_list), 1)

    mu, cluster_data = fuzzy_1d.isolate_cluster(np_data, 2)

    threshold, threshold_data = fuzzy_1d.threshold(mu, cluster_data)
    fuzzy_1d.visualize_fuzzy(np_data, threshold_data, jpg_img)

    # y, x, _ = plt.hist(data_list, bins=range(int(np.amin(np_data)), int(np.amax(np_data)) + binwidth, binwidth))

    # hist = plt.plot(x, y, 'ro')

    # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np_data, 2, 2, error=0.005, maxiter=1000, init=None)
    # for pt in cntr:
    #     print(pt, pt.size)
    # print(fpc)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: parse_netcdf.py <file> <img.jpg>\n")
        sys.exit(1)

    nc_filename = sys.argv[1]
    jpg_img = sys.argv[2]

    parse(nc_filename, jpg_img)
