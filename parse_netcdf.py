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

    # mode = 50.5-52.5

    # for (x, y) in zip(range(len(data)), range(len(data[x]))):
    #     # val = data[x][y]
    #     # if isinstance(val, numbers.Number) and val > 0.003:
    #     #     data_list.append(val)
    #
    #     pixel = Pixel(x, y, data[x][y])
    #     if isinstance(pixel.sigma0, numbers.Number) and pixel.sigma0 > 10**-3:
    #         data_list.append(pixel)

    # holy shit

    data_list = [el for sublist in [[Pixel(x, y, val) for y, val in enumerate(_data) if isinstance(val, numbers.Number) and abs(val) > 0.003] for x, _data in enumerate(data)] for el in sublist]
    np_data = np.asarray(data_list).reshape(1, len(data_list))

    # np_data = data.reshape(1, data.size)

    mu, cluster_data = fuzzy_1d.isolate_cluster(np_data, 2)

    threshold, threshold_data = fuzzy_1d.threshold(mu, cluster_data)
    fuzzy_1d.visualize_fuzzy(np_data, threshold_data, jpg_img)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: parse_netcdf.py <file> <img.jpg>\n")
        sys.exit(1)

    nc_filename = sys.argv[1]
    jpg_img = sys.argv[2]

    parse(nc_filename, jpg_img)
