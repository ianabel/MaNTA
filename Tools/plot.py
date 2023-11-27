#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys

filename = ""

if len(sys.argv) == 3:
    filename = sys.argv[1]
    time_idx = sys.argv[2]
else:
    print("Usage: ./plot.py <netcdf filename> <time index>")
    sys.exit()


nc_root = Dataset(filename, "r", format="NETCDF4")
t_var = nc_root.variables["t"]

Var = nc_root.groups["Var0"].variables["u"]

x_var = nc_root.variables["x"]

plt.plot( x_var[:], Var[time_idx,:])
plt.show()


