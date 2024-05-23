#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys

filename = ""

if len(sys.argv) == 3:
    filename   = sys.argv[1]
    scalar_var = sys.argv[2]
else:
    print("Usage: ./plot.py <netcdf filename> <Scalar variable name>")
    sys.exit()


nc_root = Dataset(filename, "r", format="NETCDF4")
t_var = nc_root.variables["t"]

Var = nc_root.variables[scalar_var]


plt.plot( t_var[:], Var[:])
plt.show()


