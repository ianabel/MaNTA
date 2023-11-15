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

Var = nc_root.variables["Var0"]

x_var = nc_root.variables["x"]

t0 = 0.2*0.2/4.0;
def gauss_pulse( x, t ):
	return np.sqrt(t0/(t+t0)) * np.exp( -x*x/(4*(t+t0)));

plt.plot( x_var[:], Var[time_idx,:])
plt.plot( x_var[:], gauss_pulse( x_var[:], t_var[time_idx]) )
plt.show()


