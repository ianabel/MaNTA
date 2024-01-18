import numpy as np
from netCDF4 import Dataset

def main():
    ncfile = Dataset("./PhysicsCases/Bfield.nc",mode="w",format="NETCDF4_CLASSIC")
    