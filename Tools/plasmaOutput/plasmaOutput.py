#!/usr/bin/env python3

# Simple python script that takes MaNTA outputs and an equilibrium file and, if necessary, generates phi_0 at all 
# points before writing a combined netCDF file

from netCDF4 import Dataset

class Plasma:
    def __init__( self, ncFileManta, ncFileEq ):
        self.MantaData = Dataset(ncFileManta, "r", format="NETCDF4")
        self.EqData = Dataset(ncFileEq, "r", format="NETCDF4")
