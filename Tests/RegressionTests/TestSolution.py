#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import os

manta_file = "../../" + os.environ["SOLVER"]
def run_manta( config_file ):
    code = os.system( manta_file + " " + config_file + " 2>/dev/null >/dev/null" )
    if( code != 0 ):
        print("Failed to run test simulation")
        sys.exit(code)


def test_soln( filename, soln_fn, tolerance ):
    print("Testing",filename)
    nc_root = Dataset(filename, "r", format="NETCDF4")
    t_var = nc_root.variables["t"]
    Var = nc_root.groups["Var0"].variables["u"]
    x_var = nc_root.variables["x"]

    # At each time t, calculate
    for t_idx in range(len(t_var)):
        diff2 = 0.0
        for x_idx in range( len( x_var ) - 1 ):
            Val_x_idx_0 = Var[t_idx,x_idx] - soln_fn( x_var[x_idx], t_var[t_idx] )
            Val_0 = Val_x_idx_0 ** 2;
            Val_x_idx_1 = Var[t_idx,x_idx + 1] - soln_fn( x_var[x_idx + 1], t_var[t_idx] )
            Val_1 = Val_x_idx_1 ** 2;
            diff2 += ( Val_0 + Val_1 )*( x_var[x_idx + 1] - x_var[x_idx] )/2.0

        l2norm = np.sqrt( diff2 )
        if( abs( l2norm ) > tolerance ):
            print("Error L_2 norm ", l2norm, " at t = ",t_var[t_idx]," is greater than ",tolerance)
            sys.exit( 1 )

def ld_soln( x, t ):
    t0 = 0.01
    return np.sqrt(t0/(t+t0)) * np.exp( -x*x/(4*(t+t0)));

print("Testing Analytic Solutions")

run_manta( "ld.conf" )
test_soln( "ld.nc", ld_soln, 1e-3 )
os.unlink( "ld.dat" )
os.unlink( "ld.nc" )

def nonlin_soln( x, t ):
    t0 = 1.1
    n = 2
    eta = x / np.sqrt( t0 + t )
    return pow( 1 - eta, 1/n )

run_manta( "nonlin.conf" )
test_soln( "nonlin.nc", nonlin_soln, 1e-3 )
os.unlink( "nonlin.dat" )
os.unlink( "nonlin.nc" )

print("All Tests Passed")
sys.exit(0)


