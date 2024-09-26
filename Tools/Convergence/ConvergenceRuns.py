#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import os
import scipy
import tomlkit

manta_file = "../../MaNTA"

def run_manta( config_file ):
    code = os.system( manta_file + " " + config_file + " 2>/dev/null >/dev/null" )
    if( code != 0 ):
        print("Failed to run test simulation with configuration in " + config_file)
        sys.exit(code)

def get_steady_state_error( filename, soln_fn ):
    nc_root = Dataset(filename, "r", format="NETCDF4")
    t_var = nc_root.variables["t"]
    Var = nc_root.groups["Var0"].variables["u"]
    x_var = nc_root.variables["x"]

    t_idx = -1
    diff2 = 0.0
    for x_idx in range( len( x_var ) - 1 ):
        Val_x_idx_0 = Var[t_idx,x_idx] - soln_fn( x_var[x_idx] )
        Val_0 = Val_x_idx_0 ** 2;
        Val_x_idx_1 = Var[t_idx,x_idx + 1] - soln_fn( x_var[x_idx + 1] )
        Val_1 = Val_x_idx_1 ** 2;
        diff2 += ( Val_0 + Val_1 )*( x_var[x_idx + 1] - x_var[x_idx] )/2.0

    l2norm = np.sqrt( diff2 )
    return l2norm

def cleanup( prefix ):
    os.unlink( prefix + ".nc" )
    os.unlink( prefix + ".dat" )
    if ( os.path.exists( prefix + ".res.dat" ) ):
       os.unlink( prefix + ".res.dat" )
    if ( os.path.exists( prefix + ".dydt.dat" ) ):
       os.unlink( prefix + ".dydt.dat" )

def nonlin_ss( x ):
    a = 6.0
    b = 0.02
    c = 0.3
    d = 50.0
    u1 = 0.3
    y = (x - c)/np.sqrt(b)
    G = (b*d/(4*a)) * ( np.exp( -(1-c)**2/b ) - np.exp( -y**2 ) ) + (d*np.sqrt( b*np.pi )/(4*a)) * ( (c-1)*scipy.special.erf( (c-1)/np.sqrt(b) ) + (1-x)*scipy.special.erf(c/np.sqrt(b)) - (x-c)*scipy.special.erf(y) )
    u2 = 1.0/np.sqrt(u1) - G
    return 1.0/(u2**2)

def write_new_config( out, template, cells, order, rtol = 1e-3, atol = 1e-3 ):
    outfile = open( out, "w" )
    templatefile = open( template, "r" )
    toml_data = tomlkit.load( templatefile )
    toml_data["configuration"]["Grid_size"] = cells
    toml_data["configuration"]["Polynomial_degree"] = order
    toml_data["configuration"]["Relative_tolerance"] = rtol
    toml_data["configuration"]["Absolute_tolerance"] = atol
    tomlkit.dump( toml_data, outfile )

for k in (2,3):
    print("# k = ",k)
    print("# nCells\tError ")
    for nCells in (2,4,8,16,32):
        if( nCells > 7 ):
            rtol = 1e-8
            atol = 1e-4
        else:
            rtol = 1e-3
            atol = 1e-2

        write_new_config( "nonlin_ss_run.conf", "nonlin_ss.template", nCells, k, rtol, atol );
        run_manta("nonlin_ss_run.conf")
        err = get_steady_state_error( "nonlin_ss_run.nc", nonlin_ss )
        print(nCells,"\t",err)
        cleanup( "nonlin_ss_run" )
    print("")
    print("")


sys.exit(0)

