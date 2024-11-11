#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import os
import scipy

manta_file = "../../" + os.environ["SOLVER"]
def run_manta( config_file ):
    code = os.system( manta_file + " " + config_file + " >/dev/null" )
    if( code != 0 ):
        print("Failed to run test simulation with configuration in " + config_file)
        sys.exit(code)

def test_ref_soln_l2( filename, ref_filename, tolerance ):
    print("Comparing " + filename + " with reference output in " + ref_filename)
    nc_root = Dataset(filename, "r", format="NETCDF4")
    nc_root_ref = Dataset(ref_filename, "r", format="NETCDF4")

    n_vars = int(nc_root.variables["nVariables"][0])
    t_var   = nc_root.variables["t"]
    x_var   = nc_root.variables["x"]

    t_var_ref   = nc_root_ref.variables["t"]
    x_var_ref   = nc_root_ref.variables["x"]

    # Loop over variables
    for v_idx in range(n_vars):
        name = "Var" + str(v_idx)
        Var     = nc_root.groups[name].variables["u"]
        Var_ref = nc_root_ref.groups[name].variables["u"]

        # At each time t, calculate || u - u_ref ||_2 and check it's within tolerance
        for t_idx in range(len(t_var)):
            diff2 = 0.0
            norm2ref = 0.0
            for x_idx in range( len( x_var ) - 1 ):
                Val_x_idx_0 = Var[t_idx,x_idx] - Var_ref[t_idx,x_idx]
                Val_0 = Val_x_idx_0 ** 2;
                Val_x_idx_1 = Var[t_idx,x_idx + 1] - Var_ref[t_idx,x_idx + 1]
                Val_1 = Val_x_idx_1 ** 2;
                diff2 += ( Val_0 + Val_1 )*( x_var[x_idx + 1] - x_var[x_idx] )/2.0 
                norm2ref += ( Var_ref[t_idx,x_idx] ** 2 + Var[t_idx,x_idx + 1] ** 2) * (x_var[x_idx+1] - x_var[x_idx] ) / 2.0

            l2norm_diff = np.sqrt( diff2 )
            l2norm_ref  = np.sqrt( norm2ref )
            if( abs( l2norm_diff/l2norm_ref ) > tolerance ):
                print("Error: L_2 norm ", l2norm_diff, " ( ref is ", l2norm_ref, " ) at t = ",t_var[t_idx]," is greater than ",tolerance)
                sys.exit( 1 )


def test_analytic_soln( filename, soln_fn, tolerance ):
    print("Testing",filename)
    nc_root = Dataset(filename, "r", format="NETCDF4")
    t_var = nc_root.variables["t"]
    Var = nc_root.groups["Var0"].variables["u"]
    x_var = nc_root.variables["x"]

    # At each time t, calculate
    for t_idx in range(len(t_var)):
        diff2 = 0.0
        norm2ref = 0.0
        for x_idx in range( len( x_var ) - 1 ):
            Val_x_idx_0 = Var[t_idx,x_idx] - soln_fn( x_var[x_idx], t_var[t_idx] )
            Val_0 = Val_x_idx_0 ** 2;
            Val_x_idx_1 = Var[t_idx,x_idx + 1] - soln_fn( x_var[x_idx + 1], t_var[t_idx] )
            Val_1 = Val_x_idx_1 ** 2;
            diff2 += ( Val_0 + Val_1 )*( x_var[x_idx + 1] - x_var[x_idx] )/2.0
            norm2ref += ( soln_fn( x_var[x_idx + 1], t_var[t_idx] ) ** 2 + soln_fn( x_var[x_idx], t_var[t_idx] ) ** 2) * (x_var[x_idx+1] - x_var[x_idx] ) / 2.0

        l2norm_diff = np.sqrt( diff2 )
        l2norm_ref  = np.sqrt( norm2ref )
        if( abs( l2norm_diff/l2norm_ref ) > tolerance ):
            print("Error: L_2 norm ", l2norm_diff, " ( ref is ", l2norm_ref, " ) at t = ",t_var[t_idx]," is greater than ",tolerance)
            sys.exit( 1 )

def test_steady_state( filename, soln_fn, tolerance ):
    print("Testing",filename)
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
    if( abs( l2norm ) > tolerance ):
        print("Error - L_2 norm ", l2norm, " at t = ",t_var[t_idx]," is greater than ",tolerance)
        sys.exit( 1 )

def cleanup( prefix ):
    os.unlink( prefix + ".nc" )
    os.unlink( prefix + ".dat" )
    if ( os.path.exists( prefix + ".res.dat" ) ):
       os.unlink( prefix + ".res.dat" )
    if ( os.path.exists( prefix + ".dydt.dat" ) ):
       os.unlink( prefix + ".dydt.dat" )

def check_ref_case( prefix ):
    print("Checking Reference Solution for "+prefix+".conf")
    run_manta( prefix + ".conf" )
    ncFileName = prefix + ".nc"
    ncRefFile  = prefix + ".ref.nc"
    test_ref_soln_l2( ncFileName, ncRefFile, 1e-3 )
    cleanup( prefix )
    
def ld_soln( x, t ):
    t0 = 0.01
    return np.sqrt(t0/(t+t0)) * np.exp( -x*x/(4*(t+t0)));

print("Testing Analytic Solutions")

run_manta( "ld.conf" )
test_analytic_soln( "ld.nc", ld_soln, 5e-3 )
cleanup( "ld" )

def nonlin_soln( x, t ):
    t0 = 1.1
    n = 2
    eta = x / np.sqrt( t0 + t )
    return pow( 1 - eta, 1/n )

run_manta( "nonlin.conf" )
test_analytic_soln( "nonlin.nc", nonlin_soln, 5e-3 )
cleanup( "nonlin" )

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

run_manta( "nonlin_ss.conf" )
test_steady_state( "nonlin_ss.nc", nonlin_ss, 5e-3 )
cleanup( "nonlin_ss" )

print("Checking Reference Solutions")

check_ref_case( "LinearDiffusion" )
check_ref_case( "MatTest" )
check_ref_case( "MatTestAlpha" )
check_ref_case( "ADTest" )
check_ref_case( "Nonlin2" )
check_ref_case( "PIDTest" )

print("\n\n----------------")
print("All Tests Passed")
print("----------------\n\n")
sys.exit(0)

