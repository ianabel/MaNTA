import pytest
import sys
from netCDF4 import Dataset
import numpy as np
from util import get_transport_system_as_module

sys.path.append("../")
import MaNTA

# Test order:
# 1) run MaNTA with a given config file
# 2) read the output file and extract the solution
# 3) compare the solution to the expected solution

def compare_ref_soln_l2( filename, ref_filename, tolerance ):
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
            diff  = abs( l2norm_diff/l2norm_ref ) if l2norm_ref > 1e-12 else abs(l2norm_diff)
            
            assert diff <= tolerance, f"Error: L_2 norm {diff} ( ref is {l2norm_ref} ) at t = {t_var[t_idx]} is greater than {tolerance}"
    # Check if adjoints were computed, if so compare those too
    if (nc_root.groups.get("G_p") is not None):
        print("  ... also checking adjoint variables")
        for var in nc_root.groups["G_p"].variables:
            G_p     = nc_root.groups["G_p"].variables[var][0]
            G_p_ref = nc_root_ref.groups["G_p"].variables[var][0]
            #print(var)
            if G_p_ref == 0.0:
                diff = abs( G_p - G_p_ref )
            else:
                diff = abs( (G_p - G_p_ref)/G_p_ref )
    
            assert diff <= tolerance, f"Error: Adjoint norm {diff} ( ref is {G_p_ref} ) for variable {var} is greater than {tolerance}"

        if (nc_root.groups.get("G_boundary") is not None):
            print("  ... also checking boundary adjoint variables")
            for var in nc_root.groups["G_boundary"].variables:
                G_bndry     = nc_root.groups["G_boundary"].variables[var][0]
                G_bndry_ref = nc_root_ref.groups["G_boundary"].variables[var][0]

                if G_bndry_ref == 0.0:
                    diff = abs( G_bndry - G_bndry_ref )
                else:
                    diff = abs( (G_bndry - G_bndry_ref)/G_bndry_ref )
        
                assert diff <= tolerance, f"Error: Boundary adjoint norm {diff} ( ref is {G_bndry_ref} ) for variable {var} is greater than {tolerance}"

def compare_solution(transport_system):
    fname = get_transport_system_as_module(transport_system+".conf")
    MaNTA.run( fname )
    compare_ref_soln_l2(transport_system+".nc", transport_system+".ref.nc", 1.0e-2)  
 
def test():
    compare_solution("JAXLinearDiffusion")







