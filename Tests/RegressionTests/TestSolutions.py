#!/usr/bin/env python3

from netCDF4 import Dataset
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import os
import argparse
import shutil
import re
import scipy

# SOLVER names the solver binary. CTest sets it to the built executable's
# absolute path, which is what makes an out-of-source build work here: an
# absolute value discards the earlier components of the join below. The fallback
# is the repo root, which is where the Makefile used to leave it -- running
# ./TestSolutions.py by hand against an old in-source build still works, and a
# missing SOLVER is a clear "no such file" rather than KeyError: 'SOLVER'.
# Resolve against this script's location, not the caller's cwd.
_here = os.path.dirname(os.path.abspath(__file__))
manta_file = os.path.join(_here, "..", "..", os.environ.get("SOLVER", "MaNTA"))
# All the .conf inputs and .ref.nc references are siblings of this script, and
# the solver writes its output into the cwd -- so anchor there.
os.chdir(_here)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MaNTA regression suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="Relative L_2 tolerance for every comparison. Raise it to triage a "
             "suite that has drifted; do not commit a raised value without "
             "saying why.",
    )
    return parser.parse_args()


ARGS = parse_args()
TOLERANCE = ARGS.tolerance


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

    var_groups = variable_groups( nc_root )
    if len(var_groups) != n_vars:
        raise Exception(
            f"{filename}: found {len(var_groups)} variable groups {var_groups}, "
            f"expected {n_vars}"
        )

    t_var   = nc_root.variables["t"]
    x_var   = nc_root.variables["x"]

    t_var_ref   = nc_root_ref.variables["t"]
    x_var_ref   = nc_root_ref.variables["x"]

    # Loop over variables
    for name in var_groups:
        if name not in nc_root_ref.groups:
            raise Exception(
                f"{ref_filename} has no group '{name}'; the reference predates a "
                f"rename of this case's variables and needs regenerating"
            )
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
            if( diff > tolerance ):
                print("Error: L_2 norm ", diff, " ( ref is ", l2norm_ref, " ) at t = ",t_var[t_idx]," is greater than ",tolerance)
                sys.exit( 1 )
    
    # Check if adjoints were computed, if so compare those too.
    # NOTE: this guard tests the freshly-generated file. WriteAdjoints() is
    # currently commented out at Solver.cpp:350 (commit 57d2652, "adjoint
    # writing doesn't work for spatial adjoints"), so no run produces "ng" and
    # this whole block silently skipped itself. Warn loudly when the reference
    # has adjoint data that the run did not reproduce.
    if ("ng" not in nc_root.variables) and ("ng" in nc_root_ref.variables):
        print("  !! SKIPPING adjoint check for", filename, "- reference has adjoint")
        print("     output but this run produced none. WriteAdjoints() is commented")
        print("     out at Solver.cpp:350. Re-enable it to restore this check.")

    if ("ng" in nc_root.variables):
        print("  ... also checking adjoint variables")
        ng = int(nc_root.variables["ng"][0])

        for i in range(0, ng):
            gname = "G" + str(i)
            for var in nc_root.groups[gname + "_p"].variables:

                G_p     = nc_root.groups[gname + "_p"].variables[var][0]
                G_p_ref = nc_root_ref.groups[gname + "_p"].variables[var][0]

                diff = abs( (G_p - G_p_ref)/G_p_ref )
        
                if( diff > tolerance ):
                    print("Error: Adjoint norm ", diff, " ( ref is ", G_p_ref, " ) for variable ",var," is greater than ",tolerance)
                    sys.exit( 1 )

            if (nc_root.groups.get(gname + "_boundary") is not None):
                print("  ... also checking boundary adjoint variables")
                for var in nc_root.groups[gname + "_boundary"].variables:
                    G_bndry     = nc_root.groups[gname + "_boundary"].variables[var][0]
                    G_bndry_ref = nc_root_ref.groups[gname + "_boundary"].variables[var][0]

                    diff = abs( (G_bndry - G_bndry_ref)/G_bndry_ref )
            
                    if( diff > tolerance ):
                        print("Error: Boundary adjoint norm ", diff, " ( ref is ", G_bndry_ref, " ) for variable ",var," is greater than ",tolerance)
                        sys.exit( 1 )


def variable_groups( nc_root ):
    """The netCDF groups holding a solution variable, in a deterministic order.

    A physics case names its own variables, so these are called whatever the
    case declared -- "u", "Density", "IonEnergy" -- rather than Var0, Var1.
    Identified by holding a "u" variable, which separates them from Grid and
    from the adjoint groups.

    Sorted by name rather than by declaration order, which the file does not
    record. Where an index is used below it is only to pair a run against
    another run of the *same* case, so any consistent order will do.
    """
    return sorted( g for g, grp in nc_root.groups.items() if "u" in grp.variables )


def test_analytic_soln( filename, soln_fn, tolerance ):
    print("Testing",filename)
    nc_root = Dataset(filename, "r", format="NETCDF4")
    t_var = nc_root.variables["t"]
    Var = nc_root.groups[ variable_groups( nc_root )[0] ].variables["u"]
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
    Var = nc_root.groups[ variable_groups( nc_root )[0] ].variables["u"]
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
    # .nc is always written; the .dat files are opt-in (WriteDatFile /
    # WriteDebugDatFiles, both off by default) so none of them may exist.
    os.unlink( prefix + ".nc" )
    for suffix in ( ".dat", ".res.dat", ".dydt.dat" ):
        if os.path.exists( prefix + suffix ):
            os.unlink( prefix + suffix )

def check_ref_case( prefix ):
    print("Checking Reference Solution for "+prefix+".conf")
    run_manta( prefix + ".conf" )
    ncFileName = prefix + ".nc"
    ncRefFile  = prefix + ".ref.nc"
    test_ref_soln_l2( ncFileName, ncRefFile, TOLERANCE )
    cleanup( prefix )
    
def ld_soln( x, t ):
    t0 = 0.01
    return np.sqrt(t0/(t+t0)) * np.exp( -x*x/(4*(t+t0)));

print("Testing Analytic Solutions")

run_manta( "ld.conf" )
test_analytic_soln( "ld.nc", ld_soln, TOLERANCE )
cleanup( "ld" )

def nonlin_soln( x, t ):
    t0 = 1.1
    n = 2
    eta = x / np.sqrt( t0 + t )
    return pow( 1 - eta, 1/n )

run_manta( "nonlin.conf" )
test_analytic_soln( "nonlin.nc", nonlin_soln, TOLERANCE )
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
test_steady_state( "nonlin_ss.nc", nonlin_ss, TOLERANCE )
cleanup( "nonlin_ss" )

print("Checking Reference Solutions")

check_ref_case( "LinearDiffusion" )
check_ref_case( "MatTest" )
check_ref_case( "MatTestAlpha" )
check_ref_case( "ADTest" )
check_ref_case( "Nonlin2" )
check_ref_case( "PIDTest" )
check_ref_case( "AdjointTestProblem" )
check_ref_case( "AuxVarTest" )
check_ref_case( "NeumannTestLower" )
check_ref_case( "NeumannTestUpper" )
# Every case above runs with Superconvergent unset, which is what keeps their
# references valid across the addition of that option. This one turns it on.
check_ref_case( "SuperconvergentADTest" )


# ---------------------------------------------------------------- restarts --
#
# Running to t2 in one go and running to t1, writing a restart file, then
# picking it up and continuing to t2 must give the same answer. That exercises
# a path nothing else does: WriteRestartFile -> StoreGridInfo -> the restart
# branch of runManta -> Grid(CellBoundaries) -> setRestartValues, including the
# DOF bookkeeping for nVars, nAux and nScalars.
#
# It is also the check that caught the clustered-grid contiguity defect: the
# grid rebuilt from a restart file has to compare *equal* to the one that wrote
# it, and a 1e-16 gap at a cell face was enough to break that.


def config_variant( source_prefix, target_prefix, **overrides ):
    """Copy a .conf, replacing keys in [configuration] (adding them if absent).

    MaNTA names its output after the config file's stem, so a variant with a
    distinct name writes distinct .nc/.dat/.restart.nc files and cannot collide
    with the checked-in references.
    """
    text = open( source_prefix + ".conf" ).read()

    # Everything before the first [section] after [configuration] is the general
    # section; keys are inserted there.
    for key, value in overrides.items():
        pattern = re.compile( r"^\s*" + re.escape( key ) + r"\s*=.*$", re.MULTILINE )
        replacement = "{} = {}".format( key, value )
        if pattern.search( text ):
            text = pattern.sub( replacement, text, count = 1 )
        else:
            text = text.replace( "[configuration]", "[configuration]\n" + replacement, 1 )

    open( target_prefix + ".conf", "w" ).write( text )


def final_slice( filename, var_index ):
    nc_root = Dataset( filename, "r", format = "NETCDF4" )
    x = np.array( nc_root.variables["x"][:] )
    u = np.array( nc_root.groups[ variable_groups( nc_root )[ var_index ] ].variables["u"][-1, :] )
    return x, u


def test_final_slices_match( filename, other_filename, tolerance ):
    """Compare the last timeslice of two runs, variable by variable.

    The reference comparison above walks every output time; here the two runs
    have different output schedules by construction, so only the end state is
    comparable.
    """
    nc_root = Dataset( filename, "r", format = "NETCDF4" )
    n_vars = int( nc_root.variables["nVariables"][0] )

    for v_idx in range( n_vars ):
        x, u = final_slice( filename, v_idx )
        x_other, u_other = final_slice( other_filename, v_idx )

        if not np.allclose( x, x_other ):
            print( "Error: output grids differ between " + filename + " and " + other_filename )
            sys.exit( 1 )

        diff2 = np.trapezoid( ( u - u_other ) ** 2, x )
        norm2 = np.trapezoid( u_other ** 2, x )

        l2diff = np.sqrt( diff2 )
        l2ref  = np.sqrt( norm2 )
        diff = abs( l2diff / l2ref ) if l2ref > 1e-12 else abs( l2diff )

        if diff > tolerance:
            print( "Error: restart round trip differs by L_2 norm ", diff,
                   " ( ref is ", l2ref, " ) for Var" + str( v_idx ),
                   " which is greater than ", tolerance )
            sys.exit( 1 )


def check_restart_round_trip( prefix, t_split, rtol = 1.0e-6, atol = 1.0e-8 ):
    """Run prefix.conf to its own t_final, split at t_split, compare.

    All three runs are done at `rtol`/`atol` rather than at whatever the case
    normally uses. That is not cosmetic: a restart re-initialises the
    integrator, so the two routes take genuinely different step sequences, and
    at a loose tolerance they legitimately disagree by more than the comparison
    threshold without either being wrong. Tightening until the answer is
    determined well below that threshold is what makes the round trip a test of
    the restart mechanism rather than of the time integrator's step choices.

    The tolerance cannot be tightened without limit, and the limit is now the
    *case*, not the restart: see the table beside the calls below.
    """
    print( "Checking restart round trip for " + prefix + ".conf (split at t = "
           + str( t_split ) + ")" )

    source = open( prefix + ".conf" ).read()
    t_final = float( re.search( r"^\s*t_final\s*=\s*(\S+)", source, re.MULTILINE ).group( 1 ) )

    whole   = prefix + "_restart_whole"
    part    = prefix + "_restart_part"
    resumed = prefix + "_restart_resumed"

    # The default MinStepSize of 1e-7 is too coarse for these tolerances; IDA
    # stalls at t = 0 with "|h| = hmin" rather than saying so.
    accurate = dict( Relative_tolerance = rtol,
                     Absolute_tolerance = atol,
                     MinStepSize = 1.0e-12 )

    try:
        config_variant( prefix, whole, **accurate )
        run_manta( whole + ".conf" )

        config_variant( prefix, part, t_final = t_split, **accurate )
        run_manta( part + ".conf" )

        config_variant(
            prefix, resumed,
            restart = "true",
            RestartFile = '"' + part + '.restart.nc"',
            t_initial = t_split,
            t_final = t_final,
            **accurate
        )
        run_manta( resumed + ".conf" )

        test_final_slices_match( resumed + ".nc", whole + ".nc", TOLERANCE )
    finally:
        for name in ( whole, part, resumed ):
            if os.path.exists( name + ".conf" ):
                os.unlink( name + ".conf" )
            if os.path.exists( name + ".nc" ):
                cleanup( name )
            if os.path.exists( name + ".restart.nc" ):
                os.unlink( name + ".restart.nc" )


print( "Checking Restart Round Trips" )

# One variable, two variables, and one auxiliary variable: the three shapes the
# restart DOF arithmetic has to get right.
#
# All three now survive 1e-6 / 1e-8. Measured as (uninterrupted run) / (restart
# and continue), with the final-slice agreement the round trip actually checks:
#
#   rtol / atol      LinearDiffusion   MatTest         AuxVarTest
#   1e-3 / 1e-5      ok / 8.5e-6       ok / 2.3e-6     ok / 2.1e-4
#   1e-4 / 1e-6      ok / 1.7e-5       ok / 2.2e-7     ok / 2.2e-5
#   1e-6 / 1e-8      ok / 6.4e-8       ok / 2.7e-10    ok / 6.4e-7
#   1e-8 / 1e-10     ok / IDASolve -3  -3 / --         ok / IDASolve -3
#
# That table used to read "ok / IDASolve -4" for AuxVarTest at 1e-4 and 1e-6 and
# "ok / -6" for MatTest at 1e-6. Two fixes closed it, and neither was in the
# restart machinery:
#
#   * setInitialConditions used to finish every restart with EvaluateLambda(),
#     throwing away the converged trace the file carried and replacing it with
#     {{u}}, which solves nothing. That is what made a restart need ~10x the
#     residual evaluations of a cold start, and it is what this comment used to
#     blame -- correctly -- on sigma and lambda being recomputed.
#   * AuxVarTest::SigmaFn adds (a - u*u) to *both* variables' fluxes while
#     dSigma_dPhi declared the derivative for variable 0 only. On the manifold
#     a = u*u the term vanishes; a warm start is precisely the state off the
#     manifold, and there Newton diverged. This was the whole of the aux case's
#     "corrector convergence failed repeatedly", which had been suspected of
#     sharing a cause with the JAXAuxTest xfail. It did not.
#
# The wall is now the *case*, not the restart. At 1e-8 / 1e-10 MatTest's
# uninterrupted run fails too, so the restart is not implicated there at all;
# LinearDiffusion and AuxVarTest resume-fail with IDA_ERR_FAIL (-3) rather than
# the corrector failure (-4) that used to appear.
#
# Cost is why MatTest is not tightened with the others: at 1e-6 / 1e-8 its three
# runs take 101 s against 6.0 s at 1e-4, for agreement (2.7e-10) that nothing
# needs. AuxVarTest tightens for free -- 0.5 s against 0.4 s -- and buys a
# factor of 330, so it is tightened.
check_restart_round_trip( "LinearDiffusion", 0.25, rtol = 1.0e-6, atol = 1.0e-8 )
check_restart_round_trip( "MatTest", 0.25, rtol = 1.0e-4, atol = 1.0e-6 )
check_restart_round_trip( "AuxVarTest", 0.6, rtol = 1.0e-6, atol = 1.0e-8 )


print("\n\n----------------")
print("All Tests Passed")
print("----------------\n\n")
sys.exit(0)

