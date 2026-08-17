"""Driving a C++ physics case from Python: manta.Runner("CaseName").

A Python case is handed to the Runner as an object. A C++ case cannot be,
because its constructor takes the `Grid` -- and configure() is what builds the
grid. So the Runner is given the *name* a config file's TransportSystem key
would give, and instantiates the case inside configure() once the mesh exists.
Its own configuration table travels in the same dict as the solver's keys,
nested under the table name the case reads (PyToml.hpp assembles the
toml::value).

The load-bearing test here is the first one: the same case, the same numbers,
run once through a config file and once through a dict, produces netCDF output
that is *identical bit for bit*. Both surfaces already share
loadSolverConfig/applySolverConfig/makeGrid, so anything that broke would be in
the new half -- a table that did not arrive, a float that became an integer, a
grid the case was handed after the fact. The tolerance is exact equality
deliberately: a tolerance is what would let a silently different configuration
through.

What is *not* covered: loading a real plugin. That needs `make install` and a
shared object compiled with the flags `pkg-config --cflags manta` reports, which
is more than a pytest should build -- and it is the same gap the TOML surface's
PhysicsPlugins key has. Only the failure path is tested here.
"""

import os

import numpy as np
import pytest
from netCDF4 import Dataset

import manta as MaNTA

from test_runner import LinearDiffusion as PythonCase


# --------------------------------------------------------------- fixtures --


def toml_scalar(value):
    """A Python value as TOML source. Enough for a config file, no more."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_scalar(v) for v in value) + "]"
    if isinstance(value, float):
        # repr, not str(): a float has to stay a float across the round trip.
        # TOML distinguishes 1 from 1.0 and toml::find<double> throws on an
        # integer node, so writing 1.0 as "1" would make the file fail where the
        # dict succeeded -- and the comparison would then be reporting a defect
        # in this helper.
        return repr(float(value))
    return str(value)


def write_config(path, case, config):
    """The same dict, as the config file runManta would read.

    Written *from* the dict rather than kept beside it, so the two surfaces are
    given one description of the run and any difference is the code's.
    """
    lines = ["[configuration]", f'TransportSystem = "{case}"']
    tables = []
    for key, value in config.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {toml_scalar(value)}")

    for name, table in tables:
        lines += ["", f"[{name}]"]
        for key, value in table.items():
            lines.append(f"{key} = {toml_scalar(value)}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_u(stem, variable):
    with Dataset(stem + ".nc") as nc:
        return np.array(nc.groups[variable].variables["u"][:])


def remove_output(stem):
    for suffix in (".nc", ".restart.nc", ".conf"):
        try:
            os.remove(stem + suffix)
        except FileNotFoundError:
            pass


# LinearDiffusion's own config file, whose numbers are chosen so the initial
# Gaussian is small at both Dirichlet ends -- exp(-10) there. Slide it onto
# [0, 1] with the default width and the initial condition is 0.29 at a boundary
# the solver is holding at zero, which is inconsistent enough that IDACalcIC
# gives up with IDA_ERR_FAIL. That is the config's fault rather than the
# plumbing's, and it is the first thing to check if this file starts failing at
# -3.
LINEAR_DIFFUSION = {
    "Polynomial_degree": 3,
    "Grid_size": 30,
    "Lower_boundary": -1.0,
    "Upper_boundary": 1.0,
    "delta_t": 0.1,
    "t_final": 1.0,
    "Relative_tolerance": 1.0e-3,
    "Absolute_tolerance": 1.0e-3,
    "DiffusionProblem": {
        "Kappa": 1.0,
        "Centre": 0.0,
        "SourceStrength": 3.5,
        "InitialWidth": 0.1,
    },
}

# Two physics tables rather than one, and a case that reads the *grid*:
# AutodiffTransportSystem's constructor takes xL and xR from it, and builds the
# initial profile between them. So this covers the half LinearDiffusion cannot
# -- that the case is handed the mesh configure() built, not one of its own. Its
# table also sets the boundary *kinds* (isLowerDirichlet = false), so a table
# that failed to arrive would change the system rather than a coefficient.
#
# solveAdjoint is on, as its own config file has it, which puts the adjoint
# problem a C++ case hands out on this path too. Note ADTestProblem would have
# been the closer analogue of the file it came from, and cannot be used: its
# checked-in Config/ADTestProblem.conf does not run on main either -- IDA gives
# up with IDA_ERR_FAIL -- so it is a broken case rather than a broken surface.
ADJOINT_TEST_PROBLEM = {
    "Polynomial_degree": 3,
    "Grid_size": 10,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "tau": 1.0,
    "delta_t": 0.25,
    "t_final": 5.0,
    "Relative_tolerance": 1.0e-4,
    "Absolute_tolerance": 1.0e-3,
    "solveAdjoint": True,
    "AutodiffTransportSystem": {
        "uL": [0.0],
        "isLowerDirichlet": False,
        "uR": [6.0],
        "isUpperDirichlet": True,
        "InitialHeights": [6.0],
        "InitialProfile": ["Uniform"],
    },
    "AdjointTestProblem": {"kappa": 2.0, "SourceCentre": 0.3, "a": -0.5},
}


# ------------------------------------------- the two surfaces agree exactly --


@pytest.mark.parametrize(
    "case,config,variable",
    [
        ("LinearDiffusion", LINEAR_DIFFUSION, "u"),
        ("AdjointTestProblem", ADJOINT_TEST_PROBLEM, "u"),
    ],
)
def test_a_dict_configures_a_cpp_case_exactly_as_a_config_file_does(
    case, config, variable
):
    """One run description, two surfaces, identical output.

    Compared through the netCDF rather than through getSolution because both
    files are written from `y` at the same output times, whereas getSolution
    reads `yJac` -- the state as of the last Jacobian evaluation, which can lag
    the final step. Comparing those would need a tolerance, and a tolerance is
    the one thing this test must not have.
    """
    from_toml, from_dict = case + "_from_toml", case + "_from_dict"
    try:
        write_config(from_toml + ".conf", case, dict(config, OutputFilename=from_toml))
        assert MaNTA.run(from_toml + ".conf") == 0

        runner = MaNTA.Runner(case)
        runner.configure(dict(config, OutputFilename=from_dict))
        runner.run()

        u_toml, u_dict = read_u(from_toml, variable), read_u(from_dict, variable)
        assert u_toml.shape == u_dict.shape
        assert np.array_equal(u_toml, u_dict), (
            "the dict surface configured this case differently from the config "
            f"file: worst difference {np.max(np.abs(u_toml - u_dict)):.3e}"
        )
        # ...and the run actually did something, so that two identically empty
        # arrays cannot pass the assertion above.
        assert np.max(np.abs(u_dict)) > 1e-3
    finally:
        remove_output(from_toml)
        remove_output(from_dict)


def test_a_cpp_cases_adjoint_problem_is_reachable_from_python():
    """createAdjointProblem, on a case this surface built rather than was given.

    Worth its own test because the adjoint object is the one thing configure()
    derives from the case and then keeps: an AutodiffAdjointProblem holds a raw
    pointer back to its transport system, so rebuilding the case on a second
    configure() has to drop the adjoint first or leave it dangling. That is not
    what this asserts -- a dangling pointer need not misbehave -- but it is what
    the two configure() calls here would exercise if it did.
    """
    runner = MaNTA.Runner("AdjointTestProblem")
    for _ in range(2):
        runner.configure(
            dict(
                ADJOINT_TEST_PROBLEM,
                OutputFilename="cpp_case_adjoint",
                WriteOutput=False,
            )
        )
        runner.run()

        G, gradients = runner.getAdjointGradients()
        assert np.all(np.isfinite(np.asarray(G)))
        assert np.all(np.isfinite(np.asarray(gradients["G_p"])))


# ------------------------------------------------------------- the registry --


def test_physics_cases_lists_what_this_build_carries():
    """The names Runner(name) accepts, which no other call can report.

    The registry is populated by static-initialisation side effects, so which
    cases exist depends on which object files were linked -- and that differs
    between MaNTA, libmanta.so, the unit tests and this extension. Asserting on
    two known members rather than the whole list, because the list is exactly
    what a new case is expected to change.
    """
    names = MaNTA.physics_cases()
    assert "LinearDiffusion" in names
    assert "ADTestProblem" in names
    assert names == sorted(names), "the map iterates sorted; this should too"


def test_a_case_registered_from_python_is_in_the_list():
    """registerPhysicsCase and the C++ cases share one map, so both appear."""
    MaNTA.registerPhysicsCase(
        "UnitTestPhysicsCasesListing", lambda config, grid: PythonCase()
    )
    assert "UnitTestPhysicsCasesListing" in MaNTA.physics_cases()


def test_an_unknown_case_name_is_refused_where_it_was_written():
    """At construction, not at configure().

    The registry is settled before any Python runs, so there is nothing to wait
    for -- and a name is a string literal in the caller's source, which is where
    the error is useful. The list of what *is* available is the other half:
    a case whose object file is not linked in produces no other diagnostic.
    """
    with pytest.raises(ValueError) as excinfo:
        MaNTA.Runner("LinearDiffusio")

    message = str(excinfo.value)
    assert "LinearDiffusio" in message
    assert "LinearDiffusion" in message, "the available cases were not listed"


def test_a_python_case_reports_no_cpp_case_name():
    """Which of the two constructors built this Runner, for a driver given one."""
    assert MaNTA.Runner(PythonCase()).physics_case == ""
    assert MaNTA.Runner("LinearDiffusion").physics_case == "LinearDiffusion"


# ------------------------------------------------- the case's own table --


def test_the_cases_table_reaches_its_constructor_and_can_change_the_spec():
    """Not merely a member: LowerNeumann decides a *boundary kind*.

    LinearDiffusion builds its SystemSpec from its table, so LowerNeumann = true
    is a different system rather than the same one with a different coefficient
    -- and the observable is structural. A Neumann end is `b = 1` against
    LowerBoundary's datum, which this case returns as zero, so q -> 0 there;
    the Dirichlet default instead pins u to zero and leaves q free. Both halves
    are asserted, because either alone would also pass if the flag were ignored
    and the run happened to be flat.
    """
    config = dict(
        LINEAR_DIFFUSION,
        OutputFilename="cpp_case_neumann",
        WriteOutput=False,
        DiffusionProblem=dict(LINEAR_DIFFUSION["DiffusionProblem"], LowerNeumann=True),
    )

    runner = MaNTA.Runner("LinearDiffusion")
    runner.configure(config)
    runner.run()

    u_lower = runner.getSolution(0, [-1.0])[0]
    q_lower = runner.getDerivative(0, [-1.0])[0]
    assert abs(q_lower) < 1e-6, f"lower end is not Neumann: q = {q_lower:.3e}"
    assert abs(u_lower) > 1e-2, f"lower end is still pinned to zero: u = {u_lower:.3e}"

    # ...and without the flag it is the other way round. Built from
    # LINEAR_DIFFUSION rather than from `config` above, which still carries
    # LowerNeumann: reusing it made this assertion read a Neumann run and fail
    # with u = 6.93, which is the right answer to the wrong question.
    plain = MaNTA.Runner("LinearDiffusion")
    plain.configure(
        dict(
            LINEAR_DIFFUSION,
            OutputFilename="cpp_case_dirichlet",
            WriteOutput=False,
        )
    )
    plain.run()
    u_dirichlet = plain.getSolution(0, [-1.0])[0]

    # 1e-5, not round-off. A Dirichlet end is an identically zero trace row with
    # the datum substituted into the cell rows, so it is `lambda` that is exactly
    # zero; getSolution evaluates the *element* polynomial there, which meets the
    # datum only to discretisation accuracy (1.6e-7 here). The contrast is what
    # the test is about -- 1.6e-7 against the 6.9 the Neumann end reaches above.
    assert abs(u_dirichlet) < 1e-5, f"u = {u_dirichlet:.3e}"
    assert abs(u_dirichlet) < abs(u_lower) / 1e4


def test_reconfiguring_rebuilds_the_case_from_the_new_table():
    """The reason the case is built in configure() rather than once.

    A C++ case reads its table in its constructor, so a driver sweeping a
    physics parameter -- the reason to want a C++ case under a Python optimiser
    at all -- has to get a new object. Instantiating once would pin the first
    call's parameters and every later run would silently repeat it.

    Kappa is the parameter here and the answer has to move the right way: a
    larger diffusivity spreads the same source further, so the peak is lower.
    """
    runner = MaNTA.Runner("LinearDiffusion")

    def peak(kappa):
        runner.configure(
            dict(
                LINEAR_DIFFUSION,
                OutputFilename="cpp_case_sweep",
                WriteOutput=False,
                DiffusionProblem=dict(
                    LINEAR_DIFFUSION["DiffusionProblem"], Kappa=kappa
                ),
            )
        )
        runner.run()
        return np.max(np.asarray(runner.getSolution(0, list(np.linspace(-1, 1, 101)))))

    slow, fast = peak(1.0), peak(8.0)
    assert fast < slow / 2.0, (
        f"Kappa = 8 gave a peak of {fast:.4f} against {slow:.4f} at Kappa = 1; "
        "the second configure() did not rebuild the case"
    )


def test_a_case_that_rejects_its_own_table_says_so():
    """The case's own validation, reported as a configuration error.

    LinearDiffusion throws if it has no [DiffusionProblem]; configure() has
    raised RuntimeError for a bad configuration since it existed, so this is
    translated rather than surfacing as the ValueError std::invalid_argument
    would otherwise become.
    """
    config = {k: v for k, v in LINEAR_DIFFUSION.items() if k != "DiffusionProblem"}
    runner = MaNTA.Runner("LinearDiffusion")
    with pytest.raises(RuntimeError, match="DiffusionProblem"):
        runner.configure(dict(config, OutputFilename="cpp_case_no_table"))


# ------------------------------------- the unknown-key sweep still applies --


def test_an_unknown_scalar_key_is_still_rejected():
    """The physics-table latitude is for tables only.

    A dict may now carry names the schema has never heard of, which is what
    lets a case be configured at all -- and that is exactly the check the schema
    exists to provide, so it is narrowed to *dict* values. A misspelled solver
    key is still an error with a suggestion.
    """
    runner = MaNTA.Runner("LinearDiffusion")
    with pytest.raises(RuntimeError) as excinfo:
        runner.configure(dict(LINEAR_DIFFUSION, Grid_sze=30, OutputFilename="x"))

    assert "Grid_sze" in str(excinfo.value)
    assert "Grid_size" in str(excinfo.value), "no did-you-mean suggestion"


def test_a_solver_key_given_as_a_table_is_a_type_error_not_a_physics_table():
    """The other half of that narrowing: the *name* decides, not the value.

    {"Grid_size": {...}} is a schema key holding the wrong type, and must be
    reported as such. Treating any dict as physics would instead drop the key
    silently and leave the run at Grid_size's default.
    """
    runner = MaNTA.Runner("LinearDiffusion")
    with pytest.raises(RuntimeError, match="Grid_size"):
        runner.configure(
            dict(LINEAR_DIFFUSION, Grid_size={"nope": 1}, OutputFilename="x")
        )


def test_the_transport_system_key_is_still_refused_in_a_dict():
    """The case is chosen when the Runner is built, either way.

    Rejected rather than accepted-and-ignored, which is the whole point of the
    ProblemSelection category: a driver passing TransportSystem to configure()
    would otherwise never learn that the Runner's own name won.
    """
    runner = MaNTA.Runner("LinearDiffusion")
    with pytest.raises(RuntimeError, match="TransportSystem"):
        runner.configure(
            dict(LINEAR_DIFFUSION, TransportSystem="LD2", OutputFilename="x")
        )


# -------------------------------------------------------------- plugins --


def test_loading_a_plugin_that_is_not_there_says_which_file():
    """Only the failure path; see this module's docstring for why."""
    with pytest.raises(RuntimeError, match="no-such-physics-plugin"):
        MaNTA.load_physics_plugin("/nonexistent/no-such-physics-plugin.so")
