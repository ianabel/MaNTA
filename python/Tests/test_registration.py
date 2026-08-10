"""Tests for registering a physics case from Python, and for the TOML bridge.

`MaNTA.registerPhysicsCase(name, factory)` puts a Python callable into the same
process-global map the C++ `REGISTER_PHYSICS_IMPL` macro writes to. The solver
then calls it with `(TomlValue, Grid)` -- which is the only route by which a
`toml::value` reaches Python, and therefore the only way `cast_toml` and
`TomlValue.__getitem__` get exercised at all.

Registration is global and cannot be undone, so every name here is prefixed to
keep it out of the way of real physics cases and of repeated test runs.
"""

import numpy as np
import pytest

import manta as MaNTA


# What the factory saw, per registered name. Module-level because the registry
# outlives any one test.
CAPTURED = {}


class Registered(MaNTA.TransportSystem):
    """Minimal working physics case that records its construction arguments."""

    def __init__(self, config, grid):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))

        # Every TOML node type, read back through cast_toml.
        CAPTURED["values"] = {
            "flag": config["flag"],
            "count": config["count"],
            "kappa": config["kappa"],
            "label": config["label"],
            "coeffs": config["coeffs"],
            "nested": config["nested"],
        }
        CAPTURED["grid_cells"] = grid.getNCells()
        self.kappa = float(config["kappa"])

    def SigmaFn(self, i, state, x, t):
        return self.kappa * state["Derivative"][i]

    def Sources(self, i, state, x, t):
        return 1.0

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, self.kappa)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


class SecondRegistered(Registered):
    """A distinguishable second class, for the duplicate-name check."""

    def __init__(self, config, grid):
        super().__init__(config, grid)
        CAPTURED["which"] = "second"


CONFIG_TEMPLATE = """
[configuration]
TransportSystem = "{name}"
Polynomial_degree = 2
Grid_size = 5
Lower_boundary = 0.0
Upper_boundary = 1.0
t_final = 0.2
delta_t = 0.1
Relative_tolerance = 1.0e-4
Absolute_tolerance = 1.0e-6
OutputPoints = 21

[Registered]
flag = true
count = 7
kappa = 0.75
label = "a string"
coeffs = [1.0, 2.5, -3.0]

[Registered.nested]
inner = 42
"""


def write_config(tmp_path, name, extra=""):
    path = tmp_path / f"{name}.conf"
    path.write_text(CONFIG_TEMPLATE.format(name=name) + extra)
    return str(path)


# ------------------------------------------------------------ registration --


def test_a_python_class_can_be_registered_and_run(tmp_path):
    """The whole round trip: register, run the solver, get a result.

    This is the path every JAX physics case takes (see
    JAXLinearDiffusion.registerTransportSystems), and it drives runManta --
    config parsing, grid construction, the solve and the netCDF output -- with
    a Python physics case at the bottom.
    """
    CAPTURED.clear()
    name = "UnitTestRegisteredCase"
    MaNTA.registerPhysicsCase(name, Registered)

    assert MaNTA.run(write_config(tmp_path, name)) == 0
    assert "values" in CAPTURED, "the factory was never called"
    assert CAPTURED["grid_cells"] == 5


def test_cast_toml_handles_every_node_type(tmp_path):
    """cast_toml is a hand-written switch over the TOML value kinds.

    Each branch produces a different Python type, and the array and table
    branches recurse -- so an omission shows up as a None or a wrong type
    rather than an error.
    """
    CAPTURED.clear()
    name = "UnitTestCastTomlCase"
    MaNTA.registerPhysicsCase(name, Registered)
    MaNTA.run(write_config(tmp_path, name))

    values = CAPTURED["values"]

    assert values["flag"] is True
    assert isinstance(values["count"], int) and values["count"] == 7
    assert isinstance(values["kappa"], float) and values["kappa"] == pytest.approx(0.75)
    assert values["label"] == "a string"

    assert isinstance(values["coeffs"], list)
    assert values["coeffs"] == pytest.approx([1.0, 2.5, -3.0])

    assert isinstance(values["nested"], dict)
    assert values["nested"] == {"inner": 42}


def test_toml_lookup_searches_the_subtables(tmp_path):
    """`config["kappa"]` finds a key inside [Registered], not at the top level.

    __getitem__ first tries the key directly and, failing that, walks the
    top-level tables looking for it. Every physics case relies on that: they
    are handed the whole document and read their own section's keys by bare
    name.
    """
    CAPTURED.clear()
    name = "UnitTestSubtableSearchCase"
    MaNTA.registerPhysicsCase(name, Registered)
    MaNTA.run(write_config(tmp_path, name))

    # "kappa" lives under [Registered], never at the document root.
    assert CAPTURED["values"]["kappa"] == pytest.approx(0.75)
    # "Polynomial_degree" lives under [configuration]; same mechanism.
    assert "count" in CAPTURED["values"]


def test_a_missing_toml_key_raises_out_of_range(tmp_path):
    """The lookup returns None internally and must not hand that back."""

    class MissingKey(Registered):
        def __init__(self, config, grid):
            MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))
            CAPTURED["error"] = None
            try:
                config["NoSuchKeyAnywhere"]
            except Exception as exc:  # noqa: BLE001 -- recording the type is the point
                CAPTURED["error"] = exc
            raise RuntimeError("stop here")

    CAPTURED.clear()
    name = "UnitTestMissingKeyCase"
    MaNTA.registerPhysicsCase(name, MissingKey)

    with pytest.raises(RuntimeError):
        MaNTA.run(write_config(tmp_path, name))

    err = CAPTURED["error"]
    assert err is not None, "looking up a missing key did not raise"
    assert isinstance(err, IndexError), f"got {type(err).__name__}: {err}"
    assert "NoSuchKeyAnywhere" in str(err)


def test_an_unregistered_name_is_reported_not_crashed(tmp_path):
    """InstantiateProblem throws; runManta catches it and returns 1.

    This used to dereference a null unique_ptr before reaching the check, so an
    unrecognised name segfaulted. The contract is an exception now rather than a
    null return, but what matters from here is unchanged: a clean 1, no crash.
    """
    path = tmp_path / "unknown.conf"
    path.write_text(CONFIG_TEMPLATE.format(name="NoSuchPhysicsCaseAtAll"))
    assert MaNTA.run(str(path)) == 1


def test_a_duplicate_registration_is_rejected(tmp_path):
    """registerPhysicsCase refuses a name that is already taken.

    Worth pinning from Python because this is where it will bite: two modules
    registering the same name is far more likely than two C++ physics cases
    doing so. It used to be a silent no-op, which left the second module's case
    unreachable with nothing said.
    """
    CAPTURED.clear()
    name = "UnitTestDuplicateRegistrationCase"
    MaNTA.registerPhysicsCase(name, Registered)

    with pytest.raises(ValueError, match=name):
        MaNTA.registerPhysicsCase(name, SecondRegistered)

    # The first registration survives and is still the one that runs.
    MaNTA.run(write_config(tmp_path, name))
    assert "which" not in CAPTURED


def test_a_nonexistent_config_file_returns_one(tmp_path):
    assert MaNTA.run(str(tmp_path / "does-not-exist.conf")) == 1


# -------------------------------------------------------------- MaNTA.Grid --


def test_grid_bindings():
    """MaNTA.Grid is what a registered factory receives as its second argument."""
    uniform = MaNTA.Grid(0.0, 1.0, 7, False, 0.0, 0.0)
    assert uniform.getNCells() == 7

    clustered = MaNTA.Grid(-1.0, 2.0, 9, True, 0.25, 0.25)
    assert clustered.getNCells() == 9

    # The default constructor exists for the same reason py::init<>() does on
    # TransportSystem: pybind needs it to build the holder.
    assert MaNTA.Grid() is not None


def test_toml_value_default_construction():
    """MaNTA.TomlValue() is bound; an empty document has no keys to find."""
    empty = MaNTA.TomlValue()
    with pytest.raises(Exception):
        empty["anything"]
