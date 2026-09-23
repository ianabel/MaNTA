"""The JAX layer is part of the installed package, not a sys.path accident.

Everything under python-examples/ and python-physics/ imports it the way an
out-of-tree user would -- `from manta.jax import ...` -- so what this file pins
is that the subpackage is importable with nothing on sys.path but the package
itself, and that it exposes the names those callers name.
"""

import pytest

pytest.importorskip("equinox")

import manta  # noqa: E402
import manta.jax  # noqa: E402


EAGER = [
    "State",
    "Integrator",
    "JAXTransportSystem",
    "JAXAdjointProblem",
    "VectorizedTransportSystem",
    "MaNTA_Decorator",
    "Physics_Decorator",
    "ScalarG_Decorator",
    "ScalarGPrime_Decorator",
    "ShiftedState_Decorator",
]


@pytest.mark.parametrize("name", EAGER)
def test_public_name_is_exported(name):
    assert hasattr(manta.jax, name), f"manta.jax does not export {name}"
    assert name in manta.jax.__all__


def test_the_base_classes_derive_from_the_compiled_ones():
    assert issubclass(manta.jax.JAXTransportSystem, manta.TransportSystem)
    assert issubclass(manta.jax.VectorizedTransportSystem, manta.TransportSystem)
    assert issubclass(manta.jax.JAXAdjointProblem, manta.AdjointProblem)


def test_importing_manta_does_not_drag_in_jax():
    """`import manta` stays numpy-only.

    manta/__init__.py must never import .jax: doing so would make JAX a hard
    dependency of every import of the package, and would create the cycle that
    the relative imports inside the subpackage currently avoid.
    """
    import ast
    import pathlib

    source = pathlib.Path(manta.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in ("jax", "manta.jax"):
            pytest.fail("manta/__init__.py imports the JAX layer")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("manta.jax")


def test_ffi_runner_is_not_imported_eagerly():
    """FFIRunner registers XLA FFI targets at module scope and raises without
    them, and those bindings exist only in an XLA_FFI build (Python.cpp:361).
    An eager import would break `from manta.jax import State` on a default
    build, so the name is served by a module-level __getattr__ instead."""
    import sys

    assert "manta.jax.ffi_runner" not in sys.modules
    with pytest.raises(AttributeError):
        manta.jax.NoSuchName


def test_ffi_runner_says_which_build_it_needs():
    """On a build without XLA_FFI, asking for FFIRunner should say so.

    Left to itself the module dies on `AttributeError: module 'manta' has no
    attribute 'runner_ffi_ops'` from inside a registration loop, which reads as
    a broken package rather than as a feature this build was not compiled with.
    """
    import importlib

    if hasattr(manta._manta, "runner_ffi_ops"):
        pytest.skip("this is an XLA_FFI build, so FFIRunner imports for real")

    # importlib rather than an `import manta.jax.ffi_runner` statement, which
    # would bind `manta` as a local and shadow the module-level name above it.
    with pytest.raises(ImportError, match="XLA_FFI"):
        importlib.import_module("manta.jax.ffi_runner")


def test_the_state_carries_the_time_derivative():
    """`du/dt` has to survive the dict -> State -> dict round trip.

    The layer's State is the only thing a JAX case sees, so a field missing here
    is a field the case cannot read whatever the solver fills. "VariableDot" is
    optional on the way in -- a dict built by hand, or by a case older than the
    key, is a legitimate caller -- and an absent one is empty rather than zero,
    which is what tells a case it is in a run that fills nothing.
    """
    import numpy as np

    from manta.jax import State

    dense = {
        "Variable": np.ones((4, 2)),
        "Derivative": np.zeros((4, 2)),
        "Flux": np.zeros((4, 2)),
        "Aux": np.zeros((4, 0)),
        "Scalars": np.zeros(0),
        "VariableDot": np.arange(8.0).reshape(4, 2),
    }

    s = State.from_manta(dense)
    assert np.allclose(np.asarray(s.VariableDot), dense["VariableDot"])
    assert np.allclose(s.to_manta()["VariableDot"], dense["VariableDot"])

    del dense["VariableDot"]
    assert State.from_manta(dense).VariableDot.size == 0


def test_the_time_derivative_is_mapped_over_points():
    """vmap_axes has to name every per-point field.

    A field left at None here would be broadcast rather than mapped, so every
    point would see the whole grid's `du/dt` and the error would be a shape
    mismatch a long way from this line -- or, worse, not a shape mismatch.
    """
    assert manta.jax.State.vmap_axes().VariableDot == 0
    assert manta.jax.State.vmap_axes().Scalars is None
