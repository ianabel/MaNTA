"""The JAX layer is part of the installed package, not a sys.path accident.

Every example under python-examples/ imports it the way an out-of-tree user
would -- `from manta.jax import ...` -- so what this file pins is that the
subpackage is importable with nothing on sys.path but the package itself, and
that it exposes the names those examples name.
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
