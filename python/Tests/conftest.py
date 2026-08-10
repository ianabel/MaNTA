"""Make the Python test suite runnable from any working directory.

The tests need three things that used to be supplied implicitly by running
`pytest` from inside this directory:

  * the built pybind11 module, python/MaNTA<suffix>.so, on sys.path
  * the JAX transport-system modules in python/ on sys.path
  * a cwd of python/Tests, because the .conf inputs name their
    PythonModuleFile relatively and the solver writes output beside them

Previously test.py and the JAX fixtures each did `sys.path.append("../")`,
which only resolves correctly when cwd is already python/Tests. With this
file, `pytest python/Tests` from the repo root works too -- which is what
`make coverage` and CI need.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_HERE)

for _p in (_PYTHON_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _check_extension_built():
    """Fail loudly and usefully if the pybind11 module has not been built.

    The repo directory is itself named MaNTA, so if its parent is on sys.path
    Python happily imports it as an empty *namespace package* that shadows the
    real extension. The symptom is a baffling
    ``AttributeError: module 'MaNTA' has no attribute 'AdjointProblem'``
    rather than a clean ImportError.
    """
    try:
        import MaNTA
    except ImportError:
        pytest.exit(
            "MaNTA extension module not found. Build it with `make python`.",
            returncode=1,
        )
    if not hasattr(MaNTA, "TransportSystem"):
        pytest.exit(
            "'MaNTA' resolved to "
            f"{getattr(MaNTA, '__path__', MaNTA.__file__)!r}, which is not the "
            "compiled extension -- most likely the repo directory imported as a "
            "namespace package. Build the module with `make python`.",
            returncode=1,
        )


_check_extension_built()


@pytest.fixture(autouse=True)
def _run_in_tests_dir():
    """Run every test with cwd = python/Tests, restoring it afterwards."""
    previous = os.getcwd()
    os.chdir(_HERE)
    try:
        yield
    finally:
        os.chdir(previous)
