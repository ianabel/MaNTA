"""Run the Python test suite against a build, from outside the source tree.

Three things every test relies on:

  * `manta` importable from a build's package directory, <build>/python. CTest
    puts that on PYTHONPATH and names it in MANTA_PYTHON_ROOT; run by hand, do
    the same, or install the package.
  * this directory on sys.path, for `util` and the case modules the fixtures
    import by name.
  * a cwd *outside* this directory. The solver writes its output into the cwd,
    so every test runs in one scratch directory under pytest's basetemp --
    <build>/python/Tests/tmp under CTest. Inputs are read from TESTS_DIR, and a
    config's PythonModuleFile resolves against the config file, not the cwd.
"""

import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _check_extension_built():
    """Fail loudly and usefully unless `manta` is the build under test.

    MANTA_PYTHON_ROOT is what makes "the build under test" checkable. Without
    it a PYTHONPATH that did not take, or one naming another build directory,
    would quietly run the suite against whichever `manta` came first --
    an installed one, most likely -- and report on that instead.
    """
    try:
        import manta
        import manta._manta
    except ImportError as err:
        pytest.exit(
            f"manta package not importable ({err}). Build it, then point Python "
            "at the build's package directory:\n"
            "    cmake --build <build> --target _manta\n"
            "    PYTHONPATH=<build>/python pytest python/Tests\n"
            "or run the suite through CTest: ctest --test-dir <build> -R '^python$'",
            returncode=1,
        )

    expected = os.environ.get("MANTA_PYTHON_ROOT")
    if expected:
        expected = os.path.abspath(expected)
        for what, path in (("manta", manta.__file__),
                           ("manta._manta", manta._manta.__file__)):
            if os.path.commonpath([expected, os.path.abspath(path)]) != expected:
                pytest.exit(
                    f"{what} was imported from {path}, not from the build under "
                    f"test at {expected}. Something ahead of PYTHONPATH on "
                    "sys.path is shadowing it.",
                    returncode=1,
                )


_check_extension_built()


@pytest.fixture(scope="session")
def _work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cwd", numbered=False)


@pytest.fixture(autouse=True)
def _run_in_work_dir(_work_dir):
    """Run every test with cwd = the session's scratch directory."""
    previous = os.getcwd()
    os.chdir(_work_dir)
    try:
        yield
    finally:
        os.chdir(previous)
