# Splitting `python/` into infrastructure and examples — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the JAX framework layer out of flat `python/` into the installed package as `manta.jax`, and move every driver, config and notebook into `python-examples/`, so that `python/` holds only the package and its tests and every example imports the way an out-of-tree user does.

**Architecture:** Six framework modules become a `manta.jax` subpackage with relative internal imports, so nothing resolves through `sys.path`. `manta.cli` gains the `PythonModuleFile` form the existing configs use, resolved relative to the config file, so no config has to be rewritten to a new convention. Each driver gets a self-contained directory under `python-examples/` holding its module, its config and a README.

**Tech Stack:** Python 3.11+, pybind11 extension built by the project Makefile, JAX/equinox/jaxtyping (optional extra), pytest, mypy, GNU make.

Spec: `docs/superpowers/specs/2026-08-12-python-examples-split-design.md`.

## Global Constraints

- **`manta/__init__.py` must never import `.jax`.** JAX stays an optional extra; a top-level re-export would make it a hard dependency and create an import cycle. Both properties depend on this one line not existing.
- **No module under `manta/jax/` may write `os.environ` at import.** Two of the moved files do today, including one that forces `JAX_PLATFORM_NAME=cpu`, which would disable the GPU path from a library import. Those writes move to the examples that want them.
- **`manta/jax/__init__.py` must not eagerly import `ffi_runner`.** `FFIRunner.py` calls `MaNTA.runner_ffi_ops()` at module scope and raises `RuntimeError` if the op is absent; that binding is `#ifdef XLA_FFI` (`Python.cpp:361`), so on a default build an eager import breaks `from manta.jax import State`. Expose it through a module-level `__getattr__`.
- **Every moved example imports `manta` and `manta.jax` absolutely.** No `sys.path` manipulation survives anywhere under `python-examples/`.
- **No test's assertions change.** If a test outcome changes, that is a defect in the move.
- Python floor is **3.11** (`pyproject.toml`), and the optional extra is named exactly **`jax`** — installed as `pip install manta[jax]`.
- Prerequisite for every task: the extension must be built. This working tree has no `python/manta/_manta*.so`; run `make python` first. Verify with `ls python/manta/*.so` against `.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'` — a mismatch there produces three unrelated-looking failures (see CLAUDE.md).
- Tests are run with the venv on `PATH`: `export PATH="$PWD/.venv/bin:$PATH"`.

---

## File Structure

**Created**

| Path | Responsibility |
|---|---|
| `python/manta/jax/__init__.py` | Public surface of the JAX layer; extra-missing error; lazy `ffi_runner` |
| `python/manta/jax/state.py` | `State` and the four decorators |
| `python/manta/jax/integrator.py` | `Integrator` |
| `python/manta/jax/transport_system.py` | `JAXTransportSystem` |
| `python/manta/jax/adjoint_problem.py` | `JAXAdjointProblem` |
| `python/manta/jax/vectorized.py` | `VectorizedTransportSystem` |
| `python/manta/jax/ffi_runner.py` | `FFIRunner`, `Platform`, the FFI registration helpers |
| `python/Tests/test_jax_layer.py` | That the subpackage imports and exposes what it claims |
| `python/Tests/test_cli_modules.py` | The `PythonModuleFile` loader |
| `python-examples/**` | The examples, one directory each |

**Modified**

| Path | Change |
|---|---|
| `python/manta/cli.py` | `load_physics_modules`, `PythonModuleFile` support |
| `python/Tests/util.py` | Delegates to `load_physics_modules` |
| `python/Tests/JAXLinearDiffusion.py`, `JAXAuxTest.py` | Import from `manta.jax`; drop `sys.path` |
| `python/Tests/conftest.py` | Docstring only |
| `pyproject.toml` | `[project.optional-dependencies] jax` |
| `mypy.ini` | `[mypy-manta.jax.*] ignore_errors`; rewritten header |
| `Makefile` | `CLEAN_DATA_DIRS`, `clean` |
| `.gitignore` | `python-examples/.ipynb_checkpoints/` |
| `README.md`, `CLAUDE.md`, `docs/*.rst`, `docs/conf.py` | Paths and pointers |

**Deleted:** `python/MantaPythonTransport.py`, `python/PyManta`.

---

## Deviations from the spec

Three, all discovered while checking the files, all folded into the tasks below. Flag them if any is unwanted.

1. **`os.environ` writes leave the framework** (Global Constraints above). The spec said imports get repointed and said nothing about these.
2. **`manta.jax.ffi_runner` is lazy**, because it cannot be imported on a non-`XLA_FFI` build at all.
3. **`notebooks/` does not exist.** Both notebooks `import MaNTA`, the retired name, and `desc_optimize-vp.ipynb` also imports `Objective2`, `Stellarator2` and `yancc_wrapper2` — so it belongs *in* `stellarator/`, not in a sibling directory it cannot import across. That leaves `AdjointAnalysis.ipynb` alone, and it is adjoint analysis, so it goes in `adjoints/`. Every directory stays liftable, which was the criterion.

---

### Task 1: The `manta.jax` subpackage

**Files:**
- Create: `python/manta/jax/{__init__,state,integrator,transport_system,adjoint_problem,vectorized,ffi_runner}.py`
- Create: `python/Tests/test_jax_layer.py`
- Create: `python-examples/jax-diffusion/jax_diffusion.py`
- Delete: `python/{State,Integrator,JAXTransportSystem,JAXAdjointProblem,VectorizedTransportSystem,FFIRunner}.py`
- Modify: `python/Tests/JAXLinearDiffusion.py`, `python/Tests/JAXAuxTest.py`, `python/Tests/conftest.py`, `pyproject.toml`, `mypy.ini`

**Interfaces:**
- Consumes: `manta.TransportSystem`, `manta.AdjointProblem`, `manta.Runner` from the built extension.
- Produces: `manta.jax` exporting `State`, `Integrator`, `JAXTransportSystem`, `JAXAdjointProblem`, `VectorizedTransportSystem`, `MaNTA_Decorator`, `Physics_Decorator`, `ScalarG_Decorator`, `ScalarGPrime_Decorator` eagerly, and `FFIRunner`, `Platform`, `register_ffi_cpu`, `register_ffi_gpu` lazily. Tasks 3 and 4 import from this surface.

- [ ] **Step 1: Write the failing test**

Create `python/Tests/test_jax_layer.py`:

```python
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
```

- [ ] **Step 2: Run it and watch it fail**

```sh
export PATH="$PWD/.venv/bin:$PATH"
pytest python/Tests/test_jax_layer.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'manta.jax'`.

- [ ] **Step 3: Move the five modules that move cleanly**

```sh
mkdir -p python/manta/jax
git mv python/State.py                     python/manta/jax/state.py
git mv python/Integrator.py                python/manta/jax/integrator.py
git mv python/JAXAdjointProblem.py         python/manta/jax/adjoint_problem.py
git mv python/VectorizedTransportSystem.py python/manta/jax/vectorized.py
git mv python/FFIRunner.py                 python/manta/jax/ffi_runner.py
```

Then rewrite exactly these import lines and nothing else:

`state.py` — replace `from Integrator import Integrator` with:

```python
from .integrator import Integrator
```

`adjoint_problem.py` — replace `from State import State, MaNTA_Decorator, Physics_Decorator` with:

```python
from .state import State, MaNTA_Decorator, Physics_Decorator
```

`vectorized.py` — replace the two flat imports with:

```python
from .adjoint_problem import JAXAdjointProblem
from .state import Physics_Decorator, State
```

`ffi_runner.py` — delete lines 2–5 entirely:

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"
```

These are process-wide policy, not library behaviour; they reappear in Task 4 at the top of the examples that want them, which is where the existing drivers already set their own (`Stellarator.py:4`, `Objective2.py:11-12`).

`integrator.py` — unchanged; it imports nothing from the project.

`import manta as MaNTA` stays as it is in every file. It is an absolute import of the parent package, which is fully initialised by the time any submodule runs, so it is not a cycle.

- [ ] **Step 4: Split `JAXTransportSystem.py`**

The framework half is lines 1–162 (`JAXTransportSystem` ends with `def createAdjointProblem(self): pass`). Create `python/manta/jax/transport_system.py` with this header, then lines 12–162 of the original verbatim:

```python
import jax
import numpy as np
import manta as MaNTA
from typing import NamedTuple
from abc import abstractmethod

from .adjoint_problem import JAXAdjointProblem
from .state import MaNTA_Decorator

"""
JAX-based transport system base class that overloads MaNTA TransportSystem.
Enables automatic differentiation of sigma and source terms using JAX.
"""
```

The original header also imported `os`, `jax.numpy as jnp`, `Any` and `State`, and set `os.environ['JAX_PLATFORM_NAME'] = 'cpu'`. The class body uses none of those four names, and the environment write must not live in a library — it would force CPU on every process that imports the layer, defeating the GPU path `ffi_runner.py` exists to provide.

The example half is lines 165–269 (from the `# Need PyTree structure for class parameters` comment to the end). Create `python-examples/jax-diffusion/jax_diffusion.py` with this header, then those lines verbatim:

```python
"""Two demo cases on manta.jax.JAXTransportSystem.

Split out of what used to be python/JAXTransportSystem.py, which held both the
framework base class and these two cases in one file. The base class is now
manta.jax.JAXTransportSystem and this is an ordinary out-of-tree case that
imports it.
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax.numpy as jnp
import manta as MaNTA
from typing import NamedTuple

from manta.jax import JAXTransportSystem, JAXAdjointProblem
```

Then delete the original:

```sh
git rm python/JAXTransportSystem.py
```

- [ ] **Step 5: Write `python/manta/jax/__init__.py`**

```python
"""JAX physics cases for MaNTA.

A case written against this layer supplies its flux and sources as JAX
functions and has every derivative hook supplied by ``jax.grad`` rather than
written by hand:

    import manta
    from manta.jax import JAXTransportSystem

    class MyCase(JAXTransportSystem):
        ...

This subpackage is the only part of ``manta`` that needs JAX, which is why it
is an optional extra rather than a dependency of the package. ``import manta``
stays numpy-only, and nothing in ``manta/__init__.py`` may import this module
-- doing so would make JAX mandatory for every user of the solver and would
turn the relative imports below into a cycle.
"""

try:
    import equinox as _equinox  # noqa: F401
    import jax as _jax  # noqa: F401
except ImportError as _e:  # pragma: no cover -- depends on the environment
    raise ImportError(
        "manta.jax needs the JAX extra: pip install manta[jax]"
    ) from _e

from .state import (  # noqa: F401
    MaNTA_Decorator,
    Physics_Decorator,
    ScalarG_Decorator,
    ScalarGPrime_Decorator,
    State,
)
from .integrator import Integrator  # noqa: F401
from .adjoint_problem import JAXAdjointProblem  # noqa: F401
from .transport_system import JAXTransportSystem  # noqa: F401
from .vectorized import VectorizedTransportSystem  # noqa: F401

__all__ = [
    "FFIRunner",
    "Integrator",
    "JAXAdjointProblem",
    "JAXTransportSystem",
    "MaNTA_Decorator",
    "Physics_Decorator",
    "Platform",
    "ScalarG_Decorator",
    "ScalarGPrime_Decorator",
    "State",
    "VectorizedTransportSystem",
    "register_ffi_cpu",
    "register_ffi_gpu",
]

# ffi_runner is loaded on demand, not with the rest. It registers XLA FFI
# targets at module scope and raises RuntimeError when it cannot find them, and
# the bindings it looks for -- runner_ffi_ops, runner_ffi_ops_cuda -- exist only
# in an XLA_FFI build (Python.cpp:361). Imported eagerly it would take
# `from manta.jax import State` down with it on every default build.
_LAZY = {
    "FFIRunner": "ffi_runner",
    "Platform": "ffi_runner",
    "register_ffi_cpu": "ffi_runner",
    "register_ffi_gpu": "ffi_runner",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
```

- [ ] **Step 6: Repoint the two test fixtures**

In `python/Tests/JAXLinearDiffusion.py` and `python/Tests/JAXAuxTest.py`, delete lines 2–3:

```python
import sys
sys.path.insert(0, '../')  # To find MaNTA module
```

and replace the two flat imports:

```python
from JAXTransportSystem import JAXTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
```

with:

```python
from manta.jax import JAXTransportSystem, JAXAdjointProblem
```

In `python/Tests/conftest.py`, delete this bullet from the module docstring, which no longer describes anything:

```
  * the JAX transport-system modules in python/ on sys.path
```

The `sys.path` insertion in `conftest.py` itself stays: it is what finds an un-installed `manta`.

- [ ] **Step 7: Declare the extra and scope mypy**

In `pyproject.toml`, after the `dependencies` line:

```toml
[project.optional-dependencies]
jax = ["jax", "equinox", "jaxtyping"]
```

In `mypy.ini`, replace the header paragraph that begins "Scoped to python/manta deliberately" with:

```
; Scoped to python/manta deliberately. The package is the part that ships and
; the part an out-of-tree case is written against, so it is the part whose
; stubs have to be true. The examples under python-examples/ are drivers and
; JAX experiments that were never written against annotations; turning mypy
; loose on them would produce a wall of findings that nobody acts on, which is
; how a check stops being read.
```

and append:

```ini
; manta.jax is the JAX layer, moved into the package from what used to be flat
; modules in python/. It is unannotated, and check_untyped_defs below would
; report every def in it. Excluded so the rest of the package stays checked --
; widen this when the layer is annotated, rather than leaving it as cover.
[mypy-manta.jax.*]
ignore_errors = True
```

- [ ] **Step 8: Run the tests**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make python
pytest python/Tests/test_jax_layer.py -v
make python_tests
make typecheck
make stubs-check
```

Expected: `test_jax_layer.py` all pass; `make python_tests` shows the same results as before the change (`test_reference_solutions.py::test_jax_aux_test` remains a `strict=True` xfail — it is a known limitation, not a regression); `typecheck` and `stubs-check` clean.

- [ ] **Step 9: Commit**

```bash
git add python/manta/jax python/Tests/test_jax_layer.py \
        python/Tests/JAXLinearDiffusion.py python/Tests/JAXAuxTest.py \
        python/Tests/conftest.py python-examples/jax-diffusion/jax_diffusion.py \
        pyproject.toml mypy.ini
git add -u python/
git commit -m "Move the JAX framework layer into the package as manta.jax"
```

---

### Task 2: `manta.cli` learns `PythonModuleFile`

**Files:**
- Modify: `python/manta/cli.py`
- Modify: `python/Tests/util.py`
- Create: `python/Tests/test_cli_modules.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `manta.cli.load_physics_modules(config_path, extra_modules=())` — imports every physics module a config names, by dotted name (`PythonModule`) or by path (`PythonModuleFile`), calling `registerTransportSystems()` on each if it defines one. Returns `None`. Raises `SystemExit` with a message naming the config key on any failure. Task 3 relies on it to run the moved configs.

- [ ] **Step 1: Write the failing test**

Create `python/Tests/test_cli_modules.py`:

```python
"""The config's physics-module keys, and how they resolve.

Two forms are supported. `PythonModule` names an importable module and is the
documented one. `PythonModuleFile` names a file, which is what every config in
this repository was written for, and is resolved *relative to the config file*
-- not to the current directory, which is what the retired PyManta script did
and why those configs only ever worked from one place.
"""

import sys

import pytest

from manta.cli import load_physics_modules


CASE_SOURCE = """
import manta

LOADED = True
REGISTERED = False

def registerTransportSystems():
    global REGISTERED
    REGISTERED = True
"""


def _write_case(directory, module_filename="case.py", **conf):
    (directory / module_filename).write_text(CASE_SOURCE)
    keys = "\n".join(f'{k} = "{v}"' for k, v in conf.items())
    config = directory / "run.conf"
    config.write_text(f'[configuration]\nTransportSystem = "X"\n{keys}\n')
    return config


def test_module_file_resolves_relative_to_the_config_not_the_cwd(tmp_path):
    config = _write_case(tmp_path, PythonModuleName="casemod",
                         PythonModuleFile="case.py")

    # conftest.py runs every test with cwd = python/Tests, so a cwd-relative
    # reading of "case.py" cannot possibly succeed here. That is the point.
    load_physics_modules(config)

    assert sys.modules["casemod"].LOADED


def test_the_legacy_registration_hook_is_called(tmp_path):
    config = _write_case(tmp_path, PythonModuleName="hookmod",
                         PythonModuleFile="case.py")

    load_physics_modules(config)

    assert sys.modules["hookmod"].REGISTERED, (
        "every example module in the tree registers through "
        "registerTransportSystems() rather than at import"
    )


def test_the_module_name_defaults_to_the_file_stem(tmp_path):
    config = _write_case(tmp_path, module_filename="stemcase.py",
                         PythonModuleFile="stemcase.py")

    load_physics_modules(config)

    assert sys.modules["stemcase"].LOADED


def test_a_missing_file_names_the_key_that_is_wrong(tmp_path):
    config = tmp_path / "run.conf"
    config.write_text('[configuration]\nPythonModuleFile = "absent.py"\n')

    with pytest.raises(SystemExit) as excinfo:
        load_physics_modules(config)

    assert "PythonModuleFile" in str(excinfo.value)


def test_a_dotted_module_still_works(tmp_path):
    config = tmp_path / "run.conf"
    config.write_text('[configuration]\nPythonModule = "json"\n')

    load_physics_modules(config)  # no exception


def test_extra_modules_are_imported_too(tmp_path):
    config = tmp_path / "run.conf"
    config.write_text("[configuration]\n")

    load_physics_modules(config, extra_modules=["json"])  # no exception
```

- [ ] **Step 2: Run it and watch it fail**

```sh
pytest python/Tests/test_cli_modules.py -v
```

Expected: collection error, `ImportError: cannot import name 'load_physics_modules' from 'manta.cli'`.

- [ ] **Step 3: Implement the loader**

In `python/manta/cli.py`, add `import importlib.util` and `from pathlib import Path` to the imports, replace `_python_module_of` with `_configuration_of`, and add `load_physics_modules`:

```python
def _configuration_of(config_path):
    """The config's [configuration] table, or an empty one."""
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        # Let runManta report it, so the message is the same either way.
        return {}
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"manta: {config_path} is not valid TOML: {e}") from e

    return config.get("configuration", {})


def _load_module_from_file(config_path, conf):
    """Import the module named by PythonModuleFile, by path.

    The path is resolved relative to the config file rather than to the current
    directory. Every config in this repository is written as though that were
    already true; the retired PyManta script resolved against the cwd, which is
    why they only ever ran from one directory.
    """
    relative = conf["PythonModuleFile"]
    path = Path(config_path).parent / relative
    name = conf.get("PythonModuleName") or path.stem

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(
            f"manta: PythonModuleFile {relative!r} (looked for {path}) is not "
            "a loadable Python module"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        del sys.modules[name]
        raise SystemExit(
            f"manta: PythonModuleFile {relative!r} not found beside "
            f"{config_path} (looked for {path})"
        ) from e
    except Exception:
        # Leaving a half-executed module in sys.modules would make the next
        # import of that name silently succeed against a broken object.
        del sys.modules[name]
        raise

    return module


def _register(module):
    """Call the legacy registration hook, if the module has one.

    A module is expected to call `manta.registerPhysicsCase` at import, the way
    a C++ case registers during static initialisation. Every example module in
    this repository instead defines `registerTransportSystems()` and waits to be
    asked, which is the convention PyManta established. Both are honoured.
    """
    hook = getattr(module, "registerTransportSystems", None)
    if callable(hook):
        hook()


def load_physics_modules(config_path, extra_modules=()):
    """Import every physics module the config names, for its registrations."""
    conf = _configuration_of(config_path)

    # The current directory first, so a case sitting next to the config file
    # works without being installed.
    if "" not in sys.path:
        sys.path.insert(0, "")

    names = list(extra_modules)
    from_config = conf.get("PythonModule")
    if from_config and from_config not in names:
        names.append(from_config)

    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError as e:
            raise SystemExit(
                f"manta: could not import physics module {name!r}: {e}"
            ) from e
        _register(module)

    if "PythonModuleFile" in conf:
        _register(_load_module_from_file(config_path, conf))
```

and reduce `main` to:

```python
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="manta", description="Run a MaNTA transport problem from a config file."
    )
    parser.add_argument("config", nargs="?", default="MaNTA.conf",
                        help="TOML configuration file (default: MaNTA.conf)")
    parser.add_argument("--module", "-m", action="append", default=[], metavar="MODULE",
                        help="import MODULE before running, for its physics-case "
                             "registrations; repeatable. Also read from the config's "
                             "PythonModule key.")
    args = parser.parse_args(argv)

    load_physics_modules(args.config, args.module)

    return manta.run(args.config)
```

Extend the module docstring to document the second form:

```
A config written for the older tooling names the module by *file* instead, and
that still works -- the path is resolved beside the config file:

    [configuration]
    TransportSystem  = "MyCase"
    PythonModuleName = "mycase"
    PythonModuleFile = "mycase.py"

In either form, a module that defines `registerTransportSystems()` has it
called after import. That is the convention every example in this repository
uses; registering at import is the documented one.
```

- [ ] **Step 4: Run the tests to verify they pass**

```sh
pytest python/Tests/test_cli_modules.py -v
```

Expected: all six pass.

- [ ] **Step 5: Make `Tests/util.py` delegate**

Replace the whole body of `python/Tests/util.py` with:

```python
"""Loading a config's physics module, for the tests that drive runManta.

This used to be a copy of the loader in manta/cli.py. It delegates now, so the
rule about where PythonModuleFile is resolved from has one implementation
rather than two that can drift.
"""

from manta.cli import load_physics_modules


def get_transport_system_as_module(config_path):
    load_physics_modules(config_path)
    return config_path
```

- [ ] **Step 6: Run the whole Python suite**

```sh
make python_tests
make typecheck
```

Expected: same results as before the change. `test_reference_solutions.py` drives `get_transport_system_as_module` and is the check that the delegation is faithful — its configs use relative `PythonModuleFile` values and sit beside their modules, so the config-relative resolution reaches the same files the cwd-relative one did.

- [ ] **Step 7: Commit**

```bash
git add python/manta/cli.py python/Tests/util.py python/Tests/test_cli_modules.py
git commit -m "Teach manta.cli the PythonModuleFile form, resolved beside the config"
```

---

### Task 3: The conf-driven examples

**Files:**
- Create: `python-examples/{linear-diffusion,toy-model,jax-diffusion,jax-linear-diffusion,jax-nonlinear-adjoint}/{<module>.py,run.conf,README.md}`
- Delete (by move): `python/{PythonLinearDiffusion,ToyModel1,JAXLinearDiffusion,NonlinearDiff_example}.py`, `python/{py,JAXLinearDiffusion,jax_example,jax_adjoint}.conf`
- Delete: `python/MantaPythonTransport.py`, `python/PyManta`

**Interfaces:**
- Consumes: `manta.jax` (Task 1), `manta.cli.load_physics_modules` (Task 2).
- Produces: five runnable example directories. Nothing later depends on their contents.

- [ ] **Step 1: Delete the two files that cannot work**

```sh
git rm python/MantaPythonTransport.py python/PyManta
```

`MantaPythonTransport.py` is a `SyntaxError` — `def InitialValue(index, x):` and `def InitialDerivative(index, x):` have empty bodies — refers to an unqualified `TransportSystem`, and is imported by nothing. `PyManta` imports the top-level `MaNTA` module, which no longer exists; Task 2 put everything it did into `manta.cli`.

- [ ] **Step 2: Move the five examples into place**

```sh
mkdir -p python-examples/{linear-diffusion,toy-model,jax-diffusion,jax-linear-diffusion,jax-nonlinear-adjoint}

git mv python/PythonLinearDiffusion.py python-examples/linear-diffusion/linear_diffusion.py
git mv python/py.conf                  python-examples/linear-diffusion/run.conf

git mv python/ToyModel1.py             python-examples/toy-model/toy_model.py   # untracked: plain mv, then git add

git mv python/jax_example.conf         python-examples/jax-diffusion/run.conf

git mv python/JAXLinearDiffusion.py    python-examples/jax-linear-diffusion/jax_linear_diffusion.py
git mv python/JAXLinearDiffusion.conf  python-examples/jax-linear-diffusion/run.conf

git mv python/NonlinearDiff_example.py python-examples/jax-nonlinear-adjoint/jax_nonlinear_adjoint.py
git mv python/jax_adjoint.conf         python-examples/jax-nonlinear-adjoint/run.conf
```

`jax-diffusion/jax_diffusion.py` already exists from Task 1. `ToyModel1.py` is untracked in this tree, so `git mv` will refuse — move it with `mv` and `git add` the destination.

- [ ] **Step 3: Repoint the example imports**

`jax_linear_diffusion.py` — replace:

```python
from JAXTransportSystem import JAXTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
```

with:

```python
from manta.jax import JAXTransportSystem, JAXAdjointProblem
```

`jax_nonlinear_adjoint.py` — replace:

```python
from JAXAdjointProblem import JAXAdjointProblem
from VectorizedTransportSystem import VectorizedTransportSystem
```

with:

```python
from manta.jax import VectorizedTransportSystem, JAXAdjointProblem
```

`linear_diffusion.py` — no change; it imports only `manta`, `numpy` and `sys`.

`toy_model.py` — the class is `PythonToyModel` but `registerTransportSystems` registers `PythonLinearDiffusion`, a name the file does not define, so registration raises `NameError`. Fix the hook to match the class and the directory:

```python
def registerTransportSystems():
    MaNTA.registerPhysicsCase("ToyModel1", PythonToyModel)
```

That is the only permitted repair to this file. It is an untracked sketch with other problems — it calls `MaNTA.TransportSystem.__init__(self)` and then `super().__init__(...)` twice, and it implements linear diffusion rather than the `d_x((a/u^{3/2}) du/dx)` its own header comment describes. If it does not run after the registration fix, record that in its README and move on rather than debugging it here.

- [ ] **Step 4: Fix the configs**

`linear-diffusion/run.conf` — replace the absolute path left over from another machine:

```toml
PythonModuleName = "linear_diffusion"
PythonModuleFile = "linear_diffusion.py"
TransportSystem = "PythonLinearDiffusion"
```

`jax-diffusion/run.conf` — the module it names was renamed in Task 1:

```toml
PythonModuleName = "jax_diffusion"
PythonModuleFile = "jax_diffusion.py"
TransportSystem = "JAXNonlinearDiffusion"
```

`jax-linear-diffusion/run.conf`:

```toml
PythonModuleName = "jax_linear_diffusion"
PythonModuleFile = "jax_linear_diffusion.py"
TransportSystem = "JAXLinearDiffusion"
```

`jax-nonlinear-adjoint/run.conf` — this one was pointing at `VectorizedTransportSystem.py`, which defines only the base class: no `JAXNonlinearDiffusion` and no registration hook, so the config could never have run. The module that defines that case is the one now in this directory:

```toml
PythonModuleName = "jax_nonlinear_adjoint"
PythonModuleFile = "jax_nonlinear_adjoint.py"
TransportSystem = "JAXNonlinearDiffusion"
```

`toy-model/run.conf` — new; the case takes no `[DiffusionProblem]` keys (it hardcodes `Centre`, `InitialWidth`, `InitialHeight` and `kappa`), and it is diffusion from a Gaussian on `[-1, 1]`:

```toml
[configuration]

Polynomial_degree = 3
Grid_size = 20
Lower_boundary = -1.0
Upper_boundary =  1.0

t_final = 0.5
delta_t = 0.1

Relative_tolerance = 1.0e-3
Absolute_tolerance = 1.0e-2

PythonModuleName = "toy_model"
PythonModuleFile = "toy_model.py"
TransportSystem = "ToyModel1"
```

- [ ] **Step 5: Run every one of them**

From inside each directory, which is the property the whole change exists to demonstrate:

```sh
MANTA=$PWD
for d in linear-diffusion toy-model jax-diffusion jax-linear-diffusion jax-nonlinear-adjoint; do
  echo "=== $d"
  ( cd python-examples/$d \
    && PYTHONPATH=$MANTA/python $MANTA/.venv/bin/python -m manta.cli run.conf ) \
    || echo "FAILED: $d"
done
```

Expected: each writes `run.nc` and `run.restart.nc` into its own directory and exits 0. (`PYTHONPATH` stands in for `pip install .`; the READMEs tell users to install.) Note the output stem comes from the *config* name, so every example produces `run.nc` — that is `Solver.cpp`'s `inputFilePath.stem()` behaviour, not a mistake.

If `toy-model` fails, say so in its README under a "Known broken" heading and leave it failing; do not debug an untracked sketch inside this task.

- [ ] **Step 6: Write the five READMEs**

Each is short and says three things: what it demonstrates, what it needs, how to run it. `python-examples/linear-diffusion/README.md`:

```markdown
# Linear diffusion, in pure Python

The smallest possible physics case: one variable, a constant-kappa flux, no
sources, and a closed-form `ExactSolution` to compare against. Nothing here
needs JAX.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/linear-diffusion
    manta run.conf

Writes `run.nc` and `run.restart.nc` beside the config. The output stem comes
from the config's name, not from `OutputFilename`.

## What to look at

`linear_diffusion.py` declares its one variable through `manta.numbered_spec(1)`
and implements `SigmaFn`, `Sources` and the four derivative hooks by hand. A
case that would rather not write those by hand should look at
`../jax-diffusion/`, where `jax.grad` supplies them.
```

Write the other four to the same shape. Their distinguishing lines:

- `toy-model/` — an unfinished sketch kept for reference; it registers as `ToyModel1` but implements the same linear diffusion as its neighbour, not the `d_x((a/u^{3/2}) du/dx)` model its header comment describes.
- `jax-diffusion/` — `JAXNonlinearDiffusion` and `JAXAuxTest` on `manta.jax.JAXTransportSystem`; derivatives by `jax.grad`; needs `pip install manta[jax]`; the config selects which of the two cases runs through `TransportSystem`.
- `jax-linear-diffusion/` — the same layer with `solveAdjoint = true`, so the run computes adjoint gradients; needs the `jax` extra.
- `jax-nonlinear-adjoint/` — `manta.jax.VectorizedTransportSystem`, the batched interface, with an adjoint problem; needs the `jax` extra. Note in the README that this example's config pointed at the wrong module until this change and had never run.

- [ ] **Step 7: Commit**

```bash
git add python-examples
git add -u python/
git commit -m "Move the conf-driven Python examples into python-examples/"
```

---

### Task 4: The script-driven examples, and the build sweep

**Files:**
- Create: `python-examples/adjoints/{jvp,spatial_adjoints,runner}.py`, `AdjointAnalysis.ipynb`, `README.md`
- Create: `python-examples/stellarator/{stellarator,stellarator2,objective,objective2,yancc_wrapper,yancc_wrapper2,desc_optimize}.py`, `run.conf`, `desc_optimize-vp.ipynb`, `README.md`
- Modify: `Makefile`, `.gitignore`

**Interfaces:**
- Consumes: `manta.jax` (Task 1).
- Produces: nothing later depends on it.

- [ ] **Step 1: Move them**

```sh
mkdir -p python-examples/adjoints python-examples/stellarator

git mv python/JVP_example.py        python-examples/adjoints/jvp.py
git mv python/SpatialAdjoints.py    python-examples/adjoints/spatial_adjoints.py
git mv python/Runner_example.py     python-examples/adjoints/runner.py
git mv python/AdjointAnalysis.ipynb python-examples/adjoints/AdjointAnalysis.ipynb

git mv python/Stellarator.py        python-examples/stellarator/stellarator.py
git mv python/Stellarator2.py       python-examples/stellarator/stellarator2.py
git mv python/Objective.py          python-examples/stellarator/objective.py
git mv python/Objective2.py         python-examples/stellarator/objective2.py
git mv python/yancc_wrapper.py      python-examples/stellarator/yancc_wrapper.py
git mv python/yancc_wrapper2.py     python-examples/stellarator/yancc_wrapper2.py
git mv python/desc-optimize.py      python-examples/stellarator/desc_optimize.py
git mv python/Stellarator.conf      python-examples/stellarator/run.conf
git mv python/desc_optimize-vp.ipynb python-examples/stellarator/desc_optimize-vp.ipynb
```

`desc-optimize.py` gains an underscore because a hyphenated name cannot be imported. Nothing imports it today, so the rename costs nothing and removes a latent trap.

`desc_optimize-vp.ipynb` goes into `stellarator/` rather than a notebooks directory because it imports `Objective2`, `Stellarator2` and `yancc_wrapper2` — it has to sit beside them. That leaves `AdjointAnalysis.ipynb` as the only other notebook, and it is adjoint analysis, so it goes in `adjoints/`.

- [ ] **Step 2: Repoint `adjoints/`**

`jvp.py` — replace:

```python
from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
from FFIRunner import FFIRunner
```

with:

```python
from manta.jax import VectorizedTransportSystem, JAXAdjointProblem, FFIRunner
```

`jvp.py` also does `from JAXLinearDiffusion import ...`, which moved to `../jax-linear-diffusion/`. Replace that import with a local definition or delete the use — check what it needs and inline it, so `adjoints/` stays liftable.

`spatial_adjoints.py` and `runner.py` — replace:

```python
from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
```

with:

```python
from manta.jax import VectorizedTransportSystem, JAXAdjointProblem
```

`AdjointAnalysis.ipynb` — change `import MaNTA` to `import manta as MaNTA` in the cell that has it. Everything else it imports (matplotlib, netCDF4, jax, toml) is unchanged.

- [ ] **Step 3: Repoint `stellarator/`**

`stellarator.py` — replace `from FFIRunner import FFIRunner` and `from State import State` with:

```python
from manta.jax import FFIRunner, State
```

and prepend the two environment writes that left `ffi_runner.py` in Task 1, above the existing `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]` line at the top of the file:

```python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
```

`stellarator2.py` — the same import replacement, and the same two environment lines. It imports `yancc_wrapper2`, which is now a sibling; that import is unchanged.

`objective.py` — `from Stellarator import StellaratorTransport` becomes `from stellarator import StellaratorTransport`; `from yancc_wrapper import yancc_data` is unchanged.

`objective2.py` — `from Stellarator2 import StellaratorTransport` becomes `from stellarator2 import StellaratorTransport`. It already sets `TF_CPP_MIN_LOG_LEVEL` and `HDF5_USE_FILE_LOCKING` itself, so add nothing.

`desc_optimize.py` — `from Stellarator2 import StellaratorTransport` becomes `from stellarator2 import StellaratorTransport`; `from Objective2 import make_objective` becomes `from objective2 import make_objective`; `from yancc_wrapper2 import yancc_data` is unchanged.

`desc_optimize-vp.ipynb` — `import MaNTA` becomes `import manta as MaNTA`; `from Objective2 import make_objective` becomes `from objective2 import ...`; `from Stellarator2 import StellaratorTransport` becomes `from stellarator2 import ...`.

`yancc_wrapper.py`, `yancc_wrapper2.py` — no project imports; unchanged.

`run.conf` — moved as-is. It names `PythonModuleName = "StellaratorTransport"` and `PythonModuleFile = "Stellarator.py"`; update the filename to `stellarator.py`, but **do not** invent a `registerTransportSystems` for `stellarator.py`, which has none. Record the discrepancy in the README instead — it cannot be verified here, and a guess committed as a fix is worse than a documented gap.

- [ ] **Step 4: Verify what can be verified**

```sh
MANTA=$PWD
( cd python-examples/adjoints \
  && for f in jvp.py spatial_adjoints.py runner.py; do
       PYTHONPATH=$MANTA/python $MANTA/.venv/bin/python -c "
import ast,sys
ast.parse(open('$f').read())
" && echo "parsed: $f"
     done )
```

then an import check of the two that do not open a window:

```sh
( cd python-examples/adjoints \
  && PYTHONPATH=$MANTA/python $MANTA/.venv/bin/python -c "import runner" \
  && PYTHONPATH=$MANTA/python $MANTA/.venv/bin/python -c "import jvp" )
```

Expected: both import. `spatial_adjoints.py` imports matplotlib and opens a plot partway through, so it is checked by parse only.

`stellarator/` cannot be verified at all: it needs `desc`, `yancc` and `interpax`, none of which are in `requirements.txt` or installable here. Check it by parse only:

```sh
( cd python-examples/stellarator \
  && for f in *.py; do $MANTA/.venv/bin/python -c "import ast; ast.parse(open('$f').read())" \
       && echo "parsed: $f"; done )
```

- [ ] **Step 5: Update the Makefile**

`CLEAN_DATA_DIRS` at `Makefile:281` — drop `python`, which holds no run output now that the drivers have left, and add the example directories:

```make
CLEAN_DATA_DIRS = . Tests/RegressionTests python/Tests \
                  python-examples $(wildcard python-examples/*/)
```

The recipe is `find $$d -maxdepth 1`, so the trailing slashes the wildcard produces are harmless. In the comment above it, `python/desc-optimize.py` is now `python-examples/stellarator/desc_optimize.py`.

In the `clean` recipe at `Makefile:245-247`, the two `python/` scratch paths follow their directories:

```make
	rm -rf python/manta/__pycache__ python/manta/jax/__pycache__ \
	       build dist python/*.egg-info
	rm -rf .pytest_cache python/__pycache__ \
	       python/Tests/__pycache__ python/Tests/.pytest_cache \
	       python-examples/.ipynb_checkpoints \
	       $(wildcard python-examples/*/__pycache__) \
	       $(wildcard python-examples/*/.ipynb_checkpoints)
```

- [ ] **Step 6: Update `.gitignore`**

Replace `python/.ipynb_checkpoints/` with:

```
python-examples/.ipynb_checkpoints/
python-examples/*/.ipynb_checkpoints/
```

- [ ] **Step 7: Confirm the sweep does not delete tracked data**

```sh
git status --short           # note what is dirty before
make clean_data
git status --short           # must be identical
```

Expected: no tracked file is removed. `python-examples/stellarator/` holds no `.nc` or `.dat`; the DESC equilibria are `.h5`, which is deliberately not in the pattern list.

- [ ] **Step 8: Write the two READMEs**

`python-examples/adjoints/README.md` — the three scripts drive the solver through `manta.Runner` rather than a config, so they are run as `python jvp.py`; they need `pip install manta[jax]`; `spatial_adjoints.py` opens a matplotlib window; `AdjointAnalysis.ipynb` reads a run's `.nc` output and plots the gradients.

`python-examples/stellarator/README.md` — must say plainly:

```markdown
## Status: unverified

These scripts need `desc`, `yancc` and `interpax`, none of which are in the
project's `requirements.txt` and none of which are installable in the
environment this repository is tested in. Their imports were repointed at the
`manta.jax` package layout by inspection and have not been run since.

Two specific things to expect:

* `run.conf` names `registerTransportSystems` in `stellarator.py`, which does
  not define one. The config has not run in its current form and was left as
  found rather than repaired on a guess.
* There are two generations of the whole stack -- `stellarator.py` /
  `objective.py` / `yancc_wrapper.py` and their `2` counterparts.
  `desc_optimize.py` drives the second; the first is reachable only from
  `desc_optimize-vp.ipynb`. How they differ is not recorded anywhere.
```

- [ ] **Step 9: Commit**

```bash
git add python-examples Makefile .gitignore
git add -u python/
git commit -m "Move the script-driven examples into python-examples/"
```

---

### Task 5: Documentation, and the final sweep

**Files:**
- Create: `python-examples/README.md`
- Modify: `README.md`, `CLAUDE.md`, `docs/python.rst`, `docs/install.rst`, `docs/out_of_tree.rst`, `docs/conf.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: nothing.

- [ ] **Step 1: Write the index**

`python-examples/README.md`:

```markdown
# MaNTA examples

Each directory here is a self-contained example: its physics case, its config
and a README. Nothing in them reaches into the MaNTA source tree — they import
`manta` and `manta.jax` the way any code outside this repository would, which
is the point. Copy one somewhere else and it still runs.

Install first, from the repository root:

    pip install .              # or `pip install .[jax]` for the JAX examples

| Directory | What it shows | Needs |
|---|---|---|
| `linear-diffusion/` | The smallest hand-written case: one variable, constant kappa, derivatives by hand | — |
| `toy-model/` | An unfinished sketch, kept for reference | — |
| `jax-diffusion/` | `manta.jax.JAXTransportSystem`; derivatives from `jax.grad` | `manta[jax]` |
| `jax-linear-diffusion/` | The same, with `solveAdjoint = true` | `manta[jax]` |
| `jax-nonlinear-adjoint/` | `manta.jax.VectorizedTransportSystem`, the batched interface | `manta[jax]` |
| `adjoints/` | Driving the solver from Python through `manta.Runner`; JVP and spatial adjoints | `manta[jax]` |
| `stellarator/` | Stellarator transport coupled to DESC and yancc. **Unverified** | `manta[jax]`, `desc`, `yancc`, `interpax` |

A config-driven example runs with the `manta` command from inside its own
directory:

    cd linear-diffusion && manta run.conf

Output lands beside the config, named after the config's stem — `run.nc` and
`run.restart.nc`.
```

- [ ] **Step 2: Correct the docs that name moved files**

`docs/python.rst:311` — replace:

```
``python/JAXTransportSystem.py`` and ``python/State.py`` wrap the dict interface
```

with:

```
``manta.jax`` wraps the dict interface
```

and add, after that paragraph:

```rst
The layer is an optional extra, because it is the only part of the package that
needs JAX::

   pip install manta[jax]

Worked examples are in ``python-examples/jax-diffusion``,
``python-examples/jax-linear-diffusion`` and
``python-examples/jax-nonlinear-adjoint``.
```

`docs/install.rst:141` — `python/MaNTA<suffix>.so` predates the package:

```rst
     - The pybind11 extension, ``python/manta/_manta<suffix>.so``.
```

`docs/install.rst:155-159` — the two `make clean` / `clean_data` descriptions name `python` among the swept directories; they are now `python/Tests`, `python-examples` and its subdirectories.

`docs/out_of_tree.rst` — after the `pip install .` block, add:

```rst
``python-examples/`` is the worked version of everything below: each directory
there is a case, its config and a README, importing ``manta`` exactly as your
own package would. Start from ``python-examples/linear-diffusion``.
```

`docs/conf.py` — add `superpowers` to `exclude_patterns`:

```python
exclude_patterns = ["_build", "requirements.txt", "superpowers"]
```

Sphinx's `source_suffix` is `.rst` and there is no myst-parser, so the specs and
plans under `docs/superpowers/` cannot become source files — but the docs build
runs `-W --keep-going` on Read the Docs, and one line is cheaper than finding
out otherwise.

- [ ] **Step 3: Correct `README.md` and `CLAUDE.md`**

`README.md` — after the testing table, a short section pointing at
`python-examples/` and the `manta` command.

`CLAUDE.md`, "Python layer" section — the sentence

```
JAX physics cases (`python/JAXTransportSystem.py`, `python/State.py`) wrap the
dict interface in equinox modules via the `MaNTA_Decorator` / `Physics_Decorator`
adapters.
```

becomes an accurate description of `manta.jax`, including the three constraints
from this plan's Global Constraints that a future editor would otherwise undo:
that `manta/__init__.py` must not import it, that `ffi_runner` is lazy because
`runner_ffi_ops` is `#ifdef XLA_FFI`, and that no module in the layer may write
`os.environ` at import. Add `python-examples/` to the Commands section's
description of the tree.

- [ ] **Step 4: The final sweep**

```sh
ls python/                    # exactly: manta  Tests
grep -rn "sys.path" python-examples/ || echo "clean"
grep -rn "^import MaNTA$\|from JAXTransportSystem\|from JAXAdjointProblem\|from VectorizedTransportSystem\|from FFIRunner\|from State import\|from Integrator import" \
     python-examples/ python/ || echo "clean"
```

Expected: `python/` holds only `manta` and `Tests`; no `sys.path` manipulation
anywhere under `python-examples/`; no flat import of a moved module survives.

- [ ] **Step 5: Run everything**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make python
make python_tests
make typecheck
make stubs-check
make test
make regression_tests
```

Expected: all pass, with `test_reference_solutions.py::test_jax_aux_test` still
the known `strict=True` xfail. The C++ suites are run because `Makefile` and
`.gitignore` were touched, not because the Python change should reach them — if
either moves, that is a finding.

Then a docs build, since five `.rst` files changed and the build is `-W`:

```sh
python3 -m venv /tmp/docsvenv
/tmp/docsvenv/bin/pip install -r docs/requirements.txt
/tmp/docsvenv/bin/sphinx-build -W -j auto -b html docs docs/_build/html
```

- [ ] **Step 6: Commit**

```bash
git add python-examples/README.md README.md CLAUDE.md docs
git commit -m "Document the package/examples split"
```

---

## Verification summary

| Claim | How it is checked |
|---|---|
| `manta.jax` importable and complete | `python/Tests/test_jax_layer.py` |
| `import manta` stays numpy-only | `test_importing_manta_does_not_drag_in_jax` (AST scan of `__init__.py`) |
| `ffi_runner` not eagerly imported | `test_ffi_runner_is_not_imported_eagerly` |
| `PythonModuleFile` resolves beside the config | `python/Tests/test_cli_modules.py` |
| No test outcome changed | `make python_tests` before and after each task |
| Five examples run out-of-tree | Task 3 Step 5, each from inside its own directory |
| `adjoints/` imports cleanly | Task 4 Step 4 |
| `stellarator/` | **Not verified** — `desc`, `yancc`, `interpax` unavailable. Parse-checked; stated in its README |
| `clean_data` deletes nothing tracked | Task 4 Step 7 |
| Docs still build under `-W` | Task 5 Step 5 |
