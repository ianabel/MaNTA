# The Python layer

The `manta` package, its trampolines and runner, `manta.jax`, the type stubs, and how the package is staged in the build tree.

## Python layer

`Python.cpp` defines `_manta`, the compiled core of the **`manta` package**;
`python/manta/__init__.py` re-exports it and adds the parts better written in
Python. Users `import manta`, and `pip install .` makes that work from anywhere
(the build shells out to CMake, reusing `build/` if it is configured, so it still
needs a working dependency setup — see `setup.py`).
A Python case declares itself with class attributes:

```python
class MyCase(manta.TransportSystem):
    variables = [manta.Field("n", "density", "m^-3", lower=manta.Neumann)]
    def __init__(self, config, grid):
        super().__init__()          # reads the class attributes
```

`super().__init__(spec)` and `super().__init__(variables=[...])` also work, for a
case whose shape depends on its config. `nVars`/`nScalars`/`nAux` are read-only.
**Only `SigmaFn` and `Sources` are required**: an absent derivative hook means
that block is identically zero, which is what the zeroed out-parameter already
gives. Four pieces to know:

* **Trampolines.** `PyTransportSystem` / `PyAdjointProblem` dispatch each virtual
  to a Python override. `PyTransportSystem::initializeOverrides` probes the
  subclass once. It is *vectorised* if it overrides both `ComputePhysics` and
  `ComputePhysicsDerivatives`; otherwise each hook is pointwise or batched
  (`SigmaFn_v`, `Sources_v`, `AuxG_v`, `dSigma`, `dSources`, `AuxGPrime_v` --
  `batched_hook_names`), freely mixed, and a requirement -- including the extra
  ones when `nScalars > 0` or `nAux > 0` -- is met by either form. Look up
  overrides with `override_for(name)`, never `method_overrides[name]` — the
  latter default-constructs a null `py::function` and calling it segfaults. And
  hand `py::get_override` a stable name pointer, never a temporary's `c_str()`:
  pybind11 caches a *missing* override keyed on the pointer, so a reused buffer
  makes one name's miss answer for the next.
* **State is a view, GlobalState is a dict.** A pointwise `State` reaches Python
  as a non-owning window onto solver memory (`PyState.hpp`) with named fields —
  `s.u`, `s.q`, `s.sigma`, `s.sigmaHat`, `s.phi`, `s.scalars` — each indexable by
  position or by declared name. It is valid only inside the call; `np.array(v,
  copy=True)` to keep anything, and note that `__array__` has to honour `copy`
  or numpy hands back a view of a destroyed temporary. There is no way to build
  one from Python, which is why the tests drive the pointwise path through the
  batched entry point. `GlobalState` still crosses as a `dict` of
  `(nPoints, nVars)` arrays — what the JAX path wants — and **its caster
  transposes in both directions** (C++ stores `(nVars, nPoints)`), so a
  round-trip test cannot detect a missing transpose; check the orientation from
  inside a batched call instead. Its `"VariableDot"` entry is `du/dt`, and it is
  an **empty** array rather than a grid of zeros in a run where no variable
  declares `sourceReadsTimeDerivatives` — and also in the calls that build the
  initial condition, which happen before there is a `dYdt` to read. A batched
  case therefore tests `.size` rather than indexing it. The pointwise view does
  not have that shape: `s.udot` is always `nVars` long and reads zero.
* **`prepareEvaluation` is optional, and dispatched with `PYBIND11_OVERRIDE`**
  like `aFn`, so a case not defining it pays an override lookup per new plan. The
  plan crosses as a copy, with numpy point sets. `regrid = manta.Regrid.InPlace`
  is a class attribute that `manta.TransportSystem.__init__` honours on both its
  paths, copying an explicit spec rather than editing it; the JAX base classes
  deliberately leave it `Fixed`, since a subclass may cache what the plan says.
  `Grid` and the plan types are bound *before* `TransportSystem` so the generated
  stub names them. A case object handed to `Runner(system)` is never told its grid
  at construction -- `configure()` builds it afterwards -- so the plan is the only
  place such a case learns the mesh, and reconfiguring one object onto another
  mesh or degree needs `InPlace`.
* **`PyRunner`** (`configure(dict)` / `run` / `run_ss` / `getSolution` / `G` /
  `getAdjointGradients`) is the API the optimisation drivers use, and the only
  route supporting repeated configure/run cycles in one process — it works by
  building a *fresh* `SystemSolver` in every `configure()` (`PyRunner.cpp:117`),
  which is load-bearing. Its parameter table is
  declarative and lives at the top of `PyRunner.cpp`.

  **A steady solve can be driven in slices from here too**, which is what makes
  the resumable continuation above reachable from Python: `start_steady` /
  `continue_steady` / `finish_steady` / `abandon_steady`, wrapped as
  `manta.SteadySolve`. Three things are load-bearing. `OutOfSteps` is *returned*
  and a `SolverFailed` throws, so a driver tells a spent budget from a dead solve
  without reading a message — which relies on `solveSteadyState` clearing
  `steadyOutcome` on entry, since `finish()` is the only thing that sets it and
  an exception from inside the residual bypasses it entirely; a stale
  `OutOfSteps` there is an infinite loop, not a wrong label. Each slice calls
  `captureState()`, because
  `getSolution` reads `yJac` and would otherwise hand back the initial condition
  — silently, since `yJac` is always *a* valid state. And a live loop owns
  SUNDIALS objects that **nothing else frees** — `~SystemSolver` does not call
  `destroySundials` — so `configure()`, `~PyRunner` and any exception out of the
  loop all abandon it explicitly; that is why the context manager is the form to
  prefer. Under `DegreeAdaptation` the loop is `runAdaptiveDegree`'s own,
  `DegreeAdaptationLoop`, a level at a time: the slice that converges a level
  finishes the run on it exactly as `runSolver` would, closes the level, and if
  the degree goes up builds and initialises the next and returns `Refined` (the
  enum's last value, so the FFI's int32s keep their meaning). One loop is what
  makes a sliced run land on `run_ss()`'s state bit for bit
  (`test_a_sliced_degree_adaptation_reaches_the_answer_run_ss_does`); a second
  copy of the sequencing would drift. Every exit clears the restart state the
  level transfers set -- `abandonAdaptation` -- or the next run on the
  configuration resumes from the last level handed on. `MeshAdaptation` and the
  ladders are still refused: they replace the mesh too.
  The same four names are FFI ops (`ffi.hpp`, CPU only like `run`/`run_ss`), so
  `manta.SteadySolve(ffi_runner)` works — `steadyStats()` and
  `objectiveEstimate()` need none, being host-side reads that touch no device
  memory. The outcome crosses as an `int32`. Eagerly `FFIRunner` makes it a
  `SteadyOutcome`, which forces the sync a Python `while` needs; under a trace
  (`jit`, a `custom_jvp`) the solve has not run, `int()` of the tracer raises,
  and it comes back as the traced scalar. **`SteadyOutcome` is bound with
  `py::native_enum` as an `enum.IntEnum` for that reason**: its members are
  ints, so `jnp.equal(outcome, SteadyOutcome.Converged)` traces, where a
  `py::enum_` member is an object JAX refuses. A single slice can therefore be
  branched on under `jit` with `lax.cond`, which is how
  `python-physics/stellarator` drives its solve; a *loop* of slices still
  cannot, and belongs in eager code or an `io_callback`.
  `test_ffi_steady_outcome.py` pins both, on an `XLA_FFI` build only.

  `G` returns the objective without the gradient. The saving is in the run, not
  in `G` itself: `integrate` calls `runAdjointSolve()` whenever `solveAdjoint` is
  set, so with `solveAdjoint = True` the gradients are already computed by the
  time `run` returns and `getAdjointGradients` only reads `G_p`. Configure with
  `solveAdjoint = False` to skip the adjoint solve, and `G` builds an
  `AdjointProblem` on demand purely to evaluate `GFn`. That object is kept in a
  separate member from `adjoint` so its presence can never be mistaken for "the
  gradients have been computed". Under `Superconvergent` it is the `u*`-based
  objective, the same number `getAdjointGradients()[0]` and
  `objectiveEstimate()["value"]` report; `G` used to evaluate the plain one and
  disagreed with them by up to 1.1% at `k = 1`, which
  `test_G_agrees_with_the_G_from_getAdjointGradients` now pins in both schemes.

**The scalar hooks are one name each, and the same in both languages:**

```
ScalarG(s, y, ydot, abscissae, weights, phiBoundary, t)          -> Value
ScalarGPrime(dG, dGdot, y, ydot, abscissae, weights, phiBoundary, t)
InitialScalarValue(s)                                            -> Value
InitialScalarDerivative(s, y, dydt)   [C++ takes DGSoln; Python takes nodal data]
dSources_dScalars(s, out, state, x, t)   — indexed by *scalar*, not by variable
```

`y`/`ydot` are the solution sampled on the element nodes; `weights` is one
quadrature weight per node, so `Int u dx` is `ScalarHooks::integrate(...)` and
its derivative with respect to node j is `weights(j)`. **A case must use these
weights rather than a rule of its own** — `ScalarTestLD3` used a global adaptive
Kronrod rule over a piecewise polynomial, which is not a smooth function of the
coefficients, and disagreed with its own Jacobian by 8%. `phiBoundary` is
`(k+1, 2)` and is the *only* way to reach a boundary point value, because the
nodes are Chebyshev points of the first kind and strictly interior;
`ScalarHooks::boundaryValue` / `addBoundaryDerivative` do that contraction.
`ScalarGPrime` reports every scalar at once, as d/d(DOF), into buffers that
arrive zeroed.

`checkScalarDerivative` in `ScalarJacobianTests.cpp` finite-differences a case's
own `ScalarG` against its `ScalarGPrime` and is the first thing to run when a
scalar system converges slowly. `python/Tests/test_scalars.py` exercises the
Python side against a closed form, algebraic and differential.

* **`manta.cli`** is the `manta` console script: it reads the config, imports the
  physics module it names, and hands over to the same `runManta` the binary uses.
  That is what lets a Python case and its driver live outside this tree.
  `load_physics_modules` accepts **two forms**, and `python/Tests/util.py`
  delegates to it so the rule has one implementation:

  * `PythonModule` — an importable dotted name, the documented form.
  * `PythonModuleFile` (+ optional `PythonModuleName`) — a path, **resolved
    relative to the config file, not the cwd**. Every config in the tree is
    written as though that were true; the retired `PyManta` resolved against the
    cwd, which is why they only ran from one directory.

  Either way the module is imported for its registrations, and if it defines
  `registerTransportSystems()` that is called afterwards. Registering at import
  via `registerPhysicsCase` is the documented convention; the hook is what every
  example under `python-examples/` actually uses, and without honouring it none
  of those configs run. `python-physics/` is neither: both systems there build a
  `manta.Runner` in Python and register nothing.

JAX physics cases use **`manta.jax`** (`python/manta/jax/`), which wraps the dict
interface in equinox modules via the `MaNTA_Decorator` / `Physics_Decorator`
adapters. It was six flat modules in `python/` until the examples split, and
three properties of the subpackage are load-bearing rather than stylistic:

* **`manta/__init__.py` must never import `.jax`.** JAX is an optional extra
  (`pip install manta[jax]`), so a top-level re-export would make it mandatory
  for every user of the solver — and it would turn the subpackage's relative
  imports into a cycle, since `transport_system.py` does `import manta`.
  `test_jax_layer.py` AST-scans `__init__.py` for exactly this.
* **`manta/jax/__init__.py` must not import `ffi_runner` eagerly.** That module
  registers XLA FFI targets at module scope, and the bindings it looks for —
  `runner_ffi_ops`, `runner_ffi_ops_cuda` — are `#ifdef XLA_FFI`
  (`Python.cpp:361`). Eager, it would take `from manta.jax import State` down on
  every default build. A module-level `__getattr__` serves `FFIRunner`,
  `Platform` and the two `register_ffi_*` helpers instead, and the module itself
  raises an `ImportError` naming the flag rather than an `AttributeError` from
  inside a loop.
* **No module in the layer may write `os.environ` at import.** `FFIRunner.py`
  set three variables and `JAXTransportSystem.py` forced
  `JAX_PLATFORM_NAME=cpu`; as library code that last one would have disabled the
  GPU path `ffi_runner` exists to provide. Those writes live in the drivers that
  need them now — `python-physics/stellarator/`.
* **A JAX case's global scalars come from one pure function**,
  `scalar(index, states, states_dot, integrator, t, params)`, beside `sigma`,
  `source` and `aux` (`manta/jax/scalars.py`). Both bases derive `ScalarG`,
  `ScalarGPrime` and `dSources_dScalars` from it and `source`, and
  `JAXAdjointProblem` derives `dScalarG_dp` — the one hook with no C++ default.
  It is optional: a case writing `ScalarG` itself, as mirror-plasma does, keeps
  its own methods, and only its adjoint stops, asking for `scalar`. Two details
  are load-bearing. `dg`'s `Scalars` entry has to be *integrated* dG/dmu, and a
  vmapped gradient carries it per point — `State.to_manta` would keep the first
  point's and drop the rest — so `JAXAdjointProblem.dg` contracts it with
  `manta.getPhysicsWeights` over the plan's grid. And with spatial parameters
  the pointwise `dSources_dScalars` finds its node's parameters by position in
  the plan's `Residual` points; with uniform fields, or a coupling that does not
  read them, a wrong lookup is exact, which is why
  `test_a_spatial_scalar_coupling_reads_each_nodes_parameters` uses neither.

`python/` holds exactly two things: `manta/` and `Tests/`. Every driver, config
and notebook is one self-contained directory under either **`python-examples/`**
or **`python-physics/`**, and the split between those two is *purpose*, not
mechanism. Both import `manta` and `manta.jax` absolutely, the way out-of-tree
code does; both survive being copied elsewhere. An example is written to be
read — small, one idea each, no dependency outside `requirements.txt`. A system
under `python-physics/` is run to get physics: `mirror-plasma/` (a package, its
own tests, needs `optimistix`) and `stellarator/` (DESC, yancc, interpax, and an
`XLA_FFI` build). Neither `python-physics/` system is reached by
the `python` CTest test — `pytest.ini` is `testpaths = python/Tests` — so nothing
in CI runs them, and their READMEs carry the status instead.

## Type stubs

`python/manta/_manta.pyi` is **generated**: `stubs` writes a fresh one into
`<build>/stubs/`, `stubs-check` diffs the committed one against it and is what CI
runs, because a stale stub is worse than none — it reports the old signature as
fact — and `stubs-update` copies it over. `typecheck` runs mypy over the build's
package, i.e. against the *committed* stub, which is what ships.
`python/manta/__init__.pyi` is hand-written and covers the Python layer, chiefly
the hook signatures a case implements.

Three things to know:

* Every hook declaration carries `# type: ignore[override]`, and that is load
  bearing rather than a workaround. The bound base method's signature is the C++
  one — the derivative hooks take a `VectorRef` out-parameter, and pybind11
  widens `int` to `SupportsInt | SupportsIndex` — while a Python case writes
  `(i, state, x, t)` and *returns* the vector. The declarations are what a
  user's subclass is checked against, which is the point; `warn_unused_ignores`
  is on, so the ignores cannot outlive the asymmetry.
* `check_untyped_defs` is what makes any of this bite. A physics case is
  unannotated Python and mypy skips unannotated defs without it.
* Two bindings were removed to keep the generated stub valid:
  `TransportSystem::ScalarGPrime` and `ComputePhysicsDerivatives` both take
  `GlobalStateMatrix`, which has no Python type, so the bound base methods were
  never callable from Python — they only put unresolvable names in the stub.
  `registerPhysicsCase` is `def`d after `TomlValue` for the same reason:
  pybind11 renders a `std::function` parameter from the types registered at that
  point, and binding it earlier left the raw toml11 C++ name in the signature.

## The package in the build tree

**The Python package lives in the build tree, one per build directory.**
`_manta` links into `<build>/python/manta/`, and `manta_package` fills the rest
of that directory with a symlink to each `.py`/`.pyi` under `python/manta/`
(`cmake/MantaStagePackage.cmake`, run on every build, so an added or deleted
file is picked up without a reconfigure). `<build>/python` is then a `sys.path`
entry, and every Python process a target or test starts gets it through
`MANTA_PYTHON_ENV` (`python/CMakeLists.txt`), together with a
`PYTHONPYCACHEPREFIX` in the build tree for the modules imported from the
source tree — the test files, and pytest's assertion rewrite of them. Links
rather than copies, so a `.py` edit reaches the next import without a rebuild,
and because Python does not resolve them — a module's `__file__` is the path it
was found at — the package's own `__pycache__` lands in the build tree too.

What that buys is that two build directories cannot see each other's
extension, which a single shared location cannot promise: CMake calls a target
up to date when the file on disk is newer than its objects, whoever wrote it.
What it leaves is the one way left to test the wrong module — something ahead
of `PYTHONPATH` on `sys.path`, an installed `manta` in a `.pth` say — so
`conftest.py` compares `manta.__file__` and `manta._manta.__file__` against
`MANTA_PYTHON_ROOT` and exits if either is elsewhere. That check is what lets a
coverage run's Python suite be trusted to have measured the instrumented
module: the failure it guards is silent, a passing suite and a report that
never mentions the binding layer.

Two things follow for anything that ships the package. `pip install .` takes
the `.py` files from `python/manta/` and copies the extension in from the build
directory (`setup.py`); pyproject's package-data deliberately does not list
`*.so`, so a stray one in the source tree cannot be shipped instead. And
`pip install -e .` has no extension to find — `PYTHONPATH=<build>/python` is
the editable form.


**The extension's ABI suffix and the interpreter that imports it come from the
same place.** `cmake/MantaPython.cmake` finds **one** interpreter -- `$VIRTUAL_ENV`,
else `build/venv`, else a `.venv` in the repo root -- and pybind11 derives the
headers and the suffix from it, so the three cannot disagree. Name a different
one with `-DPython3_EXECUTABLE=...`; `setup.py` passes `sys.executable` for the
same reason, so `pip install .` always builds for the interpreter doing the
installing. If an import of `_manta` fails, compare `ls <build>/python/manta/*.so`
with `python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'`:
the answer is a build directory configured for a different interpreter.
