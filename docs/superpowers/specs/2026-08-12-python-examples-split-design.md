# Splitting `python/` into infrastructure and examples

Date: 2026-08-12
Status: approved, ready for an implementation plan

## Why

`python/` currently holds three unrelated kinds of thing in one flat directory:
the installable `manta` package, a JAX/vectorised framework layer that several
things import, and a pile of driver scripts, configs and notebooks. Nothing
about the directory says which is which, and the flat layout forces every
consumer — drivers *and* the test fixtures — to reach the framework through
`sys.path`, with `from State import State`-style imports that only resolve when
`python/` happens to be on the path.

That was tolerable before the tree grew an out-of-tree story. It is not now.
`make install PREFIX=...` and `pip install .` exist so that a physics case and
its driver can live in the user's own repository; the drivers shipped here
demonstrate the opposite, because none of them can be lifted out of `python/`
and still run. The point of this change is that the example directory should be
the worked demonstration of the thing the build already supports.

## Decisions taken

Four choices were settled before designing, and the design below assumes them:

1. **The JAX framework layer moves into the package** as `manta.jax`, rather
   than staying flat or becoming a separate `manta_jax` distribution. This is
   what removes `sys.path` from the examples entirely.
2. **`manta.cli` learns `PythonModuleFile`** alongside `PythonModule`, so the
   existing configs keep working. The two forms converge later; this change
   does not force a conversion.
3. **`python-examples/` is one self-contained directory per example**, each with
   its case module, its config and a README, so a directory can be copied out of
   the repository and still run.
4. **JAX becomes an optional extra**, `pip install manta[jax]`. `import manta`
   stays numpy-only.

## 1. `python/manta/jax/`

Six modules move from `python/` into a new subpackage. Cross-imports become
relative, so nothing resolves through `sys.path`.

| From | To | Contents |
|---|---|---|
| `python/State.py` | `python/manta/jax/state.py` | `State`, `Physics_Decorator`, `MaNTA_Decorator`, `ScalarG_Decorator`, `ScalarGPrime_Decorator` |
| `python/Integrator.py` | `python/manta/jax/integrator.py` | `Integrator` |
| `python/JAXTransportSystem.py` lines 1–162 | `python/manta/jax/transport_system.py` | `JAXTransportSystem` |
| `python/JAXAdjointProblem.py` | `python/manta/jax/adjoint_problem.py` | `JAXAdjointProblem` |
| `python/VectorizedTransportSystem.py` | `python/manta/jax/vectorized.py` | `VectorizedTransportSystem` |
| `python/FFIRunner.py` | `python/manta/jax/ffi_runner.py` | `Platform`, `register_ffi_cpu`, `register_ffi_gpu`, `FFIRunner` |

`JAXTransportSystem.py` is the only file that splits rather than moves. The
`JAXTransportSystem` class ends at line 162; everything from the
`# Need PyTree structure for class parameters` comment at line 165 to the end of
the file is two demo physics cases (`JAXNonlinearDiffusion`, `JAXAuxTest`), a
`NamedTuple` of their parameters and a `registerTransportSystems()`, none of
which is framework. They become an example — see section 2. The import block at
the top serves both halves and is split by what each half actually uses.

`python/manta/jax/__init__.py` re-exports the public names and converts a
missing dependency into an actionable message:

```python
try:
    import equinox as _equinox  # noqa: F401
except ImportError as e:
    raise ImportError(
        "manta.jax needs the JAX extra: pip install manta[jax]"
    ) from e
```

Two properties of this layout look wrong and are not, so they are worth
recording:

* **`manta.jax` does not shadow JAX.** Python 3 has no implicit relative
  imports, so `import jax.numpy as jnp` inside `manta/jax/state.py` binds the
  real JAX. Only `from . import jax`-style spellings would reach the
  subpackage, and none are used.
* **There is no import cycle, and that depends on `manta/__init__.py` never
  importing `.jax`.** It must not, or JAX would become a hard dependency of
  every `import manta`. Because it does not, `from manta.jax import X` runs
  `manta/__init__.py` to completion first, and `from .. import TransportSystem`
  inside the subpackage is safe. Anyone adding a convenience re-export of
  `manta.jax` to the top-level `__init__.py` would break both properties at
  once.

After this, `python/` contains exactly two entries: `manta/` and `Tests/`.

## 2. `python-examples/`

```
python-examples/
  README.md
  linear-diffusion/       linear_diffusion.py       run.conf  README.md
  toy-model/              toy_model.py              run.conf  README.md
  jax-diffusion/          jax_diffusion.py          run.conf  README.md
  jax-linear-diffusion/   jax_linear_diffusion.py   run.conf  README.md
  jax-nonlinear-adjoint/  jax_nonlinear_adjoint.py  run.conf  README.md
  adjoints/               jvp.py  spatial_adjoints.py  runner.py  README.md
  stellarator/            stellarator.py   stellarator2.py
                          objective.py     objective2.py
                          yancc_wrapper.py yancc_wrapper2.py
                          desc_optimize.py run.conf  README.md
  notebooks/              AdjointAnalysis.ipynb  desc_optimize-vp.ipynb  README.md
```

Module names are distinct rather than a `case.py` in every directory, because
the config names the module and a name is more use than a position.

The complete mapping:

| From | To |
|---|---|
| `python/PythonLinearDiffusion.py` | `python-examples/linear-diffusion/linear_diffusion.py` |
| `python/py.conf` | `python-examples/linear-diffusion/run.conf` |
| `python/ToyModel1.py` (untracked) | `python-examples/toy-model/toy_model.py` |
| — | `python-examples/toy-model/run.conf` (new) |
| `python/JAXTransportSystem.py` lines 165–269 | `python-examples/jax-diffusion/jax_diffusion.py` |
| `python/jax_example.conf` | `python-examples/jax-diffusion/run.conf` |
| `python/JAXLinearDiffusion.py` | `python-examples/jax-linear-diffusion/jax_linear_diffusion.py` |
| `python/JAXLinearDiffusion.conf` | `python-examples/jax-linear-diffusion/run.conf` |
| `python/NonlinearDiff_example.py` | `python-examples/jax-nonlinear-adjoint/jax_nonlinear_adjoint.py` |
| `python/jax_adjoint.conf` | `python-examples/jax-nonlinear-adjoint/run.conf` |
| `python/JVP_example.py` | `python-examples/adjoints/jvp.py` |
| `python/SpatialAdjoints.py` | `python-examples/adjoints/spatial_adjoints.py` |
| `python/Runner_example.py` | `python-examples/adjoints/runner.py` |
| `python/Stellarator.py`, `Stellarator2.py` | `python-examples/stellarator/stellarator.py`, `stellarator2.py` |
| `python/Objective.py`, `Objective2.py` | `python-examples/stellarator/objective.py`, `objective2.py` |
| `python/yancc_wrapper.py`, `yancc_wrapper2.py` | `python-examples/stellarator/yancc_wrapper.py`, `yancc_wrapper2.py` |
| `python/desc-optimize.py` | `python-examples/stellarator/desc_optimize.py` |
| `python/Stellarator.conf` | `python-examples/stellarator/run.conf` |
| `python/AdjointAnalysis.ipynb` | `python-examples/notebooks/AdjointAnalysis.ipynb` |
| `python/desc_optimize-vp.ipynb` | `python-examples/notebooks/desc_optimize-vp.ipynb` |
| `python/MantaPythonTransport.py` | deleted |
| `python/PyManta` | deleted |

`desc-optimize.py` gains an underscore because a hyphen makes a module
unimportable; nothing imports it today, so the rename costs nothing and removes
a latent trap.

Every moved module changes its imports to the package form — `import manta`,
`from manta.jax import JAXTransportSystem` — and drops any `sys.path`
manipulation. That is the whole point: an example must import the way a user
outside this tree imports.

Each directory's README says what the example demonstrates, what it needs
installed, and how to run it. Anything conf-driven needs `pip install .` (or
`pip install -e .`) first, because the `manta` console script is what runs it.

## 3. `manta/cli.py` learns `PythonModuleFile`

The configs in the tree name a physics module by *file path*
(`PythonModuleName` + `PythonModuleFile`), a convention implemented only by the
now-deleted `PyManta` and by `python/Tests/util.py`. `manta.cli` reads
`PythonModule`, an importable module name. It gains the file form:

* If the config has `PythonModuleFile`, load it with
  `importlib.util.spec_from_file_location` and register it in `sys.modules`
  under `PythonModuleName`, or the file stem when that key is absent.
* **Resolve the path relative to the config file's directory**, not to the
  current directory. Every config in the tree is written as though this were
  already true; `PyManta` resolved against the cwd, which is why those configs
  only ever worked from one directory.
* After importing — on *either* path — call `registerTransportSystems()` if the
  module defines it. Every example module in the tree uses that idiom rather
  than registering at import, so without this the converted configs do not run.
  It is documented as the legacy hook, with import-time
  `manta.registerPhysicsCase` remaining the documented way.

Failures keep cli's existing shape: a missing file or an unimportable module
raises `SystemExit` with a message naming the config key at fault, not a
traceback.

`python/Tests/util.py` is a copy of the same loader. It becomes a call into the
cli helper, so the rule has one implementation. The tests that drive it are
unchanged in behaviour.

## 4. Repairs folded in

Two deletions, both of things that cannot work as written:

* `MantaPythonTransport.py` — a `SyntaxError` (`def InitialValue(index, x):`
  and `def InitialDerivative(index, x):` have no bodies), referring to an
  unqualified `TransportSystem`, imported by nothing.
* `PyManta` — imports the retired top-level `MaNTA` module. Section 3 makes the
  `manta` script cover everything it did.

Three configs are stale in ways the move must fix or flag:

* `py.conf` names `/home/eatocco/projects/MaNTA/python/PythonLinearDiffusion.py`.
  Becomes `linear_diffusion.py`, resolved beside the config.
* `jax_adjoint.conf` points `PythonModuleFile` at
  `VectorizedTransportSystem.py`, which defines only the base class — no
  `JAXNonlinearDiffusion`, no `registerTransportSystems`. The module that
  defines that case is `NonlinearDiff_example.py`, so the config follows it into
  `jax-nonlinear-adjoint/` and points at `jax_nonlinear_adjoint.py`.
* `Stellarator.conf` names `Stellarator.py`, which has no
  `registerTransportSystems` either. Unlike the previous two this cannot be
  verified here — the module needs `desc`, `yancc` and `interpax` — so it is
  moved as-is and its README records the discrepancy rather than a guess being
  committed as a fix.

`ToyModel1.py` registers itself as `"PythonLinearDiffusion"`, colliding by name
with the other example. It is renamed to match its directory. The file is
currently untracked; it is added as part of the move.

## 5. Build and tooling

* **`pyproject.toml`**: add

  ```toml
  [project.optional-dependencies]
  jax = ["jax", "equinox", "jaxtyping"]
  ```

  `[tool.setuptools.packages.find]` already globs `manta*`, so the subpackage
  needs no new entry.
* **`mypy.ini`**: `files = python/manta` now sweeps in the JAX layer, which is
  unannotated, under `check_untyped_defs = True`. Add

  ```ini
  [mypy-manta.jax.*]
  ignore_errors = True
  ```

  following the `manta._manta` precedent, with a comment saying to widen it when
  the layer is annotated rather than leaving it as permanent cover. The file's
  header comment ("The rest of `python/` is drivers, JAX experiments and test
  fixtures…") describes a directory that will no longer exist and is rewritten.
* **`Makefile`**:
  * `CLEAN_DATA_DIRS` — drop `python`, which will hold no run output once the
    drivers leave, and add `python-examples` plus `$(wildcard python-examples/*/)`.
    The `clean_data` recipe is `find $$d -maxdepth 1`, so trailing slashes are
    harmless. The curated comment above it is updated in the same edit: it
    names `python/desc-optimize.py` as its reason for excluding `.h5`.
  * `clean` — the `python/__pycache__` and `python/.ipynb_checkpoints` entries
    follow their directories; add `python/manta/jax/__pycache__` and the
    `python-examples` equivalents.
* **`.gitignore`**: `python-examples/.ipynb_checkpoints/` beside the existing
  `python/.ipynb_checkpoints/`, which is removed with the notebooks.
* **Unaffected**: `requirements.txt` (already pins jax, equinox and jaxtyping
  for the test suite), `setup.py`, `pytest.ini`, `make stubs` / `stubs-check`
  (they cover `manta._manta` only), and the CI workflow, which drives everything
  through make targets whose names do not change.

## 6. Tests

`python/Tests/` does not move.

* `Tests/JAXLinearDiffusion.py` and `Tests/JAXAuxTest.py` drop their
  `sys.path.insert(0, '../')` and import `from manta.jax import
  JAXTransportSystem` / `JAXAdjointProblem`.
* `Tests/conftest.py` still puts `python/` on `sys.path`, because an
  un-installed `manta` is found no other way. Its docstring bullet about "the
  JAX transport-system modules in `python/`" is no longer true and goes.
* `Tests/util.py` delegates to the cli loader (section 3).

No test's assertions change. If any test outcome changes, that is a defect in
the move, not an expected consequence of it.

## 7. Docs

* `docs/python.rst:311` and the Python-layer section of `CLAUDE.md` both name
  `python/JAXTransportSystem.py` and `python/State.py`.
* `docs/install.rst:141` still says `python/MaNTA<suffix>.so`, which predates the
  package; and its `make clean` description at :155-159 lists the directories
  that are changing.
* `docs/out_of_tree.rst` gains a pointer to `python-examples/` as the worked
  demonstration of what it describes.
* `docs/testing.rst` and `Tests/README.md` reference `python/Tests`, which does
  not move — no change expected, to be confirmed rather than assumed.
* `README.md`'s Python section gains `python-examples/`.
* New: `python-examples/README.md`, indexing the examples and stating which need
  dependencies that are not in `requirements.txt`.
* `docs/conf.py`: add `superpowers` to `exclude_patterns`. Sphinx's
  `source_suffix` is `.rst` only and there is no myst-parser, so this spec
  cannot become a source file — but the docs build runs `-W --keep-going` on
  Read the Docs, and a one-line exclusion is cheaper than discovering otherwise.

## 8. Verification — what "done" means

Passing:

* `make python`, `make python_tests`, `make typecheck`, `make stubs-check`.
* `python -c 'import manta'` with JAX uninstalled still works (the numpy-only
  guarantee), and `from manta.jax import JAXTransportSystem` in that state
  raises the `pip install manta[jax]` message rather than a bare
  `ModuleNotFoundError` on equinox.
* The four conf-driven examples — `linear-diffusion`, `toy-model`,
  `jax-diffusion`, `jax-linear-diffusion` — and `jax-nonlinear-adjoint`, whose
  config becomes runnable for the first time under the repair in section 4, each
  run to completion via `manta run.conf` executed **from inside their own
  directory**. That is the property the whole change exists to demonstrate.
* The three `adjoints/` scripts import cleanly against the new package layout.
  They are driven through `manta.Runner` rather than a config and are checked by
  import, not by a full run; `spatial_adjoints.py` additionally opens a
  matplotlib window, which is not something to run unattended.

Explicitly **not** verified: `stellarator/` and `notebooks/`. They need `desc`,
`yancc` and `interpax`, none of which are in `requirements.txt` or installable in
this environment. Their imports are repointed by inspection, and their READMEs
say plainly that they are unverified and what they require. This is recorded as a
gap rather than papered over: an example that has never been run should not sit
in a directory that implies it has.

## Out of scope

* Converting the example configs from `PythonModuleFile` to `PythonModule`.
  Decision 2 keeps both; convergence is a later change.
* Annotating the JAX layer for mypy. The `ignore_errors` entry is a marker for
  that work, not a substitute.
* `plot.py` and `CylindricalMagneticField.py` at the repository root. They are
  Python and arguably belong in an examples directory, but they were never in
  `python/` and moving them is a different change.
* Retiring the first-generation Stellarator stack (`Stellarator.py`,
  `Objective.py`, `yancc_wrapper.py`). Considered and declined: only
  `desc_optimize-vp.ipynb` reaches it, and how the two generations differ is not
  determinable from the tree.

## Risks

* **The move is large and mostly mechanical, which is where silent breakage
  hides.** Every moved file's imports change; a missed one fails only when that
  script is run, and most of them are never run in CI. Mitigation: the
  unverifiable set is enumerated above rather than assumed working.
* **`git` rename detection** will be noisy where a file both moves and has its
  imports rewritten. Doing the moves and the edits as separate commits keeps the
  history readable.
* **`manta.jax` under `check_untyped_defs`** would produce a wall of findings
  without the `ignore_errors` entry, and a wall of findings is how a check stops
  being read — the reasoning already written into `mypy.ini`.
