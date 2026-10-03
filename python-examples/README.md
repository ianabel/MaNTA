# MaNTA examples

Each directory here is a self-contained example: its physics case, its config
and a README. Nothing in them reaches into the MaNTA source tree — they import
`manta` and `manta.jax` the way any code outside this repository would, which is
the point. Copy one somewhere else and it still runs.

**These exist to show how the framework is used, not to produce physics.** They
are small enough to read in one sitting and are deliberately kept that way. The
real simulations live in [`../python-physics/`](../python-physics/), which
follows the same self-contained-directory convention.

Either install the package, from the repository root:

    pip install .              # or `pip install .[jax]` for the JAX examples

or use a build's package directory in place, with no install:

    cmake --build $BUILD_DIR --target _manta
    export PYTHONPATH=$BUILD_DIR/python

The second is the one to use while changing MaNTA itself: the package there is
the extension that build linked beside links to `python/manta/`, so an edited
`.py` file is picked up at once and a rebuilt extension on the next import. It
has no `manta` command, which `pip` writes; `python -m manta.cli` is the same
program.

| Directory | What it shows | Needs |
|---|---|---|
| `linear-diffusion/` | The smallest hand-written case: one variable, constant kappa, derivatives by hand | — |
| `toy-model/` | An unfinished sketch, kept for reference | — |
| `jax-diffusion/` | `manta.jax.JAXTransportSystem`; derivatives from `jax.grad` | `manta[jax]` |
| `jax-linear-diffusion/` | The same, with `solveAdjoint = true` | `manta[jax]` |
| `jax-nonlinear-adjoint/` | `manta.jax.VectorizedTransportSystem`, the batched interface | `manta[jax]` |
| `nc-parallel/` | A batched JAX case that lays its flux out over N threads from MaNTA's evaluation plan, and what `PhysicsParallelism` buys on such a machine | `manta[jax]` |
| `adjoints/` | Driving the solver from Python through `manta.Runner`; JVP and spatial adjoints | `manta[jax]`; `jvp.py` also needs an `XLA_FFI` build |
| `park-convergence/` | Benchmark: spatial accuracy per flux call, against Park's IDO scheme | — |
| `jardin-critical-gradient/` | Benchmark: the nonlinear solve on a stiff gradient-dependent diffusivity | — |
| `shestakov-nonlinear/` | Benchmark: degenerate diffusion, and a boundary MaNTA cannot cross | — |

The last three are a different kind of thing and are marked **Benchmark** above.
They exist to *measure* rather than to demonstrate: each reproduces a problem
from a paper in [`../refs/`](../refs/Refs.md) that has a closed-form solution,
and each ships a `benchmark.py` reporting accuracy against the number of calls
into the `TransportSystem` — the metric [`../PERFORMANCE.md`](../PERFORMANCE.md)
asks MaNTA to be judged by, and compared against the algorithms it names.
`shestakov-nonlinear/` is the odd one: the problem as its paper states it is one
MaNTA cannot integrate, so its `benchmark.py` maps where the tractable region
ends and identifies what stops it. Nothing in CI runs any of them (`pytest.ini`
is `testpaths = python/Tests`), so their READMEs carry the measured numbers.

A config-driven example runs with the `manta` command from inside its own
directory:

    cd linear-diffusion && manta run.conf     # or: python -m manta.cli run.conf

Output lands beside the config, named after the config's stem — `run.nc` and
`run.restart.nc`. The `adjoints/` scripts are not config-driven; they build a
runner in Python and are run with `python <file>`.

## Two ways a config names its case

`PythonModule` is an importable dotted name, and is the one to use once the case
lives in a package:

    PythonModule = "mypackage.mycase"

`PythonModuleFile` is a path, resolved beside the config file, which is what the
examples here use because each case is a loose file next to its config:

    PythonModuleName = "linear_diffusion"
    PythonModuleFile = "linear_diffusion.py"

Either way the module is imported for its registrations. A module that calls
`manta.registerPhysicsCase` at import needs nothing more; one that defines
`registerTransportSystems()` instead — as every example here does — has it
called after import.
