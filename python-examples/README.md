# MaNTA examples

Each directory here is a self-contained example: its physics case, its config
and a README. Nothing in them reaches into the MaNTA source tree — they import
`manta` and `manta.jax` the way any code outside this repository would, which is
the point. Copy one somewhere else and it still runs.

Install first, from the repository root:

    pip install .              # or `pip install .[jax]` for the JAX examples

| Directory | What it shows | Needs |
|---|---|---|
| `linear-diffusion/` | The smallest hand-written case: one variable, constant kappa, derivatives by hand | — |
| `toy-model/` | An unfinished sketch, kept for reference | — |
| `jax-diffusion/` | `manta.jax.JAXTransportSystem`; derivatives from `jax.grad` | `manta[jax]` |
| `jax-linear-diffusion/` | The same, with `solveAdjoint = true` | `manta[jax]` |
| `jax-nonlinear-adjoint/` | `manta.jax.VectorizedTransportSystem`, the batched interface | `manta[jax]` |
| `adjoints/` | Driving the solver from Python through `manta.Runner`; JVP and spatial adjoints | `manta[jax]`; `jvp.py` also needs an `XLA_FFI` build |
| `stellarator/` | Stellarator transport coupled to DESC and yancc. **Unverified** | `manta[jax]`, `desc`, `yancc`, `interpax` |

A config-driven example runs with the `manta` command from inside its own
directory:

    cd linear-diffusion && manta run.conf

Output lands beside the config, named after the config's stem — `run.nc` and
`run.restart.nc`. The `adjoints/` and `stellarator/` scripts are not
config-driven; they build a runner in Python and are run with `python <file>`.

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
