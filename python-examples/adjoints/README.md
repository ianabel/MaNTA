# Driving the solver from Python, and differentiating it

Three scripts and a notebook. Unlike the other examples here these are not
config-driven: they build a `manta.Runner` (or a `manta.jax.FFIRunner`) and call
it, which is the interface an optimisation loop wants — configure, run, read the
objective and its gradients, change a parameter, run again.

    pip install .[jax]       # from the repository root, once
    cd python-examples/adjoints
    python runner.py

| File | What it shows |
|---|---|
| `runner.py` | The smallest `manta.Runner` loop: configure, run, read `G` and the adjoint gradients |
| `jvp.py` | A Jacobian-vector product through the solver, with the run itself inside a JAX computation |
| `spatial_adjoints.py` | Gradients with respect to spatially varying parameters, plotted |
| `AdjointAnalysis.ipynb` | Reads a finished run's `.nc` output and analyses the gradients it recorded |

## What needs which build

`runner.py` and `AdjointAnalysis.ipynb` work with a normal build.

**`jvp.py` needs an `XLA_FFI` build of the extension.** It imports
`manta.jax.FFIRunner`, which registers the solver itself as a JAX foreign
function so a whole MaNTA run can sit inside a jitted computation, and those
bindings are compiled in only under that flag:

    make python XLA_FFI=on          # needs the jaxlib headers

Without it the import fails with a message saying so. That is not new — the
script has always needed that build — but it is now stated rather than
discovered.

`spatial_adjoints.py` opens a matplotlib window partway through, so it wants a
display.
