# Linear diffusion, in pure Python

The smallest possible physics case: one variable, a constant-`kappa` flux, no
sources, and a closed-form `ExactSolution` to compare a run against. Nothing
here needs JAX.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/linear-diffusion
    manta run.conf

Writes `run.nc` and `run.restart.nc` beside the config. The output stem comes
from the *config's* name rather than from any `OutputFilename` key, so every
example in this tree produces `run.nc`.

## What to look at

`linear_diffusion.py` declares its one variable through `manta.numbered_spec(1)`
and writes `SigmaFn`, `Sources` and all four derivative hooks by hand. A case
that would rather not write those by hand should look at `../jax-diffusion/`,
where `jax.grad` supplies them.

The config reaches the module through `PythonModuleFile`, resolved beside the
config file. `PythonModule` — an importable dotted name — works too, and is the
better choice once the case lives in a package of its own.
