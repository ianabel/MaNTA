# Toy model

An unfinished sketch, kept for reference rather than as a worked example.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/toy-model
    manta run.conf

It does run, and writes `run.nc` and `run.restart.nc`.

## What it is not

The header comment describes

    d_t u + d_x( (a/u^{3/2}) du/dx ) = S(x)

and the code implements constant-`kappa` linear diffusion with no source — the
same problem as `../linear-diffusion/`, under a different name. The nonlinear
flux was never written.

Two repairs were made when this moved out of `python/`, both mechanical: its
registration hook named a class the file does not define (`PythonLinearDiffusion`
rather than `PythonToyModel`), and its constructor called the base
`__init__` three times, the first with no spec, which raised before the other two
could run. Nothing else about it was touched.
