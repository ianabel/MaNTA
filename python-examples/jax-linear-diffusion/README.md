# Linear diffusion with an adjoint solve

`manta.jax.JAXTransportSystem` again, on a problem simple enough to check by
hand, with `solveAdjoint = true` in the config — so the run computes the
objective's parameter derivatives as well as the solution.

## Running it

    pip install .[jax]       # from the repository root, once
    cd python-examples/jax-linear-diffusion
    manta run.conf

## What to look at

`createAdjointProblem` is what makes the run an adjoint one: it builds a
`manta.jax.JAXAdjointProblem` from the case and an objective `g`, and declares
which parameters the gradient is taken with respect to. The gradients are
computed during `integrate`, so they are ready by the time the run returns.

The same machinery driven from Python rather than from a config is in
`../adjoints/`, and the C++ side of it is exercised by
`python/Tests/test_adjoint.py`.
