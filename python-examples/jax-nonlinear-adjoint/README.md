# Nonlinear diffusion on the batched interface

`manta.jax.VectorizedTransportSystem` rather than `JAXTransportSystem`. A
vectorised case overrides `ComputePhysics` and `ComputePhysicsDerivatives` and
is called once per batch of quadrature points instead of once per point, which
is what lets the whole evaluation be a single JAX computation.

## Running it

    pip install .[jax]       # from the repository root, once
    cd python-examples/jax-nonlinear-adjoint
    manta run.conf

## A note on this config

It had never run. As `python/jax_adjoint.conf` it pointed its `PythonModuleFile`
at `VectorizedTransportSystem.py`, which defines only the base class — no
`JAXNonlinearDiffusion` and no registration hook — so the case it asked for
could not be found. The module that actually defines that case is the one now
sitting beside it, and the config points there.

## What to look at

The batched hooks receive a `dict` of `(nPoints, nVars)` arrays, not the
pointwise `State` view. The caster transposes in both directions, because C++
stores `(nVars, nPoints)`; a round-trip cannot detect a missing transpose, so
check orientation from inside a batched call if you are ever unsure.
