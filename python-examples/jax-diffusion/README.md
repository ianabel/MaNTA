# Nonlinear diffusion, with derivatives from JAX

Two cases on `manta.jax.JAXTransportSystem`: `JAXNonlinearDiffusion`, a
density-dependent diffusivity with a Gaussian source, and `JAXAuxTest`, the same
with an algebraic auxiliary variable. Neither writes a derivative hook — the
base class differentiates `SigmaFn` and `Sources` with `jax.grad`.

## Running it

    pip install .[jax]       # from the repository root, once
    cd python-examples/jax-diffusion
    manta run.conf

`run.conf` selects which of the two cases runs, through its `TransportSystem`
key: `JAXNonlinearDiffusion` or `JAXAuxTest`. Both are registered by the same
module.

## What to look at

`jax_diffusion.py` was the tail of what used to be `python/JAXTransportSystem.py`
— a file that held the framework base class and these two demonstration cases
together. The base class is now `manta.jax.JAXTransportSystem` and this is an
ordinary case that imports it, with nothing on `sys.path` but the installed
package.

The case's parameters are a `NamedTuple` rather than plain attributes because
JAX needs a PyTree structure to differentiate with respect to them, which is
what makes the adjoint solve possible.
