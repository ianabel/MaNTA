# MaNTA physics

Production physics systems: the ones run to get answers about a machine, rather
than to demonstrate the framework. Each is a self-contained directory holding
its case, its driver and a README.

The difference from [`../python-examples/`](../python-examples/) is purpose, not
mechanism. Both follow the same rule — nothing here reaches into the MaNTA
source tree, everything imports `manta` and `manta.jax` the way code outside
this repository would, and a directory copied elsewhere still runs. What
separates them is that an example is written to be read and a system here is
written to be used: these are as large as the physics demands, carry their own
machine configurations, and have real external dependencies.

Install first, from the repository root:

    pip install .[jax]

| Directory | What it is | Needs | Status |
|---|---|---|---|
| `mirror-plasma/` | Centrifugal mirror: four channels, ambipolar potential, voltage controller | `manta[jax]`, `optimistix` (`desc` for its tests) | Runs without the controller; see its README |
| `stellarator/` | Stellarator neoclassical transport coupled to DESC and yancc | `manta[jax]`, `desc`, `yancc`, `interpax` | Unverified; see its README |

Neither is driven by the `manta` console script. Both build a runner in Python —
`mirror-plasma` because the voltage controller needs one, `stellarator` because
it puts the solver inside a JAX computation through `manta.jax.FFIRunner`.

## What is *not* here

The C++ physics cases, including the C++ `MirrorPlasma`, are in
`../PhysicsCases/`. The Python `mirror-plasma` here is an independent
implementation of the same physics, not a binding of that one, and the two do
not share a variable ordering — see its README.

## Dependencies

`optimistix`, `desc`, `yancc` and `interpax` are deliberately absent from the
project's `requirements.txt`. They belong to the systems that use them, not to
MaNTA, and adding them would make a solver install pull in a stellarator
equilibrium code. Each README lists what its directory needs.

## Tests

`make python_tests` does not reach here — `pytest.ini` sets
`testpaths = python/Tests`, which is the framework's own suite. A directory here
that has tests says so in its README and how to run them.
