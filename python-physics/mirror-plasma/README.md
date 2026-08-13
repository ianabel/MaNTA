# Centrifugal mirror plasma

A four-channel mirror-plasma transport model written against `manta.jax`, with
an auxiliary variable for the ambipolar potential and an optional voltage
controller carried as three global scalars. It is the Python counterpart of the
C++ `PhysicsCases/MirrorPlasma.cpp`, not a binding of it — the two are
independent implementations.

**This is work in progress.** See "State of it" below before drawing
conclusions from a run.

| | |
|---|---|
| Variables | `Density`, `AngularMomentum`, `IonEnergy`, `ElectronEnergy` |
| Auxiliary | `AmbipolarPhi`, the potential enforcing zero parallel current |
| Scalars | `VoltageError`, `VoltageErrorIntegral`, `RadialCurrent` — only with `useConstantVoltage` |
| Boundaries | Density Neumann at both ends, the other three Dirichlet |

## Running it

    pip install .[jax]         # from the repository root, once
    pip install optimistix     # not a MaNTA dependency; see "What it needs"
    cd python-physics/mirror-plasma
    python landremann.py       # once, writes land.pkl
    python run.py

`run.py` integrates `configs.CMFX` to t = 2 on 32 cells at polynomial degree 5
and writes `mirror.nc` and `mirror.restart.nc` beside itself.

This example is not config-driven — there is no `run.conf` and the `manta`
console script does not apply. It builds a `manta.Runner` in Python, which is
what the voltage controller needs.

### land.pkl

`mirror_plasma/constants.py` needs the nodes and weights of a quadrature rule
for the Pastukhov loss integrals, which `landremann.py` computes and pickles.
The pickle is generated rather than checked in (`*.pkl` is gitignored), so run
`landremann.py` once before anything else; `PlasmaConstants` raises a
`FileNotFoundError` naming the fix if you forget.

## What it needs

Beyond `manta[jax]`:

| Package | Used by | In `requirements.txt`? |
|---|---|---|
| `optimistix` | `parallel_physics.py`, solving for the centrifugal potential | no |
| `matplotlib` | `constants.py`, `landremann.py` | yes |
| `scipy` | `landremann.py`, `test_mirror.py` | yes |
| `desc` | `test_mirror.py` only, for `tree_unstack` | no |

`optimistix` and `desc` are deliberately not MaNTA dependencies — they belong to
this example, the way `desc`/`yancc`/`interpax` belong to `../stellarator/`.

## Tests

`test_mirror.py` checks the parallel physics against closed forms — Pastukhov
loss rates against `scipy.special.gammaincc`, the centrifugal potential against
its defining equation — and the flux normalisations. It is **not** collected by
`make python_tests`: `pytest.ini` sets `testpaths = python/Tests`, and this file
needs `desc` and `optimistix`, which that suite does not. Run it directly:

    cd python-physics/mirror-plasma && pytest test_mirror.py

7 tests and 12 subtests, all passing.

## State of it

The structural migration is done and verified: the case imports, builds its
`SystemSpec` from its config, and every hook it implements — batched physics,
the aux constraint, the scalar hooks, `dSources_dScalars` — dispatches through
the trampolines correctly.

**With `useConstantVoltage = False` it runs.** 32 cells, degree 5, 135 timesteps
to t = 0.1, output written.

**With `useConstantVoltage = True` — which is what every config in `configs.py`
sets — `IDACalcIC` fails with `IDA_LINESEARCH_FAIL` (-13)**, meaning the initial
condition solve cannot reach a state consistent with the algebraic constraints
from the guess it starts at. The failure is insensitive to the grid: 8 cells at
degree 3 and 32 at degree 5 fail identically. Since dropping the three control
scalars is the single change that makes it run, they are where to look —
`InitialScalarValue`, `InitialScalarDerivative` and `ScalarG` in
`mirror_plasma.py`, and whether the initial `RadialCurrent` is consistent with
the momentum equation it is supposed to satisfy.

`Tests/UnitTests/ScalarJacobianTests.cpp` documents the C++ side of this: a
case whose `ScalarGPrime` disagrees with its own `ScalarG` converges slowly or
not at all, and finite-differencing one against the other is the first thing to
try.

## A note on variable order

`Channel` here is `(Density, AngularMomentum, IonEnergy, ElectronEnergy)`. The
C++ `MirrorPlasma`'s is `(Density, IonEnergy, ElectronEnergy, AngularMomentum)`.
Nothing is shared between the two cases, so this is not a bug — but do not carry
an index from one to the other.
