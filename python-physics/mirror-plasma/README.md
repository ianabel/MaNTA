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

8 tests and 12 subtests, all passing.

## State of it

The structural migration is done and verified: the case imports, builds its
`SystemSpec` from its config, and every hook it implements — batched physics,
the aux constraint, the scalar hooks, `dSources_dScalars` — dispatches through
the trampolines correctly.

**It runs with the voltage controller on or off.** 32 cells, degree 5, 148
timesteps to t = 0.1 with `useConstantVoltage = True`, 135 with it off.

Until recently the controller could not start at all: `IDACalcIC` failed with
`IDA_LINESEARCH_FAIL` (-13), insensitive to the grid, and so did the C++
`MirrorPlasma` before it — nothing in either tree ever ran that mode. **The
cause was `VoltageError` being declared differential.** Its constraint is the
plain definition `E = V0/omega0 - Phi(u)`, containing no `dE/dt`; declared
differential, `IDA_YA_YDP_INIT` froze both `E` and `u` and solved only for
`dE/dt`, which appears nowhere at `t = 0` because `tanh(t/CurrentDecay)`
switches the controller off there. That left the row a constant residual —
4.3e-6, the gap between `InitialScalarValue` integrating `Phi` by trapezoid and
`ScalarG` integrating it with the solver's quadrature weights — and a
linesearch cannot reduce a constant. Declared algebraic, `CalcIC` solves `E`
from its own constraint and the residual is zero by construction.

`test_a_differential_scalars_constraint_reaches_a_calcic_unknown` pins the
general rule, which is worth knowing before adding a scalar: **a scalar
declared differential must have some `ydot` in its `G`.** If it does not, the
symptom is not a wrong answer but `IDACalcIC` refusing to start, and the
message points at the linesearch rather than at the declaration.

The controller demonstrably works, which completing a solve does not by itself
show: over a run to t = 0.1 it holds `Phi` to within 0.5% of `V0/omega0`, where
the same run with the controller off drifts 56% away.

`Tests/UnitTests/ScalarJacobianTests.cpp` documents the neighbouring failure
mode: a case whose `ScalarGPrime` disagrees with its own `ScalarG` converges
slowly or not at all, and finite-differencing one against the other is the
first thing to try. That was checked here and is *not* the problem — this
case's scalar Jacobian agrees with finite differences to 1e-5 relative across
every field, at both ends of the domain and at cell boundaries.

## A note on variable order

`Channel` here is `(Density, AngularMomentum, IonEnergy, ElectronEnergy)`. The
C++ `MirrorPlasma`'s is `(Density, IonEnergy, ElectronEnergy, AngularMomentum)`.
Nothing is shared between the two cases, so this is not a bug — but do not carry
an index from one to the other.
