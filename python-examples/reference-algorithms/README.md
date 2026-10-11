# Reference algorithms: ASTRA's published scheme and TGYRO's iteration

The paper's benchmarks are compared against other codes' algorithms. Where those
algorithms are published completely enough, this directory implements them and
runs them on the same problems, counting the same thing MaNTA's examples count:
**model evaluations per point**. One evaluation is the transport model at one
point, every channel at once:
- a TGYRO flux call at one radius;
- an ASTRA coefficient call at one cell face;
- a MaNTA `SigmaFn` or derivative call at one node.

| file | what it is |
|---|---|
| `problems.py` | the four benchmarks — Park, Jardin, the n–T pinch and the two-temperature critical gradient, with and without exchange — in the form a finite-difference code is handed them. The physics is imported from the example directories, not restated. |
| `astra.py` | the published ASTRA scheme, with Pereverzev and Corrigan's stabiliser |
| `tgyro.py` | TGYRO's flux-matching iteration, following `tgyro_iteration_standard.f90` |
| `compare.py` | all three methods on all the problems, plus the pinch problem's stiffness sweep |

    cd python-examples/reference-algorithms
    python compare.py
    python compare.py --json out.json

## What is implemented, and from where

**ASTRA.** Its source is not public, and its manual (IPP 5/98) does not write
down the difference scheme. `astra.py` is "the published ASTRA scheme", not
ASTRA, built from four sources:
- **Pereverzev & Corrigan, CPC 179 (2008) 579**, eqs. (2)–(3), whose examples
  ran in ASTRA: a cell-centred conservative grid with Patankar's exponential
  face flux, and backward Euler with the coefficients lagged.
- **The same paper, eq. (14):** the stiffness stabiliser, an implicit added
  diffusivity `Dbar` minus the same flux taken explicitly.
- **The same paper, eq. (19):** step control on the relative change of the
  diffusivity, with a 10% tolerance.
- **The ASTRA manual:**
  - growth factor `TAUINC = 1.1`;
  - outer iterations `ITEREX = 1`;
  - sources split into an implicit linear part and an explicit remainder.

Unpublished choices, made here and stated in the module:
- the step-cut factor, 1/2;
- `Dbar` per problem, chosen above the slope `dq/d(eta)` as PC advise;
- the initial step, 1e-3;
- the stopping test, a steady residual below 1e-10.

`Dbar` is a diffusivity and carries the geometry `V'`, as in ASTRA's equation.
Added to `V' D` instead, a constant `Dbar` swamps a diffusivity that vanishes on
the axis, and the march slows in proportion to the number of cells: on Jardin,
5.6 times the steps at 10 cells and 44 times at 80.

**TGYRO.** Candy et al., PoP 16 (2009) 060704, plus the open source
`gafusion/gacode`. Its iteration:
- flux matching on `N` radii, with unknowns the logarithmic gradients;
- profiles rebuilt by trapezoids from the boundary value, and targets by
  trapezoids on the radii;
- Newton with a forward-differenced, block-diagonal flux Jacobian (one call per
  channel per iteration);
- a per-component relaxation that is halved when that component's residual
  rises.

The code's defaults are `dz = 0.1`, `dz_max = 1` and relaxation factor 2.
TGYRO has no convergence test; this stops at a flux mismatch of 1e-10. Its
logarithmic gradients need positive profiles, so Park's and Jardin's problems,
whose wall value is 0, are shifted by 1. Both are invariant under that shift.

**ONETWO** is not implemented. The 1980 scheme (GA-A16178, §9) is clear, but
nothing public says the code still uses it. **Park's, Jardin's and Shestakov's**
own schemes are compared against their published numbers in their example
directories rather than reimplemented here.

## Results

Visits per point, and the relative L1 error (worst channel) at the finest
resolution run. MaNTA runs from 4 cells at `k = 2` up to 8 cells at `k = 5`;
ASTRA on 10–160 cells; TGYRO on 4–32 radii. MaNTA's visits do not change with
resolution, and ASTRA's and TGYRO's barely do.

| problem | MaNTA Newton | MaNTA PseudoTransient | ASTRA scheme | TGYRO, `dz = 0.1` | TGYRO, `dz = 0.01` |
|---|---|---|---|---|---|
| Park | **5** | 10 | 84 | 32–38 | 32–38 |
| Jardin | **15** | 22 | 56 (`Dbar = 30`) | 233–391 | 66–84 |
| Pinch, `D = 1` | **11** | 20 | 81–82 | 48–60 | 51–60 |
| Two-temperature | **15** | 22–24 | 80 (`Dbar = 30`) | 91–103 | 69–76 |
| Two-temperature + exchange | **11** | 16 | 76–77 (`Dbar = 30`) | 43–50 | 34–40 |

Accuracy separates the methods more than cost does:

* **Park and the pinch problem** have non-polynomial answers. MaNTA's error
  falls as `h^(k+1)`:
  - Park: 7.6e-5 from 16 points (4 cells, `k = 3`), 3.7e-9 from 48;
  - pinch: 4.1e-6 from 32 points.

  ASTRA's is second order: Park 8.6e-6 and the pinch 1.5e-5 on 160 cells, at
  13,000 evaluations against MaNTA's 160–350. TGYRO's is second order in the
  number of radii, and limited further by its profile reconstruction: Park
  8.6e-4 and the pinch 4.9e-4 on 32 radii.
* **Jardin's and the two-temperature problems** have linear answers:
  - MaNTA and ASTRA's scheme both reproduce them to round-off (MaNTA within
    1.1e-14; ASTRA 1e-12 to 7e-12, set by its stopping test).
  - TGYRO does not. A linear profile has a log-gradient that is not linear
    between radii, and its axis gradient is taken to be zero, so its error is
    4.3e-4 on Jardin and 1.4e-4 on the two-temperature problem at 32 radii.

**The stabiliser is not optional.** Without it (`Dbar = 0`), the ASTRA scheme
reaches no steady state in 20,000 steps on any of the three stiff problems.
On Jardin's, 12% of the steps are rejected by the eq. (19) control, and the
march reaches only t = 0.4 at 40 cells. That matches Pereverzev and Corrigan's
criterion, `Dbar` above the slope `dq/d(eta)`: on Jardin that slope is 28,
`Dbar = 10` fails as well, and `Dbar = 30` works.

**TGYRO's finite-difference step matters on stiff problems.** At the code's
default `dz = 0.1`, Jardin takes 79–131 iterations. At `dz = 0.01` it takes
27–30, and at `dz = 0.001` it does not converge in 500. The default is set for
gyrokinetic fluxes, whose time-averaging noise a small step cannot see past.

### Stiffness disparity

The pinch problem's answer does not depend on its particle diffusivity `D`, so
sweeping `D` changes only how much faster the density relaxes than the
temperature. Visits per point, for MaNTA at 8 cells `k = 3`, ASTRA at 40 cells
and TGYRO at 16 radii:

| D | MaNTA Newton | MaNTA PseudoTransient | ASTRA scheme | TGYRO |
|---|---|---|---|---|
| 0.01 | 11 | 18 | 130 | 59 |
| 0.1 | 11 | 18 | 106 | 59 |
| 1 | 11 | 20 | 82 | 60 |
| 10 | 11 | 18 | 74 | 67 |
| 100 | 11 | 18 | 75 | 72 |

The errors do not move with `D`. A time march pays for the slow channel:
ASTRA's cost grows as the density slows, and MaNTA's own `TimeMarch` fails at
`D = 100` (`../thermodiffusive-pinch/`). A method that solves for the steady
state directly does not care.
