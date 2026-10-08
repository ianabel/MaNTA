# Density and temperature through a thermodiffusive pinch

A **two-channel** benchmark with a closed-form, non-polynomial steady state.
Every problem the paper compares against — Park's, Jardin's, Shestakov's — has
one channel; this one couples two, in both directions, through the fluxes.

## The problem

On the cylinder `V' = x`, in the form MaNTA integrates
(`a d_t u - d_x[sigma_hat] = S`):

    d_t n - d_x[ x D (n' - C_T n T'/T) ]       = 0
    d_t T - d_x[ x n chi0 (T/T_b)^alpha T' ]    = x H

Both channels have zero flux on the axis, posed as a mixed condition with
`d = 1`, and Dirichlet values `n_b`, `T_b` at the wall. The particle flux is
diffusion plus a thermodiffusive pinch `V = C_T D T'/T`, which points inward
wherever `T` falls outward. The heat flux is gyro-Bohm-like and carries the
density.

With no particle source the particle flux vanishes identically, so
`n = n_b (T/T_b)^C_T`. The heat equation then integrates, by the Kirchhoff
transform, to

    (T/T_b)^beta = 1 + beta H (1 - x^2) / (4 n_b chi0 T_b),    beta = C_T + alpha + 1.

At the defaults (`alpha = 3/2`, `C_T = 1/2`, `H = 4`, everything else 1), that
is `T = (4 - 3x^2)^(1/3)` and `n = (4 - 3x^2)^(1/6)`.

What it exercises that a single channel cannot:

* **Off-diagonal Jacobian blocks in both directions:**
  * `d sigma_n / d q_T = -x D C_T n / T`;
  * `d sigma_n / d T`, through the pinch;
  * `d sigma_T / d n`, through the heat flux.
* **A fixed answer under changing stiffness.** The steady state does not depend
  on the particle diffusivity `D`. Sweeping `D` changes how much faster one
  channel relaxes than the other while the reference solution stays put.

`H` is set by the analytic radius rather than by the physics. The profiles have
a branch point where `4 - 3x^2 = 0`, at `x = 1.155`, i.e. 0.155 outside the
wall. At `H = 16` the profile becomes `(13 - 12x^2)^(1/3)`, whose branch point
is only 0.042 outside the wall; the error there is still pre-asymptotic at 32
cells.

## Running it

    cd python-examples/thermodiffusive-pinch
    manta run.conf           # one Newton solve, 8 cells, k = 3
    python benchmark.py      # the two tables below

## What it measures

**Accuracy per evaluation**, from Newton steady solves. The error is the
relative L1 error in each channel. A visit is the model at one point, both
channels at once.

| cells | k | visits/point | error n | error T |
|---|---|---|---|---|
| 4 | 2 | 11 | 2.0e-4 | 2.4e-4 |
| 16 | 2 | 11 | 3.4e-6 | 4.4e-6 |
| 4 | 3 | 11 | 4.7e-5 | 4.8e-5 |
| 16 | 3 | 11 | 3.0e-7 | 3.0e-7 |
| 4 | 5 | 11 | 1.4e-6 | 1.4e-6 |
| 16 | 5 | 11 | 1.0e-9 | 9.0e-10 |

The rates approach `k + 1` from below: 2.98 at `k = 2`, 3.78 at `k = 3` and
5.5 at `k = 5`, at 16 cells. The approach is slow because of the branch point.

**Cost against stiffness disparity**, at 8 cells and `k = 3`. In every row the
error is the same, 4-5e-6, whatever the method.

| D | Newton | PseudoTransient | TimeMarch |
|---|---|---|---|
| 0.01 | 11 | 18 | 190 |
| 0.1 | 11 | 18 | 167 |
| 1 | 11 | 20 | 154 |
| 10 | 11 | 18 | 203 |
| 100 | 11 | 18 | **fails** (`IDASolve could not complete`) |

Newton's cost does not see the disparity at all: it never resolves the
transient. Time marching pays for both time scales, and at `D = 100`, where
the density relaxes 100 times faster than the temperature, it fails.

[`../reference-algorithms/`](../reference-algorithms/) runs the published ASTRA
scheme and TGYRO's iteration on the same problem. At `D = 1` they cost 82 and
48–60 visits per point, and reach errors of 1.5e-5 on 160 cells and 4.9e-4 on
32 radii. Newton above gets 4e-6 from 8 cells at `k = 3`.
