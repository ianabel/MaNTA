# Two-temperature critical gradient

[Jardin's stiff critical-gradient problem](../jardin-critical-gradient/), in
two channels: ion and electron temperatures. Each channel's diffusivity is
switched on by both gradients, and the two can optionally exchange energy
collisionally. As in Jardin's problem, the stiff steady state is exactly
linear, so every resolution should reproduce it to round-off, and the cost is
the result.

## The problem

For `s` in `{i, e}`, on the cylinder `V' = x`:

    d_t T_s - d_x[ x chi_s(q_i, q_e) q_s ] = S_s

    chi_s = chi0 + sum_r kappa_sr (|q_r| - qc_r)_+^gamma

    S_i = H_i + nu (T_e - T_i),     S_e = H_e - nu (T_e - T_i)

The parameters are `chi0 = 1`, `kappa_ii = kappa_ee = 10`, `kappa_ie = 2`,
`kappa_ei = 3`, `gamma = 1/2`, `qc = (0.5, 0.8)`. Both channels have zero flux
on the axis and a Dirichlet value at the wall.

**Without exchange** (the default), with `H = (2, 3)` and both wall values 1:

* the steady gradients `g_s` solve `chi_s(g_i, g_e) g_s = H_s`;
* `g = (0.5506, 0.8365)`, 0.051 and 0.037 above the two thresholds;
* a scan of Newton starts over `(0, 4]^2` finds no other positive root.

The tangent diffusivity matrix `d(chi_s q_s)/d(q_r)` there is
`[[15.9, 2.9], [5.6, 25.5]]`. The off-diagonal entries are what a block-diagonal
or channel-by-channel scheme leaves out.

**With exchange** (`exchange=True`), a linear steady state needs `T_e - T_i` to
be a constant `delta`, which forces `g_i = g_e`:

* `H_i = 6`, `nu = 2` and `delta = 0.5` are chosen;
* `g = 0.8910` follows from the ion equation;
* `H_e = 6.2503` is then **set** so the electron equation balances.

The exchange is genuinely active, carrying `nu delta = 1` from the electrons to
the ions, but the problem is rigged to keep the answer linear. A Ti–Te problem
with exchange and unequal, non-polynomial profiles needs a manufactured
solution instead.

The initial condition is Jardin's initial gradient, 1, in each channel.

## Running it

    cd python-examples/two-temperature-critical-gradient
    manta run.conf           # one Newton solve, 10 cells, k = 3
    python benchmark.py

## What it measures

Visits per point to reach the steady state; one visit is the model at one point,
both channels at once. The worst-channel L1 error is at round-off
(4e-16 to 3e-14) in every cell of this table:

| exchange | cells | k | Newton | PseudoTransient | TimeMarch |
|---|---|---|---|---|---|
| no | 4 | 2 | 15 | 22 | 174 |
| no | 10 | 3 | 15 | 22 | 189 |
| no | 10 | 5 | 15 | 24 | 190 |
| yes | 4 | 2 | 11 | 16 | 144 |
| yes | 10 | 3 | 11 | 16 | 151 |
| yes | 10 | 5 | 11 | 16 | 151 |

Jardin's single-channel problem costs 15 (Newton) and 22 (PseudoTransient)
visits at 4 cells, `k = 2–5` (`../reference-algorithms/`). So adding a second
channel coupled through the thresholds costs nothing extra per point: the
coupled Jacobian is analytic and assembled whole.

### Near the threshold

The heating is chosen to keep the answer off the kink. `chi` has a square-root
threshold, so `d chi/d q` diverges just above it. At 4 cells, `k = 2`, with no
exchange:

| H | g - qc | Newton | PseudoTransient | TimeMarch |
|---|---|---|---|---|
| (1, 2) | (0.0054, 0.015) | 662 | 1168 | 209 |
| (1.5, 2.5) | (0.024, 0.025) | 19 | 26 | 208 |
| (2, 3) | (0.051, 0.037) | 15 | 22 | 174 |

Every run reaches round-off. At `H = (1, 2)`, though, the steady-state solvers
pay 40–50 times their usual cost; time marching does not notice.

Starting the `H = (1, 2)` case from the constant-`chi` steady state (initial
gradients 1 and 2) instead, both steady-state solvers stall. KINSOL exhausts
its iterations and 200 continuation steps end at a relative error of 9e-3, while
time marching still converges. Jardin's single-channel answer sits 0.009 above
its threshold, and that problem has no such trouble.

### The starting point

The constant-`chi` steady state is also a poor start for the exchange variant.
There it means initial gradients of 6 and 6.25, from which IDA's first step
fails (`IDASolve could not complete`, 10 cells, `k = 3`). Without exchange the
same start, gradients 2 and 3, converges in 250 visits. Jardin's initial
gradient of 1 works for every solver in both variants.
