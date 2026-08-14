# Why the error is what it is, and why the case is left alone

`README.md` describes the benchmark and reports what it costs. This records
*why*, because two of the things the README used to assert turned out to be
wrong, and because several obvious-looking improvements to the case are
improvements to MaNTA's **score** rather than to MaNTA. Every number here is
produced by `diagnostics.py`, which prints the eight sections cited below.

**The case is deliberately not tuned.** Shestakov's problem is the one in the
paper, and the point of having it in the tree is to record what MaNTA does on it
as stated. Where a change to the initial condition or the mesh would improve the
numbers, that is written down here as a measurement rather than applied to
`run.conf`. The one deviation the case does make — `n(Lx) = 0.01` against
Shestakov's `0` — is forced, and section 5 below says why it cannot be removed.

## Summary

* The error is **not** limited by the solution's regularity in the way the
  README claimed. It is a **flux deficit made entirely in the boundary cell**,
  equal to the flux the source deposits between the axis and the innermost
  collocation node, and spread uniformly over the domain by conservation.
* The **initial condition does not affect accuracy at all** — four different
  ones agree to four significant figures — but it sets the cost within a factor
  of three and decides whether `SuppressAlgebraicError` is needed.
* The shipped initial condition is **1080× too large in the flux** near the
  wall. That is what makes the flag load-bearing at `n_b = 0.01`, not the
  formulation. It is also what lets that IC reach `n_b = 0` where every better
  one fails.
* `n_b = 0` is out of reach for a reason no initial condition or mesh can
  address.

## 1. The initial condition (`diagnostics.py 1`)

`shestakov_nonlinear.py` starts from

```
n0(x) = n_b + (1 - n_b) (Lx - x)^3 (1 + 3x)
```

which satisfies the three things it was written to satisfy: `n0(0) = 1` is
Shestakov's own axis value, `n0(Lx) = n_b` meets the Dirichlet condition, and
`q0(0) = 0` matches the Neumann value on the axis. It is nonetheless a long way
from the steady state, and furthest in the flux, which is the quantity the
algebraic rows and IDA's error test see. The exact steady flux is just
`sigma_hat_e = -S0 min(x, d)`:

| x | n0 | n_e | ratio | sigma_hat0 | exact | ratio |
|---|---|---|---|---|---|---|
| 0.00 | 1.0000e+00 | 4.9147e-02 | **20.4** | 0 | 0 | — |
| 0.10 | 9.4822e-01 | 4.4622e-02 | 21.3 | −9.9102e−01 | −1.0e−01 | 9.9 |
| 0.50 | 3.1938e-01 | 2.5103e-02 | 12.7 | −3.2105e+01 | −1.0e−01 | **321** |
| 0.80 | 3.6928e-02 | 1.4957e-02 | 2.5 | −4.0289e+01 | −1.0e−01 | **403** |
| 0.90 | 1.3663e-02 | 1.2313e-02 | 1.1 | −6.5476e+00 | −1.0e−01 | 65.5 |
| 1.00 | 1.0000e-02 | 1.0000e-02 | 1.0 | **0** | **−1.0e−01** | 0 |

The cause is an amplitude, not a shape. Writing `e = Lx - x`, the exact solution
near the wall is `(n_b^(1/3) + c e)^3` with `c = (S0 d/D0)^(1/3)/3 = 0.154720`,
so its amplitude is `c^3 = 3.7037e-3`. The IC's is **4**, fixed by demanding
`n0(0) = 1` rather than by the physics. Since `sigma_hat = D0 q^3/u^2` is
homogeneous of degree 1 in that amplitude, the flux inherits the same factor
uniformly in `e`:

| n_b | sigma_hat0 at x = 0.9 | exact | ratio |
|---|---|---|---|
| 0 | −9.2017e+01 | −1.0e−01 | **920** |
| 1e−3 | −5.6945e+01 | −1.0e−01 | 569 |
| 1e−2 | −6.5476e+00 | −1.0e−01 | 65.5 |
| 1e−1 | −8.6009e−02 | −1.0e−01 | 0.9 |

and in the `n_b -> 0` limit the ratio is exactly `4/c^3 = 1080`. The implied
initial `|dn/dt|` reaches `3.9e2`, peaking at `x ~ 0.86` rather than anywhere
near the source.

## 2. Four initial conditions (`diagnostics.py 2`)

10 cells, `k = 2`, `TimeMarch`, physics evaluations per node, as
(flag off / flag on). **B** is the shipped shape with its amplitude taken from
`c^3`; **E** keeps `n(0) = 1` and `q(0) = 0` but adds the exact wall slope;
**C** is the exact steady state.

| n_b | A shipped | B wall-matched | E hot + wall slope | C exact | rel L1 error |
|---|---|---|---|---|---|
| 1e−1 | 354 / 207 | 357 / 289 | 256 / 162 | 134 / 112 | 1.059e−2 |
| 1e−2 | **fail** / 283 | **273** / 212 | fail / 244 | 164 / 151 | 1.986e−2 |
| 1e−3 | **fail** / 232 | **170** / 115 | fail / 199 | 195 / 167 | 3.298e−2 |
| 0 | fail / **357** | fail / fail | fail / fail | fail / fail | 2.557e−2 |

Three things. **The error column does not move** — accuracy is not the initial
condition's business, and section 6 explains why. Cost varies by 3×. And the
wall amplitude alone decides whether the flag is needed: with it corrected, both
`n_b = 1e-2` and Shestakov's own Section 2.2 value of `1e-3` run with
`SuppressAlgebraicError` **off**.

So the claim that this problem "cannot be integrated at all" without the flag is
a property of the shipped initial condition, not of MaNTA. Conversely the shipped
IC is the *only* one that reaches `n_b = 0`, because starting 1080× too high
keeps `u` away from the singular region for most of the run. Both are recorded;
neither is tuned away.

## 3. Shestakov's own start, and a piecewise-linear one (`diagnostics.py 3`)

Shestakov uses `n0 = 1`. Being constant that makes `D = (n_x/n)^2 = 0` and
`d(sigma_hat)/dq = 0` everywhere, so the trace system degenerates and
`IDACalcIC` has nothing to solve. The natural repair — keep `1` on `[0,a)` and
ramp linearly down — fixes the axis and breaks the wall: `u ~ (Lx - x)` with
`q ~ const` gives `sigma_hat ~ -(Lx - x)^-2`, where the true solution stays
finite by vanishing as `(Lx - x)^3`. What the solver is handed at the innermost
node of the last cell, `6.6987e-3` from the wall at 10 cells `k = 2`:

| a | ramp to | u there | q | sigma_hat | exact |
|---|---|---|---|---|---|
| 0.1 | 0 | 7.4430e−3 | −1.111 | **−2.4761e+4** | −1.0e−01 |
| 0.1 | n_b | 1.7369e−2 | −1.100 | −4.4121e+3 | −1.0e−01 |
| 0.5 | 0 | 1.3397e−2 | −2.000 | −4.4570e+4 | −1.0e−01 |
| 0.8 | 0 | 3.3494e−2 | −5.000 | **−1.1143e+5** | −1.0e−01 |

Across `a ∈ {0.1, 0.5}` and `n_b ∈ {0, 1e-2, 1e-1}`, the only runs that completed
were at `n_b = 1e-1` ramping to `n_b` rather than to 0, with the flag on — 213
visits at `a = 0.1` and 591 at `a = 0.5`. Every other combination failed, and
every combination failed with the flag off, including those two.

The ramp carries **both** obstructions, and the SUNDIALS log separates them:

* flag **off** — `IDA_ERR_FAIL` (−3) at `t = 0`, `h = 1e-7`: the error test.
  This is the same mode the shipped IC has and the one the flag exists for.
* flag **on** — `IDA_CONV_FAIL` (−4), same `t` and `h`, Newton reporting
  failure: the **corrector**, which no error-test setting can reach. At that
  wall node `d(sigma_hat)/du = -2 sigma_hat/u ~ 5e5` and `3 q^2/u^2 ~ 1e4`, so
  the linearisation Newton is handed is meaningless.

Only `n_b = 1e-1` never reaches the second mode, which is why it is the one that
runs. This is the other side of section 1: too *much* density at the wall is
survivable, too little is not, and the shipped IC survives by having 1080× too
much.

## 4. The same ramp, taken in `w = n^(1/3)` (`diagnostics.py 4`)

Shestakov substitutes `w = n^(1/3)` to linearise his own steady equation, and
outside the source region the exact solution is *exactly linear in w*:
`w = n_b^(1/3) + c (Lx - x)`. On any `w`-linear ramp

```
sigma_hat = D0 q^3/u^2 = 27 D0 (dw/dx)^3        -- constant, bounded by construction
```

so the blow-up of section 3 cannot occur. **W1** reaches `w = 1` at `x = a`;
**W2** uses the physical slope `-c`, giving `sigma_hat = -0.1` exactly; `a = 0`
is W2 with no flat region, i.e. the exact outer branch continued to the axis.
10 cells `k = 2`, (flag off / flag on):

| n_b | W1 a=0.1 | W2 a=0.1 | W2 a=0.3 | **W2 a=0** | rel L1 error |
|---|---|---|---|---|---|
| 1e−1 | 365 / 283 | 191 / 130 | 242 / 172 | 150 / **108** | 1.059e−2 |
| 1e−2 | fail / 431 | **208** / 173 | **309** / 257 | **175** / **134** | 1.986e−2 |
| 1e−3 | fail / 377 | **233** / 206 | **296** / 275 | **197** / 178 | 3.298e−2 |
| 0 | fail / fail | fail / fail | fail / fail | fail / fail | — |

`W2` with `a = 0` is the cheapest start found — 2.1× cheaper than the shipped IC
at `n_b = 1e-2` — and the only family that runs with the flag off at every
`n_b > 0`. The error is unchanged, again.

Two side notes. W1 never pays, and cannot: reaching the correct wall flux from
`n(0) = 1` would need `(1 - n_b^(1/3))/c = 5.07 > Lx`, so no single `w`-linear
ramp connects Shestakov's axis value to the physical wall slope over a unit
interval. And at `n_b = 0.1` the `a = 0` start costs **108** visits against
**112** for starting from the exact steady state — starting at the answer is not
the cheapest start, because `IDACalcIC` must build the algebraic fields either
way and the early transient is what sizes IDA's step ramp.

## 5. Where the error lives (`diagnostics.py 5`, `6`)

**Not in any element in particular.** Ten fixed bins of width 0.1, common to
every resolution (and every bin edge is a cell boundary at all four, so the
source kink at `x = d` never falls inside a cell):

| bin | 10 cells | 20 | 40 | 80 | rate | share |
|---|---|---|---|---|---|---|
| [0.0,0.1) | 4.911e−3 | 2.422e−3 | 1.202e−3 | 5.984e−4 | 1.01 | 24.7% |
| [0.3,0.4) | 2.492e−3 | 1.232e−3 | 6.123e−4 | 3.051e−4 | 1.01 | 12.6% |
| [0.6,0.7) | 9.805e−4 | 4.842e−4 | 2.405e−4 | 1.198e−4 | 1.01 | 4.9% |
| [0.9,1.0) | 9.830e−5 | 4.846e−5 | 2.404e−5 | 1.197e−5 | 1.01 | 0.5% |
| **total** | 1.983e−2 | 9.800e−3 | 4.868e−3 | 2.425e−3 | 1.01 | |

Every bin converges at the same rate and the **shares are constant to 0.1% over
an 8× refinement**. That rules out a local defect: it is one global mode.

It lives in `sigma`. The exact stored `sigma = +Gamma = S0 min(x,d)` is
piecewise linear with its kink on a cell boundary, hence representable exactly
in `P_k` for every `k >= 1` here — any error in it is the scheme's. Measured, it
is a **pure constant offset**, with `x1 = (h/2)(1 - cos(pi/2(k+1)))` the
distance from the cell edge to the outermost Chebyshev point of the first kind:

| cells | k | mean dsigma | std over x>d | /Gamma_wall | x(Gamma_h=0)/x1 | Gamma_h(0)/(−S0 x1) |
|---|---|---|---|---|---|---|
| 10 | 1 | −1.4718e−2 | 1.76e−5 | −1.4718e−1 | 0.981 | 0.971 |
| 10 | 2 | −6.5415e−3 | 7.73e−8 | −6.5415e−2 | 0.965 | 0.958 |
| 10 | 3 | −3.6784e−3 | **6.8e−17** | −3.6784e−2 | 0.959 | 0.955 |
| 10 | 4 | −2.3541e−3 | **6.2e−17** | −2.3541e−2 | 0.957 | 0.953 |
| 10 | 5 | −1.6346e−3 | **1.0e−16** | −1.6346e−2 | 0.956 | 0.953 |
| 20 | 2 | −3.2597e−3 | 1.00e−8 | −3.2597e−2 | 0.964 | 0.959 |
| 40 | 2 | −1.6261e−3 | 1.27e−9 | −1.6261e−2 | 0.964 | 0.960 |
| 80 | 2 | −8.1169e−4 | 1.60e−10 | −8.1169e−3 | 0.964 | 0.961 |

Constant to round-off for `k >= 3`, exactly `∝ h`, exactly `∝ (k+1)^-2`, and
equal to `0.96 S0 x1` — **the flux the source deposits between the axis and the
innermost collocation node.** `Gamma_h` vanishes *there* rather than at `x = 0`.
The zero-flux axis condition is in effect being imposed a distance `x1` inside
the domain, and conservation carries the resulting deficit across every cell.

Since `x1 ~ pi^2 h/16(k+1)^2`, the observed `O(h)` and `O((k+1)^-2)` are both
that one length. Fitted at `n_b = 1e-2`,

```
relative L1 error in u  ~  1.8 h0 / (k+1)^2
```

with `h0` the width of the **boundary cell** (not the mesh — see section 6):
predicted 4.43e−2 / 1.97e−2 / 7.08e−3 against measured 4.65e−2 / 1.986e−2 /
7.06e−3 at `k = 1, 2, 4`, and 2.46e−3 against 2.428e−3 at 80 cells. The
coefficient rises as `n_b` falls; the `sigma` statement above does not depend on
the fit.

**Two ingredients are needed, and `diagnostics.py 8` separates them.**

1. `sigma_hat = D q` with `D(0) = 0`, so the Neumann condition `q(0) = 0` says
   nothing about the flux — *any* `q` gives zero flux there. This is the trap
   recorded in `CLAUDE.md` as "a Neumann boundary fixes `q`, not the flux" in
   its second guise: in `../jardin-critical-gradient/` the value was wrong, here
   it is right but powerless.
2. `q_e ~ -C x^(1/3)`, whose derivative is unbounded, so no polynomial resolves
   it on the boundary cell — the max error in `q` there converges at **0.34**,
   i.e. `h^(1/3)`. Then `d(sigma_hat)/dq = 3 q^2/u^2 ~ x^(2/3)`, and at
   `x ~ x1` the flux error is `O(h^(2/3) · h^(1/3)) = O(h)`.

The control keeps the flux law and the degeneracy and removes only the second:
`S = 3 S0 d x^2` instead of a step gives `Gamma = S0 d x^3`, hence `w_x = -c x`,
`n_e = (n_b^(1/3) + c(Lx^2 - x^2)/2)^3` — a degree-6 polynomial — with
`sigma_hat_e = -S0 d x^3` and `D = (q/u)^2 ~ x^2` still vanishing on the axis:

| cells | k | dsigma/Gamma | rel L1 error | rate |
|---|---|---|---|---|
| 10 | 2 | −1.5021e−5 | 1.178e−4 | |
| 20 | 2 | −1.7889e−6 | 1.510e−5 | 2.96 |
| 40 | 2 | −2.2167e−7 | 1.925e−6 | 2.97 |
| 10 | 3 | 2.6601e−8 | 1.248e−7 | |
| 20 | 3 | 1.2013e−9 | 7.192e−9 | 4.12 |
| 10 | 6 | −1.1734e−16 | **3.675e−15** | |

Full order, and exact at `k = 6` where `n_e` lies in `P_k`. **A degenerate
diffusivity is not by itself a problem for MaNTA; a degenerate diffusivity at a
boundary where the solution carries a fractional power is.**

## 6. Refining the boundary cell alone (`diagnostics.py 7`)

Nine cells stay at `h = 0.1`; only the first is split, geometrically:

| first cell | cells | nodes | dsigma/Gamma | visits | rel L1 error |
|---|---|---|---|---|---|
| 1.000e−1 | 10 | 30 | −6.5415e−2 | 283 | 1.986e−2 |
| 5.000e−2 | 11 | 33 | −3.2597e−2 | 267 | 9.813e−3 |
| 1.250e−2 | 13 | 39 | −8.1169e−3 | 303 | 2.428e−3 |
| 7.813e−4 | 17 | 51 | −5.0546e−4 | 272 | 1.510e−4 |
| 3.052e−6 | 25 | 75 | −1.9708e−6 | 283 | 4.376e−6 |

against uniform refinement, which touches every cell:

| uniform | nodes | visits | rel L1 error |
|---|---|---|---|
| 10 cells | 30 | 283 | 1.986e−2 |
| 20 cells | 60 | 220 | 9.814e−3 |
| 40 cells | 120 | 227 | 4.875e−3 |
| 80 cells | 240 | 263 | 2.428e−3 |

The offset tracks the first cell's width and nothing else. A 13-cell graded mesh
matches uniform 80 cells with **6× fewer degrees of freedom**; 17 cells beats it
16× at the same cost. Uniform refinement past 80 cells is not merely
inefficient — **160 uniform cells fails outright**, `IDA_CONV_FAIL` at
`t = 1.5e-5`.

**This is not in `run.conf`, and should not be.** Grading a mesh at the one
point where a particular problem is singular is exactly the per-problem tuning
this benchmark exists to expose rather than to perform. What it establishes is
the diagnosis, not a setting: the boundary cell is where the error is made, so
the general fix is in how MaNTA treats a degenerate flux boundary, not in this
config file.

## 7. `n_b = 0` (`diagnostics.py 2`, `4`)

Shestakov's Section 2.1 value remains out of reach, and now for a stated reason
rather than an observation. With `n(Lx) = 0` exactly, `d(sigma_hat)/du =
-2 sigma_hat/u` diverges as the wall is approached, and no initial condition or
mesh removes a divergent Jacobian entry at a Dirichlet node. Tried and failed:
all four initial conditions of section 2, all five `w`-ramps of section 4, the
axis-graded meshes of section 6, and a mesh graded geometrically into the wall as
well (16 cells) — most failing inside `IDACalcIC`. The shipped IC reaches it at
exactly one resolution, and only because it starts three orders of magnitude
away from the singular region.

The two things that would close the gap are the ones `README.md` already names:
a positivity-preserving discretisation, or a lagged-diffusivity iteration that
never forms the divergent derivative. Both are properties of the scheme.

## What this corrected in `README.md`

* *"nothing can be done about the axis"* — wrong. Section 6: two orders of
  magnitude at equal cost. Uniform refinement was the wrong lever, and the
  README had generalised from it.
* *"the error is the solution's regularity, not the scheme"* — too kind to the
  scheme in one way and too harsh in another. Regularity alone predicts `h^(7/3)`
  in L1 from best approximation on the boundary cell; the observed `h^1` is
  worse, because the local error is amplified into a global flux deficit. And
  `(k+1)^-2` is a node position, not a regularity exponent.
* The kink at `x = d` is **exonerated**: it lies on a cell boundary at every
  resolution used here and costs nothing.
* `run.conf`'s claim that the problem "does not run at all" without
  `SuppressAlgebraicError` is a property of the shipped initial condition
  (section 2), which is why the comment there now says so.
