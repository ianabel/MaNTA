# Toy model

Nonlinear diffusion with a localised source and a closed-form steady state,
written vectorised as the test problem for a surrogate flux. In the form MaNTA
integrates, `d_t u - d_x[sigma_hat] = S`,

    sigma_hat = A u^(-3/2) du/dx,         S(x) = D exp( -(x - C)^2 / B )

    u'(0) = 0,   u(1) = U1,   u(x, 0) = U1

with `A = 6`, `B = 0.02`, `C = 0.3`, `D = 50`, `U1 = 0.3`. The diffusivity
falls as `u` rises, so the source heats the core against a conductivity that
weakens where it is hottest.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/toy-model
    manta run.conf           # time march to t = 0.5: 12 cells, k = 3
    python toy_model.py      # steady solves against the exact steady state

`run.conf` writes `run.nc` and `run.restart.nc`. By `t = 0.25` the profile is
within the discretisation error of the steady state.

## The steady state

With `w = u^(-1/2)`, the steady equation integrates twice in closed form:

    u(x)^(-1/2) = U1^(-1/2) - G(x),   G(x) = (1/2A) int_x^1 int_0^s S(r) dr ds

`ExactSteadyState` in `toy_model.py` evaluates `G` with error functions, and
agrees with nested quadrature to 5e-16. `python toy_model.py` solves for the
steady state at k = 3 on refining meshes and prints the error against it:

| cells | relative L_inf error |
|---|---|
| 6 | 3.4e-2 |
| 12 | 2.4e-3 |
| 24 | 1.5e-4 |
| 48 | 1.0e-5 |

That is order 4 = k + 1. The six-cell row is under-resolved rather than
wrong: those cells are 0.17 wide against a source of width `sqrt(B)` = 0.14,
which is why `run.conf` uses twelve.

## The interface

The case supplies `ComputePhysics` and `ComputePhysicsDerivatives`, which take
every point of a batch in one call, rather than the pointwise `SigmaFn` and
`Sources`. That is the shape a surrogate wants: a model fitted to the flux is
evaluated a batch at a time, so this is the place to swap one in.
