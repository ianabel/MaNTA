
Lists upcoming features that are in development or being thought about

- Explore adaptive mesh-refinement schemes; first to the PTC steady-state algorithm or possibly the direct newton solve as well; then work out if it's possible to add to the real TimeMarch (requires interpolating history to allow IDA's BDF methods to work, and requires the hooks for that to be in SUNDIALS)

  **Partly done, and not in the shape this entry expected.** `DegreeAdaptation`
  adapts the global *polynomial degree* between steady solves — see
  `docs/running.rst`. It went first because it is where the measured win is:
  raising `k` beat adaptive `h` by seven orders on every benchmark in this tree,
  and h-adaptivity did not pay on any of them once the cost of the extra solves
  was counted. It covers both steady modes at once, since `SteadyMode::Newton`
  is the same code with `ptcStep = infinity`.

  Still open, in the order they look worth doing: *h*-adaptivity (needs a
  transfer between meshes, which is not the same problem as between degrees —
  the projections stop composing); per-cell degrees (a much larger change, ~320
  `(k+1)` sites in the core); and TimeMarch, which is still blocked on exactly
  what this entry says.

- Scope out coupling to a magnetic field solver. From an algorithmic point of view, a self-consistent magnetic field is just a large set of algebraic constraints, which IDA / KINSOL already handle. However, because this can be a large degree of freedom system (N_{magnetics} >> N_{HDG}) we should take care about the Jacobian solve. 
Assume that the coupled magnetic field solve is a differentiable solver, and so will provide both a nonlinear solution B to

B^{GS}({MaNTA State}, B) 

as well as 

dB^{GS}/d {MaNTA State}

and the internal jacobian that is used to solve for B. From this we can construct the full Jacobian

( HDG Jacobian | A1              )
( A2           | B^{GS} Jacobian )

where A1 & A2 are the couplings. We already have a fast/robust static condensation solver for the HDG Jacobian. We could use the Woodbury formula to get an exact solution to 

J_{full} x = y

but this would almost certainly be expensive.
One approach is to assume that the coupling is weak, and iterate. This is probably a good first pass.
