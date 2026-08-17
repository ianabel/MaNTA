
Lists upcoming features that are in development or being thought about

- Explore adaptive mesh-refinement schemes; first to the PTC steady-state algorithm or possibly the direct newton solve as well; then work out if it's possible to add to the real TimeMarch (requires interpolating history to allow IDA's BDF methods to work, and requires the hooks for that to be in SUNDIALS)

  **Partly done, and not in the shape this entry expected.** `DegreeAdaptation`
  adapts the global *polynomial degree* between steady solves — see
  `docs/running.rst`. It went first because that is where the win looked to be,
  and it covers both steady modes at once, since `SteadyMode::Newton` is the same
  code with `ptcStep = infinity`.

  **The reason it went first has since been half retracted**, and the retraction
  is the more useful finding. "Raising `k` beat adaptive `h` by seven orders on
  every benchmark, and h-adaptivity did not pay on any of them" holds only for the
  *indicator-driven equidistribution loop* that was tried, and only on the smooth
  problems. A mesh **graded geometrically towards a singularity**, at an unchanged
  cell count and unchanged DOF, is worth **14900×** on Shestakov — and the reason
  the earlier loop found 1.5× is that the accuracy indicator cannot see that
  error, which `MESH-REFINEMENT.md` §3 had already measured without drawing the
  conclusion. Grading is now a configuration option (`GradedGridBoundary`).

  What the measurements settled, all in `MESH-REFINEMENT.md` §8–§10:

  * the error on a graded mesh is `0.0487 h0` in the width of the cell touching
    the singularity and in *nothing else* — not the cell count — so the useful
    move is redistribution at a fixed budget rather than refinement;
  * the modal decay rate decides whether to grade and at which end, reliably from
    `k >= 3`, and at `k = 2` the decision is *inverted* rather than merely noisy;
  * so the order of an hp scheme is forced: **p, then h, then p**;
  * per-cell degrees are **gated no** — 217 `(k+1)` sites for a few percent of DOF,
    because the smooth region's degrees were measured to buy nothing.

  Still open: the driver that runs that sequence; and TimeMarch, which is blocked
  on exactly what this entry says.

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
