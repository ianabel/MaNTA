
Lists upcoming features that are in development or being thought about

- Explore adaptive mesh-refinement schemes; first to the PTC steady-state algorithm or possibly the direct newton solve as well; then work out if it's possible to add to the real TimeMarch (requires interpolating history to allow IDA's BDF methods to work, and requires the hooks for that to be in SUNDIALS)

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
