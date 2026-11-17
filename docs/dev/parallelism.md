# Parallelism and the trace solve

`MANTA_OPENMP`, BLAS threading, and the banded factorisation of the trace system.

## Parallelism (`MANTA_OPENMP`, off by default)

**Every `#pragma omp` in the tree is inside `util/ParallelFor.hpp`.** Nothing
else may write one, and nothing else may thread either: the build defines
`EIGEN_DONT_PARALLELIZE` (`cmake/MantaDependencies.cmake`, and `manta.pc`), so
Eigen does not thread its matrix products under `-fopenmp` the way it would by
default. BLAS threading is the one exception, and it is the BLAS's, below.
`manta::parallel_for(n, body, grain)` is the only entry point, and it exists
because three properties were needed at all eight sites and present at none of
them.

**Exceptions.** A physics hook throwing is *supported*: `static_residual`
catches it, prints, and returns 1, which IDA treats as recoverable and retries
with a smaller step. It is also how a Python case's exception reaches the
solver. An exception that escapes an OpenMP structured block does not
propagate — gcc's outlined function has no handler above it, so
`__cxa_call_terminate` aborts the process. With the bare pragmas this replaced,
`MANTA_OPENMP=ON` **aborted the unit suite**:

```
__cxa_call_terminate / __cxa_throw
ThrowingDiffusion::SigmaFn(...)
TransportSystem::SigmaFn(...) [clone ._omp_fn.0]
libgomp.so.1
```

Note how it hid: run that test alone and the throwing iteration lands on the
master thread, where the exception *can* reach the handler, and it passes. It
takes a worker thread to kill the process. `parallel_for` catches per iteration,
keeps the first exception, and rethrows on the calling thread.
`utility_tests/an_exception_from_the_body_reaches_the_caller` pins it, and throws
from the *last* index for that reason.

**A trip-count floor**, because forking a team for a handful of iterations costs
more than it saves and the fixtures here are 3–10 cells. The two grains differ by
16x on purpose: `TransportSystem::physicsGrain` is 64 because an iteration is one
pointwise hook call, `SystemSolver::cellGrain` is 4 because an iteration is a
dense factorisation. A single value cannot serve both — a floor of 32 on the cell
loops would have turned off the *only* regime where threading actually pays.

**Only the solver's cell loops are threaded. The physics is never threaded, and
that is a policy rather than an omission.** The default batched wrappers in
`TransportSystem.hpp` — the fallback for a case that supplies only pointwise
hooks — are plain serial loops. A case that has not provided a batched
implementation is assumed to have a reason, and threading its hooks would call
arbitrary case code concurrently on one instance, which nothing here can check
and the case never agreed to. A case that *wants* parallel physics overrides the
batched level, which is what those methods are virtual for.

For a Python case the rule is not merely prudent but load-bearing. Every
pointwise trampoline in `PyTransportSystem` takes the GIL, so N threads serialise
on it and pay a lock handoff *per point*. Measured: with the wrappers threaded,
`MANTA_OPENMP=ON` at four threads took the Python suite from ~110 s to **over
1500 s, where it hit ctest's timeout without finishing** — a floor of 13.6x
slower, not a failure to speed up. It reached CI as a red required leg. With the
wrappers serial the same suite is **109.6 s at `OMP_NUM_THREADS=4`**, i.e. back
to indistinguishable from serial. The vectorised path was always right:
`PyTransportSystem::ComputePhysics` takes the GIL *once* for the whole grid, and
`manta.jax` goes through it.

Parallel, then: the per-cell factorisation in `updateMatricesForJacSolve`, the
per-cell solves in `solveHDGJac`, and both back-substitutions. Serial, each
saying so at the site: anything accumulating into `K_global` or `F`, because
**lambda lives on cell faces** so neighbours share the block they write; and the
adjoint's matrix build, which grows `adjoint_CEBlocks`/`adjoint_CGBlocks` with
`emplace_back` and runs once per run rather than once per Newton iteration.

**Nothing under a parallel loop calls into a physics case any more, and getting
there found real waste.** `assembleCellMatrix` — which *is* inside the threaded
cell loop — built its `X` block by integrating `alphaValue * aFn` through a
30-point Gauss rule per entry, per variable, per cell, on every Jacobian build.
`aFn` is a pointwise hook and `PyTransportSystem` overrides it with
`PYBIND11_OVERRIDE`, so a Python case's `aFn` was being called from OpenMP worker
threads, taking the GIL per quadrature point. But `initialiseMatrices` already
stores exactly that matrix unweighted in `XMats`, `MassMatrix` is linear in its
weight, and `aFn` takes no time argument — so `alphaValue * XMats[i]` is the same
quantity for none of the work. Removing the recomputation is worth **24-32%** on
its own:

| | before | after |
|---|---|---|
| k=3, 400 cells | 555 ms | 395 ms |
| k=4, 200 cells | 2147 ms | 1624 ms |
| k=8, 50 cells | 1226 ms | 828 ms |
| k=10, 20 cells | 755 ms | 512 ms |

The general point is worth more than the number: a quantity that depends on
neither the state nor the time was being rebuilt once per Newton iteration
because the assembly that needed it was written in terms of the hook rather than
in terms of the thing already derived from the hook.

**What threading is worth**, with all of the above in place. `MKL_NUM_THREADS=1`,
best of 3:

| | 1 thread | 4 threads |
|---|---|---|
| k=3, 400 cells | 482 ms | 371 ms (1.30x) |
| k=4, 200 cells | 2007 ms | 1235 ms (1.63x) |
| k=8, 50 cells | 1183 ms | 687 ms (1.72x) |
| k=10, 20 cells | 676 ms | **379 ms (1.78x)** |

Note that taking the physics *out* of the parallel region cost nothing and helped
everywhere: every absolute time above improved, the single-thread baselines
included, because `parallel_for` was charging its own overhead — an atomic flag, a
try/catch, an `exception_ptr` — on loops whose bodies were a single hook call.

**Threading changes no answers, and that is checkable rather than hoped for.**
There is no reduction anywhere in `parallel_for` — every iteration writes its own
slot — so the arithmetic is the same operations in the same order whatever the
team size. Verified the way this class of claim is verified elsewhere:
the k=10 case run by the serial build and by the OpenMP build at four threads
gives **byte-identical `.nc` and `.restart.nc`**, not merely agreement at the
regression suite's 5e-3. Re-run that after touching `util/ParallelFor.hpp` or any
loop that goes through it; the regression tolerance is far too loose to see a
change of this kind. It is also the sharp contrast with BLAS threading below,
which *does* move the last bits.

Both suites pass under `MANTA_OPENMP=ON` at `OMP_NUM_THREADS=6`, and the unit
suite is *faster* that way — 16.1 s against 26.5 s serial, because
`MMSConvergenceTests` runs at exactly the high degrees where this pays.

**Oversubscription is not a mild loss here — it is a 2x regression.** Eight
threads on 20 cells costs 1963 ms against 528 at four, and eight on 200 cells
costs 33359 against 16399 at one. Cap `OMP_NUM_THREADS` near the cell count and
well below the core count; there is no `num_threads` clause in `parallel_for`
doing it for you.

**Building with OpenMP silently turns on BLAS threading, and that changes
answers.** `-fopenmp` loads `libgomp`, which is enough for a dispatching BLAS to
start threading itself — on the development box `libblas.so.3` is
`libmkl_rt.so`, and with `OMP_NUM_THREADS=6` the multithreaded MKL changed
`dgemm` reduction order enough that `afn_tests/the_jacobian_agrees_with_the_
residual_for_a_nonunit_coefficient` failed with `IDACalcIC could not complete`.
**That test parallelises none of MaNTA's own loops** — `nCells = 3` is below
`cellGrain` and its 12 physics points are below `physicsGrain` — which is what
makes the attribution certain, and it was confirmed by separating the two
variables: `OMP_NUM_THREADS=6 MKL_NUM_THREADS=1` passes, `OMP_NUM_THREADS=1
MKL_NUM_THREADS=4` fails. Set `MKL_NUM_THREADS` (or `OPENBLAS_NUM_THREADS`)
explicitly whenever `OMP_NUM_THREADS` is set, and note the two want opposite
things: BLAS threading is the *only* thing that helps the k=4 case (1.39x) and it
hurts the k=10 one (1.7x slower).

**Nothing in CI builds with `MANTA_OPENMP=ON`**, which is why all of the above
survived to be found by hand. The `utility_tests` cases are the guard, but they
only bite in a build that sets it.

## The trace solve (`util/BandedMatrix.hpp`)

Static condensation leaves a matrix on the cell faces, `K_global`, and
`solveHDGJac` factorises it on **every Newton iteration**. It used to be a dense
`Eigen::FullPivLU` of side `nVars * (nCells + 1)` — O(nCells^3) in the one
quantity the method exists to make O(nCells) — and it was **91% of a 400-cell
k=3 run and 73% of a 200-cell k=4 one**. It is now a banded LU with partial
pivoting: `dgbtrf`/`dgbtrs` when a LAPACK was found, an equivalent built-in when
not. Measured 9.4x and 7.6x on those two cases; `TODO` has the table and the
per-phase numbers behind it.

**It is banded only in an ordering the solution vector does not use.** Lambda is
laid out `var * (nCells + 1) + node`, and in that order two nodes of different
variables sit `nCells + 1` apart — a full-width matrix. Indexed `(node, var)`
instead, cell `i` touches only nodes `i` and `i+1`, so the bandwidth is
`2 * nVars - 1` either side. The band form therefore carries **its own ordering**
and the solve gathers into it and scatters back (`toTraceMajor` /
`fromTraceMajor`). That is deliberate and worth preserving: the DOF layout, the
restart format, `DGSoln::Map` and the pybind11 casters are all untouched, and the
permutation is O(n) against a solve that is now O(n * band^2).

**`K_global` is singular, and the dense decomposition was hiding it.** A
Dirichlet end sets `Hvar`'s diagonal to zero (`initialiseMatrices`) and nothing
else writes that trace DOF, so the row *and* the column are identically zero —
the same rank deficiency `CLAUDE.md` records for the finite-differenced
Jacobian, here in the operator itself. `FullPivLU` returns the particular
solution with the free components zeroed, which is the right answer arrived at by
accident; a banded LU has no such behaviour to fall back on and reports the zero
pivot. So `imposeDirichletTraceRows` writes the constraint down — identity on the
row, zero on the right-hand side, which is correct because `delta lambda = 0` at
a face already sitting at `g_D(t)`. Measured rank 3/5 and 4/6 on the fixture
solves, with the banded answer matching the dense one to 2.7e-15. **If you touch
the boundary assembly, that identity is load-bearing**: without it the solver
throws on the first step rather than degrading quietly, which is the one mercy
here.

**The singular-matrix report is a `throw`, and it needed a catch that was not
there.** `SunLinSolWrapper::Solve` is a SUNDIALS C callback — it reaches
`solveJacEq` through a function pointer — so an escaping exception is undefined
behaviour, exactly the hazard `static_residual` was written to close. It now has
the same try/catch, returning 1, which IDA treats as a recoverable linear-solver
failure and responds to by cutting the step and re-forming the Jacobian. Worth
knowing that this closes a **pre-existing** hole too: `solveJacEq` allocates, so a
`bad_alloc` — or anything a field model threw — was already unwinding through C
frames before any of this.

**This changes answers at round-off, unlike the OpenMP work.** A different
factorisation moves the last bits and IDA's step sequence follows, so byte
comparison is the wrong check — measured worst relative difference in the netCDF
output is 4.2e-13 at k=3 and 2.7e-11 at k=10, against run tolerances of 1e-8 and
1e-6. Contrast the threading, which is byte-identical because it is the same
operations in the same order. Know which kind of change you are making before
choosing the check.

**LAPACK is optional and pinned to the BLAS's vendor, and the reason is the
dlopen trap.** `cmake/MantaDependencies.cmake` asks `FindLAPACK` for whatever
vendor the BLAS actually resolved to, copies `LAPACK_LIBRARIES` **out by value**,
and links the paths rather than `LAPACK::LAPACK`. That last part was measured, not
guessed: CMake's `FindLAPACK` creates the imported target under
`if(NOT TARGET LAPACK::LAPACK)` but sets `INTERFACE_LINK_LIBRARIES` on it
*outside* that guard, so `SUNDIALSConfig.cmake`'s `find_dependency(LAPACK)` a few
lines later **rewrites the contents of the target this project already linked**.
On the development box that put `libmkl_gf_lp64 + libmkl_gnu_thread +
libmkl_core + libgomp` — the layered link that is unsafe to `dlopen`, and the one
the BLAS block exists to avoid — onto the link line of `libmanta` and the *Python
module*, while an isolated `find_package(LAPACK)` with the same `BLA_VENDOR`
resolved cleanly. It is the BLAS trap one level deeper: not the variable this
time, the target.

The block also `unset`s `LAPACK_LIBRARIES` from the cache before every find. That
is what makes it self-healing: `FindLAPACK` short-circuits on a cached value, and
without the unset a single bad configure is permanent — the poisoned value is
read back, captured and re-pinned forever, and reconfiguring does not clear it.
Verified by poisoning a build directory and watching it recover.

**LAPACK is not faster than the built-in here, and that is expected rather than
disappointing.** Measured within noise on all four benchmarks, the built-in
marginally ahead on three (537 vs 555 ms, 1180 vs 1226, 718 vs 755, 2163 vs
2147). The bands are 1 wide for a single-variable case and 3 for two, so
`dgbtrf`'s blocking has nothing to exploit and its call overhead is comparable to
the arithmetic. LAPACK is preferred because it is the reference implementation
and someone else maintains it — not for speed — and it will start to matter for a
case with many variables, where the band is `2 * nVars - 1`. So do not "optimise"
by dropping the LAPACK path on the strength of these numbers; they say the two
are equivalent at `nVars` of 1 or 2 and nothing about `nVars` of 8.

`utility_tests` exercises **both** the LAPACK path and the built-in one in every
build, whatever the build found, through `factorizeBuiltin`/`solveInPlaceBuiltin`.
Without that the fallback would rot on every box that has LAPACK, which is most
of them — and it is the boxes that do not that need it.
