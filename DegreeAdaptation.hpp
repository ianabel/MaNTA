#ifndef DEGREEADAPTATION_HPP
#define DEGREEADAPTATION_HPP

#include <memory>

#include "SolverConfig.hpp"
#include "Types.hpp"

class AdjointProblem;
class Grid;
class SystemSolver;
class TransportSystem;

/*
 * Choosing the global polynomial degree, by solving and looking at the answer.
 *
 * Solve at k, estimate how well resolved the result is from the gap between
 * u_h and its own postprocessing u*, raise k, solve again. Measured on
 * AdjointPoster (MESH-REFINEMENT.md section 6): 2.8e-9 at 90 DOF in two
 * iterations and 3060 physics evaluations, against 2.0e-6 at 128 DOF for 16672
 * with adaptive *h*. One degree bump beats the whole h-adaptive machinery on
 * every benchmark in this tree, which is why this is where the adaptivity work
 * starts rather than with a remesh.
 *
 * The degree is *global*. Per-cell degrees are a much larger change than they
 * look -- DGSolnImpl holds one const Index k and one basis by value, and there
 * are some 320 (k+1) sites in the core -- and the measurement says most of the
 * win does not need them.
 */

// How far to raise the degree, given the current error estimate and the target.
//
// Giorgiani's rule (refs/Refs.md, "Mesh adaptivity"):
//
//     dk = ceil( log_base( E / eps ) )
//
// The point of it, and why it is used here rather than the Richardson target
// size the original plan carried, is that it assumes *no convergence order*.
// It only supposes that one more degree buys roughly a factor of `base`, which
// is a statement about the method's general behaviour rather than a calibration
// against a measured rate. That matters because u*'s rate is not dependable:
// docs/superconvergence.rst measures it falling 6.9, 11.7, 9.1, then 2.3 for a
// nonlinear flux at k = 1 -- it superconverges over the coarse grids and then
// stops. A rule calibrated on the coarse-grid ratio would over-predict the gain
// from refining and then spend its whole degree budget missing the target.
//
// Returns at least 1 whenever E exceeds eps, so a converging loop always makes
// progress; returns 0 when the target is already met.
unsigned int degreeIncrement(double E, double eps, double base);

// Solve `problem`, adapting the global polynomial degree between solves, and
// return the solver that produced the final answer.
//
// Builds and destroys one SystemSolver per level and is careful never to have
// two alive at once: Integrator's weight cache is a process-wide global keyed
// on (order, grid), and residual() calls invalidateIfStale on every single
// evaluation, so a second live solver at a different degree would clear and
// rebuild that map on every residual rather than once per level.
//
// The state crosses each level through the restart mechanism -- snapshot yJac
// into a vector, setRestartValues, destroy, build the next -- which copies both
// the vector and the Grid, so nothing points into the solver being destroyed.
// setInitialConditions then projects across the degree change. `restarting` is
// cleared before returning, since it is sticky and would otherwise make the
// *next* run on the same configuration resume from the last level instead of
// from InitialValue.
//
// `adjoint` may be null; when it is not, it is re-attached to each new solver,
// which a fresh one does not inherit.
//
// Only the caller's grid and problem outlive this. The returned solver holds a
// reference to that grid, so it must not outlive it.
std::unique_ptr<SystemSolver> runAdaptiveDegree(SolverConfig const &config,
                                                TransportSystem &problem,
                                                AdjointProblem *adjoint,
                                                Grid const &grid,
                                                unsigned int k0,
                                                double tFinal);

#endif // DEGREEADAPTATION_HPP
