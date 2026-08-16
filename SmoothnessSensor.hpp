#ifndef SMOOTHNESSSENSOR_HPP
#define SMOOTHNESSSENSOR_HPP

#include "Types.hpp"

// Eigen/Core and Eigen/Dense before the project headers, matching
// SystemSolver.hpp and Postprocessing.hpp -- see the note there. The build
// defines EIGEN_USE_BLAS, and a translation unit that reaches Eigen only
// through Basis.hpp's <Eigen/LU> sees a different set of product
// specialisations from one that includes <Eigen/Dense>.
#include <Eigen/Core>
#include <Eigen/Dense>

#include "Basis.hpp"
#include "DGSoln.hpp"

#include <vector>

/*
 * Per-cell smoothness from the decay of the modal coefficients.
 *
 * Woopen, "A Hybridized Discontinuous Galerkin Method for Unsteady Flows with
 * Shock-Capturing" (refs/HDG-hpAdaptivity.pdf) section 4.3, after Persson &
 * Peraire; algebraically identical to Capasso et al. equation (13). Writing
 * w_H for the L2 projection of a cell's solution onto P_{k-1},
 *
 *     S_K = (w - w_H, w - w_H)_K / (w, w)_K
 *
 * In a Legendre basis that projection simply drops the top coefficient, and
 * with ||P_j||^2 = 2/(2j+1) the whole thing collapses to a ratio of weighted
 * squares of the modal coefficients -- no quadrature, and no factor of |K|, so
 * it is comparable across a non-uniform mesh:
 *
 *     S_K = [ uhat_k^2 / (2k+1) ] / sum_j [ uhat_j^2 / (2j+1) ]
 *
 * This is the piece all four papers indexed under "Mesh adaptivity" in
 * refs/Refs.md need and only Woopen supplies: an *a priori* statement about how
 * well the solution is resolved, as against the a-posteriori accuracy
 * indicator built from u* - u_h. It says where refining will pay, rather than
 * where the error currently is, and unlike the accuracy indicator it is
 * available without a second solve.
 *
 * Two things it reports, and the second is the one to use.
 *
 * S_K alone is what Persson & Peraire threshold, and a fixed threshold is
 * unsafe across degrees -- which Woopen does not address. Measured on the two
 * benchmarks in this tree (MESH-REFINEMENT.md section 7), the separation
 * between a smooth case and a singular one is 389x at k = 4 and only 2.2x at
 * k = 2, because at low k neither solution is resolved and both look rough.
 * Persson & Peraire's own S* ~ 1/k^4 is calibrated for shock capture and sits
 * orders of magnitude above anything seen here.
 *
 * So the decay *rate* is the quantity to drive decisions from (Mavriplis): fit
 *
 *     log|uhat_j| = c - s log j        over j = 1..k
 *
 * within each cell. One solve, no cross-degree calibration, and larger s means
 * smoother. On AdjointPoster against Shestakov at k = 6, s runs 4.78 and above
 * everywhere on the smooth case while Shestakov's singular cell sits at 2.93
 * against its own neighbour's 8.47 -- so s < 4 separates them. The margin
 * between the two *populations* is 1.6x rather than large, and on a milder
 * singularity they could overlap; that is a limit of the sensor, recorded
 * rather than designed around.
 */
struct CellSmoothness
{
    // S_K above: the share of the cell's modal energy in the top mode. Zero for
    // a cell whose solution is exactly representable below degree k.
    double modalEnergyFraction = 0.0;

    // s above: the fitted decay exponent, larger being smoother. Two values are
    // reported rather than fitted, and they are opposite ends:
    //
    //   infinity  the top mode is at round-off, so the cell's solution is
    //             exactly representable below degree k -- a constant, or any
    //             polynomial the space contains outright. Nothing left to
    //             resolve.
    //   zero      only the top mode is above round-off, so the spectrum has no
    //             decay in it at all.
    //
    // The fit itself runs over the modes above the floor, skipping the rest;
    // see the note in the .cpp on why a floored coefficient must not be fitted
    // as though it were a measurement.
    double decayRate = 0.0;
};

// One cell, from its nodal values -- which for this basis are its coefficients.
// This is the whole sensor; everything below is a loop over it.
//
// Requires k >= 2. The fit runs over j = 1..k, so k = 1 offers a single point
// and no slope, and that is a property of the run rather than of a cell: MaNTA
// carries one global order, so if it bites at all it bites everywhere. Throwing
// is therefore better than a per-cell sentinel that every caller would have to
// test for and none would.
CellSmoothness cellSmoothness(NodalBasis const &basis,
                              Eigen::Ref<const Vector> const &nodalValues);

// Every cell of one field, in grid order. `field` may be any degree-k DGApprox
// -- u, q, sigma or an auxiliary variable -- and the caller chooses; the
// measurements in MESH-REFINEMENT.md are all on u.
std::vector<CellSmoothness> cellSmoothness(DGSoln const &Y, Index var);

#endif // SMOOTHNESSSENSOR_HPP
