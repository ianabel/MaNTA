#ifndef MESHADAPTATION_HPP
#define MESHADAPTATION_HPP

#include <memory>
#include <vector>

#include "SolverConfig.hpp"
#include "Types.hpp"
#include "gridStructures.hpp"

// DGSoln is an alias for DGSolnImpl<NodalBasis>, not a class, so it cannot be
// forward-declared as one -- and gridStructures.hpp above already pulls in enough
// that taking the definition costs nothing here.
#include "DGSoln.hpp"
#include "SmoothnessSensor.hpp"

class AdjointProblem;
class PhysicsInstance;
class SystemSolver;
class TransportSystem;

/*
 * Deciding whether to grade the mesh, and driving p -> h -> p.
 *
 * MESH-REFINEMENT.md sections 8-10 measured three things that together determine
 * this whole scheme, and none of them is a design choice:
 *
 *  1. On a mesh graded towards a singularity the error is `0.0487 * h0` in the
 *     width of the cell touching it and in *nothing else* -- not the cell count.
 *     So the useful move is to redistribute a fixed budget, not to refine: 14900x
 *     on Shestakov at an unchanged 10 cells and 60 DOF, against 4.0x for spending
 *     four times the DOF on uniform refinement.
 *  2. The per-cell modal decay rate decides whether to grade, and at which end,
 *     from one uniform solve -- reliably from k >= 3.
 *  3. **At k = 2 that decision is not merely unreliable but inverted**, so the
 *     order is forced: enough degree first, then the mesh, then degree to
 *     tolerance.
 *
 * Point 3 is why runAdaptiveMesh refuses k < 3 rather than warning. The mechanism
 * is specific and worth knowing before choosing a starting degree: the fit needs
 * at least two genuinely decaying modes, and at k = 2 it has one plus whatever the
 * first mode happens to be. A solution that is *flat* at a boundary -- which is
 * what a zero-flux axis produces, so it is the normal case here -- has a
 * suppressed linear coefficient, and a two-point fit reads that suppression as
 * slow decay. The failure is therefore systematic, in the direction of a false
 * positive, on exactly the cell being interrogated. k = 3 clears it and k = 4 puts
 * two decaying modes in the fit.
 */

enum class GradingVerdict
{
    Uniform,
    GradeLower,
    GradeUpper,
};

// What the sensor concluded, and the numbers behind it, so a caller can log or
// test the reasoning rather than just the answer.
struct GradingDecision
{
    GradingVerdict verdict = GradingVerdict::Uniform;

    // The fitted decay rate of the first and last cells, as the sensor reported
    // them. Larger is smoother. Infinity is a real value and means the cell's
    // solution is exactly representable below degree k.
    double lowerRate = 0.0;
    double upperRate = 0.0;

    // The largest rate measurable at this degree (measurableDecayRate), and the
    // median over the interior with every rate capped at it -- so finite.
    double rateCeiling = 0.0;
    double interiorMedian = 0.0;

    // interiorMedian / rate, per end, with the rate capped at the ceiling: how
    // much rougher that end is than the body of the domain. This is the quantity
    // compared against the threshold. Infinite only for an end with no decay at
    // all.
    double lowerRatio = 0.0;
    double upperRatio = 0.0;
};

// Should this mesh be graded, and at which end?
//
// Compares each end cell's decay rate against the *median over the interior*
// rather than against a fixed threshold. That choice is measured, not stylistic:
// section 7 found a fixed threshold unsafe across degrees, because S_K moves
// eight orders over k = 2..6 while the exponent stays O(1). Comparing to the
// interior also makes the test scale-free in the solution and blind to a uniform
// loss of resolution, which is the degree loop's business rather than this one's.
//
// `threshold` is the factor by which an end must be rougher to be graded.
// Measured on three problems at k >= 3: 3.09-6.80 for the one that wants grading,
// 0.97-1.19 for the two that do not, so anything in (1.2, 3.0) separates them and
// 2.0 sits in the middle. That is a gap between two populations of one and two
// problems -- enough to fix the mechanism, not a calibration.
//
// Requires at least 3 cells, so that there is an interior to compare against, and
// k >= 2 from the sensor itself.
GradingDecision gradingDecision(DGSoln const &Y, Index var, double threshold);

// The same rule over rates already measured, at degree k -- which is what the
// rule is, and what makes it testable on spectra no assigned function produces.
// Every rate is compared capped at measurableDecayRate(k): above it nothing is
// measured, only round-off on one side of the floor or the other.
GradingDecision gradingDecision(std::vector<CellSmoothness> const &cells,
                                unsigned int k, double threshold);

// The mesh a decision asks for, at the *same cell count* as the one it was made
// on -- which is the point, per (1) above.
//
// `gradedCells` of 0 means "as many as the budget allows", i.e. nCells - 1. That
// differs from what GradingCells = 0 means on the manual GradedGridBoundary path,
// where it is half the grid, and the difference is deliberate: here exactly one
// end is being graded and the error is known to be 0.0487*h0, so maximising the
// cells in the layer minimises h0 at a fixed budget. Section 9 measured 9 of 10
// beating 5 of 10 by 48x.
std::vector<Grid::Position> gradedMeshFor(GradingDecision const &decision,
                                          Grid const &uniform,
                                          Grid::Index gradedCells,
                                          double lowerFraction,
                                          double upperFraction,
                                          double ratio);

// How much of the domain the layer a decision grades into covers, as a fraction
// of the span.
//
// The end cell of the sampling mesh -- the cell the sensor flagged -- unless the
// caller gave LowerBoundaryFraction / UpperBoundaryFraction for that end. A fixed
// default fraction was wrong whenever the sampling mesh had already put a face
// somewhere that mattered. Shestakov's source switches off at x = 0.1, and a
// uniform 10-cell mesh has a face there: grading the default 20% moved that face
// and the error *rose*, 1.36e-2 to 1.58e-2 at k = 3, where grading within the
// first cell keeps it and gives 1.34e-6. The sampling mesh is the only
// information about where the faces should be that the driver has, and its end
// cell is the region the sensor actually judged rough.
//
// Throws std::logic_error for a Uniform decision, as gradedMeshFor does.
double gradingLayerFraction(GradingDecision const &decision,
                            Grid const &sampling,
                            SolverConfig const &config);

// What runAdaptiveMesh produced.
//
// Member order is load-bearing: `solver` holds a reference to `*grid`, and
// members are destroyed in reverse declaration order, so declaring the grid first
// is what makes the solver die before the mesh it points into. `grid` is always
// owned here even when no grading happened -- a copy of the caller's uniform mesh
// -- so the lifetime rule is the same either way rather than depending on the
// verdict.
struct AdaptiveMeshResult
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SystemSolver> solver;
    GradingDecision decision;
    unsigned int gradingAttempts = 0; // 0 when the mesh was left uniform
};

// Solve, decide, regrade, then adapt the degree: the p -> h -> p sequence.
//
// One solve at `k0` on the caller's uniform mesh, which is both the first `p`
// (k0 >= 3 is what makes the decision trustworthy) and the sample the decision is
// read from. Then, if an end is rough, the same cell count regraded towards it.
// Then runAdaptiveDegree to `DegreeTolerance` on whichever mesh won.
//
// **A grading that fails to solve is a rejected step, not an error.** Section 9
// measured 15 of 52 hand-built gradings failing outright, non-monotonically in
// everything -- IDA's corrector giving up at |h| = MinStepSize, and below about
// h0/span = 1e-7 nothing getting through at all. So a failed attempt softens the
// ratio towards 1 and retries, and after `MeshAdaptationAttempts` tries it falls
// back to the uniform mesh and says so. A driver without that would die on a
// third of the problems it is pointed at.
//
// The case must have been built against `uniform`. Each graded mesh, and the
// degree loop's levels, are new evaluation plans, so the case follows its
// RegridPolicy through PhysicsInstance::solverFor, and a Fixed one is refused
// before the sampling solve; see DegreeAdaptation.hpp.
//
// Throws std::invalid_argument for k0 < 3.
AdaptiveMeshResult runAdaptiveMesh(SolverConfig const &config,
                                   PhysicsInstance &physics,
                                   Grid const &uniform,
                                   unsigned int k0,
                                   double tFinal);

#endif // MESHADAPTATION_HPP
