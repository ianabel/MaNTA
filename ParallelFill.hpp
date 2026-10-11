#ifndef PARALLELFILL_HPP
#define PARALLELFILL_HPP

#include "EvaluationPlan.hpp"

#include <functional>
#include <string>

/*
    What a level of discretisation costs on a machine that evaluates the physics
    `width` points at a time, and the levels that cost no more.

    `PhysicsParallelism = N` says that a batch of M points takes ceil(M / N)
    rounds whatever M is, so a level whose batches leave the last round part
    empty is paying for evaluations it does not make. The adaptation controllers
    use that to *fill* a level: having chosen one as they always have, they take
    the largest level beyond it that costs no more rounds. The extra accuracy is
    free in physics time and may save a later level.

    The cost is counted over the batched sites a run repeats -- per residual, per
    Jacobian build, per continuation step -- each as calls x ceil(batch / N), and
    a level fills another only if it costs no more in every one of those
    cadences separately. Two kinds of site are left out, deliberately:

      * pointwise hooks, which MaNTA calls one point at a time in a serial loop,
        so no machine width applies to them;
      * sites that run once per run or per solver, or once per adjoint solve,
        which a level's cost per Newton iteration does not see.

    The plans compared are the solver's own (SystemSolver::evaluationPlanFor),
    so a candidate is costed by exactly the sites it would have.

    `MaxPhysicsBatch` is the other limit a level answers to: the most points one
    recurring batched call may carry (EvaluationPlan::largestRecurringBatch).
    Every function below that chooses a level takes it as `maxBatch`, 0 meaning
    no cap, and chooses none past it. A fill never adds rounds, but it does grow
    the batches, so the two limits are separate questions.
 */

// Rounds of `width`-wide batched evaluation per occasion, for each recurring
// cadence.
struct EvaluationRounds
{
    Index perResidual = 0;
    Index perJacobianBuild = 0;
    Index perContinuationStep = 0;

    // No more rounds than `other` in any cadence.
    bool fitsWithin(EvaluationRounds const &other) const
    {
        return perResidual <= other.perResidual &&
               perJacobianBuild <= other.perJacobianBuild &&
               perContinuationStep <= other.perContinuationStep;
    }
    bool operator==(EvaluationRounds const &) const = default;
};

EvaluationRounds roundsOf(EvaluationPlan const &plan, unsigned int width);

// Points evaluated per round slot over the residual's batched sites: 1 when
// every round is full.
double residualOccupancy(EvaluationPlan const &plan, unsigned int width);

// The plan a configured solver would make for a level: see
// SystemSolver::evaluationPlanFor.
using LevelPlan = std::function<EvaluationPlan(Grid const &, unsigned int)>;

// Does every recurring batch of `plan` fit within maxBatch? Always, at 0.
bool withinBatchCap(EvaluationPlan const &plan, Index maxBatch);

// The largest degree in [k, kMax] whose recurring batches on `grid` fit within
// maxBatch: kMax when maxBatch is 0, and k itself when nothing beyond it fits --
// including when k does not fit either, which is for initialize() to refuse.
unsigned int degreeCeiling(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                           unsigned int kMax, Index maxBatch);

// The largest degree in [k, kMax] whose plan on `grid` costs no more rounds
// than the plan at k, and fits within maxBatch. `k` itself when width is 1 or
// nothing beyond it fits.
unsigned int filledDegree(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                          unsigned int kMax, unsigned int width, Index maxBatch = 0);

// The largest cell count in [n, nMax] whose plan at degree k costs no more
// rounds than n cells, and fits within maxBatch. Cells are counted on uniform meshes over `grid`'s
// domain: a plan's batch sizes depend on how many cells there are, not on
// where they are. Searched by bisection, because rounds never fall as cells
// are added and batches never shrink, and bounded by the residual's rounds
// times the width, since every cell adds at least one point to that batch.
Grid::Index filledCellCount(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                            Grid::Index nMax, unsigned int width, Index maxBatch = 0);

// A warning for a configured level that leaves rounds part empty, naming the
// largest degree (up to kMax) and the largest cell count that would cost the
// same within maxBatch; empty when neither is larger than the level itself.
std::string underuseWarning(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                            unsigned int kMax, unsigned int width, Index maxBatch = 0);

#endif // PARALLELFILL_HPP
