#include "ParallelFill.hpp"

#include <format>
#include <limits>

namespace
{
Index ceilDiv(Index a, Index b) { return (a + b - 1) / b; }

Grid uniformOver(Grid const &grid, Grid::Index nCells)
{
    return Grid(grid.lowerBoundary(), grid.upperBoundary(), nCells);
}
} // namespace

EvaluationRounds roundsOf(EvaluationPlan const &plan, unsigned int width)
{
    const Index w = std::max<Index>(width, 1);
    EvaluationRounds r;
    for (auto const &site : plan.sites)
    {
        if (!site.isBatched())
            continue;
        const Index rounds = site.calls * ceilDiv(site.batchSize(), w);
        switch (site.cadence)
        {
        case EvaluationCadence::PerResidual:
            r.perResidual += rounds;
            break;
        case EvaluationCadence::PerJacobianBuild:
            r.perJacobianBuild += rounds;
            break;
        case EvaluationCadence::PerContinuationStep:
            r.perContinuationStep += rounds;
            break;
        default:
            break;
        }
    }
    return r;
}

double residualOccupancy(EvaluationPlan const &plan, unsigned int width)
{
    const Index w = std::max<Index>(width, 1);
    Index points = 0, slots = 0;
    for (auto const &site : plan.sites)
        if (site.isBatched() && site.cadence == EvaluationCadence::PerResidual)
        {
            points += site.calls * site.batchSize();
            slots += site.calls * ceilDiv(site.batchSize(), w) * w;
        }
    return slots == 0 ? 1.0 : static_cast<double>(points) / static_cast<double>(slots);
}

bool withinBatchCap(EvaluationPlan const &plan, Index maxBatch)
{
    return maxBatch == 0 || plan.largestRecurringBatch() <= maxBatch;
}

unsigned int degreeCeiling(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                           unsigned int kMax, Index maxBatch)
{
    if (maxBatch == 0)
        return kMax;
    unsigned int ceiling = k;
    // Batches never shrink as k rises, so the first degree past the cap ends it.
    for (unsigned int next = k + 1; next <= kMax; ++next)
    {
        if (!withinBatchCap(planAt(grid, next), maxBatch))
            break;
        ceiling = next;
    }
    return ceiling;
}

unsigned int filledDegree(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                          unsigned int kMax, unsigned int width, Index maxBatch)
{
    if (width <= 1 || k >= kMax)
        return k;
    const EvaluationRounds budget = roundsOf(planAt(grid, k), width);
    unsigned int filled = k;
    // Rounds never fall and batches never shrink as k rises, so the first degree
    // that costs more or carries too much ends it.
    for (unsigned int next = k + 1; next <= kMax; ++next)
    {
        const EvaluationPlan plan = planAt(grid, next);
        if (!roundsOf(plan, width).fitsWithin(budget) || !withinBatchCap(plan, maxBatch))
            break;
        filled = next;
    }
    return filled;
}

Grid::Index filledCellCount(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                            Grid::Index nMax, unsigned int width, Index maxBatch)
{
    const Grid::Index n = grid.getNCells();
    if (width <= 1 || n >= nMax)
        return n;
    const EvaluationRounds budget = roundsOf(planAt(uniformOver(grid, n), k), width);
    auto fits = [&](Grid::Index cells)
    {
        const EvaluationPlan plan = planAt(uniformOver(grid, cells), k);
        return roundsOf(plan, width).fitsWithin(budget) && withinBatchCap(plan, maxBatch);
    };

    // Every cell adds at least one point to the residual's batch.
    Grid::Index hi = std::min(nMax, static_cast<Grid::Index>(budget.perResidual) * width);
    if (fits(hi))
        return hi;
    // fits(lo) and !fits(hi) throughout.
    Grid::Index lo = n;
    while (hi - lo > 1)
    {
        const Grid::Index mid = lo + (hi - lo) / 2;
        (fits(mid) ? lo : hi) = mid;
    }
    return lo;
}

std::string underuseWarning(LevelPlan const &planAt, Grid const &grid, unsigned int k,
                            unsigned int kMax, unsigned int width, Index maxBatch)
{
    if (width <= 1)
        return {};
    const unsigned int kFill = filledDegree(planAt, grid, k, kMax, width, maxBatch);
    const Grid::Index nFill = filledCellCount(
        planAt, grid, k, std::numeric_limits<Grid::Index>::max(), width, maxBatch);
    if (kFill == k && nFill == grid.getNCells())
        return {};

    const EvaluationPlan plan = planAt(grid, k);
    std::string instead;
    if (kFill > k)
        instead = std::format("PolynomialDegree up to {}", kFill);
    if (nFill > grid.getNCells())
        instead += std::format("{}GridSize up to {}", instead.empty() ? "" : ", or ", nFill);
    return std::format(
        "PhysicsParallelism = {}: {} cells at k = {} fill {:.0f}% of the rounds each "
        "residual takes. {} costs no more rounds.",
        width, grid.getNCells(), k, 100.0 * residualOccupancy(plan, width), instead);
}
