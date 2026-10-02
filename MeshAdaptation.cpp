#include "MeshAdaptation.hpp"

#include "AdjointProblem.hpp"
#include "DegreeAdaptation.hpp"
#include "Logging.hpp"
#include "SmoothnessSensor.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
// Median of the finite entries, or infinity when there are none.
//
// Infinity is the right answer rather than a failure: an all-infinite interior
// means every interior cell's solution is exactly representable below degree k,
// which is Jardin's linear steady state, and the ratio it produces (median/rate)
// is then infinite or NaN for a rough end and 1 for a smooth one. Handled at the
// ratio rather than here.
double medianFinite(std::vector<double> v)
{
    auto end = std::remove_if(v.begin(), v.end(),
                              [](double x) { return !std::isfinite(x); });
    v.erase(end, v.end());
    if (v.empty())
        return std::numeric_limits<double>::infinity();

    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// median / rate, guarded at both ends.
//
// A rate of zero means the spectrum has no decay at all, which is as rough as the
// sensor can report and so should always fire; infinity means perfectly smooth and
// should never. Neither is reachable by division alone.
double roughness(double median, double rate)
{
    if (!std::isfinite(rate))
        return 0.0;              // this end is smoother than anything measurable
    if (rate <= 0.0)
        return std::numeric_limits<double>::infinity();
    if (!std::isfinite(median))
        return std::numeric_limits<double>::infinity(); // interior exact, end is not
    return median / rate;
}

const char *name(GradingVerdict v)
{
    switch (v)
    {
    case GradingVerdict::GradeLower: return "Lower";
    case GradingVerdict::GradeUpper: return "Upper";
    case GradingVerdict::Uniform:    return "none";
    }
    return "none";
}
} // namespace

GradingDecision gradingDecision(DGSoln const &Y, Index var, double threshold)
{
    if (!(threshold > 1.0))
        throw std::invalid_argument(
            "The grading threshold is a factor by which an end must be rougher "
            "than the interior, so it has to exceed 1; at or below it every mesh "
            "is graded, including one whose ends are the smoothest cells it has.");

    auto const cells = cellSmoothness(Y, var);

    if (cells.size() < 3)
        throw std::invalid_argument(
            "Deciding whether to grade needs at least 3 cells: the two ends are "
            "compared against the interior, and with fewer there is no interior.");

    GradingDecision d;
    d.lowerRate = cells.front().decayRate;
    d.upperRate = cells.back().decayRate;

    std::vector<double> interior;
    interior.reserve(cells.size() - 2);
    for (size_t i = 1; i + 1 < cells.size(); ++i)
        interior.push_back(cells[i].decayRate);

    d.interiorMedian = medianFinite(std::move(interior));
    d.lowerRatio = roughness(d.interiorMedian, d.lowerRate);
    d.upperRatio = roughness(d.interiorMedian, d.upperRate);

    // The rougher end wins if either clears the bar. A tie goes to the lower end,
    // which is arbitrary and only reachable when both ends are equally rough --
    // at which point grading one end is the wrong tool anyway and GradingEnd =
    // "Both" or explicit GridPoints is what the user wants.
    const double worst = std::max(d.lowerRatio, d.upperRatio);
    if (worst >= threshold)
        d.verdict = (d.lowerRatio >= d.upperRatio) ? GradingVerdict::GradeLower
                                                   : GradingVerdict::GradeUpper;

    return d;
}

std::vector<Grid::Position> gradedMeshFor(GradingDecision const &decision,
                                          Grid const &uniform,
                                          Grid::Index gradedCells,
                                          double lowerFraction,
                                          double upperFraction,
                                          double ratio)
{
    if (decision.verdict == GradingVerdict::Uniform)
        throw std::logic_error(
            "gradedMeshFor was asked for a mesh from a decision that said not to "
            "grade.");

    const Grid::Index nCells = uniform.getNCells();

    // As many cells in the layer as the budget allows, unless told otherwise --
    // see the header for why this differs from GradingCells = 0 on the manual
    // path.
    const Grid::Index inLayer = (gradedCells == 0) ? nCells - 1 : gradedCells;

    return gradedMeshPoints(uniform.lowerBoundary(), uniform.upperBoundary(),
                            nCells, inLayer, lowerFraction, upperFraction, ratio,
                            decision.verdict == GradingVerdict::GradeLower
                                ? GradedEnd::Lower
                                : GradedEnd::Upper);
}

namespace
{
// One solve at a fixed degree on a given mesh, returning the solver. Split out so
// that the sampling solve and the retry loop share it, and so the "never two
// solvers alive" discipline lives in one place.
std::unique_ptr<SystemSolver> solveOnce(SolverConfig const &config,
                                        TransportSystem &problem,
                                        AdjointProblem *adjoint,
                                        Grid const &grid, unsigned int k,
                                        double tFinal)
{
    auto system = std::make_unique<SystemSolver>(grid, k, &problem);
    applySolverConfig(config, *system);

    // The same check runAdaptiveDegree makes, for the same reason: what selects
    // the steady path is TerminateOnSteadyState, not the SteadyStateSolver key,
    // and a configuration that time-marches would take each stage of this
    // sequence from the previous stage's final state and integrate the interval
    // again -- a wrong answer rather than a scope violation.
    if (!system->solvesForSteadyState())
        throw std::invalid_argument(
            "MeshAdaptation needs a steady solve, but this configuration "
            "time-marches: SteadyStateTolerance is absent, so steady-state "
            "termination is never armed. Set SteadyStateTolerance or "
            "SteadyStateSolve, or call run_ss().");

    if (adjoint != nullptr)
        system->setAdjointProblem(adjoint);

    system->runSolver(tFinal);
    return system;
}
} // namespace

AdaptiveMeshResult runAdaptiveMesh(SolverConfig const &config,
                                   TransportSystem &problem,
                                   AdjointProblem *adjoint,
                                   Grid const &uniform,
                                   unsigned int k0,
                                   double tFinal)
{
    // Refused rather than warned about, because at k = 2 the decision is
    // *inverted* rather than merely uncertain -- it grades a smooth problem harder
    // than a singular one. A driver that proceeded would confidently do the wrong
    // thing. See the header for the mechanism.
    if (k0 < 3)
        throw std::invalid_argument(std::format(
            "MeshAdaptation needs PolynomialDegree >= 3, but it is {}. The "
            "grading decision is read from the decay of the modal coefficients, "
            "and at k = 2 the fit has one genuinely decaying mode plus a first "
            "mode that a solution flat at a boundary -- a zero-flux axis, say -- "
            "suppresses. Measured on three problems, the verdict at k = 2 is not "
            "merely noisy but reversed. 4 or more is better still.", k0));

    const Index var = 0;

    std::println("Mesh adaptation: sampling at k = {} on {} uniform cells",
                 k0, uniform.getNCells());

    AdaptiveMeshResult result;
    result.grid = std::make_unique<Grid>(uniform);

    // --- p: the sampling solve, at a degree the decision can be trusted at -----
    auto sample = solveOnce(config, problem, adjoint, *result.grid, k0, tFinal);

    // --- h: decide, and regrade at the same cell count -------------------------
    result.decision = gradingDecision(sample->solution(), var,
                                      config.MeshAdaptationThreshold);
    auto const &d = result.decision;

    std::println("  decay rate: lower end {:.3g}, interior median {:.3g}, upper end "
                 "{:.3g}",
                 d.lowerRate, d.interiorMedian, d.upperRate);
    std::println("  roughness vs interior: lower {:.2f}x, upper {:.2f}x, threshold "
                 "{:.2f}x -> grade {}",
                 d.lowerRatio, d.upperRatio, config.MeshAdaptationThreshold,
                 name(d.verdict));

    if (d.verdict != GradingVerdict::Uniform)
    {
        // Destroyed before the next is built. Integrator's weight cache is a
        // process-wide global keyed on (order, grid) and residual() revalidates it
        // on every evaluation, so two live solvers on different meshes would clear
        // and rebuild that map once per residual instead of once per level.
        sample.reset();

        double ratio = config.GradingRatio;
        for (unsigned int attempt = 1; attempt <= config.MeshAdaptationAttempts;
             ++attempt)
        {
            auto points = gradedMeshFor(d, uniform,
                                        static_cast<Grid::Index>(config.GradingCells),
                                        config.LowerBoundaryFraction,
                                        config.UpperBoundaryFraction, ratio);
            auto graded = std::make_unique<Grid>(points);

            const double span = uniform.upperBoundary() - uniform.lowerBoundary();
            double narrowest = span;
            for (Grid::Index i = 0; i < graded->getNCells(); ++i)
                narrowest = std::min(narrowest, (*graded)[i].h());

            std::println("  attempt {}: ratio {:.4g}, narrowest cell {:.3e} of the "
                         "domain", attempt, ratio, narrowest / span);

            try
            {
                auto trial = solveOnce(config, problem, adjoint, *graded, k0, tFinal);
                result.grid = std::move(graded);
                result.solver = std::move(trial);
                result.gradingAttempts = attempt;
                break;
            }
            catch (std::invalid_argument const &)
            {
                // A configuration error, not a solver failure. Softening the mesh
                // would not fix it and retrying would bury it.
                throw;
            }
            catch (std::exception const &e)
            {
                // The rejected-step path. Section 9 measured this as the real
                // ceiling on grading -- IDA's corrector failing at
                // |h| = MinStepSize once the narrowest cell is around 1e-6 of the
                // span -- and it is not monotone in anything, so a softer mesh is
                // worth trying rather than assuming the whole idea has failed.
                logmsg<LOG_LEVEL::WARNING>(
                    "Graded mesh attempt {} failed to solve ({}). Softening the "
                    "grading ratio and retrying.", attempt, e.what());
                std::println("  attempt {} failed; softening the ratio", attempt);

                // Towards 1, halving the distance each time, so the sequence is
                // monotone and terminates at the uniform mesh rather than
                // oscillating.
                ratio = std::sqrt(ratio);
            }
        }

        if (result.solver == nullptr)
        {
            logmsg<LOG_LEVEL::WARNING>(
                "Every graded mesh attempted failed to solve, so the run continues "
                "on the uniform mesh. The decision to grade stands -- the sensor "
                "said this problem wants it -- and what failed is the time "
                "integrator, so MinStepSize is the first thing to lower.");
            std::println("  all {} attempts failed; continuing on the uniform mesh",
                         config.MeshAdaptationAttempts);
            result.grid = std::make_unique<Grid>(uniform);
            result.gradingAttempts = config.MeshAdaptationAttempts;
        }
    }
    else
    {
        result.solver = std::move(sample);
    }

    // --- p: the degree loop, on whichever mesh won -----------------------------
    //
    // Handed the mesh rather than the config, so it never consults the grading
    // keys and cannot rebuild a different one. It owns its own solvers, so the
    // sampling solver goes first.
    result.solver.reset();

    std::println("Mesh adaptation: adapting the degree on the {} mesh",
                 result.gradingAttempts > 0 && result.decision.verdict != GradingVerdict::Uniform
                     ? "graded" : "uniform");
    result.solver = runAdaptiveDegree(config, problem, adjoint, *result.grid, k0,
                                      tFinal);
    return result;
}
