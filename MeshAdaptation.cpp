#include "MeshAdaptation.hpp"

#include "AdjointProblem.hpp"
#include "DegreeAdaptation.hpp"
#include "Logging.hpp"
#include "PhysicsInstance.hpp"
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
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// median / rate, both already capped at the measurable ceiling. A rate of zero
// means the spectrum has no decay at all, which is as rough as the sensor can
// report and so should always fire, and is not reachable by division.
double roughness(double median, double rate)
{
    if (rate <= 0.0)
        return std::numeric_limits<double>::infinity();
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

GradingDecision gradingDecision(std::vector<CellSmoothness> const &cells,
                                unsigned int k, double threshold)
{
    if (!(threshold > 1.0))
        throw std::invalid_argument(
            "The grading threshold is a factor by which an end must be rougher "
            "than the interior, so it has to exceed 1; at or below it every mesh "
            "is graded, including one whose ends are the smoothest cells it has.");

    if (cells.size() < 3)
        throw std::invalid_argument(
            "Deciding whether to grade needs at least 3 cells: the two ends are "
            "compared against the interior, and with fewer there is no interior.");

    GradingDecision d;
    d.lowerRate = cells.front().decayRate;
    d.upperRate = cells.back().decayRate;
    d.rateCeiling = measurableDecayRate(k);

    // Every rate is compared at most at the ceiling, infinite ones included. An
    // infinite rate and one just under the ceiling are the same measurement -- a
    // top mode at round-off -- and treating the first as larger than any number
    // made an end whose top mode sat a hair above the floor "infinitely rougher"
    // than an interior whose top modes sat a hair below it. That graded Jardin,
    // whose steady state the space holds outright: an interior median of
    // infinity against a wall cell at about 30, at a ceiling of 31.5, for a cost
    // of 1.27x and an answer already at 3.9e-16. Capped, the ratio is 1.05.
    auto capped = [&](double rate) { return std::min(rate, d.rateCeiling); };

    std::vector<double> interior;
    interior.reserve(cells.size() - 2);
    for (size_t i = 1; i + 1 < cells.size(); ++i)
        interior.push_back(capped(cells[i].decayRate));

    d.interiorMedian = median(std::move(interior));
    d.lowerRatio = roughness(d.interiorMedian, capped(d.lowerRate));
    d.upperRatio = roughness(d.interiorMedian, capped(d.upperRate));

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

GradingDecision gradingDecision(DGSoln const &Y, Index var, double threshold)
{
    return gradingDecision(cellSmoothness(Y, var), Y.getBasis().Order(), threshold);
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

double gradingLayerFraction(GradingDecision const &decision,
                            Grid const &sampling,
                            SolverConfig const &config)
{
    if (decision.verdict == GradingVerdict::Uniform)
        throw std::logic_error(
            "gradingLayerFraction was asked for a layer from a decision that said "
            "not to grade.");

    const bool lower = decision.verdict == GradingVerdict::GradeLower;
    if (lower ? config.LowerBoundaryFractionGiven : config.UpperBoundaryFractionGiven)
        return lower ? config.LowerBoundaryFraction : config.UpperBoundaryFraction;

    const double span = sampling.upperBoundary() - sampling.lowerBoundary();
    const Grid::Index end = lower ? 0 : sampling.getNCells() - 1;
    return sampling[end].h() / span;
}

namespace
{
// A solver for one solve at a fixed degree on a given mesh, configured but not
// yet run, around whichever instance of the case may be evaluated there. Split
// out so that the sampling solve, the retry loop and the fallback share it, and
// so the "never two solvers alive" discipline lives in one place: the caller
// destroys the last solver before asking for the next, which is also what lets
// physics replace the case under it.
std::unique_ptr<SystemSolver> configuredSolver(SolverConfig const &config,
                                               PhysicsInstance &physics,
                                               Grid const &grid, unsigned int k)
{
    auto system = physics.solverFor(grid, k, [&](SystemSolver &s)
                                    { applySolverConfig(config, s); });

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

    return system;
}
} // namespace

AdaptiveMeshResult runAdaptiveMesh(SolverConfig const &config,
                                   PhysicsInstance &physics,
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

    // Grading moves the mesh and the degree loop moves k, so a case that can
    // follow neither is refused now rather than after the sampling solve.
    physics.requireAdaptable("MeshAdaptation");

    const Index var = 0;

    std::println("Mesh adaptation: sampling at k = {} on {} uniform cells",
                 k0, uniform.getNCells());

    AdaptiveMeshResult result;
    result.grid = std::make_unique<Grid>(uniform);

    // --- p: the sampling solve, at a degree the decision can be trusted at -----
    auto sample = configuredSolver(config, physics, *result.grid, k0);
    sample->runSolver(tFinal);

    // --- h: decide, and regrade at the same cell count -------------------------
    result.decision = gradingDecision(sample->solution(), var,
                                      config.MeshAdaptationThreshold);
    auto const &d = result.decision;

    std::println("  decay rate: lower end {:.3g}, interior median {:.3g}, upper end "
                 "{:.3g}; measurable up to {:.3g}",
                 d.lowerRate, d.interiorMedian, d.upperRate, d.rateCeiling);
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
        //
        // What the graded solve starts from is taken out of the sample first: its
        // state, the mesh that state lives on, and the pseudo-transient step it
        // finished at. Started cold, the graded solve repeats the whole climb from
        // the initial condition; warm, it is two continuation steps -- 3835
        // physics evaluations against 475 on the wall-layer case of
        // MESH-REFINEMENT.md section 12, for the same answer to 1e-10. The state
        // crosses through the restart path, which L2-projects the element
        // polynomials onto the new mesh and rebuilds the trace there (section 4's
        // transfer, not the spline that section 9 measured failing).
        const std::vector<double> sampleState = sample->stateVector();
        const std::vector<double> sampleDerivative = sample->derivativeVector();
        const Grid sampleGrid = *result.grid;
        const SolverConfig warmConfig = carriedStepConfig(config, *sample);
        sample.reset();

        const double layer = gradingLayerFraction(d, uniform, config);
        std::println("  layer: {:.4g} of the domain{}", layer,
                     (d.verdict == GradingVerdict::GradeLower ? config.LowerBoundaryFractionGiven
                                                              : config.UpperBoundaryFractionGiven)
                         ? "" : ", the sampling mesh's end cell");

        double ratio = config.GradingRatio;
        for (unsigned int attempt = 1; attempt <= config.MeshAdaptationAttempts;
             ++attempt)
        {
            auto points = gradedMeshFor(d, uniform,
                                        static_cast<Grid::Index>(config.GradingCells),
                                        layer, layer, ratio);
            auto graded = std::make_unique<Grid>(points);

            const double span = uniform.upperBoundary() - uniform.lowerBoundary();
            double narrowest = span;
            for (Grid::Index i = 0; i < graded->getNCells(); ++i)
                narrowest = std::min(narrowest, (*graded)[i].h());

            std::println("  attempt {}: ratio {:.4g}, narrowest cell {:.3e} of the "
                         "domain", attempt, ratio, narrowest / span);

            // Built -- around a rebuilt case, if RebuildPhysicsOnRegrid calls for one --
            // outside the attempt's own failure handling below. A case that cannot
            // follow the mesh has not failed to solve on it, and softening the
            // grading would only bury that. The warm start goes on whichever
            // instance the solver was built around, so after it.
            std::unique_ptr<SystemSolver> trial =
                configuredSolver(warmConfig, physics, *graded, k0);
            physics.problem().setRestartValues(sampleState, sampleDerivative, sampleGrid, k0);

            try
            {
                try
                {
                    trial->runSolver(tFinal);
                    physics.problem().clearRestart();
                }
                catch (std::invalid_argument const &)
                {
                    physics.problem().clearRestart();
                    throw;
                }
                catch (std::exception const &e)
                {
                    // The warm start is a cost saving, so its failure is not
                    // evidence against the mesh: retry this mesh cold, exactly as
                    // it used to be solved, before softening anything. The failed
                    // solver goes first, so two are never alive at once.
                    physics.problem().clearRestart();
                    logmsg<LOG_LEVEL::WARNING>(
                        "Graded mesh attempt {} failed from the sample's state ({}). "
                        "Retrying it from the initial condition.", attempt, e.what());
                    std::println("  attempt {}: warm start failed; retrying cold", attempt);
                    trial.reset();
                    trial = configuredSolver(config, physics, *graded, k0);
                    trial->runSolver(tFinal);
                }
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

            // The degree loop's first level, solved here so that it is this
            // driver, which has just moved the case off the uniform mesh, that
            // brings it back.
            result.solver = configuredSolver(config, physics, *result.grid, k0);
            result.solver->runSolver(tFinal);
        }
    }
    else
    {
        result.solver = std::move(sample);
    }

    // --- p: the degree loop, on whichever mesh won -----------------------------
    //
    // Handed the mesh rather than the config, so it never consults the grading
    // keys and cannot rebuild a different one. And handed the solve already made
    // on that mesh at k0 -- the sample, or the graded trial that converged -- as
    // its first level, because that is exactly the solve the loop would open with.
    // It used to be discarded here and repeated from cold: one solve in three on
    // every graded run, and one in two on every uniform one, for an identical
    // answer. When every graded attempt failed, that is the uniform solve made
    // just above.
    std::println("Mesh adaptation: adapting the degree on the {} mesh",
                 result.gradingAttempts > 0 && result.decision.verdict != GradingVerdict::Uniform
                     ? "graded" : "uniform");
    result.solver = runAdaptiveDegree(config, physics, *result.grid, k0,
                                      tFinal, std::move(result.solver));
    return result;
}
