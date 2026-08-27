#include "DegreeAdaptation.hpp"

#include "AdjointProblem.hpp"
#include "Logging.hpp"
#include "Postprocessing.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"
#include "gridStructures.hpp"

#include <algorithm>
#include <cmath>
#include <print>
#include <stdexcept>
#include <vector>

unsigned int degreeIncrement(double E, double eps, double base)
{
    if (!(base > 1.0))
        throw std::invalid_argument(
            "The degree-adaptation base must exceed 1: at 1 the logarithm is "
            "undefined, and below it a larger error would ask for a smaller "
            "degree.");

    if (!(eps > 0.0))
        throw std::invalid_argument("The degree-adaptation tolerance must be positive.");

    // Before the stopping test below, and that ordering is the whole point: a
    // NaN compares false against everything, so `!(E > eps)` is *true* for one
    // and this returned 0 -- reporting a solve that produced garbage as
    // converged, which is the worst answer available. An infinity would have
    // fallen through instead, to a ceil() of infinity and an undefined
    // conversion to unsigned.
    //
    // Neither is an under-resolved solution, and no number of extra degrees
    // fixes either; one step is returned so the caller's ceiling still bounds
    // the loop and the run ends with a warning rather than a false success.
    if (!std::isfinite(E))
        return 1;

    // Already there. Not an error and not a bump -- the loop's stopping test.
    if (!(E > eps))
        return 0;

    const double bump = std::ceil(std::log(E / eps) / std::log(base));

    // ceil of a positive logarithm is at least 1 in exact arithmetic; the max
    // is against the case where E/eps is so close to 1 that the ratio rounds to
    // it, which would return 0 and stall a loop that has not met its target.
    return static_cast<unsigned int>(std::max(1.0, bump));
}

namespace
{
// The error estimate that drives the loop, over every variable.
//
// Relative, per variable, to its own L2 norm. That is what lets one tolerance
// mean something across variables carrying different units -- an absolute
// figure would be dominated by whichever variable happens to be large. The
// control quantity is then the worst variable, because one global degree has to
// serve all of them.
struct LevelError
{
    double relative = 0.0; // max over variables of globalL2 / scale
    double globalL2 = 0.0; // of the worst variable, unnormalised
    double worstCell = 0.0;
    Index worstVar = 0;
    Index worstCellIndex = 0;
};

// The scale to divide by: mixed, not purely relative.
//
// A purely relative measure divides by ||u_h||, and that is a division by zero
// waiting to happen -- not hypothetically. LinearDiffusion has zero Dirichlet
// data at both ends and no source, so its exact steady state is u = 0, and the
// solver duly returns it: ||u* - u_h|| came out at 1.6e-16 and ||u_h|| at
// 2.6e-15, both pure round-off, and their ratio was a meaningless 6.2e-2. The
// loop then climbed to the degree ceiling burning a solve per level, on a
// problem it had solved exactly at k = 1. A `> 0.0` guard does not catch it,
// because 2.6e-15 is greater than zero.
//
// Absolute_tolerance is the right floor and costs no new configuration key: it
// is per variable, and it already means exactly "values of u below this do not
// matter" -- the user has had to choose it for this very problem. Adding it
// rather than switching on it keeps the measure smooth, and on any problem with
// a real solution ||u_h|| dominates it and the measure is the relative one
// intended (NonlinDiffTest: 1.510e-2 before, 1.508e-2 after).
//
// Units: ||u_h||_L2 is [u] * [length]^(1/2), so the tolerance is scaled by the
// square root of the domain length to match.
double errorScale(AccuracyEstimate const &e, SolverConfig const &config,
                  Grid const &grid, Index var)
{
    // One value, or one per variable.
    const auto &atol = config.Absolute_tolerance;
    const double a = atol.empty()
                         ? 0.0
                         : atol[std::min<size_t>(var, atol.size() - 1)];

    const double L = grid.upperBoundary() - grid.lowerBoundary();
    return e.solutionL2 + a * std::sqrt(L);
}

LevelError measure(SystemSolver &system, SolverConfig const &config,
                   Grid const &grid, Index nVars)
{
    LevelError out;

    for (Index var = 0; var < nVars; ++var)
    {
        const AccuracyEstimate e = system.accuracyEstimate(var);

        const double scale = errorScale(e, config, grid, var);

        // Both numerator and denominator vanish only for a variable that is
        // identically zero *and* has a zero tolerance, which is a request not to
        // measure it. Resolved, rather than NaN.
        const double relative = scale > 0.0 ? e.globalL2 / scale : 0.0;

        if (relative >= out.relative)
        {
            Index cell = 0;
            if (e.perCell.size() > 0)
                e.perCell.maxCoeff(&cell);

            out.relative = relative;
            out.globalL2 = e.globalL2;
            out.worstCell = e.worstCell;
            out.worstVar = var;
            out.worstCellIndex = cell;
        }
    }

    return out;
}
} // namespace

std::unique_ptr<SystemSolver> runAdaptiveDegree(SolverConfig const &config,
                                                TransportSystem &problem,
                                                AdjointProblem *adjoint,
                                                Grid const &grid,
                                                unsigned int k0,
                                                double tFinal)
{
    // Only Python can arm spatial adjoint parameters, so this cannot be caught
    // in loadSolverConfig with the rest. The objection is the one that already
    // makes Superconvergent throw at SystemSolver.cpp: those parameters are
    // indexed by node, so G_p is (ng * nCells * (k+1), np) and changing the
    // degree redefines how many parameters there are. Better to refuse than to
    // return a gradient of the wrong length.
    if (adjoint != nullptr && adjoint->areParametersSpatial())
        throw std::invalid_argument(
            "DegreeAdaptation cannot be used with spatial adjoint parameters: "
            "the parameter vector is indexed by node, so changing the "
            "polynomial degree changes how many parameters there are.");

    const Index nVars = problem.getNumVars();
    const unsigned int kMax = config.MaxPolynomialDegree;
    const double eps = config.DegreeTolerance;

    if (k0 > kMax)
        throw std::invalid_argument(
            "Polynomial_degree already exceeds MaxPolynomialDegree, so degree "
            "adaptation has nothing it is allowed to do.");

    std::println("Degree adaptation: starting at k = {}, ceiling {}, relative "
                 "tolerance {:g}, base {:g}",
                 k0, kMax, eps, config.DegreeAdaptationBase);

    std::unique_ptr<SystemSolver> system;
    unsigned int k = k0;

    // What the whole adaptive run cost, as against what one level did. This is
    // the only construct in the tree where a run contains more than one steady
    // solve, so it is the only place where "the run's total" and
    // SystemSolver::lastSteadyStats() are different numbers -- and it is the
    // total the benchmarks quote, since the point of adapting the degree is that
    // the coarse levels are cheap enough to be worth paying for. A solver is
    // destroyed at the end of each level, so it has to be read before then.
    SystemSolver::SteadyStats runTotal;
    int levels = 0;

    for (int level = 0;; ++level)
    {
        system = std::make_unique<SystemSolver>(grid, k, &problem);
        applySolverConfig(config, *system);

        // Checked here, against the solver, because this is the point of truth
        // and a proxy for it is what let a transient through: loadSolverConfig
        // refuses SteadyStateSolver = "TimeMarch", but that key defaults to
        // "PseudoTransient" and the mode is only consulted once termination is
        // *armed*. A configuration that simply never set SteadyStateTolerance
        // therefore passed validation and then time-marched every level.
        //
        // That is not a scope question, it is a wrong answer. Each level would
        // take the previous one's state at t_final as its initial condition at
        // t_initial and integrate the same interval again -- so the run has
        // evolved twice. Measured on NonlinDiffTest at k = 4: u(0.9) came out
        // 0.4048 against a plain fixed-degree run's 0.3767, 7.5% apart, where
        // two runs at the same degree should agree to discretisation error.
        if (level == 0 && !system->solvesForSteadyState())
            throw std::invalid_argument(
                "DegreeAdaptation needs a steady solve, but this configuration "
                "time-marches: SteadyStateTolerance is absent, so steady-state "
                "termination is never armed and SteadyStateSolver is not "
                "consulted. Set SteadyStateTolerance, or call run_ss().");

        // A fresh solver has no adjoint problem. Forgetting this is silent: the
        // run completes and the gradients are simply never computed.
        if (adjoint != nullptr)
            system->setAdjointProblem(adjoint);

        system->runSolver(tFinal);

        // Read now: `system` is reset at the bottom of the loop, and a level
        // that breaks out leaves the last one alive but the earlier ones gone.
        {
            const auto levelStats = system->lastSteadyStats();
            runTotal.steps += levelStats.steps;
            runTotal.rejected += levelStats.rejected;
            runTotal.newtonIters += levelStats.newtonIters;
            runTotal.kinFuncEvals += levelStats.kinFuncEvals;
            runTotal.kinJacEvals += levelStats.kinJacEvals;
            runTotal.residualEvals += levelStats.residualEvals;
            runTotal.jacBuilds += levelStats.jacBuilds;
            runTotal.jacSolves += levelStats.jacSolves;
            ++levels;
        }

        // runSolver has already freed the SUNDIALS state; accuracyEstimate
        // reads yJac, which owns its own memory, and the postprocessor, which is
        // a member -- so both outlive it.
        const LevelError err = measure(*system, config, grid, nVars);

        // Both aggregates, every level. globalL2 drives the decision because it
        // is the quantity the benchmarks quote, so the tolerance means the same
        // thing as the numbers they report; worstCell is the binding constraint
        // on a *single* global degree and is the one to watch if the loop stops
        // while some corner of the domain is plainly unresolved.
        std::println("  k = {}: relative L2 error {:.3e} (variable {}), "
                     "absolute {:.3e}, worst cell {:.3e} at cell {}",
                     k, err.relative, err.worstVar, err.globalL2, err.worstCell,
                     err.worstCellIndex);

        const unsigned int bump = degreeIncrement(err.relative, eps,
                                                  config.DegreeAdaptationBase);

        if (bump == 0)
        {
            std::println("  converged at k = {} after {} solve{}", k, level + 1,
                         level == 0 ? "" : "s");
            break;
        }

        if (k >= kMax)
        {
            // Deliberately not an exception. The answer at kMax is the best one
            // available and the caller can still have it; what must not happen
            // is that the run reports success at a tolerance it never reached.
            logmsg<LOG_LEVEL::WARNING>(
                "Degree adaptation stopped at the ceiling MaxPolynomialDegree = {} "
                "with a relative L2 error of {:g} against a tolerance of {:g}. The "
                "result is the best this degree can do, not a converged one.",
                kMax, err.relative, eps);
            std::println("  stopped at the ceiling k = {}, tolerance not met", kMax);
            break;
        }

        // Capped, then clamped to the ceiling. Giorgiani's rule is free to ask
        // for a very large jump from a coarse first solve -- k = 1 with a 1e-8
        // target asks for 7 at base 10, which on NonlinDiffTest cleared a
        // ceiling of 8 in one step and learned nothing between the two. Taking
        // it in bounded steps costs solves but each one reports where it got to,
        // which is what makes a run diagnosable.
        const unsigned int capped = std::min(bump, config.MaxDegreeIncrement);
        const unsigned int next = std::min(k + capped, kMax);

        if (capped < bump)
            std::println("  raising k from {} to {} (the rule asked for +{}, capped at +{})",
                         k, next, bump, config.MaxDegreeIncrement);
        else
            std::println("  raising k from {} to {}", k, next);

        // Hand the state on. setRestartValues copies both the vector and the
        // Grid, so nothing here points into the solver about to be destroyed --
        // and the destruction has to come before the next construction, or the
        // two solvers thrash Integrator's cache between them.
        problem.setRestartValues(system->stateVector(), system->derivativeVector(),
                                 grid, k);

        system.reset();
        k = next;
    }

    // `restarting` is sticky, and the transfer above set it. Left armed, the
    // next run on this same configuration would resume from the second-to-last
    // level instead of building an initial condition -- which is the defect
    // test_reconfiguring_without_restart_clears_the_restart_state exists for,
    // one level up.
    problem.clearRestart();

    // Printed unconditionally when the diagnostics are on, even at one level,
    // where it duplicates that level's own block. The duplication is the lesser
    // evil: a log whose shape depends on how many levels the run happened to
    // take is one nothing can read mechanically, and the label says how many
    // solves went into it either way.
    if (config.SteadyStateDiagnostics && levels > 0)
    {
        std::println("Degree adaptation totals -- {} level{}, one steady solve each",
                     levels, levels == 1 ? "" : "s");
        std::println("  continuation steps      : {}  ({} rejected)",
                     runTotal.steps, runTotal.rejected);
        std::println("  KINSOL Newton iterations: {}", runTotal.newtonIters);
        std::println("  residual evaluations    : {}  (of which KINSOL: {})",
                     runTotal.residualEvals, runTotal.kinFuncEvals);
        std::println("  Jacobian builds         : {}  (KINSOL asked for {})",
                     runTotal.jacBuilds, runTotal.kinJacEvals);
        std::println("  Jacobian solves         : {}", runTotal.jacSolves);
    }

    return system;
}
