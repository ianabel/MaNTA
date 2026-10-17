// SteadyStateTest = Objective: a steady solve that stops when the objectives
// are converged, rather than the whole solution.
//
// docs/superpowers/specs/2026-10-17-objective-error-control-design.md is the
// design. The test is NewtonStep's continuation with another convergence test in
// its undamped solve: |z^T F| within the algebraic share of each objective's
// tolerance, z the adjoint state at the iterate. What these pin:
//
//  * that |z^T F| is the objective's Newton correction -- the number
//    estimateObjective forms from a forward solve, here from the adjoint;
//  * that the test fires at its threshold, on either side of it, and that the
//    error in G it leaves is inside the share it was given;
//  * the cost: Newton's sweeps and no others, and one adjoint solve per
//    objective per undamped iteration -- none of which asks the case for
//    anything;
//  * that a sliced solve resumes exactly;
//  * that a tolerance below what the arithmetic resolves ends Stalled, said and
//    returned, rather than as a claimed convergence or an exhausted budget;
//  * the refusals.

#include <boost/test/unit_test.hpp>

#include "AdjointProblem.hpp"
#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"
#include "Types.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace
{

constexpr Index k = 2, nCells = 6;

// -d_x( (1 + u^2) d_x u ) = 1 on [0, 1], u(0) = 1, u(1) = 0, from u = 1 - x:
// nonlinear enough that Newton takes several iterations from the initial
// condition. Counts its flux and derivative calls, per point.
class CountingNonlinearDiffusion : public TransportSystem
{
public:
    CountingNonlinearDiffusion() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 1.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        ++calls;
        return (1.0 + s.u(0) * s.u(0)) * s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return 1.0; }

    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        ++derivCalls;
        v[0] = 1.0 + s.u(0) * s.u(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.u(0) * s.q(0);
    }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index, Position x) const override { return 1.0 - x; }
    Value InitialDerivative(Index, Position) const override { return -1.0; }

    long calls = 0;
    long derivCalls = 0;
};

// G_0 = int u dx, a stored-energy-like objective, and G_1 = int u^2 dx, as the
// interpolatory quadrature GFn has to report for dG/dy to be its exact
// derivative.
class TwoObjectives : public AdjointProblem
{
public:
    TwoObjectives()
    {
        ng = 2;
        np = 1;
    }

    using AdjointProblem::GFn;

    Value GFn(Index g, DGSoln &y) const override
    {
        Value G = 0.0;
        for (Index i = 0; i < static_cast<Index>(y.getGrid().getNCells()); ++i)
        {
            const Vector w = y.getBasis().getIntegrationWeights(y.getGrid()[i]);
            const auto &u = y.u(0).getCoeff(i).second;
            for (Index j = 0; j < w.size(); ++j)
                G += w(j) * (g == 0 ? u(j) : u(j) * u(j));
        }
        return G;
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }
    Value gFn(Index g, const State &s, Position) const override
    {
        return g == 0 ? s.u(0) : s.u(0) * s.u(0);
    }
    void dgFn_du(Index g, VectorRef v, const State &s, Position) override
    {
        v[0] = g == 0 ? 1.0 : 2.0 * s.u(0);
    }
    void dgFn_dq(Index, VectorRef, const State &, Position) override {}
    void dgFn_dsigma(Index, VectorRef, const State &, Position) override {}
    void dgFn_dphi(Index, VectorRef, const State &, Position) override {}
    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
};

struct Options
{
    SystemSolver::SteadyTest test = SystemSolver::SteadyTest::Objective;
    SystemSolver::SteadyMode mode = SystemSolver::SteadyMode::Newton;
    std::vector<double> relative{1e-3};
    std::vector<double> absolute{0.0};
    double stepTolerance = 0.33;
    bool superconvergent = false;
    bool objective = true;      // attach the adjoint problem
    unsigned maxContinuationSteps = 200;
    unsigned newtonMaxIterations = 20;
};

struct Answer
{
    SystemSolver::SteadyOutcome outcome = SystemSolver::SteadyOutcome::NotRun;
    SystemSolver::SteadyStats stats;
    std::vector<SystemSolver::SteadyStepStats> steps;
    double G[2] = {0.0, 0.0};
    Vector u;
    long calls = 0, derivCalls = 0;
};

// A solver configured the way every test here wants it; destroyed with its
// owner, SUNDIALS objects and all.
struct Run
{
    Grid grid{0.0, 1.0, nCells};
    CountingNonlinearDiffusion problem;
    TwoObjectives objective;
    std::unique_ptr<SystemSolver> sys;

    explicit Run(Options const &o, std::string const &stem)
    {
        sys = std::make_unique<SystemSolver>(grid, k, &problem);
        sys->setTau(1.0);
        sys->resetCoeffs();
        sys->setInputFile(stem);
        sys->setOutputCadence(1.0);
        sys->setNOutput(11);
        sys->setInitialTime(0.0);
        sys->setMinStepSize(1e-12);
        sys->setTolerances({1e-8}, 1e-6);
        sys->setWriteOutput(false);
        sys->setEstimateObjectiveOnFinish(false);
        sys->setSuperconvergent(o.superconvergent);
        sys->setSteadyMode(o.mode);
        sys->setPseudoTransientInitialStep(1e-2);
        sys->setSteadyStateTolerance(1e-3); // arms the solve; unread by these tests
        sys->setSteadyTest(o.test);
        sys->setSteadyStateStepTolerance(o.stepTolerance);
        sys->setObjectiveTolerance(o.relative, o.absolute);
        sys->setMaxContinuationSteps(o.maxContinuationSteps);
        sys->setNewtonMaxIterations(o.newtonMaxIterations);
        if (o.objective)
            sys->setAdjointProblem(&objective);
    }

    ~Run()
    {
        CapturedOutput quiet;
        sys->destroySundials();
    }

    double G(Index g)
    {
        if (sys->isSuperconvergent())
        {
            sys->postprocessor->computeUStar(sys->yJac);
            return objective.GFn(g, sys->yJac, *sys->postprocessor);
        }
        return objective.GFn(g, sys->yJac);
    }

    Answer answer()
    {
        Answer a;
        a.outcome = sys->lastSteadyOutcome();
        a.stats = sys->lastSteadyStats();
        a.steps = sys->lastSteadyStepStats();
        sys->captureState();
        a.G[0] = G(0);
        a.G[1] = G(1);
        a.u.resize(9);
        for (Index i = 0; i < 9; ++i)
            a.u(i) = sys->yJac.u(0)(0.1 * (i + 1));
        a.calls = problem.calls;
        a.derivCalls = problem.derivCalls;
        return a;
    }
};

void removeOutput(std::string const &stem)
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);
}

Answer solve(Options const &o, std::string const &stem)
{
    Run run(o, stem);
    {
        CapturedOutput quiet;
        run.sys->initialize();
        run.sys->solveSteadyState();
    }
    Answer a = run.answer();
    removeOutput(stem);
    return a;
}

// The discrete fixed point, to round-off: NewtonStep at a millionth of IDA's
// corrector constant, rtol 1e-6 -- a correction of 1e-12 relative.
Answer reference(Options o, std::string const &stem)
{
    o.test = SystemSolver::SteadyTest::NewtonStep;
    o.mode = SystemSolver::SteadyMode::Newton;
    o.stepTolerance = 1e-6;
    return solve(o, stem);
}

} // namespace

BOOST_AUTO_TEST_SUITE(objective_steady_tests)

BOOST_AUTO_TEST_CASE(the_estimate_is_the_objectives_newton_correction)
{
    // z^T F, from the adjoint at the iterate, and (dG/dy) . J^-1 F, which
    // estimateObjective forms from a forward solve, are one number: J^T z =
    // dG/dy. One undamped iteration from the initial condition, which is not
    // enough to pass, and one continuation step, so the solve stops with Y
    // still at the initial condition and the estimate recorded there. Every
    // block of z -- cells, traces -- enters the dot product, so a layout error
    // shows here.
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            Run run({.relative = {1e-12}, .superconvergent = superconvergent,
                     .maxContinuationSteps = 1, .newtonMaxIterations = 1},
                    "objective_estimate_identity");
            CapturedOutput quiet;
            run.sys->initialize();
            BOOST_CHECK_THROW(run.sys->solveSteadyState(), std::runtime_error);
            const auto stats = run.sys->lastSteadyStats();
            BOOST_REQUIRE(stats.objectiveError.size() == 2);
            BOOST_TEST(stats.adjointSolves == 2);

            const auto e = run.sys->estimateObjective();
            BOOST_REQUIRE(e.valid);
            for (Index g = 0; g < 2; ++g)
            {
                const double forward = std::abs(e.value(g) - e.corrected(g));
                BOOST_TEST_MESSAGE(std::format("G_{}: |z^T F| {:.15e}, |dG/dy . J^-1 F| {:.15e}",
                                               g, stats.objectiveError(g), forward));
                BOOST_TEST(forward > 1e-3); // far from the fixed point: not vacuous
                BOOST_TEST(std::abs(stats.objectiveError(g) - forward) <= 1e-10 * forward);
            }
            removeOutput("objective_estimate_identity");
        }
    }
}

BOOST_AUTO_TEST_CASE(the_test_fires_at_its_threshold)
{
    // Newton's iterates do not depend on the tolerance, only where they stop
    // does, so the stop can be walked down the sequence: a solve that stopped at
    // iteration n with ratio r = |z^T F| / tolerance stops there again at a
    // tolerance one percent looser than r times its own, and goes on past it at
    // one percent tighter -- for every objective's ratio, the larger being the
    // one tested. Three rounds take it from 10% to the iteration before
    // round-off; past that a tighter tolerance stalls, which is the test below.
    double rel = 1e-1;
    for (int round = 0; round < 3; ++round)
    {
        const Answer a = solve({.relative = {rel}}, "objective_threshold");
        BOOST_REQUIRE((a.outcome == SystemSolver::SteadyOutcome::Converged));
        const double r = a.stats.objectiveTest;
        BOOST_TEST_MESSAGE(std::format("rel {:.3e}: {} iterations, max |z^T F|/tol = {:.3e}",
                                       rel, a.stats.newtonIters, r));
        BOOST_REQUIRE(r <= 1.0);
        BOOST_REQUIRE(r < 0.99);

        const Answer looser = solve({.relative = {rel * r * 1.01}}, "objective_threshold");
        BOOST_TEST(looser.stats.newtonIters == a.stats.newtonIters);
        const Answer tighter = solve({.relative = {rel * r * 0.99}}, "objective_threshold");
        BOOST_TEST((tighter.outcome == SystemSolver::SteadyOutcome::Converged));
        BOOST_TEST(tighter.stats.newtonIters > a.stats.newtonIters);
        BOOST_TEST(tighter.stats.objectiveTest <= 1.0);

        rel *= r * 0.99;
    }
}

BOOST_AUTO_TEST_CASE(the_error_in_G_is_inside_its_share)
{
    // Against the discrete fixed point, at every tolerance and from either
    // mode, for both objectives: the estimate is taken at the iterate the last
    // correction starts from and the correction is then applied, so the error
    // left is the next one -- quadratically smaller under full Newton. And one
    // value per objective is honoured as given: the second objective's
    // tolerance a hundred times the first's.
    const Answer ref = reference({}, "objective_share_reference");
    BOOST_REQUIRE((ref.outcome == SystemSolver::SteadyOutcome::Converged));

    for (auto mode : {SystemSolver::SteadyMode::Newton, SystemSolver::SteadyMode::PseudoTransient})
        for (double rel : {1e-2, 1e-4, 1e-6})
        {
            const Answer a = solve({.mode = mode, .relative = {rel, 100 * rel}}, "objective_share");
            BOOST_TEST_CONTEXT(std::format("{}, ObjectiveTolerance [{:g}, {:g}]",
                                           mode == SystemSolver::SteadyMode::Newton ? "Newton" : "PseudoTransient",
                                           rel, 100 * rel))
            {
                BOOST_REQUIRE((a.outcome == SystemSolver::SteadyOutcome::Converged));
                for (Index g = 0; g < 2; ++g)
                {
                    const double share = 0.5 * (g == 0 ? rel : 100 * rel);
                    const double err = std::abs(a.G[g] - ref.G[g]) / std::abs(ref.G[g]);
                    BOOST_TEST_MESSAGE(std::format("  G_{}: relative error {:.2e} against a share of {:.0e}, "
                                                   "estimate {:.2e} against {:.2e}",
                                                   g, err, share, a.stats.objectiveError(g),
                                                   a.stats.objectiveTolerance(g)));
                    BOOST_TEST(err <= share);
                    BOOST_TEST(a.stats.objectiveError(g) <= a.stats.objectiveTolerance(g));
                }
            }
        }
}

BOOST_AUTO_TEST_CASE(an_objective_solve_spends_newtons_sweeps_and_one_adjoint_solve_per_objective_per_iteration)
{
    // Newton mode, full Newton: one flux sweep builds sigma from the initial
    // condition, then each iteration evaluates the residual at its iterate and
    // builds the Jacobian there -- 1 + n flux sweeps and n derivative sweeps,
    // NewtonStep's budget exactly. The adjoint solve at each iteration, one per
    // objective, adds to neither: it is assembled from the build that
    // iteration already made.
    Run run({.relative = {1e-8}}, "objective_sweeps");
    {
        CapturedOutput quiet;
        run.sys->initialize();
        run.sys->solveSteadyState();
    }
    const Answer a = run.answer();
    const long nodes = nCells * (k + 1);
    BOOST_TEST(a.calls % nodes == 0);
    BOOST_TEST(a.derivCalls % nodes == 0);
    BOOST_TEST_MESSAGE(std::format("{} Newton iterations: {} flux sweeps, {} derivative sweeps, "
                                   "{} adjoint solves",
                                   a.stats.newtonIters, a.calls / nodes, a.derivCalls / nodes,
                                   a.stats.adjointSolves));
    BOOST_TEST((a.outcome == SystemSolver::SteadyOutcome::Converged));
    BOOST_TEST(a.stats.steps == 1);
    BOOST_TEST(a.stats.newtonIters > 2);
    BOOST_TEST(a.calls / nodes == 1 + a.stats.newtonIters);
    BOOST_TEST(a.derivCalls / nodes == a.stats.newtonIters);
    BOOST_TEST(a.stats.residualEvals == a.stats.newtonIters);
    BOOST_TEST(a.stats.jacBuilds == a.stats.newtonIters);
    BOOST_TEST(a.stats.adjointSolves == 2 * a.stats.newtonIters);
    removeOutput("objective_sweeps");

    // Under continuation the damped steps make none: one iteration each, judged
    // by nothing. Only the undamped solve pays, per iteration.
    const Answer p = solve({.mode = SystemSolver::SteadyMode::PseudoTransient}, "objective_sweeps");
    BOOST_REQUIRE((p.outcome == SystemSolver::SteadyOutcome::Converged));
    long undampedIterations = 0;
    for (auto const &s : p.steps)
    {
        BOOST_TEST(s.adjointSolves == (s.undamped ? 2 * s.newtonIters : 0));
        if (s.undamped)
            undampedIterations += s.newtonIters;
    }
    BOOST_TEST(p.stats.steps > 1);
    BOOST_TEST(p.stats.adjointSolves == 2 * undampedIterations);
}

BOOST_AUTO_TEST_CASE(newton_step_and_the_objective_test_take_the_same_continuation)
{
    // Objective changes only the test the undamped solve stops on, so the
    // damped steps of a continuation are NewtonStep's to the bit, and so is the
    // state each undamped solve starts from.
    const Answer ns = solve({.test = SystemSolver::SteadyTest::NewtonStep,
                             .mode = SystemSolver::SteadyMode::PseudoTransient},
                            "objective_same_continuation");
    const Answer ob = solve({.mode = SystemSolver::SteadyMode::PseudoTransient, .relative = {1e-9}},
                            "objective_same_continuation");
    BOOST_REQUIRE(!ns.steps.empty());
    BOOST_REQUIRE(!ob.steps.empty());
    std::size_t damped = 0;
    for (; damped < ns.steps.size() && !ns.steps[damped].undamped; ++damped)
    {
        BOOST_REQUIRE(damped < ob.steps.size());
        BOOST_TEST(!ob.steps[damped].undamped);
        BOOST_TEST(ob.steps[damped].dt == ns.steps[damped].dt);
        BOOST_TEST(ob.steps[damped].residualNorm == ns.steps[damped].residualNorm);
        BOOST_TEST(ob.steps[damped].firstCorrection == ns.steps[damped].firstCorrection);
    }
    BOOST_TEST(damped > 0);
    BOOST_REQUIRE(damped < ob.steps.size());
    BOOST_TEST(ob.steps[damped].residualNorm == ns.steps[damped].residualNorm);
}

BOOST_AUTO_TEST_CASE(a_sliced_objective_solve_resumes_exactly)
{
    // The objective test carries nothing from one step to the next -- its
    // estimate lives and dies inside one undamped solve -- so a solve cut into
    // slices of two steps takes NewtonStep's route through the slices: the same
    // steps as an uninterrupted one, and the same answer to the bit.
    constexpr int SLICE = 2;
    auto run = [&](int slice)
    {
        Run r({.mode = SystemSolver::SteadyMode::PseudoTransient, .relative = {1e-6}},
              "objective_resume");
        r.sys->setPseudoTransientInitialStep(1e-4);
        if (slice > 0)
            r.sys->setMaxContinuationSteps(slice);

        int total = 0;
        long adjointSolves = 0;
        CapturedOutput quiet;
        r.sys->initialize();
        for (int slices = 0; slices < 200; ++slices)
        {
            try
            {
                if (slices > 0)
                    r.sys->continueSteadyState();
                else
                    r.sys->solveSteadyState();
            }
            catch (std::exception const &)
            {
                // Out of steps for this slice.
            }
            total += r.sys->lastSteadyStats().steps;
            adjointSolves += r.sys->lastSteadyStats().adjointSolves;
            if (r.sys->lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged)
                break;
        }
        const bool converged = r.sys->lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged;
        const Answer a = r.answer();
        return std::tuple{converged, total, adjointSolves, a.u};
    };

    const auto [wholeOK, wholeSteps, wholeAdjoint, wholeU] = run(0);
    const auto [slicedOK, slicedSteps, slicedAdjoint, slicedU] = run(SLICE);
    BOOST_TEST_MESSAGE(std::format("uninterrupted {} steps, sliced {} steps", wholeSteps, slicedSteps));
    BOOST_REQUIRE(wholeOK);
    BOOST_REQUIRE(wholeSteps > 2 * SLICE);
    BOOST_TEST(slicedOK);
    BOOST_TEST(slicedSteps == wholeSteps);
    BOOST_TEST(slicedAdjoint == wholeAdjoint);
    for (Index i = 0; i < wholeU.size(); ++i)
        BOOST_TEST(slicedU(i) == wholeU(i));
    removeOutput("objective_resume");
}

BOOST_AUTO_TEST_CASE(a_tolerance_below_round_off_stalls_and_says_so)
{
    // A tolerance no arithmetic resolves: the iteration converges to the fixed
    // point, the estimate stops falling at round-off above it, and the solve
    // returns Stalled -- not Converged, and not a budget spent iterating to the
    // cap -- with the state at the fixed point and the shortfall reported.
    const Answer ref = reference({}, "objective_stall_reference");
    const Answer a = solve({.relative = {1e-30}}, "objective_stall");
    BOOST_TEST_MESSAGE(std::format("{} iterations, max |z^T F|/tol = {:.2e}",
                                   a.stats.newtonIters, a.stats.objectiveTest));
    BOOST_TEST((a.outcome == SystemSolver::SteadyOutcome::Stalled));
    BOOST_TEST(a.stats.steps == 1);
    BOOST_TEST(a.stats.objectiveTest > 1.0);
    BOOST_TEST(a.stats.objectiveError.size() == 2);
    BOOST_TEST((a.u - ref.u).cwiseAbs().maxCoeff() < 1e-12);
}

BOOST_AUTO_TEST_CASE(the_solver_refuses_an_objective_test_it_cannot_make)
{
    auto refused = [](Options const &o, std::string const &needle)
    {
        Run run(o, "objective_refused");
        CapturedOutput quiet;
        run.sys->initialize();
        try
        {
            run.sys->solveSteadyState();
        }
        catch (std::invalid_argument const &e)
        {
            BOOST_TEST_MESSAGE(e.what());
            return std::string(e.what()).find(needle) != std::string::npos;
        }
        return false;
    };

    // No objective attached.
    BOOST_TEST(refused({.objective = false}, "solveAdjoint"));
    // Two objectives, three tolerances.
    BOOST_TEST(refused({.relative = {1e-3, 1e-3, 1e-3}}, "one each"));
    BOOST_TEST(refused({.absolute = {0.0, 1e-3, 1e-3}}, "one each"));
    // A tolerance of nothing.
    BOOST_TEST(refused({.relative = {1e-3, 0.0}, .absolute = {0.0}}, "neither"));

    // And the setters' own checks.
    Run run({}, "objective_refused");
    BOOST_CHECK_THROW(run.sys->setObjectiveTolerance({}, {0.0}), std::logic_error);
    BOOST_CHECK_THROW(run.sys->setObjectiveTolerance({-1.0}, {0.0}), std::logic_error);
    BOOST_CHECK_THROW(run.sys->setObjectiveBudget(0.5, 0.6), std::logic_error);
    BOOST_CHECK_THROW(run.sys->setObjectiveBudget(-0.5, 1.5), std::logic_error);
    removeOutput("objective_refused");
}

BOOST_AUTO_TEST_SUITE_END()
