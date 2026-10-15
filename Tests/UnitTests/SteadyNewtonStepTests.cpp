// SteadyStateTest = NewtonStep: a steady solve that stops on IDA's corrector
// test applied to the undamped Newton correction, rather than on ||F||.
//
// docs/superpowers/specs/2026-10-11-steady-state-termination-design.md is the
// design. What these pin:
//
//  * the failure that motivated it -- a channel whose residual rows are small
//    converges invisibly under the Residual test -- and that NewtonStep does not
//    have it;
//  * that NewtonStep and a tight Residual solve agree where both converge;
//  * that the test fires at the tolerance it is given, and that tolerance
//    controls the error;
//  * the sweep budget, one flux and one derivative sweep per Newton iteration;
//  * that a sliced solve resumes exactly where it stopped.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <tuple>

namespace
{

constexpr Index k = 2, nCells = 6;

void configure(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(1.0);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(false);
    sys.setEstimateObjectiveOnFinish(false);
}

void removeOutput(std::string const &stem)
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);
}

// Two channels on very different scales. Channel 0 is nonlinear diffusion with
// a unit source, which is what makes continuation take steps. Channel 1 is
// linear diffusion whose flux and source both carry EPS, so its steady state,
// u1 = x(1 - x)/2, is O(1) while every one of its residual rows is O(EPS): the
// shape of T3D's density, whose flux vanishes at steady state.
class TwoScaleDiffusion : public TransportSystem
{
public:
    static constexpr double EPS = 1e-4;

    TwoScaleDiffusion() : TransportSystem({.variables = numberedFields(2)}) {}

    Value LowerBoundary(Index i, Time) const override { return i == 0 ? 1.0 : 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index i, const State &s, Position, Time) override
    {
        return i == 0 ? (1.0 + s.u(0) * s.u(0)) * s.q(0) : EPS * s.q(1);
    }
    Value Sources(Index i, const State &, Position, Time) override
    {
        return i == 0 ? 1.0 : EPS;
    }

    void dSigmaFn_dq(Index i, VectorRef v, const State &s, Position, Time) override
    {
        if (i == 0)
            v[0] = 1.0 + s.u(0) * s.u(0);
        else
            v[1] = EPS;
    }
    void dSigmaFn_du(Index i, VectorRef v, const State &s, Position, Time) override
    {
        if (i == 0)
            v[0] = 2.0 * s.u(0) * s.q(0);
    }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index i, Position x) const override { return i == 0 ? 1.0 - x : 0.0; }
    Value InitialDerivative(Index i, Position) const override { return i == 0 ? -1.0 : 0.0; }

    static double exactU1(double x) { return 0.5 * x * (1.0 - x); }
};

// Nonlinear diffusion that counts its flux and derivative calls, per point.
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

struct Answer
{
    bool converged = false;
    Vector u0, u1;
    SystemSolver::SteadyStats stats;
};

// One steady solve of TwoScaleDiffusion; `setup` chooses the test and mode.
template <typename Setup>
Answer solveTwoScale(std::string const &stem, Setup setup)
{
    Grid grid(0.0, 1.0, nCells);
    TwoScaleDiffusion problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    setup(sys);

    Answer a;
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.solveSteadyState();
        sys.captureState();
    }
    a.converged = sys.lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged;
    a.stats = sys.lastSteadyStats();
    a.u0.resize(9);
    a.u1.resize(9);
    for (Index i = 0; i < 9; ++i)
    {
        const double x = 0.1 * (i + 1);
        a.u0(i) = sys.yJac.u(0)(x);
        a.u1(i) = sys.yJac.u(1)(x);
    }
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput(stem);
    return a;
}

double u1Error(Answer const &a)
{
    double e = 0.0;
    for (Index i = 0; i < a.u1.size(); ++i)
        e = std::max(e, std::abs(a.u1(i) - TwoScaleDiffusion::exactU1(0.1 * (i + 1))));
    return e;
}

} // namespace

BOOST_AUTO_TEST_SUITE(steady_newton_step_tests)

BOOST_AUTO_TEST_CASE(newton_step_converges_the_channel_the_residual_test_cannot_see)
{
    // The failure in the design's "Why": a channel whose residual rows are
    // small hardly enters ||F||, so the Residual test fires on the other
    // channel and returns this one wherever continuation had got to. Its
    // steady state is quadratic, so it lies in the k = 2 space and the only
    // error left is the solver's.
    const Answer residual = solveTwoScale("newton_step_two_scale_residual", [](SystemSolver &sys)
    {
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setPseudoTransientInitialStep(1e-2);
        sys.setSteadyStateTolerance(1e-3);
        sys.setSteadyTest(SystemSolver::SteadyTest::Residual);
    });
    const Answer newtonStep = solveTwoScale("newton_step_two_scale_ns", [](SystemSolver &sys)
    {
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setPseudoTransientInitialStep(1e-2);
        sys.setSteadyStateTolerance(1e-3); // arms the solve; unread under NewtonStep
        sys.setSteadyTest(SystemSolver::SteadyTest::NewtonStep);
    });

    BOOST_TEST_MESSAGE(std::format("u1 error: Residual {:.2e} in {} steps, NewtonStep {:.2e} in {} steps",
                                   u1Error(residual), residual.stats.steps,
                                   u1Error(newtonStep), newtonStep.stats.steps));
    BOOST_TEST(residual.converged);
    BOOST_TEST(newtonStep.converged);

    // Residual stops with the small channel nowhere near its answer...
    BOOST_TEST(u1Error(residual) > 1e-2);
    // ...and NewtonStep converges it to its tolerances: rtol 1e-6 on an O(0.1)
    // answer, atol 1e-8.
    BOOST_TEST(u1Error(newtonStep) < 1e-6);
}

BOOST_AUTO_TEST_CASE(newton_step_agrees_with_a_tight_residual_solve)
{
    // Where both tests converge everything, they converge to one answer, from
    // either mode. The Residual reference is run far tighter than any
    // tolerance here so that it is the fixed point.
    const Answer reference = solveTwoScale("newton_step_reference", [](SystemSolver &sys)
    {
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setPseudoTransientInitialStep(1e-2);
        sys.setSteadyStateTolerance(1e-13);
        sys.setSteadyTest(SystemSolver::SteadyTest::Residual);
    });
    BOOST_REQUIRE(reference.converged);

    for (auto mode : {SystemSolver::SteadyMode::PseudoTransient, SystemSolver::SteadyMode::Newton})
    {
        const Answer a = solveTwoScale("newton_step_agrees", [mode](SystemSolver &sys)
        {
            sys.setSteadyMode(mode);
            sys.setPseudoTransientInitialStep(1e-2);
            sys.setSteadyStateTolerance(1e-3);
            sys.setSteadyTest(SystemSolver::SteadyTest::NewtonStep);
        });
        BOOST_TEST(a.converged);
        const double d0 = (a.u0 - reference.u0).cwiseAbs().maxCoeff();
        const double d1 = (a.u1 - reference.u1).cwiseAbs().maxCoeff();
        BOOST_TEST_MESSAGE(std::format("{}: |du0| {:.2e}, |du1| {:.2e}, {} steps",
                                       mode == SystemSolver::SteadyMode::Newton ? "Newton" : "PseudoTransient",
                                       d0, d1, a.stats.steps));
        BOOST_TEST(d0 < 1e-6);
        BOOST_TEST(d1 < 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(the_test_passes_at_the_tolerance_it_is_given)
{
    // The reported ss * ||d|| is the quantity tested, so on convergence it is
    // at or below the tolerance -- and, on a linearly convergent iteration, not
    // far below it, which is what says the test fired rather than being jumped
    // past. A hundredfold tighter tolerance (IDACalcIC's against the
    // corrector's) then buys a correspondingly smaller error. Measured:
    // ss|d| 0.11 and 6.3e-4, errors 1.5e-8 and 8.2e-11.
    const Answer reference = solveTwoScale("newton_step_tol_reference", [](SystemSolver &sys)
    {
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setPseudoTransientInitialStep(1e-2);
        sys.setSteadyStateTolerance(1e-13);
        sys.setSteadyTest(SystemSolver::SteadyTest::Residual);
    });

    double lastError = std::numeric_limits<double>::infinity();
    for (double tol : {0.33, 0.0033})
    {
        const Answer a = solveTwoScale("newton_step_tol", [tol](SystemSolver &sys)
        {
            // Newton mode: one undamped solve from the initial state, so the
            // test is made on a solve that has a long way to go.
            sys.setSteadyMode(SystemSolver::SteadyMode::Newton);
            sys.setSteadyStateTolerance(1e-3);
            sys.setSteadyTest(SystemSolver::SteadyTest::NewtonStep);
            sys.setSteadyStateStepTolerance(tol);
            // Modified Newton: one Jacobian per solve, so the iteration
            // converges linearly and the test crosses its threshold near the
            // tolerance, rather than jumping past it to round-off as the
            // quadratic convergence of full Newton does.
            sys.setNewtonJacobianReuse(1000);
            sys.setNewtonMaxIterations(200);
        });
        BOOST_REQUIRE(a.converged);
        BOOST_TEST(a.stats.correctionTest <= tol);
        BOOST_TEST(a.stats.correctionTest > 0.1 * tol);
        const double err = std::max((a.u0 - reference.u0).cwiseAbs().maxCoeff(),
                                    (a.u1 - reference.u1).cwiseAbs().maxCoeff());
        BOOST_TEST_MESSAGE(std::format("tol {:g}: ss|d| = {:.2e}, error {:.2e}", tol,
                                       a.stats.correctionTest, err));
        // Within the run's tolerances (rtol 1e-6 on O(1) values)...
        BOOST_TEST(err < 1e-6);
        // ...and the tighter test is worth at least a factor of ten.
        BOOST_TEST(err < 0.1 * lastError);
        lastError = err;
    }
}

BOOST_AUTO_TEST_CASE(a_newton_step_solve_spends_one_flux_and_one_derivative_sweep_per_iteration)
{
    // Newton mode, full Newton (NewtonJacobianReuse = 1). One flux sweep builds
    // sigma from the initial condition; then each Newton iteration evaluates
    // the residual at its iterate and builds the Jacobian there. The converged
    // iterate is never evaluated, as IDA never evaluates its converged
    // corrector -- which is what makes this 1 + n flux sweeps against the
    // Residual path's 2 + 2n.
    Grid grid(0.0, 1.0, nCells);
    CountingNonlinearDiffusion problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, "newton_step_sweeps");
    sys.setSteadyMode(SystemSolver::SteadyMode::Newton);
    sys.setSteadyStateTolerance(1e-3);
    sys.setSteadyTest(SystemSolver::SteadyTest::NewtonStep);
    sys.setNewtonMaxIterations(50);

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.solveSteadyState();
    }

    const long nodes = nCells * (k + 1);
    BOOST_TEST(problem.calls % nodes == 0);
    BOOST_TEST(problem.derivCalls % nodes == 0);
    const long flux = problem.calls / nodes, deriv = problem.derivCalls / nodes;
    const auto stats = sys.lastSteadyStats();
    BOOST_TEST_MESSAGE(std::format("{} Newton iterations: {} flux sweeps, {} derivative sweeps",
                                   stats.newtonIters, flux, deriv));

    BOOST_TEST((sys.lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged));
    BOOST_TEST(stats.rejected == 0);
    BOOST_TEST(stats.steps == 1);
    BOOST_TEST(flux == 1 + stats.newtonIters);
    BOOST_TEST(deriv == stats.newtonIters);

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput("newton_step_sweeps");
}

BOOST_AUTO_TEST_CASE(a_sliced_newton_step_solve_resumes_exactly)
{
    // Everything the loop carries from one step to the next -- dt, the SER
    // reference, the undamped-attempt state -- survives a slice, so a solve
    // cut into slices of two steps takes the same steps as an uninterrupted
    // one and reaches the same answer to the bit.
    constexpr int SLICE = 2;

    auto build = [](SystemSolver &sys)
    {
        configure(sys, "newton_step_resume");
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setPseudoTransientInitialStep(1e-4);
        sys.setSteadyStateTolerance(1e-3);
        sys.setSteadyTest(SystemSolver::SteadyTest::NewtonStep);
    };

    auto solve = [&](int slice)
    {
        Grid grid(0.0, 1.0, nCells);
        TwoScaleDiffusion problem;
        SystemSolver sys(grid, k, &problem);
        build(sys);
        if (slice > 0)
            sys.setMaxContinuationSteps(slice);

        int total = 0;
        CapturedOutput quiet;
        sys.initialize();
        for (int slices = 0; slices < 200; ++slices)
        {
            try
            {
                if (slices > 0)
                    sys.continueSteadyState();
                else
                    sys.solveSteadyState();
            }
            catch (std::exception const &)
            {
                // Out of steps for this slice.
            }
            total += sys.lastSteadyStats().steps;
            if (sys.lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged)
                break;
        }
        const bool converged = sys.lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged;
        sys.captureState();
        Vector u(9);
        for (Index i = 0; i < 9; ++i)
            u(i) = sys.yJac.u(0)(0.1 * (i + 1)) + sys.yJac.u(1)(0.1 * (i + 1));
        sys.destroySundials();
        return std::tuple{converged, total, u};
    };

    const auto [wholeOK, wholeSteps, wholeU] = solve(0);
    const auto [slicedOK, slicedSteps, slicedU] = solve(SLICE);
    BOOST_TEST_MESSAGE(std::format("uninterrupted {} steps, sliced {} steps", wholeSteps, slicedSteps));
    BOOST_REQUIRE(wholeOK);
    BOOST_REQUIRE(wholeSteps > 2 * SLICE);
    BOOST_TEST(slicedOK);
    BOOST_TEST(slicedSteps == wholeSteps);
    for (Index i = 0; i < wholeU.size(); ++i)
        BOOST_TEST(slicedU(i) == wholeU(i));

    removeOutput("newton_step_resume");
}

BOOST_AUTO_TEST_SUITE_END()
