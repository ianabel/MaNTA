// Tests for the three-phase run lifecycle:
//
//     initialize()  ->  integrate(tFinal)  ->  destroySundials()
//
// runSolver() composes those three, and used to be a single function holding
// every SUNDIALS handle in a local. The handles are now members so a caller can
// allocate, inspect, integrate and free as separate steps -- which is what makes
// PyRunner::G() possible -- and that turns a set of previously-unreachable
// states into reachable ones: destroy without initialize, destroy twice,
// integrate without initialize, initialize again on the same object. Each is
// pinned here.
//
// The other thing the split changed is that cleanup happens when the time loop
// throws. Before, every N_VDestroy and SUNLinSolFree sat after the loop, so a
// failed run leaked the whole SUNDIALS state and left its output streams open.
// Because the handles are members now, that is directly observable:
// a_failed_run_still_frees_its_resources checks they are null afterwards.
//
// A second integration on the same solver now works and is pinned here, by
// a_second_integration_on_one_solver_matches_a_fresh_one. It used to fail with
// IDA_ERR_FAIL on the first step of the second run; that test's comment has the
// diagnosis.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <ida/ida.h>
#include <kinsol/kinsol.h>

#include <cmath>
#include <exception>
#include <filesystem>
#include <print>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

using namespace toml::literals::toml_literals;

namespace
{
const toml::value lifecycle_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;

constexpr Index k = 2, nCells = 4;
constexpr double T_FINAL = 0.05;

// A solver configured for a short, cheap run. `stem` decides where the output
// lands: Solver.cpp uses inputFilePath.stem(), so the files appear in the
// working directory whatever path is given.
void configure(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(T_FINAL);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
}

// The output a run leaves behind, so each case can clean up after itself.
void removeOutput(std::string const &stem)
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);
}

// u at a handful of interior points, read out of yJac -- the only copy of the
// solution that outlives destroySundials().
Vector sample(SystemSolver &sys)
{
    Vector out(5);
    for (Index i = 0; i < 5; ++i)
        out(i) = sys.yJac.u(0)(0.1 + 0.2 * i);
    return out;
}

// The smallest objective that will drive the dG/dt gate: G = sign * Int u dx over
// TestDiffusion's single variable, so dG/dt = sign * Int u' dx and the two signs
// give opposite verdicts on the same run. Which of them rejects depends on which
// way the diffusion problem's initial condition is relaxing, and the test does
// not need to know -- it asserts that exactly one does.
class SignedIntegralObjective : public AdjointProblem
{
public:
    explicit SignedIntegralObjective(double s) : sign(s)
    {
        ng = 1;
        np = 1;
        np_boundary = 0;
    }

    using AdjointProblem::gFn;

    Value gFn(Index, const State &s, Position) const override { return sign * s.u(0); }

    Value GFn(Index, DGSoln &) const override
    {
        // Never called: the gate goes through dGdt, and solveAdjoint is off, so
        // nothing asks for G itself here.
        throw std::logic_error("GFn is not part of what this fixture exercises.");
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }

    void dgFn_du(Index, VectorRef v, const State &, Position) override
    {
        v.setZero();
        v[0] = sign;
    }
    void dgFn_dq(Index, VectorRef v, const State &, Position) override { v.setZero(); }
    void dgFn_dsigma(Index, VectorRef v, const State &, Position) override { v.setZero(); }
    void dgFn_dphi(Index, VectorRef v, const State &, Position) override { v.setZero(); }

    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }

private:
    double sign;
};

// TestDiffusion, counting physics evaluations. That is the unit PERFORMANCE.md
// asks for, and the only one in which the SER knobs below can be shown to do
// anything: they change how many continuation steps a solve takes, not what it
// converges to, and nothing exposes the step count.
class CountingDiffusion : public TestDiffusion
{
public:
    using TestDiffusion::TestDiffusion;

    Value SigmaFn(Index i, const State &s, Position x, Time t) override
    {
        ++calls;
        return TestDiffusion::SigmaFn(i, s, x, t);
    }

    int calls = 0;
};

} // namespace

BOOST_AUTO_TEST_SUITE(solver_lifecycle_tests)

BOOST_AUTO_TEST_CASE(the_three_phases_reproduce_runSolver_exactly)
{
    // The point of the refactor: runSolver() is now literally these three calls,
    // so the answers must agree bit for bit, not merely to a tolerance.
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion composedProblem(lifecycle_config);
    SystemSolver composed(grid, k, &composedProblem);
    configure(composed, "lifecycle_composed");

    TestDiffusion phasedProblem(lifecycle_config);
    SystemSolver phased(grid, k, &phasedProblem);
    configure(phased, "lifecycle_phased");

    {
        CapturedOutput quiet;
        composed.runSolver(T_FINAL);

        phased.initialize();
        phased.integrate(T_FINAL);
        phased.destroySundials();
    }

    const Vector a = sample(composed), b = sample(phased);
    for (Index i = 0; i < a.size(); ++i)
        BOOST_TEST(a(i) == b(i), boost::test_tools::tolerance(0.0));

    // Not vacuous: the solution is actually nonzero.
    BOOST_TEST(a.norm() > 1e-8);

    removeOutput("lifecycle_composed");
    removeOutput("lifecycle_phased");
}

BOOST_AUTO_TEST_CASE(destroy_is_safe_without_initialize_and_when_repeated)
{
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_destroy");

    // Never initialised: every handle is still null and there is nothing to do.
    BOOST_CHECK_NO_THROW(sys.destroySundials());

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(T_FINAL);
    }

    // Idempotent. runSolver's catch-and-rethrow depends on this: it calls
    // destroySundials() on the failure path and again on the normal one.
    BOOST_CHECK_NO_THROW(sys.destroySundials());
    BOOST_CHECK_NO_THROW(sys.destroySundials());
    BOOST_CHECK_NO_THROW(sys.destroySundials());

    removeOutput("lifecycle_destroy");
}

BOOST_AUTO_TEST_CASE(integrate_without_initialize_is_reported)
{
    // Reachable only now that the phases are separate, and the failure it would
    // otherwise produce is a null-pointer dereference inside IDA.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_uninit");

    BOOST_CHECK_THROW(sys.integrate(T_FINAL), std::logic_error);
}

BOOST_AUTO_TEST_CASE(initialize_can_be_called_again_after_destroy)
{
    // `ctx` belongs to the SystemSolver, not to a run, so it must survive
    // destroySundials(): freeing it there is what used to make a second
    // initialize() fail at IDACreate with "Sundials Initialization Error". This
    // pins that one property, and the initial condition it rebuilds.
    //
    // It stops at initialize() deliberately, and covers only what that call
    // rebuilds. Completing a second *integration* is
    // a_second_integration_on_one_solver_matches_a_fresh_one below.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_twice");

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(T_FINAL);
        sys.destroySundials();
    }
    const Vector solved = sample(sys);
    BOOST_TEST(solved.norm() > 1e-8);

    // The part that must still work: allocate again on the same object.
    // Captured without asserting inside -- Boost writes failures to stdout, so an
    // assertion that fires while captured is swallowed (Tests/README.md).
    std::string failure;
    {
        CapturedOutput quiet;
        try
        {
            sys.initialize();
        }
        catch (std::exception const &e)
        {
            failure = e.what();
        }
    }
    BOOST_TEST(failure.empty(), "second initialize() threw: " << failure);

    // And it rebuilt the initial condition rather than keeping the solved state.
    // TestDiffusion decays as exp(-t (pi/2)^2), so after T_FINAL every point is
    // strictly smaller than it started.
    const Vector restarted = sample(sys);
    for (Index i = 0; i < restarted.size(); ++i)
        BOOST_TEST(restarted(i) > solved(i),
                   "point " << i << ": re-initialised " << restarted(i)
                            << " should exceed solved " << solved(i));

    sys.destroySundials();
    removeOutput("lifecycle_twice");
}

BOOST_AUTO_TEST_CASE(yJac_holds_the_initial_condition_after_initialize_alone)
{
    // yJac used to be uninitialised memory until the end of a run, so anything
    // reading "the solution" between initialize() and integrate() -- which only
    // the split makes possible -- got garbage. setInitialConditions now seeds it.
    //
    // TestDiffusion's initial condition is a Gaussian centred at 0, so on [0, 1]
    // it is positive and decreasing. Both properties would be lost on garbage.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_ic");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    const Vector u = sample(sys);
    for (Index i = 0; i < u.size(); ++i)
    {
        BOOST_TEST(std::isfinite(u(i)));
        BOOST_TEST(u(i) > 0.0, "u(" << 0.1 + 0.2 * i << ") = " << u(i));
    }
    for (Index i = 1; i < u.size(); ++i)
        BOOST_TEST(u(i) < u(i - 1));

    sys.destroySundials();
    removeOutput("lifecycle_ic");
}

BOOST_AUTO_TEST_CASE(a_failed_run_still_frees_its_resources)
{
    // The other thing the split bought: runSolver() catches, frees and rethrows.
    // Before, every N_VDestroy and SUNLinSolFree sat after the time loop, so a
    // run that threw leaked the whole SUNDIALS state and left its output streams
    // open.
    //
    // integrate() throws when t0 is past tFinal, which is a failure it can be
    // made to hit deterministically. What is checked afterwards is the state of
    // the object: nulled handles and closed streams mean destroySundials() ran on
    // the exceptional path.
    const std::string stem = "lifecycle_failed";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setWriteDatFile(true); // so there is a stream to leave open
    sys.setInitialTime(1.0);   // after tFinal

    // Captured without asserting inside; the message is checked afterwards, which
    // also pins that this is the failure we meant to provoke rather than some
    // other runtime_error on the way.
    std::string message;
    {
        CapturedOutput quiet;
        try
        {
            sys.runSolver(T_FINAL);
        }
        catch (std::runtime_error const &e)
        {
            message = e.what();
        }
    }
    BOOST_TEST(message == "Simulation ends before it begins.", message);

    BOOST_TEST(sys.IDA_mem == nullptr);
    BOOST_TEST(sys.LS == nullptr);
    BOOST_TEST(sys.sunMat == nullptr);
    BOOST_TEST(sys.Y == nullptr);
    BOOST_TEST(sys.dYdt == nullptr);
    BOOST_TEST(sys.constraints == nullptr);
    BOOST_TEST(sys.id == nullptr);
    BOOST_TEST(sys.res == nullptr);
    BOOST_TEST(sys.absTolVec == nullptr);
    BOOST_TEST(!sys.out0.is_open());

    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(aggressive_timesteps_is_accepted_and_gives_the_same_answer)
{
    // IDASetEtaMax changes how fast IDA may grow the step, not what it converges
    // to, so the answer must agree to the integration tolerance.
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion defaultProblem(lifecycle_config);
    SystemSolver defaultSteps(grid, k, &defaultProblem);
    configure(defaultSteps, "lifecycle_eta_default");

    TestDiffusion aggressiveProblem(lifecycle_config);
    SystemSolver aggressiveSteps(grid, k, &aggressiveProblem);
    configure(aggressiveSteps, "lifecycle_eta_aggressive");
    aggressiveSteps.setAggressiveTimesteps(true);

    std::string failure;
    {
        CapturedOutput quiet;
        defaultSteps.runSolver(T_FINAL);
        try
        {
            aggressiveSteps.runSolver(T_FINAL);
        }
        catch (std::exception const &e)
        {
            failure = e.what();
        }
    }
    BOOST_TEST(failure.empty(), "aggressive timesteps threw: " << failure);

    const Vector a = sample(defaultSteps), b = sample(aggressiveSteps);
    for (Index i = 0; i < a.size(); ++i)
        BOOST_TEST(a(i) == b(i), boost::test_tools::tolerance(1e-5));

    removeOutput("lifecycle_eta_default");
    removeOutput("lifecycle_eta_aggressive");
}

BOOST_AUTO_TEST_CASE(the_dGdt_gate_skips_the_time_loop_without_disturbing_an_ungated_run)
{
    // End to end through runSolver, which is the whole point of the gate: a step
    // whose objective is already getting worse should cost initialisation and
    // nothing more.
    //
    // Observed through the solution rather than through IDA's step counter,
    // because IDA_mem is gone by the time runSolver returns. integrate() ends by
    // leaving the final solution in yJac, so a run that was skipped leaves yJac
    // holding the initial condition instead -- and that is also the assertion
    // that would catch a gate which "rejects" but integrates anyway.
    Grid grid(0.0, 1.0, nCells);

    // The initial condition alone, for comparison: initialize() seeds yJac with
    // it, so stopping there gives the state a rejected run should be left in.
    TestDiffusion icProblem(lifecycle_config);
    SystemSolver icOnly(grid, k, &icProblem);
    configure(icOnly, "lifecycle_gate_ic");
    {
        CapturedOutput quiet;
        icOnly.initialize();
    }
    const Vector initial = sample(icOnly);
    {
        CapturedOutput quiet;
        icOnly.destroySundials();
    }
    BOOST_TEST(initial.norm() > 1e-8);

    // Which sign of the objective is falling is a property of the problem, not
    // something to hardcode: run both and require exactly one rejection.
    int rejections = 0;
    for (double sign : {1.0, -1.0})
    {
        const std::string stem = sign > 0 ? "lifecycle_gate_pos" : "lifecycle_gate_neg";

        TestDiffusion problem(lifecycle_config);
        SignedIntegralObjective objective(sign);
        SystemSolver sys(grid, k, &problem);
        configure(sys, stem);
        sys.setAdjointProblem(&objective);

        // Armed tight enough that the sign of dG/dt decides, rather than the
        // slack swamping it.
        sys.setObjectiveDecreaseTolerance(1e-12);

        {
            CapturedOutput quiet;
            sys.runSolver(T_FINAL);
        }

        const Vector after = sample(sys);

        if (sys.wasRejected())
        {
            ++rejections;
            // Skipped, so the solution never moved.
            for (Index i = 0; i < after.size(); ++i)
                BOOST_TEST(after(i) == initial(i), boost::test_tools::tolerance(0.0));
            BOOST_TEST(sys.lastDGdt()(0) < 0.0);
        }
        else
        {
            // Ran, so it did. If this ever coincides with the rejected branch's
            // values the test above has stopped meaning anything.
            BOOST_TEST((after - initial).norm() > 1e-8,
                       "an accepted run left the solution where it started");
            BOOST_TEST(sys.lastDGdt()(0) >= 0.0);
        }

        removeOutput(stem);
    }

    BOOST_TEST(rejections == 1,
               "expected exactly one of the two objective signs to be rejected, got " << rejections);

    removeOutput("lifecycle_gate_ic");
}

BOOST_AUTO_TEST_CASE(the_id_vector_marks_u_differential_and_nothing_else)
{
    // IDASetId tells IDA which components are differential and which algebraic,
    // and IDA_YA_YDP_INIT uses that to decide which parts of y to solve for and
    // which parts of y' to compute. Getting it wrong does not fail loudly -- it
    // just initialises a different problem.
    //
    // Which is what happened. initialize() built the vector with
    //
    //     isDifferential.u(v).getCoeff(i).second.Constant(k + 1, 1.0);
    //
    // and Eigen's Constant is a *static factory*: legal through an instance,
    // returns a fresh expression, result discarded. So the loop did nothing, `id`
    // kept the zeros from zeroCoeffs(), and IDA was told the whole system was
    // algebraic. Nothing warned, because Constant is not [[nodiscard]] and the
    // statement declares no unused variable -- so the only thing that can catch a
    // repeat is a test that looks at the vector.
    //
    // Exact counts, not just "nonzero": u contributes one per coefficient per cell
    // and everything else -- q, sigma, lambda, and the aux and scalar blocks -- has
    // to stay algebraic, so the L1 norm pins both halves at once.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_id");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    // TestDiffusion is one variable with no scalars and no aux variables.
    const double expected = 1.0 * nCells * (k + 1);
    BOOST_TEST(N_VL1Norm(sys.id) == expected, boost::test_tools::tolerance(0.0));
    BOOST_TEST(N_VMaxNorm(sys.id) == 1.0, boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput("lifecycle_id");
}

BOOST_AUTO_TEST_CASE(initialize_leaves_the_corrected_initial_condition_in_Y)
{
    // t0 output reports the state the time integration actually starts from, which
    // means the CalcIC-corrected one, not the guess fed to CalcIC.
    //
    // IDACalcIC keeps its answer inside IDA; Y and dYdt only receive it when
    // IDAGetConsistentIC is called. That call used to happen inside the debugDat
    // branch and, later, for an armed dG/dt gate -- so the t0 timeslice reported the
    // pre-CalcIC state on an ordinary run and the corrected one under
    // WriteDebugDatFiles. It is unconditional now, and this pins that: asking IDA
    // again must produce no further change, because Y already holds what IDA has.
    //
    // The difference is confined to the algebraic fields. u is differential, so
    // CalcIC holds it fixed and the u in t0 output is unchanged by any of this --
    // which is why the regression suite, which compares only u, cannot see the
    // change at all.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_consistent_ic");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    N_Vector Ycheck = N_VClone(sys.Y), dYcheck = N_VClone(sys.dYdt);
    IDAGetConsistentIC(sys.IDA_mem, Ycheck, dYcheck);

    N_VLinearSum(1.0, sys.Y, -1.0, Ycheck, Ycheck);
    N_VLinearSum(1.0, sys.dYdt, -1.0, dYcheck, dYcheck);

    const double dY = N_VMaxNorm(Ycheck), ddY = N_VMaxNorm(dYcheck);
    N_VDestroy(Ycheck);
    N_VDestroy(dYcheck);

    BOOST_TEST(dY == 0.0, boost::test_tools::tolerance(0.0));
    BOOST_TEST(ddY == 0.0, boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput("lifecycle_consistent_ic");
}

BOOST_AUTO_TEST_CASE(at_t0_only_the_differential_part_of_dydt_exists)
{
    // A known gap, pinned so that it is visible rather than assumed away, and so
    // that whoever closes it is told the gate depends on it.
    //
    // dG/dt is assembled as a full chain rule over u, q, sigma and phi, and
    // AdjointProblemTests proves that assembly correct. But at t0 the only part of
    // dydt that carries anything is the differential one: setInitialConditions
    // zeroes dydt and writes just dydt.u and the differential scalars, and
    // IDACalcIC does not fill in the rest. So at the point the gate runs, the q,
    // sigma and phi terms contribute nothing -- an objective depending on those is
    // differentiated only through its u dependence.
    //
    // The reason is structural rather than a bug: q, sigma and phi are algebraic in
    // this formulation, and IDA's IDA_YA_YDP_INIT computes algebraic *values* and
    // differential *derivatives* -- there is no y' for an algebraic component to
    // fetch. Their true time derivatives follow from differentiating the algebraic
    // constraints, which nothing here does.
    //
    // Worth being precise about, because there *was* also a wrong id vector telling
    // IDA the whole system was algebraic, and it would have been reasonable to
    // expect that to be the cause. It is not: the id vector is correct now (see
    // the_id_vector_marks_u_differential_and_nothing_else) and these blocks are
    // still exactly zero.
    //
    // This is still strictly better than origin/optimize-mode's gate, which
    // evaluated the objective functional on the derivative vector and so was wrong
    // about the u term too whenever g was nonlinear. But it is less than the full
    // derivative, and the comment on SystemSolver::dGdt says so.
    //
    // No gate armed and no objective attached: IDAGetConsistentIC is unconditional,
    // so this holds for every run rather than only for a gated one.
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_gate_consistent");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    double uNorm = 0.0, qNorm = 0.0, sigmaNorm = 0.0;
    for (Index i = 0; i < nCells; ++i)
    {
        uNorm += sys.dydt.u(0).getCoeff(i).second.norm();
        qNorm += sys.dydt.q(0).getCoeff(i).second.norm();
        sigmaNorm += sys.dydt.sigma(0).getCoeff(i).second.norm();
    }

    // The differential part is real, so the gate has something to work with.
    BOOST_TEST(uNorm > 1e-8);

    // The algebraic parts are not. If either of these ever becomes nonzero the gap
    // above has closed, and the gate's coverage should be re-examined and this
    // test rewritten rather than relaxed.
    BOOST_TEST(qNorm == 0.0, boost::test_tools::tolerance(0.0));
    BOOST_TEST(sigmaNorm == 0.0, boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput("lifecycle_gate_consistent");
}

BOOST_AUTO_TEST_CASE(a_second_integration_on_one_solver_matches_a_fresh_one)
{
    // A second full integration on the same SystemSolver used to fail: IDASolve
    // returned IDA_ERR_FAIL (-3) on the first step of the second run. This pins
    // that it now works, and that it gives the *same answer* a fresh solver
    // would -- bit for bit, which is a stronger claim than "it completes" and is
    // the one that broke first.
    //
    // Two defects combined to produce that failure, and it took both:
    //
    //  * `id` was all zeros (Eigen's Constant used as though it mutated in
    //    place), so IDASetId told IDA the whole system was algebraic and
    //    IDA_YA_YDP_INIT solved the wrong initialisation problem. A first run
    //    survived it; and IDACalcIC's return value was discarded, so when a
    //    second run did not survive it, the run carried on from IDA's partial,
    //    constraint-violating state. IDA's error test includes the algebraic
    //    components, whose error estimate is then independent of h, so the step
    //    could not be made small enough to pass -- hence ten error-test failures
    //    and IDA_ERR_FAIL. Both fixed in 536d856.
    //
    //  * What made the *second* run different from the first: initialiseMatrices
    //    filled RF_cellwise and L_global with boundary data at a hardcoded
    //    t = 0.0, and initialize() skips it when already initialised. So on a
    //    re-initialise those arrays still held the previous run's *final-time*
    //    boundary values, and setInitialConditions solved its initial dydt out
    //    of them. They are now sized there and filled by
    //    updateBoundaryConditions(t0), which is also what makes a first run with
    //    t0 != 0 correct.
    //
    // The bit-for-bit tolerance is deliberate and load bearing: with the stale
    // boundary data the second run still completed, and still looked right to
    // any reasonable tolerance -- it was off by 1.7e-10. Only an exact
    // comparison against a fresh solver saw it.
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion freshProblem(lifecycle_config);
    SystemSolver fresh(grid, k, &freshProblem);
    configure(fresh, "lifecycle_reuse_fresh");

    TestDiffusion reusedProblem(lifecycle_config);
    SystemSolver reused(grid, k, &reusedProblem);
    configure(reused, "lifecycle_reuse_reused");

    {
        CapturedOutput quiet;
        fresh.runSolver(T_FINAL);

        // Three runs, not two: the second is the one that used to fail, and a
        // third catches anything that accumulates rather than simply differing
        // from the first.
        reused.runSolver(T_FINAL);
        reused.runSolver(T_FINAL);
        reused.runSolver(T_FINAL);
    }

    const Vector want = sample(fresh), got = sample(reused);
    for (Index i = 0; i < want.size(); ++i)
        BOOST_TEST(got(i) == want(i), boost::test_tools::tolerance(0.0));

    // Not vacuous.
    BOOST_TEST(want.norm() > 1e-8);

    removeOutput("lifecycle_reuse_fresh");
    removeOutput("lifecycle_reuse_reused");
}

BOOST_AUTO_TEST_CASE(the_initial_condition_uses_boundary_data_at_t0)
{
    // The other half of the fix above, isolated. RF_cellwise and L_global carry
    // the Dirichlet and Neumann boundary data that the initial dydt is solved out
    // of, and they are functions of time. initialiseMatrices used to fill them at
    // a hardcoded 0.0; they are now sized there and filled by
    // updateBoundaryConditions(t0) from setInitialConditions.
    //
    // What this pins is the *time they were evaluated at*, which is the whole of
    // the invariant and is not otherwise observable: with every fixture in the
    // tree starting at zero, a return to the hardcoded 0.0 would go unnoticed.
    //
    // Deliberately not asserted here: that the initial condition satisfies the t0
    // boundary conditions. TestDiffusion's InitialValue is its exact solution at
    // t = 0, so at any other t0 the initial profile and the boundary functions
    // genuinely disagree -- the same trap that got LinearDiffusion's UseMMS
    // removed. That is a property of the fixture, not of the solver.
    constexpr double T_START = 0.5;
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_t0_boundaries");
    sys.setInitialTime(T_START);

    // The boundary data really is time dependent here, or the test proves nothing.
    BOOST_TEST(problem.LowerBoundary(0, T_START) != problem.LowerBoundary(0, 0.0));

    auto boundaryData = [&sys]
    {
        Vector out(sys.RF_cellwise.size() * sys.RF_cellwise[0].size() +
                   sys.L_global.size());
        Index at = 0;
        for (auto const &rf : sys.RF_cellwise)
        {
            out.segment(at, rf.size()) = rf;
            at += rf.size();
        }
        out.segment(at, sys.L_global.size()) = sys.L_global;
        return out;
    };

    {
        CapturedOutput quiet;
        sys.initialize();
    }
    const Vector afterInitialize = boundaryData();

    sys.updateBoundaryConditions(0.0);
    const Vector atZero = boundaryData();

    sys.updateBoundaryConditions(T_START);
    const Vector atT0 = boundaryData();

    // The state initialize() left behind is the t0 one, not the t = 0 one.
    for (Index i = 0; i < afterInitialize.size(); ++i)
        BOOST_TEST(afterInitialize(i) == atT0(i), boost::test_tools::tolerance(0.0));
    BOOST_TEST((afterInitialize - atZero).norm() > 1e-8,
               "boundary data at t0 = " << T_START << " is indistinguishable from t = 0; "
               "either the fixture stopped being time dependent or the fill is hardcoded again");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput("lifecycle_t0_boundaries");
}

BOOST_AUTO_TEST_CASE(a_steady_solve_writes_its_answer_to_the_output_file)
{
    // Every output call used to live inside the time loop, so a PseudoTransient
    // or Newton run wrote nothing at all: the .nc held the single t0 timeslice
    // that initialiseNetCDF puts there during initialize(), which is the
    // *initial condition*, and the .dat held one block of the same. The answer
    // reached yJac and the restart file's Y, so the Python surface -- which
    // reads yJac -- always looked right and only the files were wrong. Every
    // steady run in this tree is driven from Python, which is how it survived.
    //
    // TestDiffusion is kappa u_xx = 0 with Dirichlet ends frozen at t0, and
    // Centre = 0 puts those at u(0) = cos(0) = 1 and u(1) = cos(pi/2) = 0. The
    // steady state is therefore u = 1 - x exactly -- degree 1, so it sits in P_k
    // with room to spare and can be checked against the closed form rather than
    // against itself.
    const std::string stem = "lifecycle_steady_output";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-10);

    {
        CapturedOutput quiet;
        sys.runSolver(T_FINAL);
    }

    // The solver reached the right answer, so anything missing below is the
    // output path rather than the solve.
    const Vector u = sample(sys);
    for (Index i = 0; i < u.size(); ++i)
    {
        const double x = 0.1 + 0.2 * i;
        BOOST_TEST(u(i) == 1.0 - x, boost::test_tools::tolerance(1e-8));
    }

    netCDF::NcFile out;
    BOOST_CHECK_NO_THROW(out.open(stem + ".nc", netCDF::NcFile::FileMode::read));

    // Two slices: the initial condition, and the converged state.
    const size_t nSlices = out.getDim("t").getSize();
    BOOST_TEST(nSlices == 2u);

    std::vector<double> t(nSlices);
    out.getVar("t").getVar(t.data());
    BOOST_TEST(t[0] == 0.0, boost::test_tools::tolerance(0.0));
    BOOST_TEST(t[1] == SystemSolver::STEADY_STATE_TIME, boost::test_tools::tolerance(0.0));

    const size_t nX = out.getDim("x").getSize();
    std::vector<double> x(nX);
    out.getVar("x").getVar(x.data());

    // The last slice, read back through the same (t, x) layout WriteTimeslice
    // wrote it in.
    std::vector<double> uOut(nX), uInitial(nX);
    netCDF::NcVar uVar = out.getGroup(problem.getVariableName(0)).getVar("u");
    uVar.getVar({nSlices - 1, 0}, {1, nX}, uOut.data());
    uVar.getVar({0, 0}, {1, nX}, uInitial.data());

    for (size_t i = 0; i < nX; ++i)
        BOOST_TEST(uOut[i] == 1.0 - x[i], boost::test_tools::tolerance(1e-8));

    // Not vacuous: the two slices really are different states, so a regression
    // that wrote the initial condition twice would fail here rather than pass.
    double spread = 0.0;
    for (size_t i = 0; i < nX; ++i)
        spread = std::max(spread, std::abs(uOut[i] - uInitial[i]));
    BOOST_TEST(spread > 1e-2,
               "the two timeslices are indistinguishable; the converged state was not written");

    out.close();
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(a_converged_steady_state_leaves_no_stale_derivative)
{
    // solveSteadyState damps through a scratch vector and never touched dYdt, so
    // on return it still held whatever IDACalcIC left at t0. Two things read it
    // afterwards: WriteRestartFile, so a restart resumed from a state whose y
    // was the steady one and whose y' was the initial one, and a physics case's
    // writeDiagnostics. Measured on AdjointPoster before the fix, ||dYdt|| was
    // 103.4 at a converged steady state.
    //
    // Stops before destroySundials(), which nulls dYdt -- the point is the state
    // integrate() leaves behind, not what cleanup does to it.
    const std::string stem = "lifecycle_steady_dydt";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-10);

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    // Not vacuous: at t0 the derivative is genuinely nonzero, so zeroing it is a
    // change rather than a coincidence of this fixture.
    BOOST_TEST(N_VMaxNorm(sys.dYdt) > 1e-3);

    {
        CapturedOutput quiet;
        sys.integrate(T_FINAL);
    }

    // Exactly zero, not merely small: it is set rather than converged to.
    BOOST_TEST(N_VMaxNorm(sys.dYdt) == 0.0, boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(the_SER_rate_and_floor_change_the_cost_and_not_the_answer)
{
    // PseudoTransientSERFloor is the least dt may grow on a step that reduced
    // the residual, and it exists because plain SER is self-perpetuating from a
    // conservative dt0: the ratio is only as large as the residual reduction,
    // and the reduction is only as large as dt allows. So the floor should be
    // visible as *cost* and invisible in the answer, and both halves are worth
    // pinning -- an option that changed the converged state would be a bug, and
    // one that changed nothing at all would be inert.
    //
    // Floor 1.0 is plain SER, i.e. no floor. The starting step is pinned so the
    // comparison is between schedules rather than between starting points.
    auto solve = [](double rate, double floorValue, int &calls)
    {
        Grid grid(0.0, 1.0, nCells);
        CountingDiffusion problem(lifecycle_config);
        SystemSolver sys(grid, k, &problem);
        configure(sys, "lifecycle_ser");
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setSteadyStateTolerance(1e-10);
        sys.setPseudoTransientInitialStep(1e-3);
        sys.setPseudoTransientSERRate(rate);
        sys.setPseudoTransientSERFloor(floorValue);

        {
            CapturedOutput quiet;
            sys.runSolver(T_FINAL);
        }
        calls = problem.calls;
        return sample(sys);
    };

    int floored = 0, plain = 0, steeper = 0;
    const Vector withFloor = solve(1.0, 2.0, floored);
    const Vector withoutFloor = solve(1.0, 1.0, plain);

    // The rate, isolated: with the floor out of the way at 1.0, leaning harder
    // on the residual ratio is the only thing left that can grow dt.
    solve(2.0, 1.0, steeper);

    BOOST_TEST_MESSAGE("physics evaluations -- rate 1 floor 2 (defaults): " << floored
                       << "; rate 1 floor 1 (plain SER): " << plain
                       << "; rate 2 floor 1: " << steeper);

    // Same steady state either way -- u = 1 - x, from Dirichlet ends of 1 and 0
    // frozen at t0.
    for (Index i = 0; i < withFloor.size(); ++i)
    {
        const double x = 0.1 + 0.2 * i;
        BOOST_TEST(withFloor(i) == 1.0 - x, boost::test_tools::tolerance(1e-8));
        BOOST_TEST(withoutFloor(i) == 1.0 - x, boost::test_tools::tolerance(1e-8));
    }

    // ...reached more cheaply with the floor. If this ever inverts, the floor is
    // no longer earning its default and the default should move, not the test.
    BOOST_TEST(floored < plain,
               "the SER floor cost " << floored << " evaluations against plain SER's "
               << plain << "; it is meant to be the cheaper of the two");

    // And a steeper rate is cheaper than plain SER for the same reason, by a
    // different route: measured 1704 against 3540 here, where the floor gets it
    // to 552. Both knobs reach the schedule, which is the claim.
    BOOST_TEST(steeper < plain,
               "SER rate 2 cost " << steeper << " evaluations against rate 1's " << plain
               << "; leaning harder on the residual ratio should grow dt faster");

    removeOutput("lifecycle_ser");
}

BOOST_AUTO_TEST_CASE(a_steady_solve_says_what_it_is_about_to_do)
{
    // The entry banner is unconditional, because TimeMarch's equivalent is:
    // "Writing output at ..." per slice, then three IDA totals. A steady run
    // printed nothing at all. The INFO-level logmsg calls inside the loop are
    // not a substitute -- Logging.hpp gates max_log_level at compile time, and
    // it is WARNING unless the build sets VERBOSE or DEBUG, so in an ordinary
    // build they emit nothing.
    //
    // Diagnostics deliberately left off here: this is the part that does not
    // need asking for.
    const std::string stem = "lifecycle_steady_banner";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-10);

    std::string log;
    {
        CapturedOutput capture;
        sys.runSolver(T_FINAL);
        log = capture.text();
    }

    BOOST_TEST(log.find("Steady solve: PseudoTransient") != std::string::npos, log);
    BOOST_TEST(log.find("initial ||F||") != std::string::npos, log);
    BOOST_TEST(log.find("SER rate") != std::string::npos, log);
    BOOST_TEST(log.find("converged") != std::string::npos, log);

    // Off by default, and this is what says so.
    BOOST_TEST(log.find("Steady solve statistics") == std::string::npos, log);

    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(the_steady_diagnostics_count_the_whole_solve_not_the_last_step)
{
    // KINSOL zeroes its own counters at the top of every KINSol call, so the
    // continuation loop has to sum them as it goes. Reading them once at the end
    // -- the obvious thing, and what this did first -- reports the final inner
    // solve alone: 1 Newton iteration against 5 continuation steps and 35
    // Jacobian solves, which is self-evidently impossible but looks like a
    // number. The invariants below are what makes that visible.
    const std::string stem = "lifecycle_steady_stats";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-10);
    sys.setSteadyStateDiagnostics(true);

    std::string log;
    {
        CapturedOutput capture;
        sys.runSolver(T_FINAL);
        log = capture.text();
    }

    BOOST_TEST(log.find("Steady solve statistics -- converged") != std::string::npos, log);
    BOOST_TEST(log.find("KINSOL Newton iterations") != std::string::npos, log);
    BOOST_TEST(log.find("Jacobian solves") != std::string::npos, log);

    const auto s = sys.lastSteadyStats();
    BOOST_TEST_MESSAGE("steps " << s.steps << ", newton " << s.newtonIters
                       << ", residual " << s.residualEvals << " (KINSOL " << s.kinFuncEvals
                       << "), jac builds " << s.jacBuilds << ", jac solves " << s.jacSolves);

    BOOST_TEST(s.steps > 0);
    BOOST_TEST(s.rejected == 0);

    // Each continuation step is one KINSol call, and a KINSol call that returns
    // has taken at least one Newton iteration. This is the invariant the
    // per-call reset broke.
    BOOST_TEST(s.newtonIters >= s.steps,
               "only " << s.newtonIters << " Newton iterations for " << s.steps
               << " continuation steps; the KINSOL counters are being read once "
               "rather than accumulated");

    // The merit function costs one residual per step plus one on entry, and
    // those are MaNTA's, not KINSOL's -- so the total strictly exceeds KINSOL's
    // own count by exactly that. Pins the snapshot being taken before the first
    // steadyNorm(), which it was not to begin with.
    BOOST_TEST(s.residualEvals == s.kinFuncEvals + s.steps + 1);

    // One linear solve per Newton iteration, with a direct solver.
    BOOST_TEST(s.jacSolves == s.newtonIters);

    // Every Jacobian build here came from KINSOL asking for one, and a build is
    // never gratuitous: KINSOL reuses a factorisation across iterations, so
    // builds are at most solves. They are *equal* on this fixture, which is the
    // uninteresting case rather than the general one -- TestDiffusion is linear,
    // so each KINSol converges in a single Newton iteration and there is nothing
    // to reuse across. The separation appears on a nonlinear problem:
    // AdjointPoster at k = 3 on 6 cells gives 7 builds against 35 solves, which
    // is what makes the two counts worth reporting separately at all.
    BOOST_TEST(s.jacBuilds == s.kinJacEvals);
    BOOST_TEST(s.jacBuilds <= s.jacSolves);

    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(a_failed_steady_solve_still_writes_the_last_state_it_reached)
{
    // A failed steady solve is exactly when the state is worth looking at, and
    // it was the one case that produced nothing: solveSteadyState threw, the
    // throw reached runSolver, and runSolver freed everything without writing.
    // The time loop has done this for a failed IDASolve all along.
    //
    // Provoked with a tolerance nothing can reach. The solve gets to ~1e-16 and
    // then stalls, which is the ordinary "ran out of continuation steps" exit --
    // deliberately that path rather than a KINSol crash, because it is
    // deterministic and the Solver.cpp catch is `catch (...)` either way.
    const std::string stem = "lifecycle_steady_failed";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-30);

    std::string message;
    {
        CapturedOutput quiet;
        try
        {
            sys.runSolver(T_FINAL);
        }
        catch (std::runtime_error const &e)
        {
            message = e.what();
        }
    }
    BOOST_TEST(message.find("Steady solve did not converge") != std::string::npos, message);

    // The stats survive the throw, which is half the point of filling them in
    // before it.
    BOOST_TEST(sys.lastSteadyStats().steps > 1);

    netCDF::NcFile out;
    BOOST_CHECK_NO_THROW(out.open(stem + ".nc", netCDF::NcFile::FileMode::read));

    const size_t nSlices = out.getDim("t").getSize();
    BOOST_TEST(nSlices == 2u);

    const size_t nX = out.getDim("x").getSize();
    std::vector<double> uOut(nX), uInitial(nX), x(nX);
    out.getVar("x").getVar(x.data());
    netCDF::NcVar uVar = out.getGroup(problem.getVariableName(0)).getVar("u");
    uVar.getVar({nSlices - 1, 0}, {1, nX}, uOut.data());
    uVar.getVar({0, 0}, {1, nX}, uInitial.data());

    // It got most of the way there before running out of steps, so the state
    // written is the near-converged one rather than the initial condition. A
    // loose tolerance: the assertion is "this is the solve's own state", not
    // "it converged".
    double worst = 0.0, spread = 0.0;
    for (size_t i = 0; i < nX; ++i)
    {
        worst = std::max(worst, std::abs(uOut[i] - (1.0 - x[i])));
        spread = std::max(spread, std::abs(uOut[i] - uInitial[i]));
    }
    BOOST_TEST(worst < 1e-6, "last state written is " << worst << " from u = 1 - x");
    BOOST_TEST(spread > 1e-2, "the two timeslices are indistinguishable");

    out.close();
    removeOutput(stem);
}

// ------------------------------------ restarting at a different degree ----
//
// A restart used to require the run's discretisation to match the file's
// exactly, and could not do otherwise: makeGrid reads both CellBoundaries and
// PolyOrder from the file and ignored Polynomial_degree, so the degrees always
// agreed and DGSoln::copy -- which throws on a different order -- was never
// asked for anything else. The config's degree is honoured now, and
// setInitialConditions projects rather than copies when it differs.

namespace
{
// Initialise a solver at degree `order`, optionally from a state left by an
// earlier one, and return its u sampled at five interior points along with the
// state it finished with.
struct RestartSnapshot
{
    Vector sampled;
    std::vector<double> Y, dYdt;
};

RestartSnapshot initialiseAt(TestDiffusion &problem, Grid const &grid, Index order,
                             std::string const &stem)
{
    SystemSolver sys(grid, order, &problem);
    configure(sys, stem);

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    RestartSnapshot out;
    out.sampled = sample(sys);

    const size_t nDOF = sys.yJac.getDoF();
    out.Y.assign(sys.yJacMem, sys.yJacMem + nDOF);
    out.dYdt.assign(sys.dydtJacMem, sys.dydtJacMem + nDOF);

    sys.destroySundials();
    return out;
}
} // namespace

BOOST_AUTO_TEST_CASE(a_restart_at_a_higher_degree_reproduces_the_state_exactly)
{
    // The sharp case, and the reason a projection is the right transfer rather
    // than a resampling: a degree-2 element polynomial lies *inside* the
    // degree-3 space, so the L2 projection onto it is not an approximation at
    // all. The refined run must reproduce the coarse state to round-off.
    //
    // Note what this compares. TestDiffusion's initial condition is
    // cos((x - Centre) * pi/2), which neither space represents exactly -- so
    // the assertion is against the *coarse run's own* state, not against the
    // exact function. Comparing to cos would be a discretisation test wearing a
    // transfer test's clothes, and would pass just as well if the transfer did
    // nothing at all and the refined run simply re-projected cos itself.
    const std::string stem = "lifecycle_restart_refine";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);

    const RestartSnapshot coarse = initialiseAt(problem, grid, 2, stem);

    problem.setRestartValues(coarse.Y, coarse.dYdt, grid, 2);
    BOOST_TEST(problem.isRestarting());

    const RestartSnapshot refined = initialiseAt(problem, grid, 3, stem);

    double worst = 0.0;
    for (Index i = 0; i < coarse.sampled.size(); ++i)
        worst = std::max(worst, std::abs(refined.sampled(i) - coarse.sampled(i)));

    BOOST_TEST_MESSAGE("k = 2 -> 3 transfer, worst |du| at five interior points: " << worst);
    BOOST_TEST(worst < 1e-12,
               "the refined run differs from the coarse state it was given by " << worst
               << "; a degree-2 polynomial is in the degree-3 space and the transfer "
               "should be exact");

    problem.clearRestart();
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(a_restart_at_a_lower_degree_lands_where_a_fresh_run_would)
{
    // Coarsening loses information -- a degree-3 polynomial does not fit in the
    // degree-2 space -- but it loses *only* that, and the statement is exact:
    // L2 projections onto nested spaces compose, so P2(P3(f)) = P2(f) whenever
    // V2 is contained in V3. Transferring a k = 3 state down to k = 2 therefore
    // lands on precisely the state a cold k = 2 run builds, to round-off, and
    // the transfer contributes no error of its own.
    //
    // That nesting is per cell and needs the same mesh. A projection onto a
    // *different* mesh would not compose, and this test would not hold -- worth
    // knowing before anyone extends the transfer to a remesh.
    const std::string stem = "lifecycle_restart_coarsen";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);

    const RestartSnapshot cold2 = initialiseAt(problem, grid, 2, stem);
    const RestartSnapshot fine3 = initialiseAt(problem, grid, 3, stem);

    problem.setRestartValues(fine3.Y, fine3.dYdt, grid, 3);
    const RestartSnapshot coarsened = initialiseAt(problem, grid, 2, stem);

    double worst = 0.0, spread = 0.0;
    for (Index i = 0; i < cold2.sampled.size(); ++i)
    {
        worst = std::max(worst, std::abs(coarsened.sampled(i) - cold2.sampled(i)));
        spread = std::max(spread, std::abs(fine3.sampled(i) - cold2.sampled(i)));
    }

    BOOST_TEST_MESSAGE("k = 3 -> 2 transfer differs from a cold k = 2 run by " << worst
                       << "; the k=3 and k=2 spaces differ by " << spread);

    // Guard first: if the two spaces did not visibly differ on this problem the
    // assertion below would hold for the wrong reason. Measured 2.8e-4.
    BOOST_TEST(spread > 1e-6,
               "k = 2 and k = 3 agree to " << spread << " here, so this fixture "
               "cannot tell a working transfer from a broken one");

    BOOST_TEST(worst < 1e-12,
               "coarsening moved the state " << worst << " away from where a cold "
               "k = 2 run lands; nested projections compose, so this should be exact");

    problem.clearRestart();
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(a_restart_at_the_same_degree_still_takes_the_copy_path)
{
    // The no-regression half. Equal degrees must keep going through
    // DGSoln::copy, bit for bit -- that is what the restart round-trip cases in
    // the regression suite compare, and it is why restartRunOrder returns the
    // file's order unchanged when the two agree rather than dropping everyone
    // into the projection.
    const std::string stem = "lifecycle_restart_same";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);

    const RestartSnapshot first = initialiseAt(problem, grid, 2, stem);

    problem.setRestartValues(first.Y, first.dYdt, grid, 2);
    const RestartSnapshot second = initialiseAt(problem, grid, 2, stem);

    for (Index i = 0; i < first.sampled.size(); ++i)
        BOOST_TEST(second.sampled(i) == first.sampled(i));

    problem.clearRestart();
    removeOutput(stem);
}

// ------------------------------------------- the steady merit function ----
//
// steadyResidualNorm is what the whole steady solve is measured against: the
// early return, the convergence test and the SER ratio all read it, and KINSOL
// is handed the same tolerance against its own norm of the same vector.
//
// It used to be a flat 2-norm, which went like sqrt(h) -- so steady_state_tol
// meant a different thing on every mesh and dt could not be carried across a
// remesh. It is now weighted by 1/sqrt(h) on the rows that are pairings against
// the basis, which is the discrete L2 norm of the equation residual. The four
// below split into what it *is* (a formula, pinned exactly), what it *is not*
// (damped), what it *does* (does not move with the mesh, and agrees with
// KINSOL), and where that claim stops holding.

BOOST_AUTO_TEST_CASE(the_steady_merit_function_is_the_undamped_residual_two_norm)
{
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_norm_formula");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    const double norm = sys.steadyResidualNorm();

    // Recomputed from the outside: zero derivative, residual at t0 and Y, then
    // sum (w_i F_i)^2 with the weights written out here rather than read from the
    // solver -- 1/sqrt(h) on the cell rows, 1 on the lambda rows and the scalars.
    // Exact equality, because it is the same arithmetic on the same data.
    //
    // Written out by hand deliberately. Reading resScale back and reusing it
    // would pin the *contraction* and say nothing about the weights, which are
    // the whole content of the change; this way a wrong weight fails here.
    N_Vector zero = N_VClone(sys.Y);
    N_Vector scratch = N_VClone(sys.Y);
    N_VConst(0.0, zero);
    sys.residual(sys.t0, sys.Y, zero, scratch);

    const sunindextype n = N_VGetLength(sys.Y);
    const Index perCell = 3 * (k + 1); // nVars = 1, nAux = 0
    const double h = 1.0 / static_cast<double>(nCells);

    double sumsq = 0.0, flat = 0.0;
    double *data = N_VGetArrayPointer(scratch);
    for (sunindextype i = 0; i < n; ++i)
    {
        const double w = (i < nCells * perCell) ? 1.0 / std::sqrt(h) : 1.0;
        sumsq += (w * data[i]) * (w * data[i]);
        flat += data[i] * data[i];
    }

    BOOST_TEST(norm == std::sqrt(sumsq));
    BOOST_TEST(norm > 0.0, "the initial condition is already a steady state; "
                           "this fixture cannot say anything about the norm");

    // ...and it is not the flat norm, nor an RMS one. Trivial arithmetic, but it
    // records which of the three steady_state_tol is quoted in, and a fixture
    // where they happened to coincide would test nothing. h = 0.25 here, so the
    // weighted norm is 2x the flat one.
    BOOST_TEST_MESSAGE("weighted ||F|| = " << norm << "; flat would be " << std::sqrt(flat)
                       << ", RMS " << std::sqrt(flat / static_cast<double>(n)));
    BOOST_TEST(norm != std::sqrt(flat));
    BOOST_TEST(norm != std::sqrt(flat / static_cast<double>(n)));

    // Undamped, whatever the continuation step currently is. steadyResidual --
    // the one KINSOL calls -- adds id*(u - uPrev)/dt, so setting a small dt and
    // a distant anchor changes that by a lot and this by nothing at all. Which
    // is the property that makes it a merit function: the damped residual can
    // be driven to zero by shrinking dt without going anywhere.
    N_VConst(-5.0, sys.uPrev);
    sys.ptcStep = 1e-4;
    BOOST_TEST(sys.steadyResidualNorm() == norm);

    sys.steadyResidual(sys.Y, scratch);
    double damped = 0.0;
    data = N_VGetArrayPointer(scratch);
    for (sunindextype i = 0; i < n; ++i)
        damped += data[i] * data[i];
    BOOST_TEST(std::sqrt(damped) > 10.0 * norm,
               "the damped residual is " << std::sqrt(damped) << " against an undamped "
               << norm << "; this fixture is not separating them");

    N_VDestroy(zero);
    N_VDestroy(scratch);
    sys.destroySundials();
    removeOutput("lifecycle_norm_formula");
}

BOOST_AUTO_TEST_CASE(the_steady_merit_function_does_not_move_with_the_mesh)
{
    // The headline property, and the whole object of weighting the norm. Same
    // physics, same order, same t0, same initial *function* -- only the mesh
    // differs, and ||F|| must not change. That is what lets steady_state_tol mean
    // one thing on every mesh and lets dt cross a remesh, since SER's ratio would
    // otherwise compare two norms measured in different units.
    //
    // This case used to assert the opposite, that the ratios were 1/sqrt(2) --
    // measured 0.70806 and 0.70742 against a flat 2-norm -- and said in its own
    // comment that it should be rewritten to say this once the norm was
    // normalised. It has been. The sqrt(h) it recorded is now divided out by
    // residualWeights().
    //
    // Both halves are asserted, because either alone is worthless. That the
    // states agree is what makes "the same physical state" true; that the norms
    // agree is then a statement about the operator rather than about the state.
    auto normOn = [](Index cells, Vector &state)
    {
        Grid grid(0.0, 1.0, cells);
        TestDiffusion problem(lifecycle_config);
        SystemSolver sys(grid, k, &problem);
        configure(sys, "lifecycle_norm_mesh");

        {
            CapturedOutput quiet;
            sys.initialize();
        }
        const double norm = sys.steadyResidualNorm();
        state = sample(sys);
        sys.destroySundials();
        return norm;
    };

    Vector coarseState, midState, fineState;
    const double coarse = normOn(nCells, coarseState);
    const double mid = normOn(2 * nCells, midState);
    const double fine = normOn(4 * nCells, fineState);

    BOOST_TEST_MESSAGE("||F|| on " << nCells << "/" << 2 * nCells << "/" << 4 * nCells
                       << " cells: " << coarse << " / " << mid << " / " << fine
                       << "; ratios " << mid / coarse << ", " << fine / mid);

    // The same state, to the accuracy the coarse mesh represents it -- these are
    // L2 projections of one function onto two spaces, corrected by IDACalcIC, so
    // they agree to discretisation error and not to round-off. Measured worst
    // case 1.0e-3 relative at k = 2 over 4 against 16 cells.
    for (Index i = 0; i < coarseState.size(); ++i)
    {
        BOOST_TEST(coarseState(i) == midState(i), boost::test_tools::tolerance(5e-3));
        BOOST_TEST(coarseState(i) == fineState(i), boost::test_tools::tolerance(5e-3));
    }

    // ...and the same number for it. Measured 1.111452 / 1.112943 / 1.113438 over
    // a 4x refinement: ratios 1.00134 and 1.00044, i.e. converging on a limit
    // rather than merely being close. Extended to 64 cells while this was being
    // written it reaches 1.111452, 1.112943, 1.113438, 1.113577, 1.113613, whose
    // departures from that last value fall by 3.2, 3.8 then 4.8 -- so the residue
    // is second-order discretisation error in the state, not a leftover mesh
    // factor, and the limit is real.
    //
    // Held to 1%, which is twenty times tighter than the sqrt(h) it replaced
    // could pass -- a flat norm gives 0.708 here, so this discriminates by a
    // factor of 300 against the defect it was written for. Not tightened to the
    // measured 0.13%, because the residue *is* the state's discretisation error
    // and a finer or coarser fixture would legitimately differ.
    BOOST_TEST(mid / coarse == 1.0, boost::test_tools::tolerance(0.01));
    BOOST_TEST(fine / mid == 1.0, boost::test_tools::tolerance(0.01));

    // The successive departures shrink -- measured by a factor of 3.0 here -- which
    // is what separates "converging on a limit" from "close on these three
    // meshes". A norm still carrying a fractional power of h would hold the ratio
    // constant instead, and the 1% window alone would not always catch that.
    const double first = std::abs(mid / coarse - 1.0);
    const double second = std::abs(fine / mid - 1.0);
    BOOST_TEST_MESSAGE("departures from 1: " << first << " then " << second
                       << ", shrinking by " << first / second);
    BOOST_TEST(second < 0.6 * first,
               "the departure from mesh-independence is not shrinking with h ("
               << first << " then " << second << "), so what is left is a mesh "
               "factor rather than discretisation error in the state");

    removeOutput("lifecycle_norm_mesh");
}

BOOST_AUTO_TEST_CASE(KINSOL_measures_the_same_thing_the_continuation_loop_does)
{
    // The coupling that makes normalising steadyResidualNorm alone a mistake.
    // KINSetFuncNormTol is given steady_state_tol, and KINSOL's own stopping test
    // is N_VWL2Norm(fval, fscale). So the way to keep the inner and outer tests
    // measuring one quantity is to hand KINSOL the *same weights* as f_scale
    // rather than to normalise one side -- which is what solveSteadyState does,
    // passing resScale where it used to pass kinScale twice. The agreement is
    // then structural instead of a coincidence that held while both were flat.
    //
    // Newton mode is what makes this checkable exactly: dt is infinite, so
    // steadyResidual's damping term is identically zero and KINSOL's residual
    // *is* the steady one. Under PseudoTransient the two would differ by the
    // damping, and any agreement would be a statement about how far dt had
    // grown rather than about the units.
    const std::string stem = "lifecycle_norm_units";
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(lifecycle_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSteadyMode(SystemSolver::SteadyMode::Newton);
    sys.setSteadyStateTolerance(1e-10);

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.solveSteadyState();
    }

    // The two vectors are different things and both matter. u_scale stays unit --
    // KINSOL uses it for the step-length test and the Newton step clamp, which are
    // about the solution's units -- while f_scale carries the residual weights.
    // Conflating them is the easy mistake here, and the previous version of this
    // case asserted all-ones on the one vector that then served as both.
    double *uScale = N_VGetArrayPointer(sys.kinScale);
    for (sunindextype i = 0; i < N_VGetLength(sys.kinScale); ++i)
        BOOST_TEST(uScale[i] == 1.0);

    // ...and f_scale is not all ones, or this case would be checking that two flat
    // norms agree, which they did before the change and would prove nothing about
    // it. h = 0.25 on this fixture, so the cell rows carry 2.
    double *fScale = N_VGetArrayPointer(sys.resScale);
    const Index perCell = 3 * (k + 1);
    BOOST_TEST(fScale[0] == 2.0);
    BOOST_TEST(fScale[nCells * perCell] == 1.0); // the first lambda row

    double kinNorm = -1.0;
    BOOST_TEST(KINGetFuncNorm(sys.kin_mem, &kinNorm) == KIN_SUCCESS);

    const double ourNorm = sys.steadyResidualNorm();
    BOOST_TEST_MESSAGE("KINSOL's ||F|| = " << kinNorm << "; ours = " << ourNorm);

    // Same vector, same weights, same norm, same point: measured bit-identical.
    // Held to a relative tolerance rather than to equality only because that
    // agreement is not something the code promises.
    //
    // Both are at round-off because TestDiffusion is linear and Newton reaches
    // the answer in one step whatever the tolerance says, so this is not a test
    // that ||F|| is *small*. It is a test of units, and it still discriminates at
    // round-off: an unweighted KINSOL against a weighted merit function would put
    // the factor of 2 above between these two.
    BOOST_TEST(kinNorm == ourNorm, boost::test_tools::tolerance(1e-10));

    sys.destroySundials();
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(the_weighted_norm_is_mesh_independent_only_near_a_solution)
{
    // Where the case above stops holding, pinned so that nobody reads
    // "mesh-independent" as more than it is -- including whoever carries dt across
    // a remesh on the strength of it.
    //
    // The weights turn a pairing <R, phi_i> ~ h R(x_i) back into a density, which
    // is right for a row that *is* such a pairing. Not every row is. Overwrite u
    // with a fixed function and leave sigma, q and lambda stale, and the q and
    // lambda rows hold the trace and derivative terms of the weak form instead --
    // O(1) per row, with no h to divide out, because the 1/h from phi' cancels the
    // h from the measure. Then the flat norm *grows* like 1/sqrt(h) and the
    // weighted one like 1/h, so weighting makes that state worse rather than
    // better.
    //
    // Measured, flat, on 4/8/16/32/64 cells: 4.275, 6.166, 8.834, 12.58, 17.86 --
    // ratios of sqrt(2) up, against sqrt(2) *down* for the consistent state in the
    // case above. Two mechanisms, opposite signs, and which dominates is a
    // property of the state. So no fixed row weighting is mesh-independent
    // everywhere, and this one is not claimed to be: what it fixes is the regime
    // the convergence test fires in, which is the one where the residual is small.
    auto normOn = [](Index cells)
    {
        Grid grid(0.0, 1.0, cells);
        TestDiffusion problem(lifecycle_config);
        SystemSolver sys(grid, k, &problem);
        configure(sys, "lifecycle_norm_farfield");

        {
            CapturedOutput quiet;
            sys.initialize();
        }

        // u alone, so the algebraic rows are left inconsistent. sigma is untouched
        // by this on purpose and stays satisfied -- TestDiffusion's sigma_hat is
        // kappa*q, which does not see u -- so what this exercises is the q and
        // lambda rows specifically.
        sys.y.AssignU([](Index, Position x) { return 0.1 * std::sin(2.0 * M_PI * x); });

        const double norm = sys.steadyResidualNorm();
        sys.destroySundials();
        return norm;
    };

    const double coarse = normOn(nCells);
    const double mid = normOn(2 * nCells);
    const double fine = normOn(4 * nCells);

    BOOST_TEST_MESSAGE("weighted ||F|| far from a solution, "
                       << nCells << "/" << 2 * nCells << "/" << 4 * nCells
                       << " cells: " << coarse << " / " << mid << " / " << fine
                       << "; ratios " << mid / coarse << ", " << fine / mid);

    // Growing, and by about 2 per doubling: the weighted norm goes like 1/h here.
    // Asserted as a bound rather than as a value, because the exact constant is
    // this fixture's and the *direction* is the finding.
    BOOST_TEST(mid > 1.5 * coarse,
               "the far-field norm is not growing with refinement (" << coarse
               << " then " << mid << "), so this fixture no longer separates the "
               "two regimes and the case above is claiming more than is measured");
    BOOST_TEST(fine > 1.5 * mid);

    removeOutput("lifecycle_norm_farfield");
}

BOOST_AUTO_TEST_CASE(an_unarmed_gate_leaves_runSolver_bit_for_bit_unchanged)
{
    // The no-regression guarantee. An AdjointProblem may be attached for other
    // reasons -- solveAdjoint, PyRunner::G -- and with no tolerance set that must
    // not change the run at all, not even by the extra objective evaluations the
    // gate would make.
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion plainProblem(lifecycle_config);
    SystemSolver plain(grid, k, &plainProblem);
    configure(plain, "lifecycle_gate_off_plain");

    TestDiffusion attachedProblem(lifecycle_config);
    SignedIntegralObjective objective(1.0);
    SystemSolver attached(grid, k, &attachedProblem);
    configure(attached, "lifecycle_gate_off_attached");
    attached.setAdjointProblem(&objective);

    {
        CapturedOutput quiet;
        plain.runSolver(T_FINAL);
        attached.runSolver(T_FINAL);
    }

    BOOST_TEST(!plain.wasRejected());
    BOOST_TEST(!attached.wasRejected());

    const Vector a = sample(plain), b = sample(attached);
    for (Index i = 0; i < a.size(); ++i)
        BOOST_TEST(a(i) == b(i), boost::test_tools::tolerance(0.0));
    BOOST_TEST(a.norm() > 1e-8);

    removeOutput("lifecycle_gate_off_plain");
    removeOutput("lifecycle_gate_off_attached");
}

BOOST_AUTO_TEST_SUITE_END()
