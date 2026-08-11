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
// One thing these tests do *not* claim: that a second integration on the same
// solver works. It does not, and it did not before this refactor either. See
// initialize_can_be_called_again_after_destroy and Tests/README.md.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <toml.hpp>

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
    // It stops at initialize() deliberately. Completing a *second integration* on
    // the same SystemSolver does not work -- IDASolve fails with
    // IDA_ERR_FAIL (-3), "the error test failed repeatedly or with |h| = hmin",
    // on the first step of the second run. That is not a consequence of the
    // three-phase split: `main` fails identically through two runSolver() calls,
    // and it has simply never been exercised, because PyRunner::configure()
    // builds a fresh SystemSolver every time (PyRunner.cpp:117) and the
    // standalone binary runs once and exits. Tests/README.md tracks it.
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
    // Two separate reasons, and both have to go for this to change:
    //
    //   * q, sigma and phi are algebraic in this formulation, and IDA's
    //     IDA_YA_YDP_INIT computes algebraic *values* and differential
    //     *derivatives* -- there is no y' for an algebraic component to fetch.
    //     Their true time derivatives follow from differentiating the algebraic
    //     constraints, which nothing here does.
    //   * IDASetId is handed an all-zero id anyway. Solver.cpp builds it with
    //     `isDifferential.u(v).getCoeff(i).second.Constant(k + 1, 1.0)`, and
    //     Eigen's Constant is a *static factory* whose result is discarded, so
    //     that line is a no-op and IDA is told the entire system is algebraic.
    //     See TODO.
    //
    // This is still strictly better than origin/optimize-mode's gate, which
    // evaluated the objective functional on the derivative vector and so was wrong
    // about the u term too whenever g was nonlinear. But it is less than the full
    // derivative, and the comment on SystemSolver::dGdt says so.
    Grid grid(0.0, 1.0, nCells);

    TestDiffusion problem(lifecycle_config);
    SignedIntegralObjective objective(1.0);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "lifecycle_gate_consistent");
    sys.setAdjointProblem(&objective);
    sys.setObjectiveDecreaseTolerance(1e-12);

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
