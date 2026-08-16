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
#include "../../FieldModel.hpp"

#include <ida/ida.h>
#include <netcdf>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <memory>
#include <numbers>
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

// ------------------------------------------------------------ the coupled pair --
//
// The lifecycle properties above have to hold with a field model attached too,
// and two of them are only reachable that way: a model may cache things across
// a run, and psi has to survive a restart.

// Linear diffusion whose diffusivity *is* the field model's geometry slot. The
// coupling has to reach u, or a difference in psi would leave the solution
// alone and the comparisons below would be measuring the uncoupled problem.
class GeometricDiffusion : public TransportSystem
{
public:
    GeometricDiffusion()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}})
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.geom(0) * s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0);
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    void dSigmaFn_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.q(0);
    }

    Value InitialValue(Index, Position x) const override
    {
        return std::sin(std::numbers::pi * x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        return std::numbers::pi * std::cos(std::numbers::pi * x);
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
};

// A field model that warm-starts from the psi it last saw, and clears that cache
// in resetForRun().
//
//     R = dpsi/dt - ( Int u dx - psi ),     psi(0) = 0.5
//
// Three properties, each load-bearing for one of the two cases below.
//
// **The row is differential**, so IDA_YA_YDP_INIT holds psi's *value* fixed and
// whatever InitialFieldValue (or a restart file) supplies is the answer rather
// than a guess. An algebraic psi would be re-solved out of the restored
// transport state by IDACalcIC, and the restart case would then pass without
// ever reading psi off disk -- vacuously, and invisibly so.
//
// **It reads the transport state**, so the coupling is two-way: dR/du is the
// quadrature weights, exactly as ManufacturedField's is.
//
// **The cache is a tripwire.** initialize() calls resetForRun() and *then*
// setInitialConditions, so with the reset honoured the warm start never fires
// and the model is a pure function of its declaration -- which is the point. A
// resetForRun() that did nothing would hand the second run its predecessor's
// final psi, which is the RF_cellwise trap one level out. Nothing else in the
// suite would notice: the run completes, and the answer is plausible.
class WarmStartingField : public FieldModel
{
public:
    static constexpr double PSI0 = 0.5;

    WarmStartingField() : FieldModel(buildSpec()) {}

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.dofs = {{"psi", "the field unknown", "1", /* differential */ true}};
        s.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}};
        s.label = "x";
        // Not the default "Field": the netCDF case below has to be able to tell
        // a group named from the spec from one named by a string literal.
        s.name = "WarmStart";
        return s;
    }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &dpsidt,
                       GlobalState const &states, std::vector<Position> const &,
                       Vector const &weights, Time) override
    {
        double integral = 0.0;
        for (Index j = 0; j < weights.size(); ++j)
            integral += weights(j) * states[j].u(0);
        out(0) = dpsidt(0) - (integral - psi(0));

        lastPsi = psi(0);
        warm = true;
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + psi(0) * std::cos(std::numbers::pi * x);
    }

    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        out(0, 0) = std::cos(std::numbers::pi * x);
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef dRddpsidt, Vector const &, Vector const &,
                            GlobalState const &, std::vector<Position> const &,
                            Vector const &weights, Time) override
    {
        dRdpsi(0, 0) = 1.0;
        dRddpsidt(0, 0) = 1.0;
        // Through Variable(), the whole (nVars, nPoints) matrix: dR[0][j] hands
        // back a State *by value* and the write would be discarded.
        dR[0].Variable().row(0) = -weights.transpose();
    }

    void InitialFieldValue(VectorRef out) override { out(0) = warm ? lastPsi : PSI0; }

    void resetForRun() override
    {
        warm = false;
        lastPsi = 0.0;
    }

    /// True once a residual has been evaluated in this run, i.e. the tripwire is
    /// armed. Asked between runs, so that a reuse test cannot go quietly vacuous
    /// if the model stops being consulted at all.
    bool isWarm() const { return warm; }

private:
    bool warm = false;
    double lastPsi = 0.0;
};

/// Everything a coupled run owns, in destruction order: the solver first, so its
/// SUNContext outlives nothing that was made in it.
struct CoupledSolver
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<GeometricDiffusion> problem;
    std::shared_ptr<WarmStartingField> field;
    std::unique_ptr<SystemSolver> sys;
};

/// A coupled solver configured like `configure` above, plus the field model.
CoupledSolver makeCoupledSolver(Index cells, Index order, std::string const &stem)
{
    CoupledSolver c;
    c.grid = std::make_unique<Grid>(0.0, 1.0, cells);
    c.problem = std::make_unique<GeometricDiffusion>();
    c.field = std::make_shared<WarmStartingField>();
    c.sys = std::make_unique<SystemSolver>(*c.grid, order, c.problem.get());

    c.sys->setTau(1.0);
    c.sys->resetCoeffs();
    c.sys->setInputFile(stem);
    c.sys->setOutputCadence(T_FINAL);
    c.sys->setNOutput(11);
    c.sys->setInitialTime(0.0);
    c.sys->setMinStepSize(1e-12);
    c.sys->setTolerances({1e-8}, 1e-6);
    c.sys->setFieldModel(c.field);
    return c;
}

/// u at five interior points followed by every field unknown, out of yJac -- the
/// only copy of the solution that outlives destroySundials(). psi is in here
/// deliberately: a defect that moved only the field block and left u alone would
/// otherwise be invisible.
Vector coupledSample(SystemSolver &sys)
{
    Vector out(5 + sys.getFieldDOF());
    for (Index i = 0; i < 5; ++i)
        out(i) = sys.getSolution().u(0)(0.1 + 0.2 * i);
    for (Index f = 0; f < sys.getFieldDOF(); ++f)
        out(5 + f) = sys.getSolution().Field(f);
    return out;
}

double maxAbsDifference(Vector const &a, Vector const &b)
{
    BOOST_REQUIRE_EQUAL(a.size(), b.size());
    return (a - b).cwiseAbs().maxCoeff();
}

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

BOOST_AUTO_TEST_CASE(a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit)
{
    // At exactly zero tolerance, like the uncoupled version above. The tolerance
    // is the point: the last defect there left the second run completing,
    // plausible, and wrong in the eleventh digit. A field model that caches
    // anything across runs and does not reset it fails here and nowhere else --
    // initialize() skips initialiseMatrices() when already initialised, so
    // resetForRun() is called from the unconditional part of it instead, and
    // nothing but this says that it is.
    //
    // WarmStartingField is exactly that model: it remembers the psi it last saw
    // and starts from it. With resetForRun() honoured the memory is cleared
    // before setInitialConditions asks, so the second run starts from PSI0 like
    // the first; without it the second run starts from the first run's *final*
    // psi, which is measurably different -- see the report for this task.
    auto fresh = makeCoupledSolver(8, 2, "lifecycle_coupled_fresh");
    auto reused = makeCoupledSolver(8, 2, "lifecycle_coupled_reused");

    bool armedBetweenRuns = false;
    {
        CapturedOutput quiet;
        fresh.sys->runSolver(T_FINAL);

        // Three runs, not two: the second is the one a missing reset breaks, and
        // a third catches anything that accumulates rather than simply differing
        // from the first.
        reused.sys->runSolver(T_FINAL);

        // The cache really was populated by the run that just finished, so the
        // next initialize() has something to clear. Without this the case would
        // go quietly vacuous the day FieldResidual stops being called.
        armedBetweenRuns = reused.field->isWarm();

        reused.sys->runSolver(T_FINAL);
        reused.sys->runSolver(T_FINAL);
    }

    BOOST_TEST(armedBetweenRuns,
               "the field model's per-run cache was never populated, so this case would "
               "pass whether or not resetForRun() is called");

    BOOST_TEST(maxAbsDifference(coupledSample(*reused.sys), coupledSample(*fresh.sys)) == 0.0,
               boost::test_tools::tolerance(0.0));

    // Not vacuous, and the coupling is live: psi has moved away from its initial
    // value, so geometry -- and therefore u -- is not the same as an uncoupled
    // run's would have been.
    const Vector want = coupledSample(*fresh.sys);
    BOOST_TEST(want.head(5).norm() > 1e-8);
    BOOST_TEST(std::abs(want(5) - WarmStartingField::PSI0) > 1e-8);

    removeOutput("lifecycle_coupled_fresh");
    removeOutput("lifecycle_coupled_reused");
}

BOOST_AUTO_TEST_CASE(psi_round_trips_through_a_restart)
{
    // The restart file has carried psi since the DoF accounting stopped being
    // open-coded -- Y is written at y.getDoF() bytes and the field block is last.
    // What it did not carry was any record of *how much* of Y that is, so the
    // reader could not shape a DGSoln over it and setInitialConditions refused
    // the combination outright. RestartData/nField is that record.
    //
    // The field DOF is differential, which is what stops this passing vacuously:
    // IDA_YA_YDP_INIT solves for algebraic *values*, so an algebraic psi would be
    // recomputed from the restored transport state and would come back right
    // whatever the file said. A differential value is held fixed, so the number
    // compared below is the number that was read off disk.
    const std::string stem = "lifecycle_coupled_restart";
    const double T_RESTART = 0.25;

    auto original = makeCoupledSolver(8, 2, stem);
    original.sys->setOutputCadence(T_RESTART);

    {
        CapturedOutput quiet;
        original.sys->initialize();
        original.sys->integrate(T_RESTART);
    }
    const double psiBefore = original.sys->getSolution().Field(0);
    const Vector uBefore = coupledSample(*original.sys).head(5);
    {
        CapturedOutput quiet;
        original.sys->destroySundials();
    }

    // Not the initial value: a run that never moved psi would round-trip through
    // anything, including a file that stored nothing at all.
    BOOST_TEST(std::abs(psiBefore - WarmStartingField::PSI0) > 1e-6);

    // ---- read it back the way runManta does ----------------------------------

    netCDF::NcFile restartFile;
    BOOST_REQUIRE_NO_THROW(
        restartFile.open(stem + ".restart.nc", netCDF::NcFile::FileMode::read));

    netCDF::NcGroup gridGroup = restartFile.getGroup("Grid");
    std::vector<Position> cellBoundaries(gridGroup.getDim("Index").getSize());
    gridGroup.getVar("CellBoundaries").getVar(cellBoundaries.data());
    Index order = 0;
    gridGroup.getVar("PolyOrder").getVar(&order);

    netCDF::NcGroup restartGroup = restartFile.getGroup("RestartData");
    const size_t nDOF_file = restartGroup.getDim("nDOF").getSize();
    std::vector<double> Y(nDOF_file), dYdt(nDOF_file);
    restartGroup.getVar("Y").getVar(Y.data());
    restartGroup.getVar("dYdt").getVar(dYdt.data());

    // The point of the format change. Without it a reader has to infer nField by
    // subtracting the uncoupled DoF formula from nDOF, which is the arithmetic
    // that was wrong in three places before y.getDoF() became the authority.
    netCDF::NcVar nFieldVar = restartGroup.getVar("nField");
    BOOST_REQUIRE(!nFieldVar.isNull());
    int nField_file = -1;
    nFieldVar.getVar(&nField_file);
    restartFile.close();

    BOOST_TEST(nField_file == 1);

    Grid restartGrid(cellBoundaries);
    GeometricDiffusion restartedProblem;
    restartedProblem.setRestartValues(Y, dYdt, restartGrid, order, nField_file);

    auto field = std::make_shared<WarmStartingField>();
    SystemSolver restarted(restartGrid, order, &restartedProblem);
    restarted.setTau(1.0);
    restarted.resetCoeffs();
    restarted.setInputFile(stem + "_resumed");
    // Not T_RESTART, and this is a trap rather than a preference. Solver.cpp
    // hands the output *cadence* to IDACalcIC's `tout1` parameter, which
    // SUNDIALS documents as the first output *time* -- it uses it only for the
    // direction and rough scale of t, and the two coincide only when t0 is zero.
    // Every fixture in the tree starts at zero, so nothing notices; a resumed run
    // whose cadence equals its t_initial gets tdist = 0 and IDACalcIC returns
    // IDA_ILL_INPUT (-22), "tout1 too close to t0". That is not about field
    // models -- it reproduces on main with no coupling at all, from any config
    // with `t_initial = delta_t` -- so it is reported rather than fixed here.
    restarted.setOutputCadence(T_RESTART / 5.0);
    restarted.setNOutput(11);
    restarted.setInitialTime(T_RESTART);
    restarted.setMinStepSize(1e-12);
    restarted.setTolerances({1e-8}, 1e-6);
    restarted.setWriteOutput(false);
    restarted.setFieldModel(field);

    // initialize() alone: yJac holds the initial condition after it, and the
    // initial condition is what a restart is. Integrating further would only add
    // the resumed run's own error to the comparison.
    {
        CapturedOutput quiet;
        restarted.initialize();
    }

    // 1e-12 is slack, not a measured bound: this comes back *bit* exact, and
    // structurally so. psi is a differential value, IDA_YA_YDP_INIT holds every
    // differential value fixed, and netCDF stores a double as a double -- so the
    // number below is the one the file carries, not one re-derived from it. If
    // this ever stops being exact, something is recomputing psi rather than
    // resuming it, and the tolerance is not the thing to adjust.
    BOOST_CHECK_SMALL(std::abs(restarted.getSolution().Field(0) - psiBefore), 1e-12);

    // ...and the transport state came back with it, so the psi above is the psi
    // of the same solution rather than of some other one.
    for (Index i = 0; i < 5; ++i)
        BOOST_CHECK_SMALL(std::abs(restarted.getSolution().u(0)(0.1 + 0.2 * i) - uBefore(i)),
                          1e-10);

    {
        CapturedOutput quiet;
        restarted.destroySundials();
    }
    removeOutput(stem);
    removeOutput(stem + "_resumed");
}

BOOST_AUTO_TEST_CASE(a_coupled_run_writes_the_field_group)
{
    // The netCDF group is the only part of the coupling a user ever sees, and
    // there is deliberately no regression case that would exercise it -- no field
    // model is registered, so no `.conf` can select one. This is its cover.
    const std::string stem = "lifecycle_coupled_output";
    auto run = makeCoupledSolver(8, 2, stem);

    {
        CapturedOutput quiet;
        run.sys->runSolver(T_FINAL);
    }

    netCDF::NcFile out;
    BOOST_REQUIRE_NO_THROW(out.open(stem + ".nc", netCDF::NcFile::FileMode::read));

    // Named from the spec, not from a string literal in NetCDFIO.cpp.
    netCDF::NcGroup group = out.getGroup("WarmStart");
    BOOST_REQUIRE(!group.isNull());

    // The coordinate the geometry is expressed against, so a run records what its
    // x meant.
    std::string label;
    group.getAtt("label").getValues(label);
    BOOST_TEST(label == "x");

    // psi is a time series -- it has no x dependence -- and the geometry slot is
    // a spatial field, like u. Getting those two the wrong way round is the
    // mistake this checks: both would still be "present".
    netCDF::NcVar psi = group.getVar("psi");
    BOOST_REQUIRE(!psi.isNull());
    BOOST_TEST(psi.getDimCount() == 1);

    netCDF::NcVar g = group.getVar("g");
    BOOST_REQUIRE(!g.isNull());
    BOOST_TEST(g.getDimCount() == 2);

    std::string units;
    psi.getAtt("units").getValues(units);
    BOOST_TEST(units == "1");

    const size_t nTimes = psi.getDim(0).getSize();
    BOOST_TEST(nTimes >= 2u);

    std::vector<double> psiSeries(nTimes);
    psi.getVar(psiSeries.data());
    BOOST_TEST(psiSeries[0] == WarmStartingField::PSI0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(std::abs(psiSeries[nTimes - 1] - WarmStartingField::PSI0) > 1e-8);

    // g = 1 + psi cos(pi x) at the final psi, sampled at the output grid. Checked
    // at the ends, where cos(pi x) is +-1 and the two would differ by 2 psi if
    // the geometry were written at the wrong time or the wrong psi.
    const size_t nx = g.getDim(1).getSize();
    std::vector<double> gFinal(nx);
    g.getVar({nTimes - 1, 0}, {1, nx}, gFinal.data());
    BOOST_TEST(gFinal.front() == 1.0 + psiSeries[nTimes - 1], boost::test_tools::tolerance(1e-12));
    BOOST_TEST(gFinal.back() == 1.0 - psiSeries[nTimes - 1], boost::test_tools::tolerance(1e-12));

    out.close();
    removeOutput(stem);
}

BOOST_AUTO_TEST_SUITE_END()
