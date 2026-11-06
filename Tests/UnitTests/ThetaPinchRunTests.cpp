#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "../../PhysicsCases/AutodiffAdjointProblem.hpp"
#include "../../PhysicsCases/ThetaPinchDiffusion.hpp"
#include "../../FieldModelEquilibrium.hpp"
#include "../../PhysicsCases/ThetaPinchEquilibrium.hpp"
#include "../../PhysicsCases/ThetaPinchPressureBalance.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <netcdf>
#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

int static_residual(sunrealtype tres, N_Vector Y, N_Vector dYdt, N_Vector resval, void *user_data);

/*
    The registered theta-pinch pair, run end to end: what a coupled run does
    that the Jacobian checks cannot see -- a restart, a reused solver, the
    output, the two field solves over a whole integration.
 */

namespace
{

using namespace toml::literals::toml_literals;

// A pressure high enough that the field is far from vacuum: B on axis falls to
// about 0.85 of the wall value by t = 0.5.
const toml::value thetaPinchConfig = u8R"(
    [ThetaPinchDiffusion]
    Temperature = 0.3
    D0 = 0.5
    Gamma = 2.0
)"_toml;

struct Options
{
    Index nCells = 6;
    Index k = 3;
    Index intervalsPerCell = 2;
    bool iterative = true;
    // Tier 0: the same model through FieldModelEquilibrium, FieldCoupling =
    // "consistent".
    bool consistent = false;
    bool writeOutput = false;
    double cadence = 0.25;
    double t0 = 0.0;
    // ThetaPinchDiffusion's GeometryDerivatives: Analytic, Differenced or None.
    std::string derivatives = "Analytic";
    // Tier 1: the model through FieldModelEquilibrium, FieldCoupling =
    // "reduced", with its own response or the differenced default.
    bool reduced = false;
    bool differencedResponse = false;
    // FieldSolve = krylov, at tier 1 or 2; overrides `iterative`.
    bool krylov = false;
    // Count what the run asks of the case (CountingThetaPinch).
    bool count = false;
    // Tier 1 with a transposed response that does nothing.
    bool blindTranspose = false;
    // The native model, ThetaPinchPressureBalance, in place of the field model:
    // at tier 1 unless `consistent`, with its own response unless
    // `differencedResponse`.
    bool native = false;
};

/// The tier-1 adapter, counting its equilibrium solves -- including the ones
/// the differenced response makes -- and optionally blind in the transposed
/// direction, which is what an adjoint that never applied G'^T would compute.
class CountingEquilibrium : public FieldModelEquilibrium
{
public:
    using FieldModelEquilibrium::FieldModelEquilibrium;

    std::unique_ptr<Equilibrium> Solve(GlobalState const &states, std::vector<Position> const &points,
                                       Vector const &weights, Time t, Equilibrium const *warm) override
    {
        ++solves;
        return FieldModelEquilibrium::Solve(states, points, weights, t, warm);
    }
    void applyResponseTranspose(GlobalState &dp, Equilibrium const &e, GlobalState const &states, Matrix const &dg,
                                std::vector<Position> const &inputPoints, Vector const &weights, Time t,
                                std::vector<Position> const &geometryPoints) override
    {
        if (!blind)
            FieldModelEquilibrium::applyResponseTranspose(dp, e, states, dg, inputPoints, weights, t,
                                                          geometryPoints);
    }

    long solves = 0;
    bool blind = false;
};

/// ThetaPinchDiffusion counting the physics a run asks for, in the cost
/// metric: points of batched value evaluations, points of batched derivative
/// evaluations, and nodes at which the analytic geometry hooks were called (one
/// dSigmaFn_dGeometry per node per Jacobian build).
class CountingThetaPinch : public ThetaPinchDiffusion
{
public:
    using ThetaPinchDiffusion::ThetaPinchDiffusion;

    PhysicsOutput ComputePhysics(GlobalState const &s, std::vector<Position> const &x, Time t) override
    {
        values += static_cast<long>(x.size());
        return ThetaPinchDiffusion::ComputePhysics(s, x, t);
    }
    void ComputePhysicsDerivatives(std::array<std::reference_wrapper<GlobalStateMatrix>, NPHYSICS_FUNCTIONS> &&out,
                                   GlobalState const &s, std::vector<Position> const &x, Time t) override
    {
        derivatives += static_cast<long>(x.size());
        ThetaPinchDiffusion::ComputePhysicsDerivatives(std::move(out), s, x, t);
    }
    void dSigmaFn_dGeometry(Index i, VectorRef v, const State &s, Position x, Time t) override
    {
        if (i == 0)
            ++geometryHookNodes;
        ThetaPinchDiffusion::dSigmaFn_dGeometry(i, v, s, x, t);
    }

    /// Values at 1 and derivatives at 2 per point; the hooks priced separately.
    long cost() const { return values + 2 * derivatives; }

    long values = 0, derivatives = 0, geometryHookNodes = 0;
};

struct Run
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<ThetaPinchDiffusion> problem;
    std::shared_ptr<ThetaPinchEquilibrium> field;
    std::shared_ptr<CountingEquilibrium> equilibrium;
    std::shared_ptr<ThetaPinchPressureBalance> native;
    std::unique_ptr<SystemSolver> sys;
};

Run makeRun(Options const &o, std::string const &stem,
            std::function<void(ThetaPinchDiffusion &, Grid const &)> const &prepare = {})
{
    Run r;
    r.grid = std::make_unique<Grid>(0.0, 1.0, o.nCells);
    toml::value config = thetaPinchConfig;
    config.as_table().at("ThetaPinchDiffusion").as_table()["GeometryDerivatives"] = o.derivatives;
    if (o.count)
        r.problem = std::make_unique<CountingThetaPinch>(config, *r.grid);
    else
        r.problem = std::make_unique<ThetaPinchDiffusion>(config, *r.grid);
    if (prepare)
        prepare(*r.problem, *r.grid);
    r.field = std::make_shared<ThetaPinchEquilibrium>(
        ThetaPinchEquilibrium::Parameters{.intervalsPerCell = o.intervalsPerCell}, *r.grid);
    r.sys = std::make_unique<SystemSolver>(*r.grid, o.k, r.problem.get());

    SystemSolver &sys = *r.sys;
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(o.cadence);
    sys.setNOutput(11);
    sys.setInitialTime(o.t0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(o.writeOutput);
    sys.setWriteDatFile(false);
    if (o.native)
    {
        r.native = std::make_shared<ThetaPinchPressureBalance>(
            ThetaPinchPressureBalance::Parameters{.differencedResponse = o.differencedResponse}, *r.grid);
        sys.setEquilibriumModel(r.native, o.consistent ? SystemSolver::EquilibriumCoupling::Consistent
                                                       : SystemSolver::EquilibriumCoupling::Reduced);
    }
    else if (o.consistent)
        sys.setEquilibriumModel(std::make_shared<FieldModelEquilibrium>(r.field));
    else if (o.reduced)
    {
        r.equilibrium = std::make_shared<CountingEquilibrium>(r.field, o.differencedResponse);
        r.equilibrium->blind = o.blindTranspose;
        sys.setEquilibriumModel(r.equilibrium, SystemSolver::EquilibriumCoupling::Reduced);
    }
    else
        sys.setFieldModel(r.field);
    if (!o.consistent)
        sys.setFieldSolveMode(o.krylov      ? SystemSolver::FieldSolveMode::Krylov
                              : o.iterative ? SystemSolver::FieldSolveMode::Iterative
                                            : SystemSolver::FieldSolveMode::Exact);
    return r;
}

/// N at five interior points, then every field unknown, out of yJac -- the copy
/// of the solution that outlives destroySundials().
Vector sample(SystemSolver &sys)
{
    Vector out(5 + sys.getFieldDOF());
    for (Index i = 0; i < 5; ++i)
        out(i) = sys.getSolution().u(0)(0.1 + 0.2 * i);
    for (Index f = 0; f < sys.getFieldDOF(); ++f)
        out(5 + f) = sys.getSolution().Field(f);
    return out;
}

void removeOutput(std::string const &stem)
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);
}

struct RestartFile
{
    std::vector<Position> cellBoundaries;
    Index order = 0;
    std::vector<double> Y, dYdt;
    int nField = -1;
};

/// The restart file read the way runManta reads it.
RestartFile readRestart(std::string const &stem)
{
    RestartFile f;
    netCDF::NcFile file(stem + ".restart.nc", netCDF::NcFile::FileMode::read);
    netCDF::NcGroup grid = file.getGroup("Grid");
    f.cellBoundaries.resize(grid.getDim("Index").getSize());
    grid.getVar("CellBoundaries").getVar(f.cellBoundaries.data());
    grid.getVar("PolyOrder").getVar(&f.order);
    netCDF::NcGroup data = file.getGroup("RestartData");
    const size_t n = data.getDim("nDOF").getSize();
    f.Y.resize(n);
    f.dYdt.resize(n);
    data.getVar("Y").getVar(f.Y.data());
    data.getVar("dYdt").getVar(f.dYdt.data());
    data.getVar("nField").getVar(&f.nField);
    return f;
}

/// Objectives that read geometry. 0: the field alone, B(x)(1 + x^2), weighted
/// so that dg/dB varies across a cell; 1: N V'(x), transport and geometry
/// together.
void addGeometricObjectives(AutodiffAdjointProblem &objective)
{
    objective.addGWithGeometry(
        [](Position x, RealVector &, RealVector &, RealVector &, RealVector &, RealVector &geom) -> Real
        { return geom(ThetaPinchDiffusion::Bfield) * (1.0 + x * x); });
    objective.addGWithGeometry(
        [](Position, RealVector &u, RealVector &, RealVector &, RealVector &, RealVector &geom) -> Real
        { return u(0) * geom(ThetaPinchDiffusion::Vprime); });
    objective.setNp(2);
}

} // namespace

BOOST_AUTO_TEST_SUITE(theta_pinch_run_tests)

BOOST_AUTO_TEST_CASE(a_reused_theta_pinch_solver_matches_a_fresh_one_bit_for_bit)
{
    auto fresh = makeRun({}, "theta_pinch_fresh");
    auto reused = makeRun({}, "theta_pinch_reused");
    {
        CapturedOutput quiet;
        fresh.sys->runSolver(0.5);
        reused.sys->runSolver(0.5);
        reused.sys->runSolver(0.5);
    }
    const Vector want = sample(*fresh.sys), got = sample(*reused.sys);
    BOOST_TEST((want - got).cwiseAbs().maxCoeff() == 0.0, boost::test_tools::tolerance(0.0));

    // Not vacuous: the field has left the vacuum state, V' = pi everywhere.
    BOOST_TEST((want.tail(fresh.sys->getFieldDOF() - 1).array() - M_PI).abs().maxCoeff() > 1e-2);
}

BOOST_AUTO_TEST_CASE(exact_and_iterative_field_solves_give_the_same_run)
{
    // Not merely the same answer to tolerance: the iterative solve stops at
    // FieldSolveTolerance, so Newton sees a slightly different correction, but
    // the step sequence is set by IDA's error test, which that difference does
    // not reach. Measured: the same 166 steps, and the same answer to 1.2e-11.
    //
    // This is also the run that pins the sweep's test as *relative* to the
    // increment. An absolute bound of 1e-2 error-weight units -- far inside
    // the 0.33 at which Newton stops -- took 658 steps here and differed by
    // 2e-7: late in a Newton solve the update is itself that small, and IDA's
    // rate estimate read the sweep's error as divergence.
    auto exact = makeRun({.iterative = false}, "theta_pinch_exact");
    auto iterative = makeRun({.iterative = true}, "theta_pinch_iterative");
    std::string logExact, logIterative;
    {
        CapturedOutput capture;
        exact.sys->runSolver(0.5);
        logExact = capture.text();
    }
    {
        CapturedOutput capture;
        iterative.sys->runSolver(0.5);
        logIterative = capture.text();
    }
    auto steps = [](std::string const &log)
    {
        const auto at = log.find("Total Number of Timesteps");
        return at == std::string::npos ? std::string("?") : log.substr(at, log.find('\n', at) - at);
    };
    const Vector a = sample(*exact.sys), b = sample(*iterative.sys);
    BOOST_TEST_MESSAGE("exact: " << steps(logExact) << "; iterative: " << steps(logIterative)
                                 << "; max difference " << (a - b).cwiseAbs().maxCoeff() << " against max "
                                 << a.cwiseAbs().maxCoeff());
    BOOST_TEST((a - b).cwiseAbs().maxCoeff() < 1e-9 * a.cwiseAbs().maxCoeff());
}

/// N and B at five interior points of the final state, from either coupling.
Vector sampleWithGeometry(SystemSolver &sys)
{
    Vector out(10), g(4);
    for (Index i = 0; i < 5; ++i)
    {
        const double x = 0.1 + 0.2 * i;
        out(i) = sys.getSolution().u(0)(x);
        g.setZero();
        sys.geometryAt(g, sys.getSolution(), x, 0.5);
        out(5 + i) = g(ThetaPinchEquilibrium::Bfield);
    }
    return out;
}

BOOST_AUTO_TEST_CASE(a_tier_zero_theta_pinch_reaches_the_coupled_answer)
{
    // The same DAE, solved without differentiating through the field: Newton
    // converges more slowly and costs more residual evaluations, but each step
    // is held to the same tolerances, so the two runs agree to the integration
    // error.
    auto coupled = makeRun({}, "theta_pinch_tier2");
    auto tier0 = makeRun({.consistent = true}, "theta_pinch_tier0");
    {
        CapturedOutput quiet;
        coupled.sys->runSolver(0.5);
        tier0.sys->runSolver(0.5);
    }
    const Vector a = sampleWithGeometry(*coupled.sys), b = sampleWithGeometry(*tier0.sys);
    const double dN = (a.head(5) - b.head(5)).cwiseAbs().maxCoeff() / a.head(5).cwiseAbs().maxCoeff();
    const double dB = (a.tail(5) - b.tail(5)).cwiseAbs().maxCoeff();
    BOOST_TEST_MESSAGE("tier 0 against coupled at t = 0.5: N " << dN << " relative, B " << dB
                                                                << "; equilibrium solves "
                                                                << tier0.sys->getEquilibriumSolves());
    // At rtol 1e-6 / atol 1e-8, measured 4.3e-7 and 4.1e-8 as the runs default
    // (AlgebraicMax, algebraic error suppressed); 2.9e-6 and 1.1e-7 with the
    // algebraic rows in the error test, 8.8e-8 and 6.6e-9 under Pointwise:
    // integration error each way, and which is smallest is a matter of which
    // steps each run happened to take.
    BOOST_TEST(dN < 1e-5);
    BOOST_TEST(dB < 1e-6);
    // Not vacuous: the field has left the vacuum, B = 1.
    BOOST_TEST((a.tail(5).array() - 1.0).abs().maxCoeff() > 1e-2);
    BOOST_TEST(tier0.sys->getFieldDOF() == 0);
}

BOOST_AUTO_TEST_CASE(a_tier_zero_steady_solve_reaches_the_coupled_answer)
{
    for (auto mode : {SystemSolver::SteadyMode::Newton, SystemSolver::SteadyMode::PseudoTransient})
    {
        auto coupled = makeRun({}, "theta_pinch_tier2_ss");
        auto tier0 = makeRun({.consistent = true}, "theta_pinch_tier0_ss");
        for (Run *r : {&coupled, &tier0})
        {
            r->sys->setSteadyStateTolerance(1e-8);
            r->sys->setSteadyMode(mode);
            CapturedOutput quiet;
            r->sys->runSolver(0.5);
        }
        const Vector a = sampleWithGeometry(*coupled.sys), b = sampleWithGeometry(*tier0.sys);
        const double dN = (a.head(5) - b.head(5)).cwiseAbs().maxCoeff() / a.head(5).cwiseAbs().maxCoeff();
        BOOST_TEST_MESSAGE("steady tier 0 against coupled: N " << dN << " relative");
        // Measured 3.6e-10 under Newton and 6.4e-13 under PseudoTransient.
        BOOST_TEST(dN < 1e-8);
    }
}

// The solver differencing the case's values in each geometry slot, in place of
// the case's own derivatives, changes the Jacobian by the difference's error and
// nothing else -- so the same steps, and the same answer to far below the run's
// tolerance -- at nGeometry + 1 batched evaluations per Jacobian build.
BOOST_AUTO_TEST_CASE(differenced_geometry_derivatives_give_the_analytic_run)
{
    auto steps = [](std::string const &log)
    {
        const auto at = log.find("Total Number of Timesteps");
        return at == std::string::npos ? std::string("?") : log.substr(at, log.find('\n', at) - at);
    };

    auto analytic = makeRun({}, "theta_pinch_analytic");
    auto differenced = makeRun({.derivatives = "Differenced"}, "theta_pinch_differenced");
    std::string logA, logD;
    {
        CapturedOutput capture;
        analytic.sys->runSolver(0.5);
        logA = capture.text();
    }
    {
        CapturedOutput capture;
        differenced.sys->runSolver(0.5);
        logD = capture.text();
    }
    const Vector a = sample(*analytic.sys), d = sample(*differenced.sys);
    const double transient = (a - d).cwiseAbs().maxCoeff() / a.cwiseAbs().maxCoeff();
    BOOST_TEST_MESSAGE("transient: analytic " << steps(logA) << "; differenced " << steps(logD)
                                              << "; relative difference " << transient);
    BOOST_TEST(steps(logA) == steps(logD));
    BOOST_TEST(transient < 1e-9);

    auto newtonA = makeRun({}, "theta_pinch_analytic_ss");
    auto newtonD = makeRun({.derivatives = "Differenced"}, "theta_pinch_differenced_ss");
    for (Run *r : {&newtonA, &newtonD})
    {
        r->sys->setSteadyStateTolerance(1e-8);
        r->sys->setSteadyMode(SystemSolver::SteadyMode::Newton);
        CapturedOutput quiet;
        r->sys->runSolver(0.5);
    }
    const Vector sa = sample(*newtonA.sys), sd = sample(*newtonD.sys);
    const double steady = (sa - sd).cwiseAbs().maxCoeff() / sa.cwiseAbs().maxCoeff();
    BOOST_TEST_MESSAGE("steady Newton: analytic " << newtonA.sys->lastSteadyStats().newtonIters
                                                  << " Newton iterations, differenced "
                                                  << newtonD.sys->lastSteadyStats().newtonIters
                                                  << "; relative difference " << steady);
    BOOST_TEST(steady < 1e-9);

    // The plan says what a build costs: one batched evaluation per geometry slot
    // and one at the state.
    const EvaluationPlan plan = differenced.sys->evaluationPlan();
    const auto site = std::find_if(plan.sites.begin(), plan.sites.end(), [](EvaluationSite const &s)
                                   { return s.kind == EvaluationKind::FieldCoupling; });
    BOOST_REQUIRE(site != plan.sites.end());
    BOOST_TEST(site->calls == differenced.sys->getGeometrySlots() + 1);
}

/// A "Total Number of ..." count from a run's report, as a number.
long countIn(std::string const &log, std::string const &what)
{
    const auto at = log.find("Total Number of " + what);
    if (at == std::string::npos)
        return -1;
    return std::stol(log.substr(log.find(':', at) + 1));
}

long stepsIn(std::string const &log) { return countIn(log, "Timesteps"); }

// Tier 1 carries the coupling in its Jacobian, so IDA's Newton converges as it
// does at tier 2 rather than linearly as at tier 0 -- with the model's own
// response or the differenced default, under either solve. What that saves is
// Newton iterations, i.e. residual evaluations, each an equilibrium solve.
// Measured, with the algebraic error suppressed as a run with a field is by
// default: 128 steps and 197 residuals under iterative, 146 and 208 under
// krylov, against tier 2's 126 and 166 and tier 0's 136 and 262; N within
// 5.7e-8 and 2.7e-7 of tier 2's.
//
// Steps alone no longer separate the tiers. With the algebraic rows in IDA's
// error test, tier 0's linearly converged algebraic unknowns read as local error
// and it crawled -- 321 steps against tier 1's 163 -- but suppressed, its
// Newton's slower convergence costs residuals and not steps. Hence the step
// bound against tier 2 alone, at 1.5 times since step counts move by tens of
// percent with the last bits of the arithmetic, and residuals against tier 0.
BOOST_AUTO_TEST_CASE(a_reduced_theta_pinch_reaches_the_coupled_answer)
{
    auto run = [](Options const &o, std::string const &stem, long &steps, long &residuals)
    {
        auto r = makeRun(o, stem);
        CapturedOutput capture;
        r.sys->runSolver(0.5);
        steps = stepsIn(capture.text());
        residuals = countIn(capture.text(), "Residual Evaluations");
        return r;
    };
    long coupledSteps = 0, tier0Steps = 0, coupledResiduals = 0, tier0Residuals = 0;
    auto coupled = run({}, "theta_pinch_tier2_t1", coupledSteps, coupledResiduals);
    auto tier0 = run({.consistent = true}, "theta_pinch_tier0_t1", tier0Steps, tier0Residuals);
    const Vector want = sampleWithGeometry(*coupled.sys);

    for (bool differenced : {false, true})
        for (bool krylov : {false, true})
        {
            long steps = 0, residuals = 0;
            auto reduced = run({.reduced = true, .differencedResponse = differenced, .krylov = krylov},
                               "theta_pinch_tier1", steps, residuals);
            const Vector got = sampleWithGeometry(*reduced.sys);
            const double dN =
                (want.head(5) - got.head(5)).cwiseAbs().maxCoeff() / want.head(5).cwiseAbs().maxCoeff();
            const double dB = (want.tail(5) - got.tail(5)).cwiseAbs().maxCoeff();
            const auto stats = reduced.sys->getFieldSweepStats();
            BOOST_TEST_MESSAGE("tier 1, differenced response = "
                               << differenced << ", krylov = " << krylov << ": " << steps << " steps, "
                               << residuals << " residuals (tier 2 " << coupledSteps << ", " << coupledResiduals
                               << "; tier 0 " << tier0Steps << ", " << tier0Residuals << "); N " << dN
                               << " relative, B "
                               << dB << "; " << stats.iterations << " sweeps / " << stats.solves << " solves, "
                               << stats.krylovIterations << " GMRES iterations / " << stats.krylovSolves
                               << ", " << stats.responses << " responses, " << stats.fallbacks
                               << " fallbacks");
            BOOST_TEST(dN < 1e-5);
            BOOST_TEST(dB < 1e-6);
            BOOST_TEST(reduced.sys->getFieldDOF() == 0);
            BOOST_TEST(steps <= coupledSteps + coupledSteps / 2);
            BOOST_TEST(residuals < tier0Residuals);
        }
}

BOOST_AUTO_TEST_CASE(a_reduced_steady_solve_reaches_the_coupled_answer)
{
    for (auto mode : {SystemSolver::SteadyMode::Newton, SystemSolver::SteadyMode::PseudoTransient})
        for (bool differenced : {false, true})
        {
            auto coupled = makeRun({}, "theta_pinch_tier2_ss1");
            auto reduced = makeRun({.reduced = true, .differencedResponse = differenced}, "theta_pinch_tier1_ss");
            for (Run *r : {&coupled, &reduced})
            {
                r->sys->setSteadyStateTolerance(1e-8);
                r->sys->setSteadyMode(mode);
                CapturedOutput quiet;
                r->sys->runSolver(0.5);
            }
            const Vector a = sampleWithGeometry(*coupled.sys), b = sampleWithGeometry(*reduced.sys);
            const double dN = (a.head(5) - b.head(5)).cwiseAbs().maxCoeff() / a.head(5).cwiseAbs().maxCoeff();
            BOOST_TEST_MESSAGE("steady tier 1 (differenced response = "
                               << differenced << ") against coupled: N " << dN << " relative; Newton iterations "
                               << reduced.sys->lastSteadyStats().newtonIters << " against "
                               << coupled.sys->lastSteadyStats().newtonIters);
            BOOST_TEST(dN < 1e-8);
        }
}

BOOST_AUTO_TEST_CASE(a_reused_reduced_theta_pinch_solver_matches_a_fresh_one_bit_for_bit)
{
    for (bool differenced : {false, true})
    {
        auto fresh = makeRun({.reduced = true, .differencedResponse = differenced}, "theta_pinch_tier1_fresh");
        auto reused = makeRun({.reduced = true, .differencedResponse = differenced}, "theta_pinch_tier1_reused");
        {
            CapturedOutput quiet;
            fresh.sys->runSolver(0.5);
            reused.sys->runSolver(0.5);
            reused.sys->runSolver(0.5);
        }
        const Vector want = sampleWithGeometry(*fresh.sys), got = sampleWithGeometry(*reused.sys);
        BOOST_TEST((want - got).cwiseAbs().maxCoeff() == 0.0, boost::test_tools::tolerance(0.0));
    }
}

// What a mode spends on the case, pinned exactly. Every mode pays one value
// sweep per residual, one derivative sweep per Jacobian build and its coupling
// site per build, plus the same fixed sweeps to start: so the cost less those
// per-evaluation terms is the same constant in every mode, and the equilibrium
// work tier 1 adds -- solves, responses, the differenced response's extra
// solves -- costs the case nothing.
BOOST_AUTO_TEST_CASE(each_mode_spends_one_sweep_per_residual_and_per_build_and_no_more)
{
    for (std::string derivatives : {"Analytic", "Differenced"})
    {
        std::vector<long> fixedValues, fixedDerivatives;
        for (int mode = 0; mode < 4; ++mode)
        {
            Options o{.derivatives = derivatives, .count = true};
            o.reduced = mode >= 1;
            o.differencedResponse = mode == 2;
            o.krylov = mode == 3;
            auto r = makeRun(o, "theta_pinch_counted");
            {
                CapturedOutput quiet;
                r.sys->runSolver(0.5);
            }
            auto const &c = static_cast<CountingThetaPinch const &>(*r.problem);
            const long points = static_cast<long>(r.sys->evaluationPlan().grid.getNCells()) * (o.k + 1);
            const long coupling = derivatives == "Differenced" ? r.sys->getGeometrySlots() + 1 : 0;
            const long perValues = r.sys->nResidualEvals + coupling * r.sys->nJacBuilds;
            fixedValues.push_back(c.values - points * perValues);
            fixedDerivatives.push_back(c.derivatives - points * r.sys->nJacBuilds);
            BOOST_TEST_MESSAGE(derivatives << ", mode " << mode << ": " << r.sys->nResidualEvals << " residuals, "
                                           << r.sys->nJacBuilds << " builds; values " << c.values
                                           << ", derivatives " << c.derivatives << ", hook nodes "
                                           << c.geometryHookNodes << "; fixed " << fixedValues.back() << " / "
                                           << fixedDerivatives.back());
            if (derivatives == "Analytic")
                BOOST_TEST(c.geometryHookNodes == points * r.sys->nJacBuilds);
            else
                BOOST_TEST(c.geometryHookNodes == 0);
        }
        for (size_t m = 1; m < fixedValues.size(); ++m)
        {
            BOOST_TEST(fixedValues[m] == fixedValues[0]);
            BOOST_TEST(fixedDerivatives[m] == fixedDerivatives[0]);
        }
    }
}

// The plan says what a build costs at tier 1 as at tier 2: the case's coupling
// site, and nothing for the equilibrium.
BOOST_AUTO_TEST_CASE(a_reduced_plan_lists_the_coupling_site)
{
    auto analytic = makeRun({.reduced = true}, "theta_pinch_tier1_plan");
    auto differenced = makeRun({.derivatives = "Differenced", .reduced = true}, "theta_pinch_tier1_plan");
    auto tier0 = makeRun({.consistent = true}, "theta_pinch_tier0_plan");
    auto site = [](SystemSolver &sys)
    {
        const EvaluationPlan plan = sys.evaluationPlan();
        return std::find_if(plan.sites.begin(), plan.sites.end(), [](EvaluationSite const &s)
                            { return s.kind == EvaluationKind::FieldCoupling; }) != plan.sites.end();
    };
    BOOST_TEST(site(*analytic.sys));
    BOOST_TEST(site(*differenced.sys));
    BOOST_TEST(!site(*tier0.sys));
}

BOOST_AUTO_TEST_CASE(a_reused_tier_zero_theta_pinch_solver_matches_a_fresh_one_bit_for_bit)
{
    // The adapter's Newton starts from the warm psi, and the theta pinch's rows
    // are nonlinear, so a warm start leaking from the last run moves the answer:
    // by 1.6e-8, measured with initialize()'s cache reset deleted.
    auto fresh = makeRun({.consistent = true}, "theta_pinch_tier0_fresh");
    auto reused = makeRun({.consistent = true}, "theta_pinch_tier0_reused");
    {
        CapturedOutput quiet;
        fresh.sys->runSolver(0.5);
        reused.sys->runSolver(0.5);
        reused.sys->runSolver(0.5);
    }
    const Vector want = sampleWithGeometry(*fresh.sys), got = sampleWithGeometry(*reused.sys);
    BOOST_TEST((want - got).cwiseAbs().maxCoeff() == 0.0, boost::test_tools::tolerance(0.0));
}

BOOST_AUTO_TEST_CASE(a_tier_zero_run_writes_its_geometry_and_no_unknowns)
{
    // The field group carries the geometry slots per output slice, from the
    // cached equilibrium, and nothing for psi, which is not in the state.
    const std::string stem = "theta_pinch_tier0_output";
    {
        auto run = makeRun({.consistent = true, .writeOutput = true}, stem);
        CapturedOutput quiet;
        run.sys->runSolver(0.5);
    }
    {
        netCDF::NcFile file(stem + ".nc", netCDF::NcFile::FileMode::read);
        netCDF::NcGroup group = file.getGroup("ThetaPinch");
        BOOST_REQUIRE(!group.isNull());
        for (char const *slot : {"Vprime", "dVprime_dx", "gradx2", "B"})
            BOOST_TEST(!group.getVar(slot).isNull(), slot);
        BOOST_TEST(group.getVar("C").isNull());

        netCDF::NcVar B = group.getVar("B");
        const size_t nt = B.getDim(0).getSize(), nx = B.getDim(1).getSize();
        std::vector<double> b(nx);
        B.getVar({nt - 1, 0}, {1, nx}, b.data());
        BOOST_TEST(*std::min_element(b.begin(), b.end()) < 0.99);

        int nField = -1;
        netCDF::NcFile restart(stem + ".restart.nc", netCDF::NcFile::FileMode::read);
        restart.getGroup("RestartData").getVar("nField").getVar(&nField);
        BOOST_TEST(nField == 0);
    }
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(a_restart_onto_another_degree_carries_psi)
{
    // The projection path transfers u, q, aux, the scalars and the trace. It
    // used to leave psi at the zero resetCoeffs put there -- InitialFieldValue
    // is not called on a restart -- which for this model is V' = 0, and the
    // first geometry evaluation threw.
    const std::string stem = "theta_pinch_projected_restart";
    const double T_RESTART = 0.25, T_LATER = 0.5;
    {
        auto original = makeRun({.k = 3, .writeOutput = true, .cadence = T_RESTART}, stem);
        CapturedOutput quiet;
        original.sys->runSolver(T_RESTART);
    }
    const RestartFile file = readRestart(stem);
    BOOST_REQUIRE_EQUAL(file.order, 3);
    BOOST_REQUIRE_EQUAL(file.nField, 12 + 4);
    const Vector psiFile =
        Eigen::Map<const Vector>(file.Y.data() + file.Y.size() - file.nField, file.nField);
    // Not vacuous: the field the file carries is well away from vacuum.
    BOOST_REQUIRE_GT((psiFile.head(file.nField - 1).array() - M_PI).abs().maxCoeff(), 1e-2);

    Grid restartGrid(file.cellBoundaries);
    // A restart takes the cadence as IDA's first step, and this one changes
    // degree, so it is projected: IDACalcIC makes the values consistent and
    // consistentAlgebraicDerivatives solves for the algebraic components of y'
    // it leaves at zero. Without that solve, this cadence fails the first step
    // ten times over -- the error estimate falls far more slowly than h (35049,
    // 18835, 11123 per fourfold cut) -- so the run's surviving it is part of
    // what this checks.
    auto restarted = makeRun({.k = 4, .cadence = T_RESTART / 5.0, .t0 = T_RESTART}, stem + "_resumed",
                             [&](ThetaPinchDiffusion &p, Grid const &)
                             { p.setRestartValues(file.Y, file.dYdt, restartGrid, file.order, file.nField); });
    {
        CapturedOutput quiet;
        BOOST_REQUIRE_NO_THROW(restarted.sys->initialize());
    }
    // The field is algebraic, so IDACalcIC re-solves it against the projected
    // transport state; it moves by the difference between the two
    // discretisations, not by the distance from vacuum.
    Vector psiStart(file.nField);
    for (Index f = 0; f < file.nField; ++f)
        psiStart(f) = restarted.sys->getSolution().Field(f);
    BOOST_TEST_MESSAGE("psi moved by " << (psiStart - psiFile).cwiseAbs().maxCoeff() << " across the restart");
    BOOST_TEST((psiStart - psiFile).cwiseAbs().maxCoeff() < 1e-3);

    // ...and the run goes on, to where an uninterrupted k = 4 run gets.
    {
        CapturedOutput quiet;
        restarted.sys->integrate(T_LATER);
        restarted.sys->destroySundials();
    }
    auto straight = makeRun({.k = 4}, stem + "_straight");
    {
        CapturedOutput quiet;
        straight.sys->runSolver(T_LATER);
    }
    // The yardstick is how far apart the two degrees are on their own: the
    // restarted run spent its first half at k = 3, so it should land between a
    // straight k = 3 run and a straight k = 4 one, and nearer the latter.
    auto coarse = makeRun({.k = 3}, stem + "_coarse");
    {
        CapturedOutput quiet;
        coarse.sys->runSolver(T_LATER);
    }
    const Vector fine = sample(*straight.sys), resumed = sample(*restarted.sys), lower = sample(*coarse.sys);
    const double gap = (fine - lower).cwiseAbs().maxCoeff();
    const double off = (fine - resumed).cwiseAbs().maxCoeff();
    BOOST_TEST_MESSAGE("at t = " << T_LATER << ": restarted vs straight k = 4 " << off
                                 << ", straight k = 3 vs k = 4 " << gap);
    BOOST_TEST(off < gap);

    removeOutput(stem);
    removeOutput(stem + "_resumed");
    removeOutput(stem + "_straight");
    removeOutput(stem + "_coarse");
}

BOOST_AUTO_TEST_CASE(the_diagnostics_evaluate_the_aux_equation_with_geometry)
{
    // AutodiffTransportSystem writes AuxG, the auxiliary residual, at every
    // output. This case's aux equation reads V', and the states the diagnostics
    // sampled carried no geometry rows. Now they are filled from the run's psi,
    // and the residual comes out as small as the solve left it.
    const std::string stem = "theta_pinch_diagnostics";
    {
        auto run = makeRun({.writeOutput = true}, stem);
        CapturedOutput quiet;
        run.sys->runSolver(0.5);
    }
    netCDF::NcFile file(stem + ".nc", netCDF::NcFile::FileMode::read);
    netCDF::NcVar G = file.getGroup("AuxG").getVar("Aux0");
    netCDF::NcVar p = file.getVar("p");
    const size_t nt = G.getDim(0).getSize(), nx = G.getDim(1).getSize();
    std::vector<double> g(nx), pv(nx);
    G.getVar({nt - 1, 0}, {1, nx}, g.data());
    p.getVar({nt - 1, 0}, {1, nx}, pv.data());
    file.close();

    double worst = 0.0, pmax = 0.0;
    for (size_t i = 0; i < nx; ++i)
    {
        worst = std::max(worst, std::abs(g[i]));
        pmax = std::max(pmax, std::abs(pv[i]));
    }
    BOOST_TEST_MESSAGE("final AuxG " << worst << " against p up to " << pmax);
    // The output grid samples between the nodes the aux equation is enforced
    // at, so what is left there is the interpolation error of p against
    // T N / V' -- measured 1.4e-4 of p, at k = 3 on six cells -- and not the
    // solve's. Reading geometry out of bounds would be nowhere near it.
    BOOST_TEST(pmax > 0.1);
    BOOST_TEST(worst < 1e-3 * pmax);
    removeOutput(stem);
}

BOOST_AUTO_TEST_CASE(a_non_positive_vprime_is_a_recoverable_residual_failure)
{
    // The model throws; the residual callback IDA calls must turn that into
    // IDA's "recoverable" return, so a step that wanders there is cut rather
    // than the run aborted.
    auto run = makeRun({}, "theta_pinch_bad_field");
    {
        CapturedOutput quiet;
        run.sys->initialize();
    }
    SystemSolver &sys = *run.sys;
    N_Vector Y = N_VClone(sys.Y), dYdt = N_VClone(sys.Y), res = N_VClone(sys.Y);
    N_VScale(1.0, sys.Y, Y);
    N_VConst(0.0, dYdt);
    DGSoln y(sys.nVars, *run.grid, 3, N_VGetArrayPointer(Y), sys.nScalars, sys.nAux, sys.getFieldDOF());
    BOOST_REQUIRE_EQUAL(static_residual(0.0, Y, dYdt, res, &sys), 0);
    y.Field(4) = -10.0;
    int retval = 0;
    {
        CapturedOutput quiet;
        retval = static_residual(0.0, Y, dYdt, res, &sys);
    }
    BOOST_TEST(retval == 1);
    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(res);
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

// The coupled adjoint against finite differences, at a steady state. D0 enters
// the flux; T enters only the pressure the field rows read, so dG/dT reaches
// the objective through A2^T, B^T and A1^T and nothing else -- a missing or
// mistransposed block there is a wrong dG/dT beside a perfectly good G.
BOOST_AUTO_TEST_CASE(the_coupled_gradient_matches_finite_differences)
{
    // The adjoint transposes A1 whichever way it was taken, so a differenced A1
    // gives the gradient too, to the difference's accuracy.
    for (auto [iterative, derivatives] : {std::pair{true, "Analytic"}, std::pair{false, "Analytic"},
                                          std::pair{true, "Differenced"}})
    {
        BOOST_TEST_CONTEXT("FieldSolve = " << (iterative ? "iterative" : "exact") << ", " << derivatives)
        {
            // G, and with `gradient` the adjoint's dG/dp, at a steady state with
            // parameter `which` moved by `delta`.
            auto solve = [&](Index which, double delta, Vector *gradient)
            {
                auto run = makeRun({.iterative = iterative, .derivatives = derivatives}, "theta_pinch_gradient");
                run.problem->setPval(which, Real(run.problem->getPval(which).val + delta));
                AutodiffAdjointProblem objective(run.problem.get());
                objective.addG([](Position, RealVector &u, RealVector &, RealVector &, RealVector &) -> Real
                               { return u(0) * u(0); });
                objective.setNp(2);

                SystemSolver &sys = *run.sys;
                sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
                sys.setSteadyStateTolerance(1e-11);
                sys.setTolerances({1e-11}, 1e-9);
                sys.setAdjointProblem(&objective);
                sys.setSolveAdjoint(gradient != nullptr);
                {
                    CapturedOutput quiet;
                    sys.initialize();
                    sys.integrate(0.0);
                }
                if (gradient)
                    *gradient = sys.G_p.row(0).transpose();
                const double G = objective.GFn(0, sys.yJac);
                {
                    CapturedOutput quiet;
                    sys.destroySundials();
                }
                return G;
            };

            Vector gradient;
            solve(0, 0.0, &gradient);
            BOOST_REQUIRE_EQUAL(gradient.size(), 2);
            for (Index which : {ThetaPinchDiffusion::pD0, ThetaPinchDiffusion::pTemperature})
            {
                const double h = 1e-5;
                const double fd = (solve(which, +h, nullptr) - solve(which, -h, nullptr)) / (2.0 * h);
                BOOST_TEST_MESSAGE("parameter " << which << ": adjoint " << gradient(which) << " against " << fd);
                BOOST_TEST(std::abs(fd) > 1e-3, "parameter " << which << " barely moves G");
                BOOST_TEST(gradient(which) == fd, boost::test_tools::tolerance(1e-6));
            }
        }
    }
    removeOutput("theta_pinch_gradient");
}


// G_field is the derivative of the G the run reports, component by component.
// Central differences of objectiveValue in each psi component pin both halves
// at once: the value has to be evaluated on geometry-filled states, and the
// derivative weighted by diag(w) at the same nodes -- the weights G itself is
// summed with -- rather than by anything that merely integrates the same thing.
BOOST_AUTO_TEST_CASE(g_field_is_the_derivative_of_the_reported_objective)
{
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto run = makeRun({}, "theta_pinch_g_field");
            run.sys->setSuperconvergent(superconvergent);
            AutodiffAdjointProblem objective(run.problem.get());
            addGeometricObjectives(objective);
            SystemSolver &sys = *run.sys;
            {
                CapturedOutput quiet;
                sys.initialize();
                sys.integrate(0.25);
            }
            const Index nField = sys.getFieldDOF();
            std::vector<double> base(sys.yJac.data(), sys.yJac.data() + sys.yJac.getDoF());
            for (Index g = 0; g < 2; ++g)
            {
                Vector Gmu, Gpsi;
                DGSoln state(sys.nVars, *run.grid, sys.k, base.data(), 0, sys.nAux, nField);
                sys.assembleObjectiveGradient(objective, g, state, sys.jt, &Gmu, &Gpsi);
                BOOST_REQUIRE_EQUAL(Gpsi.size(), nField);
                Vector fd(nField);
                for (Index m = 0; m < nField; ++m)
                {
                    std::vector<double> plus(base), minus(base);
                    DGSoln up(sys.nVars, *run.grid, sys.k, plus.data(), 0, sys.nAux, nField);
                    DGSoln down(sys.nVars, *run.grid, sys.k, minus.data(), 0, sys.nAux, nField);
                    const double h = 1e-6 * std::max(1.0, std::abs(up.Field(m)));
                    up.Field(m) += h;
                    down.Field(m) -= h;
                    fd(m) = (sys.objectiveValue(objective, g, up, sys.jt) -
                             sys.objectiveValue(objective, g, down, sys.jt)) / (2.0 * h);
                }
                BOOST_TEST_MESSAGE("objective " << g << ": |G_field - fd| = " << (Gpsi - fd).cwiseAbs().maxCoeff()
                                                << " against |fd| = " << fd.cwiseAbs().maxCoeff());
                BOOST_TEST(fd.cwiseAbs().maxCoeff() > 1e-2);
                BOOST_TEST((Gpsi - fd).cwiseAbs().maxCoeff() < 1e-8 * fd.cwiseAbs().maxCoeff());
            }
            CapturedOutput quiet;
            sys.destroySundials();
        }
    }
}

// The whole gradient of an objective that reads the field alone. Its dg/du is
// zero, so G_y is zero and the adjoint is driven by G_field and nothing else:
// without it G_p would be exactly zero, beside a perfectly good G.
BOOST_AUTO_TEST_CASE(an_objective_reading_only_the_field_gets_its_gradient)
{
    auto solve = [&](Index which, double delta, Vector *gradient)
    {
        auto run = makeRun({}, "theta_pinch_field_gradient");
        run.problem->setPval(which, Real(run.problem->getPval(which).val + delta));
        AutodiffAdjointProblem objective(run.problem.get());
        addGeometricObjectives(objective);
        SystemSolver &sys = *run.sys;
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setSteadyStateTolerance(1e-11);
        sys.setTolerances({1e-11}, 1e-9);
        sys.setAdjointProblem(&objective);
        sys.setSolveAdjoint(gradient != nullptr);
        {
            CapturedOutput quiet;
            sys.initialize();
            sys.integrate(0.0);
        }
        if (gradient)
            *gradient = sys.G_p.row(0).transpose();
        const double G = sys.objectiveValue(objective, 0, sys.yJac, sys.jt);
        {
            CapturedOutput quiet;
            sys.destroySundials();
        }
        return G;
    };

    Vector gradient;
    solve(0, 0.0, &gradient);
    for (Index which : {ThetaPinchDiffusion::pD0, ThetaPinchDiffusion::pTemperature})
    {
        const double h = 1e-5;
        const double fd = (solve(which, +h, nullptr) - solve(which, -h, nullptr)) / (2.0 * h);
        BOOST_TEST_MESSAGE("parameter " << which << ": adjoint " << gradient(which) << " against " << fd);
        BOOST_TEST(std::abs(fd) > 1e-3);
        BOOST_TEST(gradient(which) == fd, boost::test_tools::tolerance(1e-6));
    }
    removeOutput("theta_pinch_field_gradient");
}

// The objective estimate's forward form, (dG/dy) . J^-1 F, is the adjoint's
// z^T F only if it carries the field's share, G_field . delta psi, as it
// carries the scalars' through G_mu.
BOOST_AUTO_TEST_CASE(the_objective_estimate_carries_the_field_share)
{
    auto run = makeRun({}, "theta_pinch_estimate");
    AutodiffAdjointProblem objective(run.problem.get());
    addGeometricObjectives(objective);
    SystemSolver &sys = *run.sys;
    sys.setAdjointProblem(&objective);
    {
        CapturedOutput quiet;
        sys.initialize();
    }
    // Away from any steady state, so F is not small.
    const auto e = sys.estimateObjective();
    BOOST_REQUIRE(e.valid);

    const Index nField = sys.getFieldDOF();
    DGSoln state(sys.nVars, *run.grid, sys.k, N_VGetArrayPointer(sys.Y), 0, sys.nAux, nField);
    sys.solveAdjointAt(objective, {0, 1}, state);

    // The iterative adjoint sweep converged on its own. Its iteration matrix
    // amplifies the first increment tenfold before contracting at 0.14, and an
    // Irons-Tuck step taken from that pair diverged to a backward error of 1e68,
    // leaving the exact solve to rescue every gradient -- correct, and paying
    // for both. ironsTuck now accelerates only along aligned increments.
    const auto sweeps = sys.getFieldSweepStats();
    BOOST_TEST_MESSAGE("adjoint sweeps: " << sweeps.adjointSweeps);
    BOOST_TEST(!sweeps.adjointFellBack);
    BOOST_TEST(sweeps.adjointSweeps <= 30);
    N_Vector zero = N_VClone(sys.Y), F = N_VClone(sys.Y);
    N_VConst(0.0, zero);
    sys.residual(sys.t0, sys.Y, zero, F);
    DGSoln Fmap(sys.nVars, *run.grid, sys.k, N_VGetArrayPointer(F), 0, sys.nAux, nField);
    double const *f = N_VGetArrayPointer(F);
    const Index cellDOF = static_cast<Index>(sys.localDOF);
    const Index nCells = static_cast<Index>(run.grid->getNCells());

    for (Index g = 0; g < 2; ++g)
    {
        auto const *z = sys.adjointSolution(g, state);
        BOOST_REQUIRE(z != nullptr);
        double zF = 0.0;
        for (Index i = 0; i < nCells; ++i)
            zF += z->squ[i].dot(Eigen::Map<const Vector>(f + i * cellDOF, cellDOF));
        zF += z->lambda.dot(Eigen::Map<const Vector>(f + nCells * cellDOF, nCells + 1));
        const double fieldShare = z->field.dot(Vector(Fmap.getField()));
        zF += fieldShare;

        const double forward = e.value(g) - e.corrected(g);
        BOOST_TEST_MESSAGE("objective " << g << ": (dG/dy) . J^-1 F = " << forward << ", z^T F = " << zF
                                        << ", of which the field rows " << fieldShare);
        // The two forms agree to the forward sweep's tolerance, not to
        // roundoff: it stops where IDA needs it to, about 1e-7 relative in this
        // estimate. Dropping G_field . delta psi from the forward form moves
        // them apart by 100% and 1.3% for the two objectives, so 1e-6 still
        // separates a missing field share by four orders.
        BOOST_TEST(std::abs(forward) > 1e-4);
        BOOST_TEST(std::abs(forward - zF) <= 1e-6 * std::abs(forward));
    }
    N_VDestroy(zero);
    N_VDestroy(F);
    CapturedOutput quiet;
    sys.destroySundials();
}


// ------------------------------------------------- tier 1: the adjoint --

/// A steady theta pinch with an adjoint: the three objectives -- u^2, then the
/// two of addGeometricObjectives -- at the converged state, and with `gradient`
/// the adjoint's G_p, all three rows. G is the run's own objectiveValue, which
/// at tiers 1 and 2 reads the geometry.
Vector steadyObjectives(Options const &o, Index which, double delta, Matrix *gradient, std::string const &stem)
{
    auto run = makeRun(o, stem);
    run.problem->setPval(which, Real(run.problem->getPval(which).val + delta));
    AutodiffAdjointProblem objective(run.problem.get());
    objective.addG([](Position, RealVector &u, RealVector &, RealVector &, RealVector &) -> Real
                   { return u(0) * u(0); });
    addGeometricObjectives(objective);
    SystemSolver &sys = *run.sys;
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-11);
    sys.setTolerances({1e-11}, 1e-9);
    sys.setAdjointProblem(&objective);
    sys.setSolveAdjoint(gradient != nullptr);
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(0.0);
    }
    if (gradient)
        *gradient = sys.G_p;
    Vector G(3);
    for (Index g = 0; g < 3; ++g)
        G(g) = sys.objectiveValue(objective, g, sys.yJac, sys.jt);
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    return G;
}

// The tier-1 gradient against central differences of G, with the adapter's
// exact transpose and with the assembled default, under each solve; and the
// same G_p as tier 2's, which with the exact response is the same adjoint
// with psi eliminated. Three objectives: u^2 reaches the adjoint through G_y
// alone, B (1 + x^2) through G_geometry alone, and u V' through both.
BOOST_AUTO_TEST_CASE(the_reduced_gradient_matches_finite_differences_and_tier_two)
{
    Matrix coupled;
    steadyObjectives({}, 0, 0.0, &coupled, "theta_pinch_t1_gradient");

    struct Case
    {
        const char *name;
        Options o;
    };
    const std::vector<Case> cases = {
        {"exact G', iterative", {.reduced = true}},
        {"exact G', exact", {.iterative = false, .reduced = true}},
        {"exact G', krylov", {.reduced = true, .krylov = true}},
        {"assembled G', iterative", {.reduced = true, .differencedResponse = true}},
    };
    for (Case const &c : cases)
    {
        BOOST_TEST_CONTEXT(c.name)
        {
            Matrix gradient;
            steadyObjectives(c.o, 0, 0.0, &gradient, "theta_pinch_t1_gradient");
            BOOST_REQUIRE_EQUAL(gradient.rows(), 3);
            const double againstTier2 = (gradient - coupled).cwiseAbs().maxCoeff() / coupled.cwiseAbs().maxCoeff();
            BOOST_TEST_MESSAGE(c.name << ": |G_p - G_p(tier 2)| / |G_p| = " << againstTier2);
            BOOST_TEST(againstTier2 < (c.o.differencedResponse ? 1e-6 : 1e-8));

            for (Index which : {ThetaPinchDiffusion::pD0, ThetaPinchDiffusion::pTemperature})
            {
                const double h = 1e-5;
                const Vector fd = (steadyObjectives(c.o, which, +h, nullptr, "theta_pinch_t1_gradient") -
                                   steadyObjectives(c.o, which, -h, nullptr, "theta_pinch_t1_gradient")) /
                                  (2.0 * h);
                // The assembled transpose is a forward difference, good to about
                // 1e-7 of G'. That is an error on the scale of the whole
                // gradient: measured 1.1e-7 of max |G_p| against tier 2, which
                // is 6e-6 of objective 1's own entries, 300 times smaller. So
                // it is bounded on that scale; the model's transpose agrees with
                // the differences entry by entry, to 1e-9.
                const double scale = gradient.cwiseAbs().maxCoeff();
                for (Index g = 0; g < 3; ++g)
                {
                    BOOST_TEST_MESSAGE("  objective " << g << ", parameter " << which << ": adjoint "
                                                      << gradient(g, which) << " against " << fd(g));
                    BOOST_TEST(std::abs(fd(g)) > 1e-4);
                    if (c.o.differencedResponse)
                        BOOST_TEST(std::abs(gradient(g, which) - fd(g)) < 1e-6 * scale);
                    else
                        BOOST_TEST(gradient(g, which) == fd(g), boost::test_tools::tolerance(1e-6));
                }
            }
        }
    }
    removeOutput("theta_pinch_t1_gradient");
}

// The vacuity guards, on the gradient. A_g^T zeroed, or a transposed response
// that does nothing, is the frozen-field gradient, and it is wrong.
BOOST_AUTO_TEST_CASE(dropping_the_transposed_response_makes_the_reduced_gradient_wrong)
{
    auto relative = [](Vector const &a, Vector const &b) { return (a - b).norm() / b.norm(); };
    Matrix good;
    steadyObjectives({.reduced = true}, 0, 0.0, &good, "theta_pinch_t1_guard");

    Matrix blind;
    steadyObjectives({.reduced = true, .blindTranspose = true}, 0, 0.0, &blind, "theta_pinch_t1_guard");

    // A_g^T: zeroed after the run, then solved and accumulated again for the
    // last objective, whose G_y the run left behind.
    auto run = makeRun({.reduced = true}, "theta_pinch_t1_guard");
    AutodiffAdjointProblem objective(run.problem.get());
    objective.addG([](Position, RealVector &u, RealVector &, RealVector &, RealVector &) -> Real
                   { return u(0) * u(0); });
    addGeometricObjectives(objective);
    SystemSolver &sys = *run.sys;
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-11);
    sys.setTolerances({1e-11}, 1e-9);
    sys.setAdjointProblem(&objective);
    sys.setSolveAdjoint(true);
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(0.0);
    }
    BOOST_TEST(relative(sys.G_p.row(2).transpose(), good.row(2).transpose()) < 1e-12);
    for (Matrix &block : sys.Ag_transpose_cellwise)
    {
        BOOST_REQUIRE_GT(block.cwiseAbs().maxCoeff(), 0.0);
        block.setZero();
    }
    sys.solveAdjointState();
    sys.accumulateAdjointGradients(2);
    const Vector noAg = sys.G_p.row(2).transpose();
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }

    for (Index g = 0; g < 3; ++g)
        BOOST_TEST_MESSAGE("objective " << g << ": with no transposed response, relative error "
                                        << relative(blind.row(g).transpose(), good.row(g).transpose()));
    BOOST_TEST_MESSAGE("objective 2 with A_g^T zeroed: relative error " << relative(noAg, good.row(2).transpose()));
    for (Index g = 0; g < 3; ++g)
        BOOST_TEST(relative(blind.row(g).transpose(), good.row(g).transpose()) > 1e-2);
    BOOST_TEST(relative(noAg, good.row(2).transpose()) > 1e-2);
    removeOutput("theta_pinch_t1_guard");
}

// An adjoint on demand at tier 1 makes no physics call, as at tier 2, and the
// equilibrium solves it makes are pinned: one for the objective's geometry,
// and with the assembled default one per input component per point besides,
// once per equilibrium -- a second solve at the same operator makes none.
BOOST_AUTO_TEST_CASE(a_reduced_adjoint_on_demand_makes_no_physics_call)
{
    for (bool differenced : {false, true})
    {
        BOOST_TEST_CONTEXT("differenced response = " << differenced)
        {
            Options o{.reduced = true, .differencedResponse = differenced, .count = true};
            auto run = makeRun(o, "theta_pinch_t1_on_demand");
            AutodiffAdjointProblem objective(run.problem.get());
            addGeometricObjectives(objective);
            SystemSolver &sys = *run.sys;
            {
                CapturedOutput quiet;
                sys.runSolver(0.5);
            }
            auto const &c = static_cast<CountingThetaPinch const &>(*run.problem);
            const long values = c.values, derivatives = c.derivatives, hooks = c.geometryHookNodes;
            const long solves = run.equilibrium->solves;

            DGSoln state = sys.getSolution();
            std::vector<double> copy(state.data(), state.data() + state.getDoF());
            DGSoln at(sys.nVars, *run.grid, sys.k, copy.data(), 0, sys.nAux, 0);
            sys.solveAdjointAt(objective, {0, 1}, at);
            const long first = run.equilibrium->solves - solves;
            sys.solveAdjointAt(objective, {0, 1}, at);
            const long second = run.equilibrium->solves - solves - first;

            const long components = 3 * sys.nVars + sys.nAux;
            const long inputs = components * static_cast<long>(run.grid->getNCells()) * (o.k + 1);
            BOOST_TEST_MESSAGE("physics calls " << c.values - values << " / " << c.derivatives - derivatives << " / "
                                                << c.geometryHookNodes - hooks << "; equilibrium solves "
                                                << first << " then " << second << " (" << inputs
                                                << " input components)");
            BOOST_TEST(c.values == values);
            BOOST_TEST(c.derivatives == derivatives);
            BOOST_TEST(c.geometryHookNodes == hooks);
            // One solve for the objective's own geometry: dg reads it at the
            // state asked about, and the cache holds the last residual's state,
            // which after a time march is an iterate of IDA's rather than the
            // output it interpolated. Then the assembly, once.
            BOOST_TEST(first == 1 + (differenced ? inputs : 0));
            BOOST_TEST(second == 0);
            BOOST_TEST(sys.adjointSolution(0, at) != nullptr);
        }
    }
    removeOutput("theta_pinch_t1_on_demand");
}

BOOST_AUTO_TEST_CASE(a_reused_reduced_adjoint_matches_a_fresh_one_bit_for_bit)
{
    for (bool differenced : {false, true})
    {
        auto gradientOf = [&](int runs)
        {
            auto run = makeRun({.reduced = true, .differencedResponse = differenced}, "theta_pinch_t1_reuse");
            AutodiffAdjointProblem objective(run.problem.get());
            addGeometricObjectives(objective);
            SystemSolver &sys = *run.sys;
            sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
            sys.setSteadyStateTolerance(1e-10);
            sys.setAdjointProblem(&objective);
            sys.setSolveAdjoint(true);
            for (int r = 0; r < runs; ++r)
            {
                CapturedOutput quiet;
                sys.runSolver(0.0);
            }
            return Matrix(sys.G_p);
        };
        const Matrix fresh = gradientOf(1), reused = gradientOf(2);
        BOOST_TEST((fresh - reused).cwiseAbs().maxCoeff() == 0.0, boost::test_tools::tolerance(0.0));
    }
    removeOutput("theta_pinch_t1_reuse");
}

// ------------------------------- a native equilibrium: the pressure balance --
//
// ThetaPinchPressureBalance, the registered native EquilibriumModel, through the
// run tests tiers 0 and 1 of the field model take: reuse, the gradient and the
// adjoint on demand.

BOOST_AUTO_TEST_CASE(a_reused_native_theta_pinch_solver_matches_a_fresh_one_bit_for_bit)
{
    struct Case
    {
        const char *name;
        Options o;
    };
    for (Case const &c : {Case{"tier 1, model G'", {.reduced = true, .native = true}},
                          Case{"tier 1, differenced G'", {.reduced = true, .differencedResponse = true, .native = true}},
                          Case{"tier 0", {.consistent = true, .native = true}}})
        BOOST_TEST_CONTEXT(c.name)
        {
            auto fresh = makeRun(c.o, "theta_pinch_native_fresh");
            auto reused = makeRun(c.o, "theta_pinch_native_reused");
            {
                CapturedOutput quiet;
                fresh.sys->runSolver(0.5);
                reused.sys->runSolver(0.5);
                reused.sys->runSolver(0.5);
            }
            const Vector want = sampleWithGeometry(*fresh.sys), got = sampleWithGeometry(*reused.sys);
            BOOST_TEST((want - got).cwiseAbs().maxCoeff() == 0.0, boost::test_tools::tolerance(0.0));
            // Not vacuous: the field has left the vacuum, B = 1.
            BOOST_TEST((want.tail(5).array() - 1.0).abs().maxCoeff() > 1e-2);
        }
}

// The native tier-1 gradient against central differences of G, with the model's
// exact transpose and with the base class's assembly of its differenced
// response, on the three objectives of the_reduced_gradient_matches_finite_
// differences_and_tier_two.
BOOST_AUTO_TEST_CASE(the_native_gradient_matches_finite_differences)
{
    for (bool differenced : {false, true})
        BOOST_TEST_CONTEXT("differenced response = " << differenced)
        {
            const Options o{.reduced = true, .differencedResponse = differenced, .native = true};
            Matrix gradient;
            steadyObjectives(o, 0, 0.0, &gradient, "theta_pinch_native_gradient");
            BOOST_REQUIRE_EQUAL(gradient.rows(), 3);
            const double scale = gradient.cwiseAbs().maxCoeff();
            for (Index which : {ThetaPinchDiffusion::pD0, ThetaPinchDiffusion::pTemperature})
            {
                const double h = 1e-5;
                const Vector fd = (steadyObjectives(o, which, +h, nullptr, "theta_pinch_native_gradient") -
                                   steadyObjectives(o, which, -h, nullptr, "theta_pinch_native_gradient")) /
                                  (2.0 * h);
                for (Index g = 0; g < 3; ++g)
                {
                    BOOST_TEST_MESSAGE("objective " << g << ", parameter " << which << ": adjoint "
                                                    << gradient(g, which) << " against " << fd(g));
                    BOOST_TEST(std::abs(fd(g)) > 1e-4);
                    // As for the field model's: the assembled transpose of a
                    // forward difference is good to about 1e-7 of G', so on the
                    // scale of the whole gradient; the exact one entry by entry.
                    if (differenced)
                        BOOST_TEST(std::abs(gradient(g, which) - fd(g)) < 1e-6 * scale);
                    else
                        BOOST_TEST(gradient(g, which) == fd(g), boost::test_tools::tolerance(1e-6));
                }
            }
        }
    removeOutput("theta_pinch_native_gradient");
}

// The native adjoint on demand makes no physics call, and its equilibrium
// solves are pinned: one for the objective's geometry and, with the exact
// transpose, no more -- it assembles G' from duals, not from Solve. With the
// differenced response the base class's assembly makes one per input component
// per point, once per equilibrium.
BOOST_AUTO_TEST_CASE(a_native_adjoint_on_demand_makes_no_physics_call)
{
    for (bool differenced : {false, true})
        BOOST_TEST_CONTEXT("differenced response = " << differenced)
        {
            Options o{.reduced = true, .differencedResponse = differenced, .count = true, .native = true};
            auto run = makeRun(o, "theta_pinch_native_on_demand");
            AutodiffAdjointProblem objective(run.problem.get());
            addGeometricObjectives(objective);
            SystemSolver &sys = *run.sys;
            {
                CapturedOutput quiet;
                sys.runSolver(0.5);
            }
            auto const &c = static_cast<CountingThetaPinch const &>(*run.problem);
            const long values = c.values, derivatives = c.derivatives, hooks = c.geometryHookNodes;
            const long solves = run.native->solves();

            DGSoln state = sys.getSolution();
            std::vector<double> copy(state.data(), state.data() + state.getDoF());
            DGSoln at(sys.nVars, *run.grid, sys.k, copy.data(), 0, sys.nAux, 0);
            sys.solveAdjointAt(objective, {0, 1}, at);
            const long first = run.native->solves() - solves;
            sys.solveAdjointAt(objective, {0, 1}, at);
            const long second = run.native->solves() - solves - first;

            const long components = 3 * sys.nVars + sys.nAux;
            const long inputs = components * static_cast<long>(run.grid->getNCells()) * (o.k + 1);
            BOOST_TEST_MESSAGE("physics calls " << c.values - values << " / " << c.derivatives - derivatives
                                                << " / " << c.geometryHookNodes - hooks << "; equilibrium solves "
                                                << first << " then " << second);
            BOOST_TEST(c.values == values);
            BOOST_TEST(c.derivatives == derivatives);
            BOOST_TEST(c.geometryHookNodes == hooks);
            BOOST_TEST(first == 1 + (differenced ? inputs : 0));
            BOOST_TEST(second == 0);
            BOOST_TEST(sys.adjointSolution(0, at) != nullptr);
        }
    removeOutput("theta_pinch_native_on_demand");
}

BOOST_AUTO_TEST_SUITE_END()
