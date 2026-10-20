#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "../../PhysicsCases/AutodiffAdjointProblem.hpp"
#include "../../PhysicsCases/ThetaPinchDiffusion.hpp"
#include "../../PhysicsCases/ThetaPinchEquilibrium.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <netcdf>
#include <nvector/nvector_serial.h>

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
    Index intervals = 12;
    bool iterative = true;
    bool writeOutput = false;
    double cadence = 0.25;
    double t0 = 0.0;
};

struct Run
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<ThetaPinchDiffusion> problem;
    std::shared_ptr<ThetaPinchEquilibrium> field;
    std::unique_ptr<SystemSolver> sys;
};

Run makeRun(Options const &o, std::string const &stem,
            std::function<void(ThetaPinchDiffusion &, Grid const &)> const &prepare = {})
{
    Run r;
    r.grid = std::make_unique<Grid>(0.0, 1.0, o.nCells);
    r.problem = std::make_unique<ThetaPinchDiffusion>(thetaPinchConfig, *r.grid);
    if (prepare)
        prepare(*r.problem, *r.grid);
    r.field = std::make_shared<ThetaPinchEquilibrium>(
        ThetaPinchEquilibrium::Parameters{.intervals = o.intervals}, *r.grid);
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
    sys.setFieldModel(r.field);
    sys.setFieldSolveMode(o.iterative ? SystemSolver::FieldSolveMode::Iterative
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
    // not reach. Measured: the same 156 steps, and the same answer to 5e-15.
    auto exact = makeRun({.iterative = false}, "theta_pinch_exact");
    auto iterative = makeRun({.iterative = true}, "theta_pinch_iterative");
    {
        CapturedOutput quiet;
        exact.sys->runSolver(0.5);
        iterative.sys->runSolver(0.5);
    }
    const Vector a = sample(*exact.sys), b = sample(*iterative.sys);
    BOOST_TEST_MESSAGE("exact vs iterative: " << (a - b).cwiseAbs().maxCoeff());
    BOOST_TEST((a - b).cwiseAbs().maxCoeff() < 1e-9 * a.cwiseAbs().maxCoeff());
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
    // A small cadence. A restart takes the cadence as IDA's first step, and
    // this one changes degree, so it is projected: the copy path resumes the
    // file's y', but here the algebraic components of y' start at zero and
    // IDACalcIC's IDA_YA_YDP_INIT holds them there. The first step's error
    // estimate then falls far more slowly than h (35049, 18835, 11123 per
    // fourfold cut), and at T_RESTART / 5 it fails the error test ten times
    // over.
    auto restarted = makeRun({.k = 4, .cadence = T_RESTART / 50.0, .t0 = T_RESTART}, stem + "_resumed",
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
    for (bool iterative : {true, false})
    {
        BOOST_TEST_CONTEXT("FieldSolve = " << (iterative ? "iterative" : "exact"))
        {
            // G, and with `gradient` the adjoint's dG/dp, at a steady state with
            // parameter `which` moved by `delta`.
            auto solve = [&](Index which, double delta, Vector *gradient)
            {
                auto run = makeRun({.iterative = iterative}, "theta_pinch_gradient");
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
        BOOST_TEST(std::abs(forward) > 1e-4);
        BOOST_TEST(std::abs(forward - zF) <= 1e-9 * std::abs(forward));
    }
    N_VDestroy(zero);
    N_VDestroy(F);
    CapturedOutput quiet;
    sys.destroySundials();
}


BOOST_AUTO_TEST_SUITE_END()
