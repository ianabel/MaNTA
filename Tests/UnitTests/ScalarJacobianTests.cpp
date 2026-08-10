// Settles the open question about solveJacEq's Woodbury/bordered elimination.
//
// The background: SolveJacTests.cpp finite-differences `residual` and requires
// the vector the linear solve returns to satisfy J dy = g. With nScalars = 0
// that passes at 3e-10, so solveHDGJac's static condensation is verified. With
// nScalars = 3 (ScalarTestLD3) the same check came out O(1) -- and not only in
// the scalar rows, which ruled nothing out on its own. Two candidates:
//
//   (a) the bordered elimination in solveJacEq is wrong, or
//   (b) the physics case's hand-written ScalarGPrimeExtended disagrees with its
//       own ScalarGExtended, which corrupts the whole solve rather than just the
//       scalar rows -- matching what was observed.
//
// PIDTest cannot distinguish them: a wrong Jacobian only slows Newton, it still
// converges to the right answer, so a reference comparison stays green either
// way.
//
// This file separates them by supplying a scalar system whose Jacobian is known
// exactly, so any failure can only be (a). It then adds a reusable consistency
// check -- finite-difference ScalarGExtended and compare against
// ScalarGPrimeExtended -- which answers (b) for any physics case.
//
// Result: solveJacEq is correct. Both minimal systems, algebraic and
// differential, satisfy J dy = g to round-off. ScalarTestLD3's dG_0/du had the
// wrong sign; with that fixed its case passes too.

#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/ScalarTestLD3.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "PyIntegrator.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <boost/math/quadrature/gauss.hpp>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <functional>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

using namespace toml::literals::toml_literals;

namespace
{

constexpr double KAPPA = 1.3;
constexpr double COUPLING = 0.8; // dS/dmu -- fills the v vectors
constexpr double BETA = 0.6;     // dG/du  -- fills the w vectors

boost::math::quadrature::gauss<double, 30> integrator;

// A transport system with exactly one global scalar and an exactly known
// Jacobian.
//
//   d_t u = d_x( kappa d_x u ) + COUPLING * mu
//   G     = mu - BETA * Int_0^1 u dx
//
// Every entry of the bordered system is a constant or a basis integral:
//
//   v (dS/dmu)                  = COUPLING, uniformly
//   w (dG/du against phi_{i,l}) = -BETA * Int_{cell i} phi_{i,l} dx
//   N (dG/dmu)                  = 1
//
// Both couplings are nonzero -- a system with v = 0 or w = 0 would leave the
// elimination untested -- and neither depends on the state, so there is no
// question of the reference being wrong.
class MinimalScalarSystem : public TransportSystem
{
public:
    // One scalar. It is algebraic here -- G = mu - BETA * Int u dx has no
    // explicit dmu/dt -- and differential in the subclass below, which is the
    // whole point of the pair, so the kind is a constructor argument rather
    // than the isScalarDifferential override it used to be.
    explicit MinimalScalarSystem(bool differential = false)
        : TransportSystem({.variables = numberedFields(1),
                           .scalars = numberedScalars(1, differential)})
    {
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return KAPPA * s.q(0);
    }
    Value Sources(Index, const State &s, Position, Time) override
    {
        return COUPLING * s.scalar(0);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = KAPPA;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dScalars(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = COUPLING;
    }

    /// Int_0^1 u dx, on the framework's node quadrature.
    ///
    /// This was a cell-by-cell adaptive rule applied to y.u(0). The rule had to
    /// be per cell, because u is only piecewise polynomial and a single rule
    /// over the whole domain does not integrate it exactly -- and then the
    /// derivative of the computed mass is not what ScalarGPrime reports, which
    /// made the fixture disagree with itself by a couple of percent. Using the
    /// weights the hook is handed removes the question: the derivative of
    /// `weights . u` with respect to u_j is exactly weights(j).
    static double mass(GlobalState const &y, Vector const &weights)
    {
        return ScalarHooks::integrate(y.Variable().row(0), weights);
    }

    // G = mu - BETA * Int u dx. Algebraic, so the dY'/dt half is zero.
    Value ScalarG(Index, GlobalState const &y, GlobalState const &,
                  std::vector<Position> const &, Values const &weights, Matrix const &,
                  Time) override
    {
        return y.Scalars()(0) - BETA * mass(y, weights);
    }

    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &,
                      Values const &weights, Matrix const &, Time) override
    {
        dG[0].Variable().row(0) = -BETA * weights.transpose();
        dG[0].Scalars()(0) = 1.0;
    }

    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; }
    Value InitialScalarValue(Index) const override { return 0.25; }
};

// The same system with the scalar made *differential*:
//
//   G = d(mu)/dt - BETA * Int_0^1 u dx
//
// so dG/dmu = 0 and dG/dmu' = 1, giving N = alpha rather than 1. That is the
// only way to test that the dY'/dt half of the scalar row is weighted by alpha
// (= IDA's cj) rather than dropped or double counted -- with an algebraic scalar
// the term is identically zero and any handling of it looks correct.
class DifferentialScalarSystem : public MinimalScalarSystem
{
public:
    DifferentialScalarSystem() : MinimalScalarSystem(true) {}

    Value ScalarG(Index, GlobalState const &y, GlobalState const &ydot,
                  std::vector<Position> const &, Values const &weights, Matrix const &,
                  Time) override
    {
        return ydot.Scalars()(0) - BETA * mass(y, weights);
    }

    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &,
                      Values const &weights, Matrix const &, Time) override
    {
        dG[0].Variable().row(0) = -BETA * weights.transpose();
        dGdot[0].Scalars()(0) = 1.0;
    }

    Value InitialScalarDerivative(Index, const DGSoln &y, const DGSoln &) const override
    {
        // Still a DGSoln hook, so it samples the nodes itself to reach the same
        // mass the constraint uses.
        return BETA * mass(y.evalOnNodes(),
                           Integrator::getIntegrationWeights(y.getBasis(), y.getGrid()));
    }
};

// Everything a bordered-solve test needs, set up and torn down.
struct ScalarFixture
{
    Grid grid;
    TransportSystem &problem;
    SystemSolver sys;
    SUNContext ctx = nullptr;
    N_Vector Y = nullptr, dYdt = nullptr;
    Index n = 0;
    double cj;

    ScalarFixture(TransportSystem &p, Grid g, Index k, double tau, double cj_)
        : grid(g), problem(p), sys(grid, k, &p), cj(cj_)
    {
        sys.setTau(tau);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext_Create(SUN_COMM_NULL, &ctx);

        DGSoln shape(p.getNumVars(), grid, k, p.getNumScalars(), p.getNumAux());
        n = shape.getDoF();

        Y = N_VNew_Serial(n, ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);

        sys.setJacTime(0.0);
        sys.setAlpha(cj);
        sys.setJacEvalY(Y, dYdt);
        sys.updateBoundaryConditions(0.0);
        sys.updateMatricesForJacSolve();
    }

    ~ScalarFixture()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
};

// Run the J dy = g check for several right-hand sides and return the worst
// relative residual, split into the DG field rows and the scalar rows.
struct Residuals
{
    double total = 0.0;
    double field = 0.0;
    double scalar = 0.0;
};

Residuals checkBorderedSolve(ScalarFixture &f, int trials = 3)
{
    const Matrix J = fdjac::jacobian(f.sys, f.Y, f.dYdt, 0.0, f.cj);
    const auto skip = fdjac::undefinedRows(J);
    const Index nS = f.problem.getNumScalars();

    Residuals worst;
    for (int trial = 0; trial < trials; ++trial)
    {
        N_Vector g = N_VNew_Serial(f.n, f.ctx), dy = N_VClone(g);
        double *ga = N_VGetArrayPointer(g);
        for (Index i = 0; i < f.n; ++i)
            ga[i] = std::cos(0.4 + i * (trial + 1) * 0.53);
        const Vector gVec = fdjac::toVector(g);

        f.sys.solveJacEq(g, dy);
        const Vector dyVec = fdjac::toVector(dy);

        for (Index i = 0; i < f.n; ++i)
            BOOST_TEST_REQUIRE(std::isfinite(dyVec(i)));

        const Vector r = J * dyVec - gVec;
        double fnum = 0.0, fden = 0.0, snum = 0.0, sden = 0.0;
        for (Index i = 0; i < f.n; ++i)
        {
            if (fdjac::isUndefined(skip, i))
                continue;
            if (i >= f.n - nS)
            {
                snum += r(i) * r(i);
                sden += gVec(i) * gVec(i);
            }
            else
            {
                fnum += r(i) * r(i);
                fden += gVec(i) * gVec(i);
            }
        }

        worst.total = std::max(worst.total,
                               fdjac::relativeResidual(J, dyVec, gVec, skip));
        worst.field = std::max(worst.field, std::sqrt(fnum / fden));
        worst.scalar = std::max(worst.scalar, std::sqrt(snum / sden));

        N_VDestroy(g);
        N_VDestroy(dy);
    }
    return worst;
}

// --------------------------------------------------------------------------
// Finite-difference ScalarG and compare against ScalarGPrime.
//
// This is the check that separates a broken elimination from a physics case
// that misreports its own derivative, and it works for any case: perturb one
// solution coefficient at a time and difference the scalar constraint.
//
// The hooks now report every scalar's derivative in one call, indexed by node
// rather than being asked once per (cell, basis function), so the reported
// value for cell i node l is column i*(k+1)+l -- the same flattening the
// solver reads back into the w vectors, which is exactly the indexing an
// off-by-one here would hide.
//
// Returns the largest disagreement found, and reports each one.
double checkScalarDerivative(TransportSystem &problem, DGSoln &y, DGSoln &dydt,
                             Grid const &grid, Index k, double t, double tolerance)
{
    const Index nVars = problem.getNumVars();
    const Index nScalars = problem.getNumScalars();
    const Index nAux = problem.getNumAux();
    const Index nCells = static_cast<Index>(grid.getNCells());

    const Vector &weights = Integrator::getIntegrationWeights(y.getBasis(), grid);
    const Matrix &phiBoundary = Integrator::getPhiBoundary(y.getBasis(), grid);

    double worst = 0.0;

    // What the case says its derivative is -- all scalars, all nodes, one call.
    GlobalStateMatrix reported(nScalars), reported_dt(nScalars);
    for (Index s = 0; s < nScalars; ++s)
    {
        reported.add(nCells, k, nVars, nScalars, nAux);
        reported_dt.add(nCells, k, nVars, nScalars, nAux);
    }
    problem.ScalarGPrime(reported, reported_dt, y.evalOnNodes(), dydt.evalOnNodes(),
                         y.getPoints(), weights, phiBoundary, t);

    // Re-sampled on every call: the coefficient being perturbed has to reach
    // the nodal values the constraint actually reads.
    auto G = [&](Index s)
    {
        return problem.ScalarG(s, y.evalOnNodes(), dydt.evalOnNodes(), y.getPoints(), weights,
                               phiBoundary, t);
    };

    auto compare = [&](const char *what, Index s, Index cell, Index l, Index var,
                       double expected, double *coefficient)
    {
        const double h = 1e-6 * std::max(1.0, std::abs(*coefficient));
        const double original = *coefficient;

        *coefficient = original + h;
        const double gp = G(s);
        *coefficient = original - h;
        const double gm = G(s);
        *coefficient = original;

        const double fd = (gp - gm) / (2.0 * h);
        const double err = std::abs(fd - expected);
        worst = std::max(worst, err);

        BOOST_TEST(err < tolerance,
                   "dG_" << s << "/d" << what << " (cell " << cell << ", node " << l
                         << ", var " << var << "): reported " << expected
                         << ", finite difference " << fd);
    };

    for (Index s = 0; s < nScalars; ++s)
    {
        for (Index cell = 0; cell < nCells; ++cell)
        {
            for (Index l = 0; l < k + 1; ++l)
            {
                const Index node = cell * (k + 1) + l;

                for (Index v = 0; v < nVars; ++v)
                {
                    compare("u", s, cell, l, v, reported[s].Variable()(v, node),
                            &y.u(v).getCoeff(cell).second(l));
                    compare("q", s, cell, l, v, reported[s].Derivative()(v, node),
                            &y.q(v).getCoeff(cell).second(l));
                    compare("sigma", s, cell, l, v, reported[s].Flux()(v, node),
                            &y.sigma(v).getCoeff(cell).second(l));
                }
                for (Index a = 0; a < nAux; ++a)
                    compare("aux", s, cell, l, a, reported[s].Aux()(a, node),
                            &y.Aux(a).getCoeff(cell).second(l));
            }
        }

        // The scalar-scalar block, and the dY'/dt half.
        for (Index m = 0; m < nScalars; ++m)
        {
            compare("mu", s, -1, -1, m, reported[s].Scalars()(m), &y.Scalar(m));
            compare("dmu/dt", s, -1, -1, m, reported_dt[s].Scalars()(m), &dydt.Scalar(m));
        }
    }

    return worst;
}

} // namespace

BOOST_AUTO_TEST_SUITE(scalar_jacobian_tests)

// ------------------------------------------------ the bordered elimination --

BOOST_AUTO_TEST_CASE(the_reported_scalar_derivative_is_the_real_one)
{
    // Before testing the elimination, confirm the fixture is not lying about
    // its own Jacobian -- otherwise a failure below would be ambiguous in
    // exactly the way this file exists to resolve.
    Grid grid(0.0, 1.0, 3);
    const Index k = 2;
    MinimalScalarSystem problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);
    DGSoln shape(1, grid, k, Index(1), Index(0));
    N_Vector Y = N_VNew_Serial(shape.getDoF(), ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    DGSoln y(1, grid, k, N_VGetArrayPointer(Y), Index(1), Index(0));
    DGSoln dydt(1, grid, k, N_VGetArrayPointer(dYdt), Index(1), Index(0));

    const double worst = checkScalarDerivative(problem, y, dydt, grid, k, 0.0, 1e-7);
    BOOST_TEST_MESSAGE("MinimalScalarSystem: worst derivative error " << worst);

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(solve_jac_eq_inverts_the_bordered_jacobian)
{
    // The decisive test. Every entry of the bordered system is exactly known,
    // so if J dy = g fails here the fault can only be in solveJacEq's
    // Woodbury elimination.
    MinimalScalarSystem problem;
    ScalarFixture f(problem, Grid(0.0, 1.0, 4), 2, 1.0, 2.5);

    const Residuals r = checkBorderedSolve(f);
    BOOST_TEST_MESSAGE("algebraic scalar: total " << r.total << "  field " << r.field
                                                  << "  scalar " << r.scalar);

    // The finite-difference Jacobian itself carries O(h^2) ~ 1e-8 error with
    // h = 1e-6, so this is close to the achievable floor rather than a loose
    // bound; the observed value is printed above.
    BOOST_TEST(r.total < 1e-7);
    BOOST_TEST(r.field < 1e-7);
    BOOST_TEST(r.scalar < 1e-7);
}

BOOST_AUTO_TEST_CASE(solve_jac_eq_weights_the_time_derivative_block_by_alpha)
{
    // With a differential scalar N = alpha rather than 1, so this fails if the
    // dY'/dt half of the scalar row is dropped, double counted, or weighted by
    // the wrong factor. Two values of alpha, because a single one cannot
    // distinguish "weighted by alpha" from "weighted by some other constant".
    DifferentialScalarSystem problem;

    for (double cj : {1.0, 3.7})
    {
        ScalarFixture f(problem, Grid(0.0, 1.0, 4), 2, 1.0, cj);
        const Residuals r = checkBorderedSolve(f);
        BOOST_TEST_MESSAGE("differential scalar, alpha = " << cj << ": total " << r.total
                                                           << "  field " << r.field
                                                           << "  scalar " << r.scalar);
        BOOST_TEST(r.total < 1e-7, "alpha = " << cj);
        BOOST_TEST(r.scalar < 1e-7, "alpha = " << cj);
    }
}

BOOST_AUTO_TEST_CASE(the_bordered_solve_is_linear)
{
    // The whole elimination is a linear operator, whatever else it is.
    MinimalScalarSystem problem;
    ScalarFixture f(problem, Grid(0.0, 1.0, 4), 2, 1.0, 1.5);

    N_Vector g = N_VNew_Serial(f.n, f.ctx), g2 = N_VClone(g);
    N_Vector dy = N_VClone(g), dy2 = N_VClone(g);

    double *ga = N_VGetArrayPointer(g);
    double *g2a = N_VGetArrayPointer(g2);
    for (Index i = 0; i < f.n; ++i)
    {
        ga[i] = std::sin(0.3 * i + 0.2);
        g2a[i] = -2.5 * ga[i];
    }

    f.sys.solveJacEq(g, dy);
    f.sys.solveJacEq(g2, dy2);

    const double *a = N_VGetArrayPointer(dy);
    const double *b = N_VGetArrayPointer(dy2);
    double worst = 0.0, scale = 0.0;
    for (Index i = 0; i < f.n; ++i)
    {
        worst = std::max(worst, std::abs(b[i] + 2.5 * a[i]));
        scale = std::max(scale, std::abs(a[i]));
    }
    BOOST_TEST(worst <= 1e-9 * std::max(1.0, scale));

    // Zero in, zero out -- so the scalar block is not picking up a stray
    // constant from the previous solve.
    N_VConst(0.0, g);
    f.sys.solveJacEq(g, dy);
    BOOST_TEST(N_VMaxNorm(dy) == 0.0);

    for (N_Vector v : {g, g2, dy, dy2})
        N_VDestroy(v);
}

// ------------------------------------------------------- the physics case --

BOOST_AUTO_TEST_CASE(scalar_test_ld3_reports_its_own_scalar_derivatives_correctly)
{
    const toml::value scalar_config = u8R"(
        [DiffusionProblem]
        Kappa = 1.0
        alpha = 0.2
        beta = 1.0
        gamma = 1.0
        u0 = 0.1
    )"_toml;

    for (auto [k, nCells] : std::vector<std::pair<Index, Index>>{
             {2, 4}, {3, 8}, {4, 16}, {6, 32}})
    {
        Grid grid(-1.0, 1.0, nCells);
        ScalarTestLD3 problem(scalar_config, grid);

        SystemSolver sys(grid, k, &problem);
        sys.setTau(1.0);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext ctx;
        SUNContext_Create(SUN_COMM_NULL, &ctx);
        DGSoln shape(problem.getNumVars(), grid, k, problem.getNumScalars(),
                     problem.getNumAux());
        N_Vector Y = N_VNew_Serial(shape.getDoF(), ctx), dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);

        DGSoln y(problem.getNumVars(), grid, k, N_VGetArrayPointer(Y),
                 problem.getNumScalars(), problem.getNumAux());
        DGSoln dydt(problem.getNumVars(), grid, k, N_VGetArrayPointer(dYdt),
                    problem.getNumScalars(), problem.getNumAux());

        const double worst =
            checkScalarDerivative(problem, y, dydt, grid, k, 0.0, 1e-6);
        BOOST_TEST_MESSAGE("ScalarTestLD3 derivative k=" << k << " n=" << nCells
                                                         << ": worst " << worst);

        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
}

BOOST_AUTO_TEST_CASE(solve_jac_eq_handles_a_three_scalar_physics_case)
{
    const toml::value scalar_config = u8R"(
        [DiffusionProblem]
        Kappa = 1.0
        alpha = 0.2
        beta = 1.0
        gamma = 1.0
        u0 = 0.1
    )"_toml;

    // ScalarTestLD3 hardcodes its domain as [-1, 1].
    for (auto [k, nCells] : std::vector<std::pair<Index, Index>>{
             {2, 4}, {3, 8}, {4, 16}, {6, 32}})
    {
        Grid grid(-1.0, 1.0, nCells);
        ScalarTestLD3 problem(scalar_config, grid);
        ScalarFixture f(problem, grid, k, 1.0, 2.5);
        const Residuals r = checkBorderedSolve(f, 2);
        BOOST_TEST_MESSAGE("ScalarTestLD3 k=" << k << " n=" << nCells << ": total "
                                              << r.total << "  field " << r.field
                                              << "  scalar " << r.scalar);

        // Bounded rather than fixed: the finite-difference Jacobian carries
        // O(h^2) error, and the residual is measured relative to a right-hand
        // side whose scale does not shrink with refinement.
        BOOST_TEST(r.total < 1e-7, "k = " << k << ", n = " << nCells);
        BOOST_TEST(r.field < 1e-7, "k = " << k << ", n = " << nCells);
        BOOST_TEST(r.scalar < 1e-7, "k = " << k << ", n = " << nCells);
    }
}

BOOST_AUTO_TEST_SUITE_END()
