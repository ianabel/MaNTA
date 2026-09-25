// Tests for the HDG linear solve -- the largest untested piece of the solver.
//
// SystemSolver::solveHDGJac performs static condensation onto the lambda trace
// unknowns and back-substitutes; solveJacEq wraps it in a Woodbury/bordered
// elimination for the global scalars. Neither ever forms the Jacobian, so
// SUNDIALS never checks them and an error here shows up only as degraded Newton
// convergence -- or as a wrong answer.
//
// The strategy is to build the Jacobian explicitly by finite-differencing
// SystemSolver::residual, solve the same system densely with Eigen, and require
// the two answers to agree. IDA's Jacobian is
//
//     J = dF/dY + cj dF/dY'
//
// (cj is `alpha` here), so each column is obtained by perturbing Y[j] by h and
// Y'[j] by cj*h together.

#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/ScalarTestLD3.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <toml.hpp>

using namespace toml::literals::toml_literals;

namespace
{
const toml::value diffusion_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;

// ScalarTestLD3: linear diffusion carrying three global scalars.
const toml::value scalar_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    alpha = 0.2
    beta = 1.0
    gamma = 1.0
    u0 = 0.1
)"_toml;

// The finite-difference Jacobian helpers now live in
// FiniteDifferenceJacobian.hpp, shared with ScalarJacobianTests.cpp.
using fdjac::jacobian;

} // namespace

BOOST_AUTO_TEST_SUITE(solve_jac_tests)

BOOST_AUTO_TEST_CASE(solve_hdg_jac_agrees_with_a_dense_solve)
{
    const Index k = 1, nCells = 4;
    const double tau = 0.5, cj = 3.7, t = 0.0;

    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(diffusion_config);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(tau);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx);
    N_Vector dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(Y, dYdt);
    sys.updateBoundaryConditions(t);
    sys.updateMatricesForJacSolve();

    const Matrix J = jacobian(sys, Y, dYdt, t, cj);

    Eigen::FullPivLU<Matrix> lu(J);
    BOOST_TEST_MESSAGE("FD Jacobian: n = " << n << ", rank = " << lu.rank());

    // residual() does not write the two Dirichlet boundary rows -- those
    // constraints are imposed inside the linear solve (ApplyDirichletBCs /
    // H_global), not in the residual. So the finite-differenced Jacobian is
    // rank-deficient by exactly the number of Dirichlet boundaries, and those
    // rows must be excluded before comparing against what solveHDGJac returns.
    //
    // Identify them rather than hardcoding indices, and assert that the set is
    // exactly what we expect: the first and last lambda entry of each variable.
    std::vector<Index> emptyRows;
    for (Index i = 0; i < n; ++i)
        if (J.row(i).norm() == 0.0)
            emptyRows.push_back(i);

    const Index lambdaBase = static_cast<Index>(nCells) * 3 * (k + 1);
    std::vector<Index> expectedEmpty{lambdaBase, lambdaBase + static_cast<Index>(nCells)};
    BOOST_TEST(emptyRows == expectedEmpty, boost::test_tools::per_element());

    // Try several right-hand sides, including ones that excite every block.
    for (int trial = 0; trial < 3; ++trial)
    {
        N_Vector g = N_VNew_Serial(n, ctx);
        N_Vector dy = N_VClone(g);
        double *ga = N_VGetArrayPointer(g);
        for (Index i = 0; i < n; ++i)
            ga[i] = std::sin(1.0 + i * (trial + 1) * 0.7);

        Vector gVec(n);
        for (Index i = 0; i < n; ++i)
            gVec(i) = ga[i];

        sys.solveHDGJac(g, dy);

        Vector dyHDG(n);
        const double *dya = N_VGetArrayPointer(dy);
        for (Index i = 0; i < n; ++i)
            dyHDG(i) = dya[i];

        // The meaningful check: on every row the residual actually defines, the
        // vector solveHDGJac returns must satisfy J dy = g. This is what says
        // the hand-rolled static condensation really inverts the Jacobian.
        const Vector r = J * dyHDG - gVec;
        double num = 0.0, den = 0.0;
        for (Index i = 0; i < n; ++i)
        {
            if (std::find(emptyRows.begin(), emptyRows.end(), i) != emptyRows.end())
                continue;
            num += r(i) * r(i);
            den += gVec(i) * gVec(i);
        }
        const double resid = std::sqrt(num) / std::sqrt(den);
        BOOST_TEST_MESSAGE("  trial " << trial << ": ||J dy - g|| / ||g|| = " << resid
                                      << " (over defined rows)");
        BOOST_TEST(resid < 1e-6);

        N_VDestroy(g);
        N_VDestroy(dy);
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(solve_hdg_jac_is_linear_and_maps_zero_to_zero)
{
    const Index k = 2, nCells = 5;
    const double tau = 1.25, cj = 2.0, t = 0.0;

    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(diffusion_config);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(tau);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(Y, dYdt);
    sys.updateBoundaryConditions(t);
    sys.updateMatricesForJacSolve();

    N_Vector g = N_VNew_Serial(n, ctx), dy = N_VClone(g);

    // Zero in, zero out.
    N_VConst(0.0, g);
    sys.solveHDGJac(g, dy);
    BOOST_TEST(N_VMaxNorm(dy) == 0.0);

    // Linearity: solving for 2g must give twice the solution for g.
    N_Vector g2 = N_VClone(g), dy2 = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    double *g2a = N_VGetArrayPointer(g2);
    for (Index i = 0; i < n; ++i)
    {
        ga[i] = std::cos(0.3 * i);
        g2a[i] = 2.0 * ga[i];
    }

    sys.solveHDGJac(g, dy);
    sys.solveHDGJac(g2, dy2);

    const double *a = N_VGetArrayPointer(dy);
    const double *b = N_VGetArrayPointer(dy2);
    double maxdiff = 0.0, scale = 0.0;
    for (Index i = 0; i < n; ++i)
    {
        maxdiff = std::max(maxdiff, std::abs(b[i] - 2.0 * a[i]));
        scale = std::max(scale, std::abs(a[i]));
    }
    BOOST_TEST(maxdiff <= 1e-9 * std::max(1.0, scale));

    N_VDestroy(g);
    N_VDestroy(dy);
    N_VDestroy(g2);
    N_VDestroy(dy2);
    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(update_matrices_for_jac_solve_is_idempotent)
{
    // JacSetup calls this on every Jacobian evaluation; calling it twice with
    // the same state must not accumulate into the cached blocks.
    const Index k = 2, nCells = 4;
    const double tau = 0.75, cj = 1.5, t = 0.0;

    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(diffusion_config);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(tau);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(Y, dYdt);
    sys.updateBoundaryConditions(t);

    N_Vector g = N_VNew_Serial(n, ctx), dyOnce = N_VClone(g), dyTwice = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = 1.0 + 0.1 * i;

    sys.updateMatricesForJacSolve();
    sys.solveHDGJac(g, dyOnce);

    sys.updateMatricesForJacSolve();
    sys.updateMatricesForJacSolve();
    sys.solveHDGJac(g, dyTwice);

    const double *a = N_VGetArrayPointer(dyOnce);
    const double *b = N_VGetArrayPointer(dyTwice);
    for (Index i = 0; i < n; ++i)
        BOOST_TEST(a[i] == b[i], boost::test_tools::tolerance(1e-12));

    N_VDestroy(g);
    N_VDestroy(dyOnce);
    N_VDestroy(dyTwice);
    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

// ------------------------------------------ the superconvergent Jacobian --

namespace
{
// A flux and a source that both depend on u, so the chain rule through the
// postprocessing is actually exercised: with the superconvergent scheme u* enters
// both, and u* depends on the cell's u coefficients (through B12) *and* on its q
// coefficients (through B11). B11 is the only genuinely new coupling the scheme
// introduces, and a linear constant-coefficient case cannot see it at all.
class NonlinearDiffusion : public TransportSystem
{
public:
    NonlinearDiffusion() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    // sigma_hat = ( 1 + u^2 ) q
    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        const double u = s.u(0);
        return (1.0 + u * u) * s.q(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.u(0) * s.q(0);
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        const double u = s.u(0);
        v[0] = 1.0 + u * u;
    }

    // S = 1 + u - u^3 + 0.3 q, so the source depends on u and on q directly.
    Value Sources(Index, const State &s, Position x, Time) override
    {
        const double u = s.u(0);
        return 1.0 + u - u * u * u + 0.3 * s.q(0) + std::sin(3.0 * x);
    }
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        const double u = s.u(0);
        v[0] = 1.0 - 3.0 * u * u;
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.3;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }

    Value InitialValue(Index, Position x) const override
    {
        return 0.4 * std::sin(std::numbers::pi * x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        return 0.4 * std::numbers::pi * std::cos(std::numbers::pi * x);
    }
};

// A Jardin-shaped critical-gradient flux with the singularity
// regularised away: chi = 1 + 10[(t+eps)^0.5 - eps^0.5], t = max(|q| - qc, 0).
// The subtraction keeps chi continuous at the threshold and eps = 1e-3 caps
// dchi/dq at 158, so the derivative is bounded and smooth and a finite
// difference of the residual is trustworthy -- which it is not at eps = 0,
// where dchi/dq is unbounded and a difference quotient averages across it.
//
// This is the state a superconvergent steady solve fails from: the initial
// condition is 1 - x + sin(8 pi x), whose q sweeps through the critical
// gradient many times.
class KinkedDiffusion : public TransportSystem
{
public:
    KinkedDiffusion() : TransportSystem({.variables = numberedFields(1)}) {}

    static constexpr double chi0 = 1.0, kap = 10.0, qc = 0.5, eps = 1e-3, a = 0.5;

    static double chi(double q)
    {
        const double t = std::max(std::abs(q) - qc, 0.0);
        return t == 0.0 ? chi0 : chi0 + kap * (std::pow(t + eps, a) - std::pow(eps, a));
    }
    static double dchi(double q)
    {
        const double t = std::max(std::abs(q) - qc, 0.0);
        return t == 0.0 ? 0.0
                        : kap * a * std::pow(t + eps, a - 1.0) * (q > 0 ? 1.0 : -1.0);
    }

    Value LowerBoundary(Index, Time) const override { return 1.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position x, Time) override
    {
        return x * chi(s.q(0)) * s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return 1.0; }

    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position x, Time) override
    {
        v[0] = x * (chi(s.q(0)) + dchi(s.q(0)) * s.q(0));
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override
    {
        return 1.0 - x + std::sin(8.0 * std::numbers::pi * x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        return -1.0 + 8.0 * std::numbers::pi * std::cos(8.0 * std::numbers::pi * x);
    }
};

// The J dy = g check of the first test case in this file, parameterised on the
// flag and the physics so it can be run both ways.
template <class Problem>
double solveResidualRatio(Index k, Index nCells, double tau, double cj,
                          bool superconvergent, int trial)
{
    const double t = 0.0;
    Grid grid(0.0, 1.0, nCells);
    Problem problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(tau);
    sys.setSuperconvergent(superconvergent);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(Y, dYdt);
    sys.updateBoundaryConditions(t);
    sys.updateMatricesForJacSolve();

    const Matrix J = jacobian(sys, Y, dYdt, t, cj);

    std::vector<Index> emptyRows;
    for (Index i = 0; i < n; ++i)
        if (J.row(i).norm() == 0.0)
            emptyRows.push_back(i);

    N_Vector g = N_VNew_Serial(n, ctx), dy = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = std::sin(1.0 + i * (trial + 1) * 0.7);

    Vector gVec(n);
    for (Index i = 0; i < n; ++i)
        gVec(i) = ga[i];

    sys.solveHDGJac(g, dy);

    Vector dyHDG(n);
    const double *dya = N_VGetArrayPointer(dy);
    for (Index i = 0; i < n; ++i)
        dyHDG(i) = dya[i];

    const Vector r = J * dyHDG - gVec;
    double num = 0.0, den = 0.0;
    for (Index i = 0; i < n; ++i)
    {
        if (std::find(emptyRows.begin(), emptyRows.end(), i) != emptyRows.end())
            continue;
        num += r(i) * r(i);
        den += gVec(i) * gVec(i);
    }

    N_VDestroy(g);
    N_VDestroy(dy);
    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);

    return std::sqrt(num) / std::sqrt(den);
}
} // namespace

BOOST_AUTO_TEST_CASE(the_superconvergent_jacobian_matches_a_finite_difference_of_its_residual)
{
    // The decisive check on the chain rule through the postprocessing. The
    // Jacobian is never assembled, so an error in it costs Newton iterations
    // rather than accuracy -- which is precisely why it survives a passing
    // regression suite and has to be caught here.
    //
    // Finite-differencing residual() with the flag on and requiring
    // solveHDGJac to satisfy J dy = g pins every term at once: A9, V, B11 and
    // B12, and the star node set the derivatives are evaluated on.
    for (Index k : {1, 2, 3})
    {
        for (int trial = 0; trial < 3; ++trial)
        {
            const double resid = solveResidualRatio<NonlinearDiffusion>(
                k, 4, 0.5, 3.7, true, trial);
            BOOST_TEST_MESSAGE("superconvergent, k = " << k << ", trial " << trial
                                                       << ": ||J dy - g|| / ||g|| = "
                                                       << resid);
            BOOST_TEST(resid < 1e-6,
                       "k = " << k << ", trial " << trial
                              << ": the superconvergent Jacobian disagrees with a "
                                 "finite difference of its own residual (relative "
                                 "residual "
                              << resid << ")");
        }
    }
}

BOOST_AUTO_TEST_CASE(the_flag_off_jacobian_is_unaffected_by_the_new_code_path)
{
    // The same nonlinear case with the flag off, so a regression in the shared
    // plumbing is attributed correctly rather than blamed on the chain rule.
    for (Index k : {1, 2, 3})
    {
        const double resid =
            solveResidualRatio<NonlinearDiffusion>(k, 4, 0.5, 3.7, false, 0);
        BOOST_TEST_MESSAGE("flag off, k = " << k << ": ||J dy - g|| / ||g|| = "
                                           << resid);
        BOOST_TEST(resid < 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(the_superconvergent_jacobian_is_right_where_its_own_solve_fails)
{
    // Superconvergent = true cannot reach a steady state on Jardin's flux from
    // a perturbed initial condition, by either residual-driven solver and at
    // every degree, where the plain method converges every time. The obvious
    // suspicion is a wrong block in the chain rule through the postprocessing,
    // reachable only off the constraint manifold -- the case above differences
    // a *smooth* flux at a mild state, which is exactly where such a block
    // would not show.
    //
    // It is not that. This differences the residual at the state the solve
    // fails from, on the flux it fails on, and the superconvergent Jacobian is
    // as good as the plain one to seven digits. What the flag costs there is
    // the width of the Newton basin, not the accuracy of the linearisation:
    // pseudo-transient continuation with a small enough initial step converges
    // with the flag on, at 5090 transport-model calls against 5280 with it off.
    // See PERFORMANCE.md.
    //
    // Run at cj = 0 as well as cj != 0 deliberately, because cj = 0 is the
    // operator a steady solve actually uses and the mass term at cj = 3.7 would
    // otherwise dominate the comparison.
    for (Index k : {2, 3})
        for (double cj : {0.0, 3.7})
        {
            double plain = 0.0, star = 0.0;
            for (int trial = 0; trial < 3; ++trial)
            {
                plain = std::max(plain, solveResidualRatio<KinkedDiffusion>(
                                            k, 10, 1.0, cj, false, trial));
                star = std::max(star, solveResidualRatio<KinkedDiffusion>(
                                          k, 10, 1.0, cj, true, trial));
            }
            BOOST_TEST_MESSAGE(std::format(
                "  k={} cj={:<4}  ||J dy - g||/||g||: plain {:.3e}, superconvergent {:.3e}",
                k, cj, plain, star));

            BOOST_TEST(star < 1e-5,
                       "k = " << k << ", cj = " << cj
                              << ": the superconvergent Jacobian disagrees with a "
                                 "finite difference of its own residual ("
                              << star << ")");

            // And not merely small but no worse than the plain one, which is
            // the comparison that would catch a block that is right to within
            // the finite-difference floor and wrong beyond it. A factor of ten
            // of headroom, because both are finite-difference measurements.
            BOOST_TEST(star < 10.0 * plain,
                       "k = " << k << ", cj = " << cj << ": superconvergent " << star
                              << " against plain " << plain);
        }
}

BOOST_AUTO_TEST_CASE(solve_jac_eq_without_a_field_model_is_the_transport_solve)
{
    // The zero-coupling invariant, at the one place the field work edits the
    // path every existing run takes. solveJacEq now dispatches -- transport
    // solve, exact Schur solve, or (later) the iterative one -- and with no
    // field model attached it has to be the first of those *bit for bit*, not
    // merely to within a tolerance. A dispatcher that dropped through to a
    // coupled path with nField = 0 would produce the same answer by a different
    // route and cost every uncoupled run the difference.
    //
    // The coupled side of the same dispatch is checked in FieldJacobianTests.
    const Index k = 2, nCells = 4;
    const double tau = 0.75, cj = 1.5, t = 0.0;

    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(diffusion_config);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(tau);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(Y, dYdt);
    sys.updateBoundaryConditions(t);
    sys.updateMatricesForJacSolve();

    N_Vector g = N_VNew_Serial(n, ctx);
    N_Vector viaJacEq = N_VClone(g), viaTransport = N_VClone(g), viaHDG = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = std::sin(1.0 + i * 0.7);

    sys.solveJacEq(g, viaJacEq);
    sys.solveTransportJac(g, viaTransport);
    sys.solveHDGJac(g, viaHDG);

    const double *a = N_VGetArrayPointer(viaJacEq);
    const double *b = N_VGetArrayPointer(viaTransport);
    const double *c = N_VGetArrayPointer(viaHDG);
    for (Index i = 0; i < n; ++i)
    {
        BOOST_TEST(a[i] == b[i]);
        // ...and with no scalars either, the transport solve is solveHDGJac.
        BOOST_TEST(a[i] == c[i]);
    }

    N_VDestroy(g);
    N_VDestroy(viaJacEq);
    N_VDestroy(viaTransport);
    N_VDestroy(viaHDG);
    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

// ------------------------------------------- the scalar (Woodbury) path --
//
// Moved to ScalarJacobianTests.cpp, which settles it properly: the check here
// used to assert only that the solve returned finite numbers, because with
// ScalarTestLD3 the J dy = g residual came out O(1) and nothing distinguished a
// broken elimination from a physics case misreporting its own derivatives.
//
// It was the latter, three times over -- a sign error in dG_0/du, an
// uninitialised entry in dSources_dScalars, and dSources_dScalars_Mat
// integrating exactly where the residual interpolates. solveJacEq itself is
// correct, and is now verified against scalar systems whose Jacobians are known
// in closed form.

BOOST_AUTO_TEST_SUITE_END()
