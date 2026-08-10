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
        const double u = s.Variable[0];
        return (1.0 + u * u) * s.Derivative[0];
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.Variable[0] * s.Derivative[0];
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        const double u = s.Variable[0];
        v[0] = 1.0 + u * u;
    }

    // S = 1 + u - u^3 + 0.3 q, so the source depends on u and on q directly.
    Value Sources(Index, const State &s, Position x, Time) override
    {
        const double u = s.Variable[0];
        return 1.0 + u - u * u * u + 0.3 * s.Derivative[0] + std::sin(3.0 * x);
    }
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        const double u = s.Variable[0];
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
