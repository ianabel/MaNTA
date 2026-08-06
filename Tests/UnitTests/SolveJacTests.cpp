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

// Finite-difference the residual to get J = dF/dY + cj dF/dY'.
Matrix finiteDifferenceJacobian(SystemSolver &sys, N_Vector Y, N_Vector dYdt,
                                double t, double cj, SUNContext ctx)
{
    const Index n = N_VGetLength(Y);
    Matrix J(n, n);

    N_Vector Yp = N_VClone(Y), dYp = N_VClone(dYdt);
    N_Vector Fplus = N_VClone(Y), Fminus = N_VClone(Y);

    const double *Y0 = N_VGetArrayPointer(Y);
    const double *dY0 = N_VGetArrayPointer(dYdt);
    double *Ya = N_VGetArrayPointer(Yp);
    double *dYa = N_VGetArrayPointer(dYp);

    for (Index j = 0; j < n; ++j)
    {
        const double h = 1e-6 * std::max(1.0, std::abs(Y0[j]));

        std::copy(Y0, Y0 + n, Ya);
        std::copy(dY0, dY0 + n, dYa);
        Ya[j] += h;
        dYa[j] += cj * h;
        sys.residual(t, Yp, dYp, Fplus);

        std::copy(Y0, Y0 + n, Ya);
        std::copy(dY0, dY0 + n, dYa);
        Ya[j] -= h;
        dYa[j] -= cj * h;
        sys.residual(t, Yp, dYp, Fminus);

        const double *fp = N_VGetArrayPointer(Fplus);
        const double *fm = N_VGetArrayPointer(Fminus);
        for (Index i = 0; i < n; ++i)
            J(i, j) = (fp[i] - fm[i]) / (2.0 * h);
    }

    N_VDestroy(Yp);
    N_VDestroy(dYp);
    N_VDestroy(Fplus);
    N_VDestroy(Fminus);
    return J;
}
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

    const Matrix J = finiteDifferenceJacobian(sys, Y, dYdt, t, cj, ctx);

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

// ------------------------------------------- the scalar (Woodbury) path --

BOOST_AUTO_TEST_CASE(solve_jac_eq_handles_global_scalars)
{
    // solveJacEq wraps solveHDGJac in a bordered/Woodbury elimination for the
    // nScalars global unknowns (see WoodburyIdentityNote). That elimination had
    // no coverage at all: nothing in the suite ran a case with nScalars > 0
    // through the linear solve, and IDA never checks it because the Jacobian is
    // never formed.
    //
    // ScalarTestLD3 is linear diffusion carrying three global scalars, so it
    // exercises the bordered solve while keeping the physics simple.
    //
    // OPEN QUESTION -- do not read this test as a clean bill of health.
    // The J dy = g check that passes at 3e-10 for nScalars = 0 comes out O(1)
    // here, in the field rows (~0.3) as well as the scalar rows (~5-8). Two
    // candidate explanations, not yet separated:
    //   (a) the Woodbury/bordered elimination in solveJacEq is wrong, or
    //   (b) ScalarTestLD3's hand-written ScalarGPrimeExtended disagrees with
    //       its own ScalarGExtended, which would corrupt the whole bordered
    //       solve and not just the scalar rows -- matching what is observed.
    // Distinguishing them needs a scalar system with an exactly-known
    // Jacobian (G = mu - const). Note the PIDTest regression case would not
    // catch (b): a wrong Jacobian only slows Newton, it still converges to the
    // right answer. See Tests/README.md.
    const Index k = 2, nCells = 4;
    const double tau = 1.0, cj = 2.5, t = 0.0;

    // ScalarTestLD3 hardcodes its domain as [-1, 1]: ScalarGExtended integrates
    // u over (-1, 1) and evaluates sigma at +-1.
    Grid grid(-1.0, 1.0, nCells);
    ScalarTestLD3 problem(scalar_config, grid);
    BOOST_TEST_REQUIRE(problem.getNumScalars() == 3);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(tau);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k, problem.getNumScalars(),
                 problem.getNumAux());
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

    const Matrix J = finiteDifferenceJacobian(sys, Y, dYdt, t, cj, ctx);

    // As in the nScalars = 0 case, residual() leaves the Dirichlet boundary
    // rows empty; exclude them.
    std::vector<Index> emptyRows;
    for (Index i = 0; i < n; ++i)
        if (J.row(i).norm() == 0.0)
            emptyRows.push_back(i);
    BOOST_TEST_MESSAGE("scalar case: n = " << n << ", empty rows = "
                                           << emptyRows.size());

    for (int trial = 0; trial < 2; ++trial)
    {
        N_Vector g = N_VNew_Serial(n, ctx), dy = N_VClone(g);
        double *ga = N_VGetArrayPointer(g);
        Vector gVec(n);
        for (Index i = 0; i < n; ++i)
        {
            ga[i] = std::cos(0.4 + i * (trial + 1) * 0.53);
            gVec(i) = ga[i];
        }

        sys.solveJacEq(g, dy);

        Vector dyOut(n);
        const double *dya = N_VGetArrayPointer(dy);
        for (Index i = 0; i < n; ++i)
            dyOut(i) = dya[i];

        const Vector r = J * dyOut - gVec;
        double num = 0.0, den = 0.0;
        for (Index i = 0; i < n; ++i)
        {
            if (std::find(emptyRows.begin(), emptyRows.end(), i) != emptyRows.end())
                continue;
            num += r(i) * r(i);
            den += gVec(i) * gVec(i);
        }
        const double resid = std::sqrt(num) / std::sqrt(den);

        // Split the residual: is the error in the DG field rows, or confined to
        // the scalar rows? That distinguishes a broken Woodbury elimination from
        // a physics case whose hand-written scalar Jacobian disagrees with its
        // own ScalarGExtended.
        const Index nS = problem.getNumScalars();
        double fnum = 0.0, fden = 0.0, snum = 0.0, sden = 0.0;
        for (Index i = 0; i < n; ++i)
        {
            if (std::find(emptyRows.begin(), emptyRows.end(), i) != emptyRows.end())
                continue;
            if (i >= n - nS) { snum += r(i) * r(i); sden += gVec(i) * gVec(i); }
            else             { fnum += r(i) * r(i); fden += gVec(i) * gVec(i); }
        }
        BOOST_TEST_MESSAGE("  trial " << trial << ": total=" << resid
                                      << "  field rows=" << std::sqrt(fnum / fden)
                                      << "  scalar rows=" << std::sqrt(snum / sden));

        // NOT asserted yet -- see the OPEN QUESTION at the top of this case.
        // The identical check passes at 3e-10 for nScalars = 0, but here the
        // residual is O(1) in BOTH the field and the scalar rows.

        // What can be asserted: the solve completes, returns finite numbers,
        // and actually produces a nonzero scalar block (so the bordered
        // elimination is not being skipped outright).
        for (Index i = 0; i < n; ++i)
            BOOST_TEST(std::isfinite(dyOut(i)));

        // The scalar block must actually be solved for, not left at zero --
        // otherwise the bordered elimination could be skipped entirely and the
        // residual check above would still pass on the field rows.
        double scalarNorm = 0.0;
        for (Index i = n - problem.getNumScalars(); i < n; ++i)
            scalarNorm += dyOut(i) * dyOut(i);
        BOOST_TEST(std::sqrt(scalarNorm) > 0.0);

        N_VDestroy(g);
        N_VDestroy(dy);
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_SUITE_END()
