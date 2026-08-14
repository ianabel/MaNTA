// Tests for aFn -- the coefficient a_i(x) on du/dt.
//
// The equation MaNTA integrates is
//
//     a_i(x) d_t u_i - d_x[ sigma_hat_i ] = S_i
//
// and a_i is the one physics hook with a *default* rather than a pure virtual:
// TransportSystem::aFn returns 1.0 unless a case overrides it. That combination --
// optional, defaulted to the identity, and reached from only two places -- is what
// makes it a bitrot risk. A case that never overrides it cannot tell whether it
// still works, and until this file the only test asserted that the *default* is 1
// (TransportSystemTests, aFn_defaults_to_one), which is true of a hook that has
// been disconnected entirely.
//
// What protection already existed, and what it misses. `ADTestProblem` overrides
// aFn as `afn_test * x` and is the only case in the tree that does; it runs in the
// regression suite as ADTest.conf and SuperconvergentADTest.conf, both with
// afn_test at its default 1.0, so a position-dependent a_i = x is compared against
// checked-in references on both the ordinary and the superconvergent path. That
// would catch aFn being dropped from the *residual*.
//
// It would not catch aFn being dropped from the *Jacobian*. There are two
// consumers -- SystemSolver.cpp builds XMats from aFn for the residual's X dudt
// term, and assembleCellMatrix builds alpha * aFn for the u block of MX -- and
// they have to agree, because the second is the derivative of the first with
// respect to u given that d(dudt)/du = alpha. Drop it from the Jacobian alone and
// the converged answer is unchanged: the regression references still pass and only
// Newton slows. That is the same failure mode CLAUDE.md records for every Jacobian
// block, and the finite-difference test at the end of this file is what closes it.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <boost/math/quadrature/gauss.hpp>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <filesystem>
#include <numbers>
#include <vector>

namespace
{

// Decoupled linear diffusion, one equation per variable, homogeneous Dirichlet at
// both ends and no source:
//
//     a_i d_t u_i = kappa d_xx u_i,     u_i(0) = u_i(1) = 0
//
// With u_i(x, 0) = sin(pi x) that has the closed-form solution
//
//     u_i(x, t) = sin(pi x) exp(-kappa pi^2 t / a_i)
//
// so a_i is a pure rescaling of time and nothing else. Two things follow, and the
// tests below use both: the solution at time t with a_i = A must equal the one at
// t/A with a_i = 1, and the steady state (zero) must not depend on a_i at all.
//
// The rescaling comparison is the sharper of the two. Both runs discretise the
// same eigenfunction on the same mesh, so they share the same *discrete*
// eigenvalue; the spatial error cancels between them exactly, and what is left is
// limited by the time integrator rather than by k.
class ACoefficient : public TransportSystem
{
public:
    ACoefficient(std::vector<double> a, bool weightByX = false, double kappa_ = 1.0)
        : TransportSystem({.variables = numberedFields(static_cast<Index>(a.size()))}),
          coeff(std::move(a)), byX(weightByX), kappa(kappa_)
    {
    }

    // The hook under test.
    Value aFn(Index i, Position x) override
    {
        return coeff.at(static_cast<size_t>(i)) * (byX ? x : 1.0);
    }

    Value SigmaFn(Index i, const State &s, Position, Time) override { return kappa * s.q(i); }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time) override
    {
        v[i] = kappa;
    }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

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

    std::vector<double> coeff;
    bool byX;
    double kappa;
};

const std::vector<double> SAMPLE{0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875};

// Integrate to `tFinal` and return u_var at SAMPLE.
Vector solved(TransportSystem &problem, double tFinal, std::string const &stem,
              Index var = 0, Index order = 4, Index nCells = 8)
{
    Grid grid(0.0, 1.0, nCells);
    SystemSolver sys(grid, order, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(tFinal);
    sys.setNOutput(5);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-12}, 1e-11);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(tFinal);
        sys.destroySundials();
    }
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);

    Vector out(static_cast<Eigen::Index>(SAMPLE.size()));
    for (size_t i = 0; i < SAMPLE.size(); ++i)
        out(static_cast<Eigen::Index>(i)) = sys.yJac.u(var)(SAMPLE[i]);
    return out;
}

// Assemble only, for the tests that read a matrix rather than an answer.
struct Assembled
{
    Grid grid;
    SystemSolver sys;

    Assembled(TransportSystem &problem, Index order, Index nCells, double tau = 1.0)
        : grid(0.0, 1.0, nCells), sys(grid, order, &problem)
    {
        sys.setTau(tau);
        sys.resetCoeffs();
        sys.initialiseMatrices();
    }
};

} // namespace

BOOST_AUTO_TEST_SUITE(afn_tests)

BOOST_AUTO_TEST_CASE(the_hook_exists_and_is_reached)
{
    // The weakest possible statement, first: an override is actually called, with
    // the variable index and the position it claims to take. A hook that had been
    // disconnected from the solver would still pass this, which is why it is only
    // the start.
    ACoefficient problem({2.0, 5.0}, /* weightByX */ true);
    BOOST_TEST(problem.aFn(0, 3.0) == 6.0);
    BOOST_TEST(problem.aFn(1, 3.0) == 15.0);

    // And the base-class default is the identity, so a case that ignores aFn gets
    // the equation it expects.
    struct Plain : public ACoefficient
    {
        Plain() : ACoefficient({1.0}) {}
        using TransportSystem::aFn; // un-hide the base version
    };
    Plain plain;
    BOOST_TEST(static_cast<TransportSystem &>(plain).aFn(0, 0.7) == 1.0);
}

BOOST_AUTO_TEST_CASE(a_constant_coefficient_rescales_time)
{
    // The load-bearing test that aFn reaches the *residual*. a_i = A makes the
    // equation the A = 1 equation with t -> t/A, so these two runs must agree.
    //
    // They share a mesh and an eigenfunction, hence a discrete eigenvalue, so the
    // spatial error cancels exactly between them: what is compared is the effect
    // of A alone.
    const double T = 0.02, A = 4.0;

    ACoefficient one({1.0});
    ACoefficient four({A});

    const Vector u1 = solved(one, T, "afn_rescale_1");
    const Vector u4 = solved(four, A * T, "afn_rescale_4");

    BOOST_TEST(u1.norm() > 1e-6); // the run has to have done something
    BOOST_TEST((u1 - u4).cwiseAbs().maxCoeff() < 1e-8);
}

BOOST_AUTO_TEST_CASE(the_decay_rate_matches_the_closed_form)
{
    // The same statement against an absolute reference rather than against another
    // run, so that a *pair* of runs both wrong in the same way cannot pass. Looser
    // than the comparison above because the spatial error no longer cancels.
    const double T = 0.02, A = 3.0, kappa = 1.0;
    ACoefficient problem({A}, false, kappa);
    const Vector got = solved(problem, T, "afn_closed_form");

    const double decay = std::exp(-kappa * std::numbers::pi * std::numbers::pi * T / A);
    Vector want(static_cast<Eigen::Index>(SAMPLE.size()));
    for (size_t i = 0; i < SAMPLE.size(); ++i)
        want(static_cast<Eigen::Index>(i)) = std::sin(std::numbers::pi * SAMPLE[i]) * decay;

    BOOST_TEST((got - want).cwiseAbs().maxCoeff() < 1e-6);

    // And the sign of the effect: a larger a_i must decay *slower*, so a run that
    // divided by a_i the wrong way round cannot pass either.
    ACoefficient faster({1.0}, false, kappa);
    const Vector quick = solved(faster, T, "afn_closed_form_fast");
    BOOST_TEST(quick.cwiseAbs().maxCoeff() < got.cwiseAbs().maxCoeff());
}

BOOST_AUTO_TEST_CASE(the_coefficient_is_per_variable)
{
    // Two decoupled variables with different a_i. A mis-slotted index -- var 1
    // getting var 0's coefficient, say -- makes them relax at the same rate, which
    // nothing in a single-variable test could see. numberedFields gives both the
    // same boundary conditions and the fixture the same initial condition, so the
    // only difference between them is aFn.
    const double T = 0.02, A = 4.0;
    ACoefficient problem({1.0, A});

    const Vector fast = solved(problem, T, "afn_pervar_a", /* var */ 0);
    ACoefficient again({1.0, A});
    const Vector slow = solved(again, A * T, "afn_pervar_b", /* var */ 1);

    // var 1 at A*T must match var 0 at T.
    BOOST_TEST((fast - slow).cwiseAbs().maxCoeff() < 1e-8);

    // ... and must *not* match var 0 at the same time, or the two coefficients are
    // being treated alike.
    ACoefficient third({1.0, A});
    const Vector sameTime = solved(third, T, "afn_pervar_c", /* var */ 1);
    BOOST_TEST((fast - sameTime).cwiseAbs().maxCoeff() > 1e-3);
}

BOOST_AUTO_TEST_CASE(a_position_dependent_coefficient_is_integrated_not_sampled)
{
    // aFn is a *weight* on the mass matrix, X_ij = Int a(x) phi_i phi_j dx, not a
    // scalar multiplying it. A version that evaluated a at the cell centre instead
    // would be right to O(h^2) and wrong in a way no convergence study would
    // notice, so this compares against an independent quadrature.
    //
    // ADTestProblem, the only case in the tree overriding aFn, returns
    // afn_test * x, so this is the shape that is actually in use.
    const Index order = 3, nCells = 4;
    ACoefficient problem({1.0}, /* weightByX */ true);
    Assembled a(problem, order, nCells);

    auto const &basis = a.sys.y.getBasis();
    boost::math::quadrature::gauss<double, 12> quad;

    for (Index cell = 0; cell < nCells; ++cell)
    {
        Interval const &I(a.grid[cell]);
        for (Index i = 0; i <= order; ++i)
            for (Index j = 0; j <= order; ++j)
            {
                const double want = quad.integrate(
                    [&](double x) { return x * basis.Evaluate(I, i, x) * basis.Evaluate(I, j, x); },
                    I.x_l, I.x_u);
                BOOST_TEST(a.sys.XMats[cell](i, j) == want,
                           boost::test_tools::tolerance(1e-11));
            }
    }

    // Not merely proportional to the unweighted mass matrix: check the boundary
    // cell, where the weight varies most in relative terms.
    ACoefficient unweighted({1.0}, /* weightByX */ false);
    Assembled u(unweighted, order, nCells);
    const double mid = 0.5 * (a.grid[0].x_l + a.grid[0].x_u);
    BOOST_TEST((a.sys.XMats[0] - mid * u.sys.XMats[0]).cwiseAbs().maxCoeff() > 1e-6);
}

BOOST_AUTO_TEST_CASE(the_jacobian_agrees_with_the_residual_for_a_nonunit_coefficient)
{
    // The gap the regression references cannot close. aFn feeds two places that
    // must agree: XMats, which the residual applies to dudt, and alpha * aFn in
    // the u block of MX. Drop it from the second and the converged answer does not
    // move -- only Newton slows -- so ADTest.conf would still pass.
    //
    // Finite-difference the residual and require J dy = g, with alpha nonzero so
    // the mass term is actually present in the Jacobian, and a_i both nonunit and
    // position-dependent so that neither a missing factor nor a mis-weighted one
    // could pass.
    for (bool superconvergent : {false, true})
    {
        ACoefficient problem({2.5, 1.0}, /* weightByX */ true);
        const Index order = 3, nCells = 3;
        Grid grid(0.0, 1.0, nCells);
        SystemSolver sys(grid, order, &problem);
        sys.setTau(1.0);
        sys.setSuperconvergent(superconvergent);
        sys.resetCoeffs();
        sys.setInputFile("afn_fdjac");
        sys.setOutputCadence(1.0);
        sys.setNOutput(3);
        sys.setInitialTime(0.0);
        sys.setMinStepSize(1e-14);
        sys.setTolerances({1e-10}, 1e-8);
        sys.setWriteOutput(false);
        sys.setWriteDatFile(false);

        SUNContext ctx = nullptr;
        SUNContext_Create(SUN_COMM_NULL, &ctx);
        {
            CapturedOutput quiet;
            sys.initialize();
        }

        const Index n = N_VGetLength(sys.Y);
        const double t = 0.0, cj = 3.0; // cj = alpha; nonzero, so X is in MX
        sys.setJacTime(t);
        sys.setAlpha(cj);
        sys.setJacEvalY(sys.Y, sys.dYdt);
        sys.updateBoundaryConditions(t);
        sys.updateMatricesForJacSolve();

        const Matrix J = fdjac::jacobian(sys, sys.Y, sys.dYdt, t, cj);
        const std::vector<Index> skip = fdjac::undefinedRows(J);

        // Dirichlet at both ends of both variables: four undefined rows.
        BOOST_TEST(skip.size() == 4u);

        N_Vector g = N_VNew_Serial(n, ctx);
        N_Vector dy = N_VClone(g);
        double *ga = N_VGetArrayPointer(g);
        for (Index i = 0; i < n; ++i)
            ga[i] = 0.05 + 0.01 * static_cast<double>((i * 11) % 17);
        {
            CapturedOutput quiet;
            sys.solveJacEq(g, dy);
        }

        Vector dyv(n), gv(n);
        const double *dya = N_VGetArrayPointer(dy);
        for (Index i = 0; i < n; ++i)
        {
            dyv(i) = dya[i];
            gv(i) = ga[i];
        }
        const double res = fdjac::relativeResidual(J, dyv, gv, skip);
        BOOST_TEST_MESSAGE("superconvergent = " << superconvergent
                                               << ": ||J dy - g|| / ||g|| = " << res);
        BOOST_TEST(res < 1e-6);

        N_VDestroy(g);
        N_VDestroy(dy);
        {
            CapturedOutput quiet;
            sys.destroySundials();
        }
        SUNContext_Free(&ctx);
    }
}

BOOST_AUTO_TEST_SUITE_END()
