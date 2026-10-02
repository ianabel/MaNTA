// Tests for tauScaling = Diffusive: the per-face tau = tau * (kappa/h + floor).
//
// Three things can go wrong, and each has its own test here:
//
//   * the formula -- the wrong kappa, the wrong h, the wrong face or the wrong
//     floor. faceTau is compared against the closed form on a non-uniform grid
//     with a kappa that varies from face to face, so an indexing slip between a
//     cell's two faces or between cells cannot hide behind a constant.
//   * the state kappa is read at -- the trace, and the Dirichlet datum where the
//     trace row is not solved. Checked against a u-dependent kappa.
//   * the wiring. The residual and the Jacobian each apply tau from their own
//     state into separate copies of the tau blocks, so one copy can be updated and
//     the other forgotten. With a kappa that depends on x but not on the state,
//     tau is state independent and the Jacobian has to satisfy J dy = g against a
//     finite difference of the residual exactly as a constant tau does.
//   * d tau / dy. With a kappa that depends on u, the finite difference of the
//     residual sees tau move and the Jacobian has to carry it; and with tau
//     frozen per continuation step neither moves, so the match is exact again.
//
// What a Constant scaling does is pinned elsewhere, and more strongly: the
// regression configs produce byte-identical output with the scaling machinery in
// place, which no tolerance here could say.

#include <boost/test/unit_test.hpp>

#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace
{
using fdjac::jacobian;

// sigma_hat = kappa(x) q with kappa = 1 + 3 x^2: linear in the state, so tau does
// not depend on it, but different on every face.
class VariableKappa : public TransportSystem
{
public:
    VariableKappa() : TransportSystem({.variables = numberedFields(1)}) {}

    static double kappa(Position x) { return 1.0 + 3.0 * x * x; }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position x, Time) override { return kappa(x) * s.q(0); }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position x, Time) override { v[0] = kappa(x); }

    Value Sources(Index, const State &, Position x, Time) override { return std::sin(3.0 * x); }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return 0.4 * std::sin(std::numbers::pi * x); }
    Value InitialDerivative(Index, Position x) const override
    {
        return 0.4 * std::numbers::pi * std::cos(std::numbers::pi * x);
    }
};

// sigma_hat = (1 + u^2) q, Dirichlet u = 0.3 at the upper end: kappa depends on
// the state, and on which u a face reads.
class StateKappa : public TransportSystem
{
public:
    StateKappa() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.3; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return (1.0 + s.u(0) * s.u(0)) * s.q(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.u(0) * s.q(0);
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 1.0 + s.u(0) * s.u(0);
    }

    Value Sources(Index, const State &s, Position x, Time) override
    {
        return 1.0 + s.u(0) - s.u(0) * s.u(0) * s.u(0) + std::sin(3.0 * x);
    }
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 1.0 - 3.0 * s.u(0) * s.u(0);
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return 0.3 * x + 0.4 * std::sin(std::numbers::pi * x); }
    Value InitialDerivative(Index, Position x) const override
    {
        return 0.3 + 0.4 * std::numbers::pi * std::cos(std::numbers::pi * x);
    }
};

// sigma_hat = x q, zero flux on the axis: kappa vanishes at x = 0 for a reason
// that has nothing to do with resolution, which is the case the flux-face rule in
// faceKappaOverH exists for.
class AxisKappa : public TransportSystem
{
public:
    AxisKappa()
        : TransportSystem({.variables = {{"u", "", "", BoundaryCondition::mixed(0.0, 0.0, 1.0),
                                          BoundaryKind::Dirichlet}}})
    {
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position x, Time) override { return x * s.q(0); }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position x, Time) override { v[0] = x; }

    Value Sources(Index, const State &, Position, Time) override { return 1.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return 1.0 - x * x; }
    Value InitialDerivative(Index, Position x) const override { return -2.0 * x; }
};

const std::vector<Position> uneven{0.0, 0.1, 0.25, 0.5, 0.8, 1.0};

// The initial state of `problem` on `grid`, in a solver configured Diffusive.
struct Setup
{
    Grid grid;
    SystemSolver sys;
    SUNContext ctx;
    N_Vector Y, dYdt;

    Setup(TransportSystem &problem, Index k, double tau, double floorFraction,
          SystemSolver::TauUpdate update = SystemSolver::TauUpdate::Residual)
        : grid(uneven), sys(grid, k, &problem)
    {
        sys.setTau(tau);
        sys.setTauScaling(SystemSolver::TauScaling::Diffusive, floorFraction, update);
        sys.resetCoeffs();
        sys.initialiseMatrices();
        SUNContext_Create(SUN_COMM_NULL, &ctx);
        DGSoln shape(problem.getNumVars(), grid, k);
        Y = N_VNew_Serial(shape.getDoF(), ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);
    }
    ~Setup()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
    DGSoln view() const { return DGSoln(1, grid, sys.getOrder(), N_VGetArrayPointer(Y)); }
};

double hOf(Grid const &grid, Index i) { return grid[i].x_u - grid[i].x_l; }

// ||J dy - g|| / ||g|| over the rows the residual defines, as SolveJacTests does.
double jacobianMismatch(Setup &s, double cj, int trial)
{
    const double t = 0.0;
    s.sys.setJacTime(t);
    s.sys.setAlpha(cj);
    s.sys.setJacEvalY(s.Y, s.dYdt);
    s.sys.updateBoundaryConditions(t);
    s.sys.updateMatricesForJacSolve();

    const Matrix J = jacobian(s.sys, s.Y, s.dYdt, t, cj);
    const Index n = J.rows();

    N_Vector g = N_VClone(s.Y), dy = N_VClone(s.Y);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = std::sin(1.0 + i * (trial + 1) * 0.7);
    Vector gVec = Eigen::Map<Vector>(ga, n);

    s.sys.solveHDGJac(g, dy);
    const Vector r = J * Eigen::Map<Vector>(N_VGetArrayPointer(dy), n) - gVec;

    double num = 0.0, den = 0.0;
    for (Index i = 0; i < n; ++i)
    {
        if (J.row(i).norm() == 0.0)
            continue; // a Dirichlet trace row; see SolveJacTests.cpp
        num += r(i) * r(i);
        den += gVec(i) * gVec(i);
    }
    N_VDestroy(g);
    N_VDestroy(dy);
    return std::sqrt(num / den);
}
} // namespace

BOOST_AUTO_TEST_SUITE(local_tau_tests)

BOOST_AUTO_TEST_CASE(a_constant_scaling_puts_the_configured_tau_on_every_face)
{
    VariableKappa problem;
    Grid grid(uneven);
    SystemSolver sys(grid, 2, &problem);
    sys.setTau(0.7);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    DGSoln Y(1, grid, 2);
    const Matrix tau = sys.faceTau(Y, 0.0);
    BOOST_TEST(tau.rows() == 1);
    BOOST_TEST(tau.cols() == 2 * static_cast<Index>(grid.getNCells()));
    BOOST_TEST((tau.array() == 0.7).all());
    BOOST_TEST((sys.tauRes.array() == 0.7).all());
    BOOST_TEST((sys.tauJac.array() == 0.7).all());
}

BOOST_AUTO_TEST_CASE(diffusive_tau_is_kappa_over_h_plus_the_floor_on_each_face)
{
    VariableKappa problem;
    const double tauMult = 2.5, floorFraction = 0.01;
    Setup s(problem, 3, tauMult, floorFraction);

    const Matrix tau = s.sys.faceTau(s.view(), 0.0);
    const Index nCells = s.grid.getNCells();

    double scale = 0.0;
    for (Index i = 0; i < nCells; ++i)
        for (Position x : {s.grid[i].x_l, s.grid[i].x_u})
            scale = std::max(scale, VariableKappa::kappa(x) / hOf(s.grid, i));

    for (Index i = 0; i < nCells; ++i)
    {
        const double h = hOf(s.grid, i);
        const double lower = tauMult * (VariableKappa::kappa(s.grid[i].x_l) / h + floorFraction * scale);
        const double upper = tauMult * (VariableKappa::kappa(s.grid[i].x_u) / h + floorFraction * scale);
        BOOST_TEST(tau(0, 2 * i) == lower, boost::test_tools::tolerance(1e-13));
        BOOST_TEST(tau(0, 2 * i + 1) == upper, boost::test_tools::tolerance(1e-13));
    }

    // One-sided: the two cells meeting at a face see the same kappa and their
    // own h, so on this grid the two sides of every interior face differ.
    for (Index i = 0; i + 1 < nCells; ++i)
        BOOST_TEST(tau(0, 2 * i + 1) != tau(0, 2 * (i + 1)));

    // setInitialConditions has already put it where the dydt solve reads it.
    BOOST_TEST((s.sys.tauRes - tau).norm() == 0.0);
}

BOOST_AUTO_TEST_CASE(diffusive_tau_reads_the_trace_and_the_dirichlet_datum)
{
    StateKappa problem;
    const double tauMult = 1.0, floorFraction = 1e-3;
    Setup s(problem, 2, tauMult, floorFraction);
    DGSoln Y = s.view();

    const Matrix tau = s.sys.faceTau(Y, 0.0);
    const Index nCells = s.grid.getNCells();

    // u on each face: the trace inside, the boundary datum at both ends (both
    // are Dirichlet in this case).
    auto uFace = [&](Index node) -> double {
        if (node == 0)
            return problem.LowerBoundary(0, 0.0);
        if (node == nCells)
            return problem.UpperBoundary(0, 0.0);
        return Y.lambda(0)[node];
    };
    Vector kh(2 * nCells);
    for (Index i = 0; i < nCells; ++i)
    {
        kh(2 * i) = (1.0 + std::pow(uFace(i), 2)) / hOf(s.grid, i);
        kh(2 * i + 1) = (1.0 + std::pow(uFace(i + 1), 2)) / hOf(s.grid, i);
    }
    const Vector expected = tauMult * (kh.array() + floorFraction * kh.maxCoeff()).matrix();
    for (Index p = 0; p < 2 * nCells; ++p)
        BOOST_TEST(tau(0, p) == expected(p), boost::test_tools::tolerance(1e-13));
}

BOOST_AUTO_TEST_CASE(a_flux_condition_face_takes_kappa_from_its_cell_where_its_own_vanishes)
{
    // kappa = x is zero on the axis. Left to its own kappa the axis face would
    // sit at the floor; it takes the cell's other face instead. The Dirichlet
    // end is untouched by the rule -- there tau weighs the boundary mismatch
    // against the flux, and a small one is the point.
    AxisKappa problem;
    const double tauMult = 1.0, floorFraction = 1e-3;
    Setup s(problem, 2, tauMult, floorFraction);
    const Matrix tau = s.sys.faceTau(s.view(), 0.0);
    const Index nCells = s.grid.getNCells();

    Vector kh(2 * nCells);
    for (Index i = 0; i < nCells; ++i)
    {
        kh(2 * i) = s.grid[i].x_l / hOf(s.grid, i);
        kh(2 * i + 1) = s.grid[i].x_u / hOf(s.grid, i);
    }
    kh(0) = kh(1); // the rule
    const double scale = kh.maxCoeff();
    for (Index p = 0; p < 2 * nCells; ++p)
        BOOST_TEST(tau(0, p) == tauMult * (kh(p) + floorFraction * scale),
                   boost::test_tools::tolerance(1e-13));
    BOOST_TEST(tau(0, 0) > 100 * tauMult * floorFraction * scale);
}

BOOST_AUTO_TEST_CASE(with_a_state_independent_tau_the_diffusive_jacobian_is_exact)
{
    // The wiring check. Any tau block the Jacobian forgot to refresh -- MBlocks'
    // D, CEBlocks' E, CG_cellwise's G, H_jac_cellwise -- is left at the constant
    // initialiseMatrices wrote, which on this grid is nowhere near kappa/h.
    VariableKappa problem;
    for (Index k : {1, 2, 3})
    {
        Setup s(problem, k, 1.0, 1e-3);
        for (int trial = 0; trial < 3; ++trial)
        {
            const double resid = jacobianMismatch(s, 3.7, trial);
            BOOST_TEST_MESSAGE("k = " << k << ", trial " << trial << ": ||J dy - g|| / ||g|| = " << resid);
            BOOST_TEST(resid < 1e-6);
        }
    }
}

BOOST_AUTO_TEST_CASE(with_a_state_dependent_tau_the_jacobian_carries_dtau_dy)
{
    // tau follows the residual, so the finite difference of the residual sees
    // d tau / dy; the Jacobian carries it by a finite difference of its own over
    // the face state. What it deliberately leaves out is the floor's dependence on
    // the grid maximum, so the match is limited by the floor -- which is why this
    // runs at two -- and by the two finite differences. Without the term at all
    // the mismatch here was 4.2e-4.
    StateKappa problem;
    for (double floorFraction : {1e-3, 1e-9})
    {
        Setup s(problem, 2, 1.0, floorFraction);
        for (int trial = 0; trial < 3; ++trial)
        {
            const double resid = jacobianMismatch(s, 3.7, trial);
            BOOST_TEST_MESSAGE("floor " << floorFraction << ", trial " << trial
                                        << ": ||J dy - g|| / ||g|| = " << resid);
            // At a negligible floor, the finite-difference noise of a constant
            // tau: nothing else is missing.
            BOOST_TEST(resid < (floorFraction < 1e-6 ? 1e-7 : 1e-5));
        }
    }
}

BOOST_AUTO_TEST_CASE(a_tau_frozen_for_the_step_gives_an_exact_jacobian)
{
    // setInitialConditions freezes tau at the initial state and nothing in the
    // residual or the Jacobian build moves it, so this is a fixed-coefficient
    // problem again and the Jacobian has nothing to approximate.
    StateKappa problem;
    Setup s(problem, 2, 1.0, 1e-3, SystemSolver::TauUpdate::ContinuationStep);
    const Matrix frozen = s.sys.tauRes;
    BOOST_TEST((s.sys.tauJac - frozen).norm() == 0.0);
    for (int trial = 0; trial < 3; ++trial)
    {
        const double resid = jacobianMismatch(s, 3.7, trial);
        BOOST_TEST_MESSAGE("frozen, trial " << trial << ": ||J dy - g|| / ||g|| = " << resid);
        BOOST_TEST(resid < 1e-6);
    }
    BOOST_TEST((s.sys.tauRes - frozen).norm() == 0.0);
    BOOST_TEST((s.sys.tauJac - frozen).norm() == 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
