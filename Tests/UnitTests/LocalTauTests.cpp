// Tests for tauScaling = Diffusive: the per-face tau = tau * (kappa/h + floor).
//
// Most of them run under both TauKappa sources -- Nodal, the cell's nodal
// d sigma_hat / d q extrapolated to its faces, and Face, evaluated at the face.
// A kappa polynomial in x of degree at most k is extrapolated exactly, which is
// what lets one closed form serve both.
//
// What can go wrong, and the test for each:
//
//   * the formula -- the wrong kappa, the wrong h, the wrong face or the wrong
//     floor. faceTau is compared against the closed form on a non-uniform grid
//     with a kappa that varies from face to face, so an indexing slip between a
//     cell's two faces or between cells cannot hide behind a constant.
//   * the state kappa is read at. Face reads the trace, and the Dirichlet datum
//     where the trace row is not solved; Nodal reads the cell's own u. Checked
//     against a u-dependent kappa, which tells the two apart.
//   * the extrapolation going non-positive, which Nodal replaces by the
//     extrapolation of log kappa. Checked at k = 1, where that has a closed form.
//   * the wiring. The residual and the Jacobian each apply tau from their own
//     state into separate copies of the tau blocks, so one copy can be updated and
//     the other forgotten. With a kappa that depends on x but not on the state,
//     tau is state independent and the Jacobian has to satisfy J dy = g against a
//     finite difference of the residual exactly as a constant tau does.
//   * d tau / dy. With a kappa that depends on the state, the finite difference
//     of the residual sees tau move and the Jacobian has to carry it; and with
//     tau frozen per continuation step neither moves, so the match is exact
//     again. Under Nodal it is chained through the cell's coefficients, so it
//     is checked with the superconvergent scheme -- where u* reaches q as well
//     as u -- with a kappa reading q, and with one reading an aux variable.
//
// What a Constant scaling does is pinned elsewhere, and more strongly: the
// regression configs produce byte-identical output with the scaling machinery in
// place, which no tolerance here could say.

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <algorithm>
#include <cmath>
#include <numbers>

// For BOOST_DATA_TEST_CASE's log, which finds it by argument-dependent lookup
// in the enum's own (global) namespace.
static std::ostream &operator<<(std::ostream &os, SystemSolver::TauKappa kappa)
{
    return os << (kappa == SystemSolver::TauKappa::Nodal ? "Nodal" : "Face");
}

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

// sigma_hat = (1 + u) q with u > 0: kappa is affine in u, so the cell's
// interpolant of its nodal kappa is exactly 1 + u_h, and its value at a face is
// the cell's own one-sided u -- not the trace -- under Nodal.
class AffineStateKappa : public StateKappa
{
public:
    Value SigmaFn(Index, const State &s, Position, Time) override { return (1.0 + s.u(0)) * s.q(0); }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override { v[0] = s.q(0); }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override { v[0] = 1.0 + s.u(0); }
};

// sigma_hat = (1 + u^2 + q^2 / 4) q: kappa depends on q as well as u, which is
// the component the superconvergent u* adds a path through.
class GradientKappa : public StateKappa
{
public:
    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return (1.0 + s.u(0) * s.u(0) + 0.25 * s.q(0) * s.q(0)) * s.q(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.u(0) * s.q(0);
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 1.0 + s.u(0) * s.u(0) + 0.75 * s.q(0) * s.q(0);
    }
};

// sigma_hat = (1 + phi) q with the auxiliary constraint phi = u^2: kappa reads
// only the aux variable.
class AuxKappa : public TransportSystem
{
public:
    AuxKappa() : TransportSystem({.variables = numberedFields(1), .aux = numberedAux(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.3; }

    Value SigmaFn(Index, const State &s, Position, Time) override { return (1.0 + s.phi(0)) * s.q(0); }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override { v[0] = 1.0 + s.phi(0); }
    void dSigma_dPhi(Index, VectorRef v, const State &s, Position, Time) override { v[0] = s.q(0); }

    Value Sources(Index, const State &s, Position x, Time) override { return 1.0 - s.u(0) + std::sin(3.0 * x); }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = -1.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dPhi(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value AuxG(Index, const State &s, Position, Time) override { return s.phi(0) - s.u(0) * s.u(0); }
    void AuxGPrime(Index, State &out, const State &s, Position, Time) override
    {
        out.u(0) = -2.0 * s.u(0);
        out.phi(0) = 1.0;
    }

    static double u0(Position x) { return 0.3 * x + 0.4 * std::sin(std::numbers::pi * x); }
    Value InitialValue(Index, Position x) const override { return u0(x); }
    Value InitialDerivative(Index, Position x) const override
    {
        return 0.3 + 0.4 * std::numbers::pi * std::cos(std::numbers::pi * x);
    }
    Value InitialAuxValue(Index, Position x) const override { return u0(x) * u0(x); }
};

// sigma_hat = ((x - 0.03)^2 + 1e-4) (1 + u^2) q. On the uneven grid's first
// cell, [0, 0.1], kappa has its minimum just inside the axis face, and at k = 1
// the line through the two nodes extrapolates to a negative value there.
class DippingKappa : public StateKappa
{
public:
    static double shape(Position x) { return (x - 0.03) * (x - 0.03) + 1e-4; }
    Value SigmaFn(Index, const State &s, Position x, Time) override
    {
        return shape(x) * (1.0 + s.u(0) * s.u(0)) * s.q(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position x, Time) override
    {
        v[0] = shape(x) * 2.0 * s.u(0) * s.q(0);
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position x, Time) override
    {
        v[0] = shape(x) * (1.0 + s.u(0) * s.u(0));
    }
};

const std::vector<Position> uneven{0.0, 0.1, 0.25, 0.5, 0.8, 1.0};

using Kappa = SystemSolver::TauKappa;
const Kappa bothKappas[] = {Kappa::Nodal, Kappa::Face};
char const *name(Kappa kappa) { return kappa == Kappa::Nodal ? "Nodal" : "Face"; }

// The initial state of `problem` on `grid`, in a solver configured Diffusive.
struct Setup
{
    Grid grid;
    SystemSolver sys;
    SUNContext ctx;
    N_Vector Y, dYdt;
    Index nAux;

    Setup(TransportSystem &problem, Index k, double tau, double floorFraction,
          SystemSolver::TauUpdate update = SystemSolver::TauUpdate::Residual,
          Kappa kappa = Kappa::Nodal, bool superconvergent = false)
        : grid(uneven), sys(grid, k, &problem), nAux(problem.getNumAux())
    {
        sys.setTau(tau);
        sys.setTauScaling(SystemSolver::TauScaling::Diffusive, floorFraction, update, kappa);
        sys.setSuperconvergent(superconvergent);
        sys.resetCoeffs();
        sys.initialiseMatrices();
        SUNContext_Create(SUN_COMM_NULL, &ctx);
        DGSoln shape(problem.getNumVars(), grid, k, Index{0}, problem.getNumAux());
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
    DGSoln view() const
    {
        return DGSoln(1, grid, sys.getOrder(), N_VGetArrayPointer(Y), 0, nAux);
    }
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

BOOST_DATA_TEST_CASE(diffusive_tau_is_kappa_over_h_plus_the_floor_on_each_face,
                     boost::unit_test::data::make(bothKappas), kappa)
{
    // kappa = 1 + 3 x^2 is quadratic, so at k = 3 the nodal extrapolation is
    // exact and the same closed form holds for both sources.
    VariableKappa problem;
    const double tauMult = 2.5, floorFraction = 0.01;
    Setup s(problem, 3, tauMult, floorFraction, SystemSolver::TauUpdate::Residual, kappa);

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
        BOOST_TEST(tau(0, 2 * i) == lower, boost::test_tools::tolerance(1e-12));
        BOOST_TEST(tau(0, 2 * i + 1) == upper, boost::test_tools::tolerance(1e-12));
    }

    // One-sided: the two cells meeting at a face see the same kappa and their
    // own h, so on this grid the two sides of every interior face differ.
    for (Index i = 0; i + 1 < nCells; ++i)
        BOOST_TEST(tau(0, 2 * i + 1) != tau(0, 2 * (i + 1)));

    // setInitialConditions has already put it where the dydt solve reads it.
    BOOST_TEST((s.sys.tauRes - tau).norm() == 0.0);
}

BOOST_AUTO_TEST_CASE(face_kappa_reads_the_trace_and_the_dirichlet_datum)
{
    StateKappa problem;
    const double tauMult = 1.0, floorFraction = 1e-3;
    Setup s(problem, 2, tauMult, floorFraction, SystemSolver::TauUpdate::Residual, Kappa::Face);
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

BOOST_AUTO_TEST_CASE(nodal_kappa_reads_the_cells_own_u)
{
    // kappa = 1 + u is affine in u, so the interpolant of its nodal values is
    // 1 + u_h exactly, and at a face it is the cell's own one-sided u_h -- which
    // is neither the trace nor the Dirichlet datum, and differs from both here.
    AffineStateKappa problem;
    const double tauMult = 1.0, floorFraction = 1e-3;
    Setup s(problem, 2, tauMult, floorFraction);
    DGSoln Y = s.view();

    const Matrix tau = s.sys.faceTau(Y, 0.0);
    const Index nCells = s.grid.getNCells();

    Vector kh(2 * nCells);
    for (Index i = 0; i < nCells; ++i)
    {
        Interval const &I = s.grid[i];
        auto const &coeffs = Y.u(0).getCoeff(i).second;
        double lower = 0.0, upper = 0.0;
        for (Index j = 0; j < 3; ++j)
        {
            lower += coeffs(j) * Y.getBasis().Evaluate(I, j, I.x_l);
            upper += coeffs(j) * Y.getBasis().Evaluate(I, j, I.x_u);
        }
        kh(2 * i) = (1.0 + lower) / hOf(s.grid, i);
        kh(2 * i + 1) = (1.0 + upper) / hOf(s.grid, i);
    }
    const Vector expected = tauMult * (kh.array() + floorFraction * kh.maxCoeff()).matrix();
    for (Index p = 0; p < 2 * nCells; ++p)
        BOOST_TEST(tau(0, p) == expected(p), boost::test_tools::tolerance(1e-12));

    // And it is not what the face would read: the trace at an interior node
    // differs from both cells' one-sided values.
    const double traceKappa = 1.0 + Y.lambda(0)[1];
    BOOST_TEST(std::abs(kh(1) * hOf(s.grid, 0) - traceKappa) > 1e-6);
}

BOOST_AUTO_TEST_CASE(where_the_nodal_line_goes_negative_log_kappa_is_extrapolated)
{
    // k = 1, so each cell's interpolant is the line through its two nodes and
    // the log branch has a closed form: the line through (x_j, log kappa_j).
    DippingKappa problem;
    const double tauMult = 1.0, floorFraction = 1e-3;
    Setup s(problem, 1, tauMult, floorFraction);
    DGSoln Y = s.view();
    const Matrix tau = s.sys.faceTau(Y, 0.0);

    Interval const &I = s.grid[0];
    std::vector<Position> const points = Y.getPoints();
    const Position x0 = points[0], x1 = points[1];
    auto kappaAt = [&](Index node, Position x)
    {
        const double u = Y.u(0).getCoeff(0).second(node);
        return DippingKappa::shape(x) * (1.0 + u * u);
    };
    const double k0 = kappaAt(0, x0), k1 = kappaAt(1, x1);
    const double slope = (I.x_l - x0) / (x1 - x0);

    // The premise: the straight line is negative at the axis face.
    BOOST_TEST_REQUIRE(k0 + (k1 - k0) * slope < 0.0);
    // The axis carries a flux condition only if declared so; this one is
    // Dirichlet at both ends, so the face keeps its own kappa.
    const double logKappa = std::exp(std::log(k0) + (std::log(k1) - std::log(k0)) * slope);
    const double scale = (tau.row(0).array() / tauMult).maxCoeff() / (1.0 + floorFraction);
    BOOST_TEST(tau(0, 0) > 0.0);
    BOOST_TEST(tau(0, 0) == tauMult * (logKappa / hOf(s.grid, 0) + floorFraction * scale),
               boost::test_tools::tolerance(1e-10));
}

BOOST_DATA_TEST_CASE(a_flux_condition_face_takes_kappa_from_its_cell_where_its_own_vanishes,
                     boost::unit_test::data::make(bothKappas), kappa)
{
    // kappa = x is zero on the axis. Left to its own kappa the axis face would
    // sit at the floor; it takes the cell's other face instead. The Dirichlet
    // end is untouched by the rule -- there tau weighs the boundary mismatch
    // against the flux, and a small one is the point.
    AxisKappa problem;
    const double tauMult = 1.0, floorFraction = 1e-3;
    Setup s(problem, 2, tauMult, floorFraction, SystemSolver::TauUpdate::Residual, kappa);
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
                   boost::test_tools::tolerance(1e-12));
    BOOST_TEST(tau(0, 0) > 100 * tauMult * floorFraction * scale);
}

BOOST_DATA_TEST_CASE(with_a_state_independent_tau_the_diffusive_jacobian_is_exact,
                     boost::unit_test::data::make(bothKappas), kappa)
{
    // The wiring check. Any tau block the Jacobian forgot to refresh -- MBlocks'
    // D, CEBlocks' E, CG_cellwise's G, H_jac_cellwise -- is left at the constant
    // initialiseMatrices wrote, which on this grid is nowhere near kappa/h.
    VariableKappa problem;
    for (Index k : {1, 2, 3})
    {
        Setup s(problem, k, 1.0, 1e-3, SystemSolver::TauUpdate::Residual, kappa);
        for (int trial = 0; trial < 3; ++trial)
        {
            const double resid = jacobianMismatch(s, 3.7, trial);
            BOOST_TEST_MESSAGE(name(kappa) << ", k = " << k << ", trial " << trial
                                           << ": ||J dy - g|| / ||g|| = " << resid);
            BOOST_TEST(resid < 1e-6);
        }
    }
}

BOOST_DATA_TEST_CASE(with_a_state_dependent_tau_the_jacobian_carries_dtau_dy,
                     boost::unit_test::data::make(bothKappas), kappa)
{
    // tau follows the residual, so the finite difference of the residual sees
    // d tau / dy; the Jacobian carries it by a finite difference of its own over
    // the state kappa is read from. What it deliberately leaves out is the
    // floor's dependence on the grid maximum, so the match is limited by the
    // floor -- which is why this runs at two -- and by the two finite
    // differences. Without the term at all the mismatch here was 4.2e-4.
    StateKappa problem;
    for (double floorFraction : {1e-3, 1e-9})
    {
        Setup s(problem, 2, 1.0, floorFraction, SystemSolver::TauUpdate::Residual, kappa);
        for (int trial = 0; trial < 3; ++trial)
        {
            const double resid = jacobianMismatch(s, 3.7, trial);
            BOOST_TEST_MESSAGE(name(kappa) << ", floor " << floorFraction << ", trial " << trial
                                           << ": ||J dy - g|| / ||g|| = " << resid);
            // At a negligible floor, the finite-difference noise of a constant
            // tau: nothing else is missing.
            BOOST_TEST(resid < (floorFraction < 1e-6 ? 1e-7 : 1e-5));
        }
    }
}

namespace
{
// The nodal d tau / dy against a finite difference of the residual, at a
// negligible floor, so nothing is missing from it but finite-difference noise.
template <class Problem>
void checkNodalDtauDy(char const *label, Index k, bool superconvergent)
{
    Problem problem;
    Setup s(problem, k, 1.0, 1e-9, SystemSolver::TauUpdate::Residual, Kappa::Nodal,
            superconvergent);
    for (int trial = 0; trial < 3; ++trial)
    {
        const double resid = jacobianMismatch(s, 3.7, trial);
        BOOST_TEST_MESSAGE(label << ", k = " << k << (superconvergent ? ", superconvergent" : "")
                                 << ", trial " << trial << ": ||J dy - g|| / ||g|| = " << resid);
        BOOST_TEST(resid < 1e-7, label << " k = " << k << " sc = " << superconvergent);
    }
}
} // namespace

BOOST_AUTO_TEST_CASE(the_nodal_dtau_dy_is_chained_through_every_component)
{
    // Through u and q (GradientKappa), the aux variable (AuxKappa) and the log
    // branch (DippingKappa, at k = 1 where its axis face takes it). Each with
    // and without the superconvergent scheme, whose u* reaches q through B11:
    // a missing path there is the error this would show.
    for (bool sc : {false, true})
    {
        checkNodalDtauDy<GradientKappa>("GradientKappa", 2, sc);
        checkNodalDtauDy<GradientKappa>("GradientKappa", 3, sc);
        checkNodalDtauDy<AuxKappa>("AuxKappa", 2, sc);
        checkNodalDtauDy<DippingKappa>("DippingKappa", 1, sc);
    }
}

BOOST_DATA_TEST_CASE(a_tau_frozen_for_the_step_gives_an_exact_jacobian,
                     boost::unit_test::data::make(bothKappas), kappa)
{
    // setInitialConditions freezes tau at the initial state and nothing in the
    // residual or the Jacobian build moves it, so this is a fixed-coefficient
    // problem again and the Jacobian has nothing to approximate.
    StateKappa problem;
    Setup s(problem, 2, 1.0, 1e-3, SystemSolver::TauUpdate::ContinuationStep, kappa);
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
