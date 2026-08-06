// Tests for AdjointProblem's defaults and for the adjoint derivative vectors
// SystemSolver builds from them.
//
// Context worth knowing before reading: end-to-end adjoint output is currently
// dead -- WriteAdjoints() is commented out at Solver.cpp:350 (commit 57d2652,
// "adjoint writing doesn't work for spatial adjoints"), so no run emits the
// gradients and both the regression and Python suites skip their adjoint
// comparison. That makes unit coverage the *only* thing standing behind this
// machinery, and the existing AdjointTests.cpp only ever asserts that
// dGdq_Vec/dGdsigma_Vec/dGdaux_Vec are zero -- true for AdjointTestProblem,
// whose integrand genuinely has no q, sigma or phi dependence, but vacuous as
// a test. The mock here gives all four a nonzero derivative.

#include <boost/test/unit_test.hpp>

#include "AdjointProblem.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"
#include "Types.hpp"

#include <boost/math/quadrature/gauss.hpp>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <stdexcept>

namespace
{

// Two variables, one aux variable, so every derivative slot has somewhere
// distinct to go.
class AdjointHostSystem : public TransportSystem
{
public:
    AdjointHostSystem()
    {
        nVars = 2;
        nAux = 1;
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    bool isLowerBoundaryDirichlet(Index) const override { return true; }
    bool isUpperBoundaryDirichlet(Index) const override { return true; }

    Value SigmaFn(Index i, const State &s, Position, Time) override
    {
        return s.Derivative[i];
    }
    Value Sources(Index, const State &s, Position, Time) override { return s.Aux[0]; }

    void dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
        v[i] = 1.0;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    }
    void dSources_dPhi(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
        v[0] = 1.0;
    }
    void dSigma_dPhi(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    }
    Value AuxG(Index, const State &s, Position, Time) override
    {
        return s.Aux[0] - s.Variable[0];
    }
    void AuxGPrime(Index, State &out, const State &, Position, Time) override
    {
        out.zero();
        out.Variable[0] = -1.0;
        out.Aux[0] = 1.0;
    }

    Value InitialValue(Index i, Position x) const override
    {
        return (1.0 + i) * x * (1.0 - x) + 0.3;
    }
    Value InitialDerivative(Index i, Position x) const override
    {
        return (1.0 + i) * (1.0 - 2.0 * x);
    }
    Value InitialAuxValue(Index, Position x) const override { return 0.7 + x; }
};

// Implements only the pure virtuals, so every default below is the base one.
// np = 3 with np_boundary = 1 makes getNpInternal() = 2, which is the split the
// dSigma/dSources defaults loop over -- if they used np instead they would run
// off the end of the p-derivative block.
class MockAdjoint : public AdjointProblem
{
public:
    MockAdjoint()
    {
        ng = 2;
        np = 3;
        np_boundary = 1;
    }

    // The pointwise overrides below would otherwise hide the base's batched
    // and vector-valued overloads of the same names, which are the defaults
    // this suite exists to test.
    using AdjointProblem::dGFndp;
    using AdjointProblem::gFn;

    Value GFn(Index gIndex, DGSoln &) const override { return 100.0 + gIndex; }
    Value dGFndp(Index gIndex, Index pIndex, DGSoln &) const override
    {
        return 1.0 + 10.0 * gIndex + pIndex;
    }

    // A g whose derivatives are nonzero in *every* slot, so the four dgFn_*
    // hooks can be told apart in the assembled vectors.
    Value gFn(Index gIndex, const State &s, Position x) const override
    {
        return (1.0 + gIndex) * (s.Variable[0] * s.Variable[0] + 2.0 * s.Derivative[0] +
                                 3.0 * s.Flux[0] + 4.0 * s.Aux[0] + x);
    }

    void dgFn_du(Index gIndex, VectorRef v, const State &s, Position) override
    {
        v.setZero();
        v[0] = (1.0 + gIndex) * 2.0 * s.Variable[0];
    }
    void dgFn_dq(Index gIndex, VectorRef v, const State &, Position x) override
    {
        v.setZero();
        v[0] = (1.0 + gIndex) * (2.0 + x);
    }
    void dgFn_dsigma(Index gIndex, VectorRef v, const State &, Position x) override
    {
        v.setZero();
        v[0] = (1.0 + gIndex) * (3.0 - 0.5 * x);
    }
    void dgFn_dphi(Index gIndex, VectorRef v, const State &, Position x) override
    {
        v.setZero();
        v[0] = (1.0 + gIndex) * (4.0 + 2.0 * x);
    }

    void dSigmaFn_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        out = 1.0 + i + 10.0 * pIndex + x + s.Variable[0];
    }
    void dSources_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        out = -2.0 - i - 20.0 * pIndex + 2.0 * x + s.Derivative[0];
    }
};

GlobalState makeStates(Index nCells, Index k, Index nVars, Index nAux)
{
    GlobalState g(nCells, k, nVars, 0, nAux);
    for (Index j = 0; j < static_cast<Index>(g.size()); ++j)
    {
        State s(nVars, 0, nAux);
        for (Index v = 0; v < nVars; ++v)
        {
            s.Variable[v] = 0.2 * (j + 1) + v;
            s.Derivative[v] = -0.4 * (j + 1) + v;
            s.Flux[v] = 0.6 * (j + 1) - v;
        }
        for (Index a = 0; a < nAux; ++a)
            s.Aux[a] = 0.11 * (j + 1);
        g.setWithState(j, s);
    }
    return g;
}

std::vector<Position> makePoints(Index n)
{
    std::vector<Position> p(n);
    for (Index j = 0; j < n; ++j)
        p[j] = 0.09 * (j + 1);
    return p;
}

} // namespace

BOOST_AUTO_TEST_SUITE(adjoint_problem_tests)

// ------------------------------------------------------ index bookkeeping --

BOOST_AUTO_TEST_CASE(parameter_index_split_is_internal_then_boundary)
{
    // np is split into np_boundary trailing boundary parameters and the rest
    // internal. Several defaults loop to getNpInternal(), and F_p is assembled
    // with the boundary block handled separately, so this split has to hold.
    MockAdjoint adj;
    BOOST_TEST(adj.getNg() == 2);
    BOOST_TEST(adj.getNp() == 3);
    BOOST_TEST(adj.getNpBoundary() == 1);
    BOOST_TEST(adj.getNpInternal() == 2);

    BOOST_TEST(adj.isAdjointIndexInternal(0));
    BOOST_TEST(adj.isAdjointIndexInternal(1));
    BOOST_TEST(!adj.isAdjointIndexInternal(2));

    BOOST_TEST(adj.areParametersSpatial() == false);
    BOOST_TEST(adj.getName(2) == "p2");
    BOOST_TEST(adj.computeUpperBoundarySensitivity(0, 0) == false);
    BOOST_TEST(adj.computeLowerBoundarySensitivity(0, 0) == false);
}

BOOST_AUTO_TEST_CASE(vector_dgfndp_gathers_the_scalar_form_over_internal_indices)
{
    // The row-vector overload fills only the internal parameters and leaves the
    // boundary entries at zero -- their sensitivity comes from a different
    // route entirely.
    MockAdjoint adj;
    Grid grid(0.0, 1.0, 2);
    DGSoln y(2, grid, 1);

    const Matrix row = adj.dGFndp(1, y);
    BOOST_TEST_REQUIRE(row.rows() == 1);
    BOOST_TEST_REQUIRE(row.cols() == adj.getNp());

    for (Index p = 0; p < adj.getNpInternal(); ++p)
        BOOST_TEST(row(0, p) == adj.dGFndp(1, p, y));
    BOOST_TEST(row(0, adj.getNp() - 1) == 0.0);
}

BOOST_AUTO_TEST_CASE(dgfndp_over_a_global_state_is_python_only_and_says_so)
{
    // The batched dgFndp has no C++ implementation; it exists for the pybind11
    // trampoline to override. Calling it from C++ must fail loudly rather than
    // return garbage.
    MockAdjoint adj;
    GlobalState states = makeStates(2, 1, 2, 1);
    const auto pts = makePoints(static_cast<Index>(states.size()));
    BOOST_CHECK_THROW(adj.dgFndp(0, states, pts), std::runtime_error);
}

// -------------------------------------------------------- batched defaults --

BOOST_AUTO_TEST_CASE(batched_g_matches_the_pointwise_form)
{
    MockAdjoint adj;
    GlobalState states = makeStates(2, 2, 2, 1);
    const auto pts = makePoints(static_cast<Index>(states.size()));

    for (Index g = 0; g < adj.getNg(); ++g)
    {
        const Values out = adj.gFn(g, states, pts);
        BOOST_TEST_REQUIRE(out.size() == static_cast<Index>(pts.size()));
        for (size_t j = 0; j < pts.size(); ++j)
            BOOST_TEST(out(j) == adj.gFn(g, states[j], pts[j]),
                       boost::test_tools::tolerance(1e-14));
    }
}

BOOST_AUTO_TEST_CASE(dg_routes_each_hook_to_its_own_state_slice)
{
    // dgFn_du -> Variable, dgFn_dq -> Derivative, dgFn_dsigma -> Flux,
    // dgFn_dphi -> Aux. Those four land in different blocks of the adjoint
    // right-hand side, so a swap would silently solve the wrong adjoint.
    MockAdjoint adj;
    const Index nCells = 2, k = 2, nVars = 2, nAux = 1;
    GlobalState states = makeStates(nCells, k, nVars, nAux);
    const auto pts = makePoints(static_cast<Index>(states.size()));

    GlobalState out(nCells, k, nVars, 0, nAux);
    adj.dg(1, out, states, pts);

    Vector refV(nVars), refA(nAux);
    for (size_t j = 0; j < states.size(); ++j)
    {
        const State s = states[j];
        adj.dgFn_du(1, refV, s, pts[j]);
        BOOST_TEST((out.Variable(j) - refV).norm() < 1e-14);
        adj.dgFn_dq(1, refV, s, pts[j]);
        BOOST_TEST((out.Derivative(j) - refV).norm() < 1e-14);
        adj.dgFn_dsigma(1, refV, s, pts[j]);
        BOOST_TEST((out.Flux(j) - refV).norm() < 1e-14);
        adj.dgFn_dphi(1, refA, s, pts[j]);
        BOOST_TEST((out.Aux(j) - refA).norm() < 1e-14);
    }

    // Not vacuous: all four blocks are nonzero and mutually different.
    BOOST_TEST(out.Variable().norm() > 1e-3);
    BOOST_TEST(out.Derivative().norm() > 1e-3);
    BOOST_TEST(out.Flux().norm() > 1e-3);
    BOOST_TEST(out.Aux().norm() > 1e-3);
}

BOOST_AUTO_TEST_CASE(parameter_derivatives_fill_only_the_internal_indices)
{
    // dSigma/dSources reuse the Variable slot to hold p-derivatives -- the
    // comment in AdjointProblem.hpp calls this out. They loop pIndex over
    // getNpInternal(), so out.Variable(j)(np-1) must stay untouched.
    MockAdjoint adj;
    const Index nCells = 2, k = 1, nVars = 3, nAux = 1;
    GlobalState states = makeStates(nCells, k, nVars, nAux);
    const auto pts = makePoints(static_cast<Index>(states.size()));

    GlobalState out(nCells, k, nVars, 0, nAux);
    // Poison the block so an untouched entry is distinguishable from a zero.
    out.Variable().setConstant(-999.0);

    adj.dSigma(0, out, states, pts);
    for (size_t j = 0; j < states.size(); ++j)
    {
        for (Index p = 0; p < adj.getNpInternal(); ++p)
        {
            Value expected = 0.0;
            adj.dSigmaFn_dp(0, p, expected, states[j], pts[j]);
            BOOST_TEST(out.Variable(j)(p) == expected,
                       boost::test_tools::tolerance(1e-14));
        }
        BOOST_TEST(out.Variable(j)(adj.getNp() - 1) == -999.0);
    }

    out.Variable().setConstant(-999.0);
    adj.dSources(1, out, states, pts);
    for (size_t j = 0; j < states.size(); ++j)
        for (Index p = 0; p < adj.getNpInternal(); ++p)
        {
            Value expected = 0.0;
            adj.dSources_dp(1, p, expected, states[j], pts[j]);
            BOOST_TEST(out.Variable(j)(p) == expected,
                       boost::test_tools::tolerance(1e-14));
        }
}

BOOST_AUTO_TEST_CASE(compute_physics_derivatives_dispatches_to_all_three)
{
    // The default ComputePhysicsDerivatives loops dSigma_vals.size() (not
    // nVars) and dAux_vals.size(), so the sizes come from the caller's
    // GlobalStateMatrix rather than the problem. Pin that.
    MockAdjoint adj;
    const Index nCells = 2, k = 1, nVars = 2, nAux = 1;
    GlobalState states = makeStates(nCells, k, nVars, nAux);
    const auto pts = makePoints(static_cast<Index>(states.size()));

    GlobalStateMatrix dSigma(nVars), dSource(nVars), dAux(nAux);
    for (Index v = 0; v < nVars; ++v)
    {
        dSigma.add(nCells, k, nVars, 0, nAux);
        dSource.add(nCells, k, nVars, 0, nAux);
    }
    for (Index a = 0; a < nAux; ++a)
        dAux.add(nCells, k, nVars, 0, nAux);

    // dAux_dp has no C++ implementation, so the aux leg must fail loudly.
    BOOST_CHECK_THROW(adj.ComputePhysicsDerivatives({dSigma, dSource, dAux}, states, pts),
                      std::logic_error);

    // With no aux parameters requested the other two legs run cleanly.
    GlobalStateMatrix noAux(0);
    BOOST_CHECK_NO_THROW(
        adj.ComputePhysicsDerivatives({dSigma, dSource, noAux}, states, pts));

    for (Index i = 0; i < nVars; ++i)
        for (size_t j = 0; j < states.size(); ++j)
            for (Index p = 0; p < adj.getNpInternal(); ++p)
            {
                Value expected = 0.0;
                adj.dSigmaFn_dp(i, p, expected, states[j], pts[j]);
                BOOST_TEST(dSigma[i].Variable(j)(p) == expected,
                           boost::test_tools::tolerance(1e-14));
            }
}

BOOST_AUTO_TEST_CASE(missing_aux_parameter_derivative_is_reported)
{
    // The default dAux_dp exists only to complain. It must actually throw --
    // constructing a std::logic_error and dropping it on the floor would leave
    // the caller's output at whatever was in the buffer, which is how a silent
    // wrong-gradient bug gets into the adjoint solve.
    MockAdjoint adj;
    Value out = 0.0;
    State s(2, 0, 1);
    BOOST_CHECK_THROW(adj.dAux_dp(0, 0, out, s, 0.5), std::logic_error);
}

// -------------------------------- the adjoint vectors SystemSolver builds --

BOOST_AUTO_TEST_CASE(adjoint_derivative_vectors_match_gauss_quadrature)
{
    // dGdu_Vec / dGdq_Vec / dGdsigma_Vec / dGdaux_Vec each compute
    //     Vec(var*(k+1) + j) = Int_I dg/dZ_var * phi_j dx
    // by the basis's own Gauss rule. AdjointTests.cpp checks dGdu_Vec this way
    // but only asserts the other three are *zero*, which they are for its
    // fixture; here all four integrands are nonzero and each is compared with
    // an independent 30-point rule.
    const Index k = 2, nCells = 4, nVars = 2;
    Grid grid(0.0, 1.0, nCells);
    AdjointHostSystem problem;
    MockAdjoint adjoint;

    SystemSolver sys(grid, k, &problem);
    sys.setAdjointProblem(&adjoint);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(nVars, grid, k, Index(0), Index(1));
    N_Vector Y = N_VNew_Serial(shape.getDoF(), ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    auto integrator = boost::math::quadrature::gauss<double, 30>();
    const Index gIndex = 1;

    Vector actual(nVars * (k + 1));
    for (Index cell = 0; cell < nCells; ++cell)
    {
        Interval const &I = grid[cell];

        auto reference = [&](void (AdjointProblem::*hook)(Index, VectorRef, const State &,
                                                          Position),
                             Index nComponents)
        {
            Vector ref(nVars * (k + 1));
            ref.setZero();
            for (Index var = 0; var < nComponents; ++var)
                for (Index j = 0; j < k + 1; ++j)
                {
                    auto integrand = [&](double x)
                    {
                        Values grad(std::max(nVars, nComponents));
                        grad.setZero();
                        State s = sys.y.eval(x);
                        (adjoint.*hook)(gIndex, grad, s, x);
                        return grad(var) * sys.y.getBasis().Evaluate(I, j, x);
                    };
                    ref(var * (k + 1) + j) = integrator.integrate(integrand, I.x_l, I.x_u);
                }
            return ref;
        };

        sys.dGdu_Vec(gIndex, actual, sys.y, cell);
        BOOST_TEST((actual - reference(&AdjointProblem::dgFn_du, nVars)).norm() < 1e-10,
                   "dGdu_Vec, cell " << cell);
        BOOST_TEST(actual.norm() > 1e-6);

        sys.dGdq_Vec(gIndex, actual, sys.y, cell);
        BOOST_TEST((actual - reference(&AdjointProblem::dgFn_dq, nVars)).norm() < 1e-10,
                   "dGdq_Vec, cell " << cell);
        BOOST_TEST(actual.norm() > 1e-6);

        sys.dGdsigma_Vec(gIndex, actual, sys.y, cell);
        BOOST_TEST((actual - reference(&AdjointProblem::dgFn_dsigma, nVars)).norm() < 1e-10,
                   "dGdsigma_Vec, cell " << cell);
        BOOST_TEST(actual.norm() > 1e-6);

        // dGdaux_Vec loops over nAux, not nVars, so only the first nAux blocks
        // of the vector are written.
        sys.dGdaux_Vec(gIndex, actual, sys.y, cell);
        BOOST_TEST((actual - reference(&AdjointProblem::dgFn_dphi, 1)).norm() < 1e-10,
                   "dGdaux_Vec, cell " << cell);
        BOOST_TEST(actual.norm() > 1e-6);
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(the_interpolatory_derivative_sub_vector_agrees_with_the_quadrature_one)
{
    // DerivativeSubVector has the same two-overload structure as
    // DerivativeSubMatrix: one integrates the hook by quadrature, the other
    // projects precomputed nodal values. They differ in general (interpolation
    // versus exact integration) but must agree when the integrand is a
    // polynomial the basis represents -- dgFn_dq here is affine in x.
    const Index k = 2, nCells = 3, nVars = 2;
    Grid grid(0.0, 1.0, nCells);
    AdjointHostSystem problem;
    MockAdjoint adjoint;

    SystemSolver sys(grid, k, &problem);
    sys.setAdjointProblem(&adjoint);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(nVars, grid, k, Index(0), Index(1));
    N_Vector Y = N_VNew_Serial(shape.getDoF(), ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    GlobalState dGdvars(nCells, k, nVars, 0, 1);
    adjoint.dg(1, dGdvars, sys.y.evalOnNodes(), sys.y.getPoints());

    Vector viaQuadrature(nVars * (k + 1)), viaNodes(nVars * (k + 1));
    for (Index cell = 0; cell < nCells; ++cell)
    {
        sys.dGdq_Vec(1, viaQuadrature, sys.y, cell);
        sys.DerivativeSubVector(1, viaNodes, dGdvars.cellwiseDerivative(cell), sys.y, cell);

        // InterpolateOntoBasis reproduces an affine integrand exactly, so the
        // two routes coincide to round-off here. A tolerance any looser would
        // stop distinguishing "same formula" from "same ballpark".
        BOOST_TEST((viaQuadrature - viaNodes).norm() < 1e-10,
                   "cell " << cell << ": quadrature and nodal forms differ by "
                           << (viaQuadrature - viaNodes).norm());
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_SUITE_END()
