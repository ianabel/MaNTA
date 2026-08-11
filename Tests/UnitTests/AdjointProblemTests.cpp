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
        : TransportSystem({.variables = numberedFields(2), .aux = numberedAux(1)})
    {
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index i, const State &s, Position, Time) override
    {
        return s.q(i);
    }
    Value Sources(Index, const State &s, Position, Time) override { return s.phi(0); }

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
        return s.phi(0) - s.u(0);
    }
    void AuxGPrime(Index, State &out, const State &, Position, Time) override
    {
        out.zero();
        out.u(0) = -1.0;
        out.phi(0) = 1.0;
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
        return (1.0 + gIndex) * (s.u(0) * s.u(0) + 2.0 * s.q(0) +
                                 3.0 * s.sigma(0) + 4.0 * s.phi(0) + x);
    }

    void dgFn_du(Index gIndex, VectorRef v, const State &s, Position) override
    {
        v.setZero();
        v[0] = (1.0 + gIndex) * 2.0 * s.u(0);
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
        out = 1.0 + i + 10.0 * pIndex + x + s.u(0);
    }
    void dSources_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        out = -2.0 - i - 20.0 * pIndex + 2.0 * x + s.q(0);
    }
};

// An objective that is a real functional of the state, whose GFn integrates its
// own gFn and whose dgFn_* hooks really are that gFn's derivatives.
//
// MockAdjoint above cannot serve here on either count. Its GFn returns the
// constant 100 + gIndex, so it does not depend on the state at all, and its
// dgFn_dq/dsigma/dphi carry x-dependence its gFn does not -- deliberately, so
// that the four hooks can be told apart in the assembled vectors. A
// finite-difference check needs all of it to line up.
//
// g is nonlinear in u and depends on q, sigma and phi as well. Every part of that
// is load-bearing. A g linear in u cannot tell a correct dG/dt from evaluating
// the objective functional on the derivative vector, which is what
// origin/optimize-mode's gate did and which happens to be right only in the
// linear case. And a g that ignored q, sigma or phi could not tell a full chain
// rule from one missing those terms -- the same blind spot that let the gate be
// evaluated where dydt.q, dydt.sigma and dydt.phi are all still zero.
class QuadratureAdjoint : public AdjointProblem
{
public:
    explicit QuadratureAdjoint(Grid const &g) : grid(g)
    {
        ng = 1;
        np = 1;
        np_boundary = 0;
    }

    using AdjointProblem::gFn;

    // Every term is quadratic at worst, so that each of the four derivatives is
    // degree k or less and the interpolant of that derivative *is* the derivative.
    // dGdt projects with InterpolateOntoBasis rather than integrating exactly --
    // the route the adjoint assembly takes, and the only one a Python case can
    // support -- so the two agree here for the right reason. A cubic term (phi^3
    // was one) makes its derivative degree 2k, which P_k cannot represent, and the
    // comparison against an exactly-integrated GFn then fails on the interpolation
    // error rather than on anything about the chain rule.
    Value gFn(Index, const State &s, Position x) const override
    {
        return s.u(0) * s.u(0)              // nonlinear in u
               + 0.5 * s.q(0) * s.q(0)      // and in q
               + (2.0 + x) * s.sigma(0)     // linear in sigma, x-weighted
               + s.phi(0) * s.phi(0)        // and in the auxiliary variable
               + 0.25 * s.u(1);             // a second variable, so var != 0 counts too
    }

    Value GFn(Index gIndex, DGSoln &Y) const override
    {
        // Independent of the solver's own quadrature: a 30-point rule per cell,
        // where dGdt goes through the basis's rule. If the two agree to
        // finite-difference accuracy the assembly is right for the reason we want
        // rather than because both share a mistake.
        auto integrator = boost::math::quadrature::gauss<double, 30>();
        Value total = 0.0;
        for (Grid::Index i = 0; i < grid.getNCells(); ++i)
        {
            Interval const &I = grid[i];
            auto integrand = [&](double x) { return gFn(gIndex, Y.eval(x), x); };
            total += integrator.integrate(integrand, I.x_l, I.x_u);
        }
        return total;
    }

    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }

    void dgFn_du(Index, VectorRef v, const State &s, Position) override
    {
        v.setZero();
        v[0] = 2.0 * s.u(0);
        v[1] = 0.25;
    }
    void dgFn_dq(Index, VectorRef v, const State &s, Position) override
    {
        v.setZero();
        v[0] = s.q(0);
    }
    void dgFn_dsigma(Index, VectorRef v, const State &, Position x) override
    {
        v.setZero();
        v[0] = 2.0 + x;
    }
    void dgFn_dphi(Index, VectorRef v, const State &s, Position) override
    {
        v.setZero();
        v[0] = 2.0 * s.phi(0);
    }

    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }

private:
    Grid const &grid;
};

// A deterministic, everywhere-nonzero derivative vector.
//
// Deliberately not what setInitialConditions leaves behind: that zeroes dydt and
// then fills in only the differential part, so dydt.q, dydt.sigma and dydt.phi
// stay at zero until IDACalcIC runs. Testing the chain rule against that would
// exercise one of its four terms.
void fillWithNonzeroPattern(N_Vector v)
{
    sunrealtype *p = N_VGetArrayPointer(v);
    const sunindextype n = N_VGetLength(v);
    for (sunindextype i = 0; i < n; ++i)
        p[i] = 0.05 * std::sin(1.0 + 0.7 * static_cast<double>(i)) + 0.11;
}

GlobalState makeStates(Index nCells, Index k, Index nVars, Index nAux)
{
    GlobalState g(nCells, k, nVars, 0, nAux);
    for (Index j = 0; j < static_cast<Index>(g.size()); ++j)
    {
        State s(nVars, 0, nAux);
        for (Index v = 0; v < nVars; ++v)
        {
            s.u(v) = 0.2 * (j + 1) + v;
            s.q(v) = -0.4 * (j + 1) + v;
            s.sigma(v) = 0.6 * (j + 1) - v;
        }
        for (Index a = 0; a < nAux; ++a)
            s.phi(a) = 0.11 * (j + 1);
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
    const Index k = 2, nCells = 4, nVars = 2, nAux = 1;
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
            // Sized by nComponents, not nVars: dGdaux_Vec's output has one
            // block per auxiliary variable, so for it the two differ.
            Vector ref(nComponents * (k + 1));
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

        // dGdaux_Vec writes one block per *auxiliary* variable, so its result is
        // nAux*(k+1) long -- which for this fixture (nVars = 2, nAux = 1) is not
        // the same length as the other three. Passing the nVars-sized `actual`
        // used to be accepted only because the size assert inside read nVars;
        // it now demands nAux, matching what the solver's own call site in
        // initializeMatricesForAdjointSolve supplies.
        Vector actualAux(nAux * (k + 1));
        sys.dGdaux_Vec(gIndex, actualAux, sys.y, cell);
        BOOST_TEST((actualAux - reference(&AdjointProblem::dgFn_dphi, nAux)).norm() < 1e-10,
                   "dGdaux_Vec, cell " << cell);
        BOOST_TEST(actualAux.norm() > 1e-6);
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
}

// ------------------------------------------------------------------ dG/dt --

namespace
{

// The shared setup for the two dG/dt cases: a solver with a real objective on it,
// an initial condition, and an everywhere-nonzero dYdt.
struct DGdtFixture
{
    static constexpr Index k = 2, nCells = 4, nVars = 2, nAux = 1;

    Grid grid{0.0, 1.0, nCells};
    AdjointHostSystem problem;
    QuadratureAdjoint adjoint{grid};
    SystemSolver sys{grid, k, &problem};
    SUNContext ctx = nullptr;
    N_Vector Y = nullptr, dYdt = nullptr;

    DGdtFixture()
    {
        sys.setAdjointProblem(&adjoint);
        sys.setTau(1.0);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext_Create(SUN_COMM_NULL, &ctx);

        DGSoln shape(nVars, grid, k, Index(0), Index(nAux));
        Y = N_VNew_Serial(shape.getDoF(), ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);
        fillWithNonzeroPattern(dYdt);
    }

    ~DGdtFixture()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(dGdt_matches_a_finite_difference_of_the_objective)
{
    // dG/dt is the directional derivative of G along dy/dt, so a central
    // difference of G along that direction has to reproduce it. The state vector
    // *is* the coefficient vector, which is what makes the comparison exact
    // rather than approximate: perturbing Y by h*dYdt is precisely the
    // perturbation the chain rule linearises.
    DGdtFixture f;

    const Value analytic = f.sys.dGdt(0, f.sys.y, f.sys.dydt);

    N_Vector perturbed = N_VClone(f.Y);
    DGSoln probe(f.nVars, f.grid, f.k, N_VGetArrayPointer(perturbed), Index(0), Index(f.nAux));

    auto G_at = [&](double h)
    {
        N_VLinearSum(1.0, f.Y, h, f.dYdt, perturbed);
        probe.Map(N_VGetArrayPointer(perturbed));
        return f.adjoint.GFn(0, probe);
    };

    const double h = 1e-6;
    const Value numeric = (G_at(h) - G_at(-h)) / (2.0 * h);

    N_VDestroy(perturbed);

    BOOST_TEST(std::abs(analytic) > 1e-6,
               "dG/dt is ~zero, so this fixture would pass with any implementation");
    BOOST_TEST(std::abs(analytic - numeric) < 1e-6 * std::max(1.0, std::abs(numeric)),
               "dGdt = " << analytic << " but the central difference of G gives " << numeric);
}

BOOST_AUTO_TEST_CASE(dGdt_accounts_for_q_sigma_and_phi_not_just_u)
{
    // The anti-regression for the two defects that made origin/optimize-mode's
    // gate untrustworthy. Zeroing the q, sigma and aux blocks of dYdt has to move
    // the answer: if it does not, either those terms are missing from the chain
    // rule or the derivative being read has nothing in them -- which is exactly
    // the state dydt is in before IDACalcIC, where that gate was evaluated.
    DGdtFixture f;

    const Value full = f.sys.dGdt(0, f.sys.y, f.sys.dydt);

    for (Index i = 0; i < f.nCells; ++i)
    {
        for (Index var = 0; var < f.nVars; ++var)
        {
            f.sys.dydt.q(var).getCoeff(i).second.setZero();
            f.sys.dydt.sigma(var).getCoeff(i).second.setZero();
        }
        for (Index a = 0; a < f.nAux; ++a)
            f.sys.dydt.Aux(a).getCoeff(i).second.setZero();
    }

    const Value uOnly = f.sys.dGdt(0, f.sys.y, f.sys.dydt);

    BOOST_TEST(std::abs(full - uOnly) > 1e-6,
               "dGdt is unchanged when dydt's q, sigma and phi are zeroed, so those terms are not reaching it");
}

BOOST_AUTO_TEST_CASE(dGdt_without_an_adjoint_problem_is_reported)
{
    // The gate is armed by a tolerance and the objective by a separate call, so
    // the two can disagree. Better a named error than a null dereference.
    Grid grid(0.0, 1.0, 3);
    AdjointHostSystem problem;
    SystemSolver sys(grid, 2, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    BOOST_CHECK_THROW(sys.dGdt(0, sys.y, sys.dydt), std::logic_error);

    // And the gate itself, which is the path a user actually reaches it by.
    sys.setObjectiveDecreaseTolerance(1e-3);
    BOOST_CHECK_THROW(sys.objectiveIsDecreasing(), std::logic_error);
}

BOOST_AUTO_TEST_CASE(the_gate_is_off_until_a_tolerance_arms_it)
{
    // No tolerance set means no objective evaluation at all -- note this holds
    // even with no AdjointProblem, so an ordinary run cannot be made to throw by
    // machinery it never asked for.
    Grid grid(0.0, 1.0, 3);
    AdjointHostSystem problem;
    SystemSolver sys(grid, 2, &problem);

    BOOST_TEST(!sys.objectiveIsDecreasing());
    BOOST_TEST(!sys.wasRejected());
}

BOOST_AUTO_TEST_CASE(the_gate_compares_dGdt_against_the_tolerance_it_was_given)
{
    // The decision is a one-sided band around zero: reject below -tol, accept
    // above it. Driving it from both sides with the same state pins the sign
    // convention (G is maximised, so falling is bad) and the slack.
    DGdtFixture f;

    const Value dGdt = f.sys.dGdt(0, f.sys.y, f.sys.dydt);
    BOOST_TEST(std::abs(dGdt) > 1e-6);

    // A tolerance well inside |dG/dt| rejects when dG/dt is negative and accepts
    // when positive; one well outside it accepts either way.
    const double tight = std::abs(dGdt) / 2.0;
    const double loose = std::abs(dGdt) * 2.0;

    f.sys.setObjectiveDecreaseTolerance(tight);
    BOOST_TEST(f.sys.objectiveIsDecreasing() == (dGdt < 0.0),
               "dG/dt = " << dGdt << " with tolerance " << tight);
    BOOST_TEST(f.sys.wasRejected() == (dGdt < 0.0));
    BOOST_TEST(f.sys.lastDGdt().size() == 1);
    BOOST_TEST(std::abs(f.sys.lastDGdt()(0) - dGdt) < 1e-12);

    f.sys.setObjectiveDecreaseTolerance(loose);
    BOOST_TEST(!f.sys.objectiveIsDecreasing(),
               "tolerance " << loose << " is larger than |dG/dt| so nothing should be rejected");
    BOOST_TEST(!f.sys.wasRejected());
}

BOOST_AUTO_TEST_CASE(the_gate_does_not_depend_on_the_output_cadence)
{
    // Defect 2's regression test. origin/optimize-mode compared dt * dG/dt against
    // its threshold, with dt the netCDF output cadence -- so an I/O setting decided
    // whether a step was rejected, and setOutputCadence(0.0) zeroed the product
    // and disarmed the gate silently. The verdict must be identical across
    // cadences, including zero.
    const double cadences[] = {0.0, 1e-4, 0.25, 1000.0};

    bool verdicts[std::size(cadences)];
    Value values[std::size(cadences)];

    for (size_t i = 0; i < std::size(cadences); ++i)
    {
        DGdtFixture f;
        f.sys.setOutputCadence(cadences[i]);
        // Tight enough that the decision is genuinely driven by dG/dt rather than
        // by slack swamping it.
        f.sys.setObjectiveDecreaseTolerance(std::abs(f.sys.dGdt(0, f.sys.y, f.sys.dydt)) / 2.0);
        verdicts[i] = f.sys.objectiveIsDecreasing();
        values[i] = f.sys.lastDGdt()(0);
    }

    for (size_t i = 1; i < std::size(cadences); ++i)
    {
        BOOST_TEST(verdicts[i] == verdicts[0],
                   "output cadence " << cadences[i] << " changed the gate's verdict");
        BOOST_TEST(std::abs(values[i] - values[0]) < 1e-12,
                   "output cadence " << cadences[i] << " changed dG/dt itself");
    }
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
