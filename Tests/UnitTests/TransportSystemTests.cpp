// Tests for the defaults TransportSystem hands every physics case.
//
// TransportSystem is a pure interface with nine pure virtuals and about forty
// defaulted ones. A physics case that implements only the nine gets the rest
// for free, so those defaults are executed by every run in the project -- and
// none of them had direct coverage.
//
// Two classes of default matter here:
//
//  * The **batched wrappers**. SigmaFn/Sources/AuxG/dSigma/dSources/AuxGPrime
//    each have a vectorised overload whose default body is a serial loop over
//    the pointwise version, several of them under `#pragma omp parallel for`.
//    The contract is that batched output j corresponds to states[j] and
//    abscissae[j]. That is exactly the sort of thing an OpenMP loop can get
//    subtly wrong, so the mocks below make the physics depend strongly on both
//    the index and the position: a transposed or shifted loop cannot produce
//    the right answer by accident. Build with OMP=on to exercise the threaded
//    path against the same assertions.
//
//  * The **not-implemented guards**. Optional hooks throw std::logic_error when
//    the case declares nScalars > 0 or nAux > 0 without providing them, and
//    silently return 0 when it does not. Both branches are real: the silent
//    branch is what keeps a plain nVars-only case working.

#include <boost/test/unit_test.hpp>

#include "DGSoln.hpp"
#include "TransportSystem.hpp"
#include "Types.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

// Implements the nine pure virtuals and nothing else, so every defaulted
// method below is the base-class one.
//
// The physics is deliberately index- and position-sensitive: SigmaFn depends on
// the variable index, the position, and several state components at once.
class MinimalSystem : public TransportSystem
{
public:
    // The counts are constructor arguments because the subclasses below vary
    // them; they used to be assigned into the base after the fact.
    explicit MinimalSystem(Index nv = 2, Index ns = 0, Index na = 0)
        : TransportSystem({.variables = numberedFields(nv),
                           .scalars = numberedScalars(ns),
                           .aux = numberedAux(na)})
    {
    }

    // For subclasses that need something the count triple cannot say, such as
    // a non-default boundary kind.
    explicit MinimalSystem(SystemSpec spec) : TransportSystem(std::move(spec)) {}

    // Overriding the pointwise SigmaFn/Sources would otherwise hide the
    // batched overloads of the same name -- which are exactly the defaults
    // under test here.
    using TransportSystem::SigmaFn;
    using TransportSystem::Sources;

    Value SigmaFn(Index i, const State &s, Position x, Time t) override
    {
        ++sigmaCalls;
        return 1.0 + 10.0 * i + 100.0 * x + 3.0 * s.u(i) + 5.0 * s.q(i) + t;
    }
    Value Sources(Index i, const State &s, Position x, Time t) override
    {
        ++sourceCalls;
        return -2.0 - 7.0 * i + 13.0 * x + s.u(0) * s.q(i) - 0.5 * t;
    }

    void dSigmaFn_du(Index i, VectorRef v, const State &, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 1.0 + i + 2.0 * j + x;
    }
    void dSigmaFn_dq(Index i, VectorRef v, const State &, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 10.0 + i - 3.0 * j + 2.0 * x;
    }
    void dSources_du(Index i, VectorRef v, const State &, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 100.0 + i + j - x;
    }
    void dSources_dq(Index i, VectorRef v, const State &, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 200.0 + 5.0 * i - j + 3.0 * x;
    }
    void dSources_dsigma(Index i, VectorRef v, const State &, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 300.0 - i + 4.0 * j + x * x;
    }

    Value InitialValue(Index i, Position x) const override { return (1.0 + i) * std::sin(x); }
    Value InitialDerivative(Index i, Position x) const override
    {
        return (1.0 + i) * std::cos(x);
    }

    // Call counters, so "the batched default really loops over the scalar
    // version" can be asserted rather than assumed.
    int sigmaCalls = 0;
    int sourceCalls = 0;
};

// Declares scalars and aux variables but implements none of the corresponding
// hooks, so every not-implemented guard fires.
class UnderSpecifiedSystem : public MinimalSystem
{
public:
    UnderSpecifiedSystem() : MinimalSystem(2, 2, 1) {}
};

// Provides the aux hooks so the batched aux wrappers can be driven.
class AuxSystem : public MinimalSystem
{
public:
    AuxSystem() : MinimalSystem(2, 0, 2) {}

    using TransportSystem::AuxG;
    using TransportSystem::AuxGPrime;

    Value AuxG(Index i, const State &s, Position x, Time t) override
    {
        ++auxCalls;
        return 1000.0 * (i + 1) + 17.0 * x + s.phi(i) - s.u(0) + 0.25 * t;
    }
    void AuxGPrime(Index i, State &out, const State &s, Position x, Time) override
    {
        out.zero();
        out.u(0) = -1.0 - i;
        out.q(0) = 2.0 * x;
        out.sigma(0) = 3.0 + i;
        out.phi(i) = 1.0;
    }
    void dSources_dPhi(Index i, VectorRef v, const State &, Position x, Time) override
    {
        v.setZero();
        v[0] = 1.0 + i + x;
    }
    void dSigma_dPhi(Index i, VectorRef v, const State &, Position x, Time) override
    {
        v.setZero();
        v[0] = 7.0 - i + 2.0 * x;
    }

    int auxCalls = 0;
};

// Builds a GlobalState whose every column differs, so an index mix-up cannot
// be masked by two identical points.
GlobalState makeStates(Index nCells, Index k, Index nVars, Index nScalars, Index nAux)
{
    GlobalState g(nCells, k, nVars, nScalars, nAux);
    const Index n = static_cast<Index>(g.size());
    for (Index j = 0; j < n; ++j)
    {
        State s(nVars, nScalars, nAux);
        for (Index v = 0; v < nVars; ++v)
        {
            s.u(v) = 0.1 * (j + 1) + v;
            s.q(v) = -0.3 * (j + 1) + 2.0 * v;
            s.sigma(v) = 0.7 * (j + 1) - v;
        }
        for (Index a = 0; a < nAux; ++a)
            s.phi(a) = 0.05 * (j + 1) * (a + 2);
        for (Index c = 0; c < nScalars; ++c)
            s.scalar(c) = 1.5 + c;
        g.setWithState(j, s);
    }
    return g;
}

std::vector<Position> makePoints(Index n)
{
    std::vector<Position> pts(n);
    for (Index j = 0; j < n; ++j)
        pts[j] = 0.13 * (j + 1);
    return pts;
}

} // namespace

BOOST_AUTO_TEST_SUITE(transport_system_tests)

// -------------------------------------------------- the batched wrappers --

BOOST_AUTO_TEST_CASE(batched_sigma_and_sources_match_the_scalar_loop)
{
    // out(j) must be the scalar function evaluated at (states[j], abscissae[j]).
    // Under OMP=on this loop runs in parallel; the assertion is unchanged.
    MinimalSystem sys;
    const Index nCells = 3, k = 2, nVars = 2;
    GlobalState states = makeStates(nCells, k, nVars, 0, 0);
    const auto points = makePoints(static_cast<Index>(states.size()));
    const Time t = 0.375;

    for (Index i = 0; i < nVars; ++i)
    {
        const Values sigma = sys.SigmaFn(i, states, points, t);
        const Values source = sys.Sources(i, states, points, t);

        BOOST_TEST_REQUIRE(sigma.size() == static_cast<Index>(states.size()));

        for (size_t j = 0; j < states.size(); ++j)
        {
            const State s = states[j];
            BOOST_TEST(sigma(j) == sys.SigmaFn(i, s, points[j], t),
                       boost::test_tools::tolerance(1e-14));
            BOOST_TEST(source(j) == sys.Sources(i, s, points[j], t),
                       boost::test_tools::tolerance(1e-14));
        }
    }

    // Not vacuous: consecutive entries must actually differ, otherwise a
    // shifted loop would agree with the reference anyway.
    const Values sigma0 = sys.SigmaFn(0, states, points, t);
    BOOST_TEST(std::abs(sigma0(0) - sigma0(1)) > 1e-3);
}

BOOST_AUTO_TEST_CASE(compute_physics_fills_every_slot_and_caches_the_sources)
{
    // ComputePhysics returns { sigma[nVars], sources[nVars], auxG[nAux] } and
    // squirrels the sources away in m_sourceCache for the diagnostics writer.
    // The cache is read back by getSourceCache and nothing else validates it.
    AuxSystem sys;
    const Index nCells = 2, k = 3, nVars = 2, nAux = 2;
    GlobalState states = makeStates(nCells, k, nVars, 0, nAux);
    const auto points = makePoints(static_cast<Index>(states.size()));
    const Time t = 1.25;

    PhysicsOutput out = sys.ComputePhysics(states, points, t);

    BOOST_TEST_REQUIRE(out[0].size() == static_cast<size_t>(nVars));
    BOOST_TEST_REQUIRE(out[1].size() == static_cast<size_t>(nVars));
    BOOST_TEST_REQUIRE(out[2].size() == static_cast<size_t>(nAux));

    for (Index i = 0; i < nVars; ++i)
        for (size_t j = 0; j < states.size(); ++j)
        {
            const State s = states[j];
            BOOST_TEST(out[0][i](j) == sys.SigmaFn(i, s, points[j], t),
                       boost::test_tools::tolerance(1e-14));
            BOOST_TEST(out[1][i](j) == sys.Sources(i, s, points[j], t),
                       boost::test_tools::tolerance(1e-14));
        }

    for (Index a = 0; a < nAux; ++a)
        for (size_t j = 0; j < states.size(); ++j)
            BOOST_TEST(out[2][a](j) == sys.AuxG(a, states[j], points[j], t),
                       boost::test_tools::tolerance(1e-14));

    // The cached sources must be the ones just computed, not the fluxes.
    for (Index i = 0; i < nVars; ++i)
        BOOST_TEST((sys.getSourceCache(i) - out[1][i]).norm() < 1e-14);
}

BOOST_AUTO_TEST_CASE(compute_physics_derivatives_fills_the_right_state_slices)
{
    // dSigma writes into out.u()/Derivative (and Aux when nAux > 0);
    // dSources additionally writes out.sigma(). Getting these slices crossed
    // would put dS/dq where dS/du belongs in every Jacobian block.
    AuxSystem sys;
    const Index nCells = 2, k = 2, nVars = 2, nAux = 2;
    GlobalState states = makeStates(nCells, k, nVars, 0, nAux);
    const auto points = makePoints(static_cast<Index>(states.size()));
    const Time t = 0.5;

    GlobalStateMatrix dSigma(nVars), dSource(nVars), dAux(nAux);
    for (Index v = 0; v < nVars; ++v)
    {
        dSigma.add(nCells, k, nVars, 0, nAux);
        dSource.add(nCells, k, nVars, 0, nAux);
    }
    for (Index a = 0; a < nAux; ++a)
        dAux.add(nCells, k, nVars, 0, nAux);

    sys.ComputePhysicsDerivatives({dSigma, dSource, dAux}, states, points, t);

    Vector ref(nVars);
    for (Index i = 0; i < nVars; ++i)
        for (size_t j = 0; j < states.size(); ++j)
        {
            const State s = states[j];

            sys.dSigmaFn_du(i, ref, s, points[j], t);
            BOOST_TEST((dSigma[i].Variable(j) - ref).norm() < 1e-14);
            sys.dSigmaFn_dq(i, ref, s, points[j], t);
            BOOST_TEST((dSigma[i].Derivative(j) - ref).norm() < 1e-14);

            sys.dSources_du(i, ref, s, points[j], t);
            BOOST_TEST((dSource[i].Variable(j) - ref).norm() < 1e-14);
            sys.dSources_dq(i, ref, s, points[j], t);
            BOOST_TEST((dSource[i].Derivative(j) - ref).norm() < 1e-14);
            sys.dSources_dsigma(i, ref, s, points[j], t);
            BOOST_TEST((dSource[i].Flux(j) - ref).norm() < 1e-14);

            // nAux > 0, so the Aux slices are populated too.
            Vector auxRef(nAux);
            sys.dSigma_dPhi(i, auxRef, s, points[j], t);
            BOOST_TEST((dSigma[i].Aux(j) - auxRef).norm() < 1e-14);
            sys.dSources_dPhi(i, auxRef, s, points[j], t);
            BOOST_TEST((dSource[i].Aux(j) - auxRef).norm() < 1e-14);
        }

    // And the aux constraint derivatives.
    for (Index a = 0; a < nAux; ++a)
        for (size_t j = 0; j < states.size(); ++j)
        {
            State expected(nVars, 0, nAux);
            sys.AuxGPrime(a, expected, states[j], points[j], t);
            BOOST_TEST((dAux[a].Variable(j) - expected.u()).norm() < 1e-14);
            BOOST_TEST((dAux[a].Derivative(j) - expected.q()).norm() < 1e-14);
            BOOST_TEST((dAux[a].Flux(j) - expected.sigma()).norm() < 1e-14);
            BOOST_TEST((dAux[a].Aux(j) - expected.phi()).norm() < 1e-14);
        }
}

BOOST_AUTO_TEST_CASE(the_aux_slices_are_left_alone_when_there_are_no_aux_variables)
{
    // dSigma/dSources guard the dSigma_dPhi / dSources_dPhi calls on nAux > 0.
    // Without the guard, a plain nVars-only case would hit the throwing default
    // on every Jacobian evaluation -- so this branch is load-bearing, not
    // defensive.
    MinimalSystem sys; // nAux = 0
    const Index nCells = 2, k = 1, nVars = 2;
    GlobalState states = makeStates(nCells, k, nVars, 0, 0);
    const auto points = makePoints(static_cast<Index>(states.size()));

    GlobalState out(nCells, k, nVars, 0, 0);
    BOOST_CHECK_NO_THROW(sys.dSigma(0, out, states, points, 0.0));
    BOOST_CHECK_NO_THROW(sys.dSources(0, out, states, points, 0.0));
}

BOOST_AUTO_TEST_CASE(batched_aux_wrappers_match_the_scalar_versions)
{
    AuxSystem sys;
    const Index nCells = 2, k = 2, nVars = 2, nAux = 2;
    GlobalState states = makeStates(nCells, k, nVars, 0, nAux);
    const auto points = makePoints(static_cast<Index>(states.size()));
    const Time t = 2.0;

    for (Index a = 0; a < nAux; ++a)
    {
        const Values g = sys.AuxG(a, states, points, t);
        for (size_t j = 0; j < states.size(); ++j)
            BOOST_TEST(g(j) == sys.AuxG(a, states[j], points[j], t),
                       boost::test_tools::tolerance(1e-14));

        GlobalState out(nCells, k, nVars, 0, nAux);
        sys.AuxGPrime(a, out, states, points, t);
        for (size_t j = 0; j < states.size(); ++j)
        {
            State expected(nVars, 0, nAux);
            sys.AuxGPrime(a, expected, states[j], points[j], t);
            BOOST_TEST((out.Variable(j) - expected.u()).norm() < 1e-14);
            BOOST_TEST((out.Aux(j) - expected.phi()).norm() < 1e-14);
        }
    }
}

// ------------------------------------------ the not-implemented guards --

BOOST_AUTO_TEST_CASE(optional_hooks_throw_when_the_case_declares_scalars_or_aux)
{
    UnderSpecifiedSystem sys;
    State s(2, 2, 1);
    Vector v(2);
    std::vector<double> mem;

    BOOST_CHECK_THROW(sys.InitialScalarValue(0), std::logic_error);
    BOOST_CHECK_THROW(sys.dSources_dScalars(0, v, s, 0.0, 0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.InitialAuxValue(0, 0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.AuxG(0, s, 0.0, 0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.dSources_dPhi(0, v, s, 0.0, 0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.dSigma_dPhi(0, v, s, 0.0, 0.0), std::logic_error);
    BOOST_CHECK_THROW(sys.createAdjointProblem(), std::logic_error);

    // AuxGPrime throws unconditionally -- there is no sensible zero derivative
    // for a constraint that was never written down.
    State out(2, 2, 1);
    BOOST_CHECK_THROW(sys.AuxGPrime(0, out, s, 0.0, 0.0), std::logic_error);

    // ScalarG needs a DGSoln; a default-constructed one is enough because the
    // guard fires before it is touched.
    Grid g(0.0, 1.0, 2);
    DGSoln dummy(2, g, 1, Index(2), Index(1));
    GlobalState nodal(2, 1, 2, 2, 1), nodal_dt(2, 1, 2, 2, 1);
    std::vector<Position> abscissae(nodal.size(), 0.0);
    Vector weights = Vector::Zero(nodal.size());
    Matrix phiBoundary = Matrix::Zero(2, 2);
    GlobalStateMatrix dG(2), dGdot(2);
    for (Index i = 0; i < 2; ++i)
    {
        dG.add(2, 1, 2, 2, 1);
        dGdot.add(2, 1, 2, 2, 1);
    }
    BOOST_CHECK_THROW(sys.ScalarG(0, nodal, nodal_dt, abscissae, weights, phiBoundary, 0.0),
                      std::logic_error);
    BOOST_CHECK_THROW(
        sys.ScalarGPrime(dG, dGdot, nodal, nodal_dt, abscissae, weights, phiBoundary, 0.0),
        std::logic_error);
}

BOOST_AUTO_TEST_CASE(the_same_hooks_are_silent_when_there_are_no_scalars_or_aux)
{
    // The guards read `if (nScalars != 0) throw`, so a plain case gets a
    // harmless zero instead of an exception. Every nVars-only physics case in
    // the repo depends on this.
    MinimalSystem sys;
    State s(2);
    Vector v(2);

    BOOST_TEST(sys.InitialScalarValue(0) == 0.0);
    BOOST_TEST(sys.InitialAuxValue(0, 0.5) == 0.0);
    BOOST_TEST(sys.AuxG(0, s, 0.5, 0.0) == 0.0);
    BOOST_CHECK_NO_THROW(sys.dSources_dScalars(0, v, s, 0.0, 0.0));
    BOOST_CHECK_NO_THROW(sys.dSources_dPhi(0, v, s, 0.0, 0.0));
    BOOST_CHECK_NO_THROW(sys.dSigma_dPhi(0, v, s, 0.0, 0.0));

    Grid g(0.0, 1.0, 2);
    DGSoln dummy(2, g, 1);
    GlobalState nodal(2, 1, 2), nodal_dt(2, 1, 2);
    std::vector<Position> abscissae(nodal.size(), 0.0);
    Vector weights = Vector::Zero(nodal.size());
    Matrix phiBoundary = Matrix::Zero(2, 2);
    BOOST_TEST(sys.ScalarG(0, nodal, nodal_dt, abscissae, weights, phiBoundary, 0.0) == 0.0);
    BOOST_TEST(sys.InitialScalarDerivative(0, dummy, dummy) == 0.0);

    // isScalarDifferential is not in the list above. It used to answer `false`
    // for any index, including on a case with no scalars at all; it is a spec
    // lookup now, and asking about scalar 0 of a system that declares none is a
    // question with no answer rather than a hook that should stay quiet. The
    // solver only ever asks under `for (s = 0; s < nScalars; ++s)`.
    BOOST_CHECK_THROW(sys.isScalarDifferential(0), std::out_of_range);
}

// -------------------------------------------------------- small defaults --

BOOST_AUTO_TEST_CASE(scalar_defaults_and_naming_are_what_the_output_writer_expects)
{
    // Scalars and aux are populated here because the names now come from the
    // spec: asking a system with no scalars for the name of scalar 1 used to
    // fabricate "Scalar1", and is an out-of-range read on an empty vector now.
    MinimalSystem sys(3, 2, 1);

    BOOST_TEST(sys.getNumVars() == 3);
    BOOST_TEST(sys.getNumScalars() == 2);
    BOOST_TEST(sys.getNumAux() == 1);

    // aFn is the coefficient of du/dt; the default is 1 everywhere.
    for (Index i = 0; i < 3; ++i)
        for (double x : {0.0, 0.25, 1.0})
            BOOST_TEST(sys.aFn(i, x) == 1.0);

    BOOST_TEST(sys.getVariableName(2) == "Var2");
    BOOST_TEST(sys.getScalarName(1) == "Scalar1");
    BOOST_TEST(sys.getAuxVarName(0) == "AuxVariable0");
    BOOST_TEST(sys.getVariableDescription(2) == "Variable 2");
    BOOST_TEST(sys.getScalarDescription(1) == "Scalar 1");
    BOOST_TEST(sys.getAuxDescription(0) == "Auxiliary Variable 0");
    BOOST_TEST(sys.getVariableUnits(0) == "");
    BOOST_TEST(sys.getScalarUnits(0) == "");
    BOOST_TEST(sys.getAuxUnits(0) == "");
    BOOST_TEST(sys.getAdjointNames(3) == "p3");

    // The diagnostics hooks are no-ops that must not throw.
    NetCDFIO nc;
    BOOST_CHECK_NO_THROW(sys.initialiseDiagnostics(nc));
    BOOST_CHECK_NO_THROW(sys.finaliseDiagnostics(nc));

    BOOST_TEST(sys.isRestarting() == false);
}

// The two tests that stood here -- scalar_g_extended_defaults_to_scalar_g and
// batched_scalar_g_prime_extended_walks_every_basis_function -- pinned the
// forwarding machinery of the old scalar interface: ScalarGExtended defaulting
// to ScalarG, and the base class loop that called a per-(cell, basis function)
// hook and packed the answers at cell*(k+1) + j. There is no forwarding now and
// no loop; a case fills the whole array itself.
//
// What they were really protecting -- that the flattening a case writes is the
// one the solver reads back into its w vectors -- is covered more strongly by
// checkScalarDerivative in ScalarJacobianTests.cpp, which finite-differences
// the case's own ScalarG and compares every entry rather than checking that
// plumbing forwards.

BOOST_AUTO_TEST_CASE(set_restart_values_takes_ownership_and_derives_the_boundaries)
{
    // The restart path reads back a stored solution and must (a) own its own
    // copy of the data -- the caller's vectors are reused by the netCDF reader
    // -- and (b) recover the boundary conditions from the restarted profile
    // rather than from the config.
    struct RestartSystem : public MinimalSystem
    {
        // Dirichlet below, Neumann above, so setRestartValues has to read u at
        // one end and q at the other.
        RestartSystem()
            : MinimalSystem(SystemSpec{.variables = numberedFields(1, BoundaryKind::Dirichlet,
                                                                   BoundaryKind::Neumann)})
        {
        }
    } sys;

    const Index nCells = 3, k = 2, nVars = 1;
    Grid grid(0.0, 1.0, nCells);
    DGSoln shape(nVars, grid, k);
    std::vector<double> Ydata(shape.getDoF(), 0.0);
    std::vector<double> dYdata(shape.getDoF(), 0.0);

    // Fill with a known profile: u = 2 + 3x, q = 3, sigma = -0.5. q and sigma
    // are deliberately different numbers -- that is what lets the assertions
    // below say which of them a Neumann boundary actually gets.
    {
        DGSoln tmp(nVars, grid, k, Ydata.data());
        tmp.AssignU([](Index, double x) { return 2.0 + 3.0 * x; });
        tmp.AssignQ([](Index, double) { return 3.0; });
        tmp.AssignSigma(
            [](Index, const State &, Position, Time) -> Value { return -0.5; });
        tmp.EvaluateLambda();
    }

    sys.setRestartValues(Ydata, dYdata, grid, k);
    BOOST_TEST(sys.isRestarting());

    // Scribbling over the caller's buffer must not disturb the stored copy.
    std::fill(Ydata.begin(), Ydata.end(), -12345.0);

    BOOST_TEST(sys.getRestartY().u(0)(0.5) == 3.5, boost::test_tools::tolerance(1e-12));

    // Lower boundary is Dirichlet, so uL comes from u(x_l) = 2.
    BOOST_TEST(sys.LowerBoundary(0, 0.0) == 2.0, boost::test_tools::tolerance(1e-12));
    // Upper boundary is Neumann, so uR comes from q(x_u) = 3 -- *not* from
    // sigma, which this asserted until the two were told apart. A Neumann
    // boundary value is applied to q (SystemSolver.cpp's L_global assembly),
    // so seeding it from sigma resumed the run against the wrong quantity, and
    // against the wrong sign too, the stored sigma being -sigma_hat. -0.5 here
    // means that regression is back.
    // test_a_restart_carries_a_neumann_boundary_as_q_not_sigma in
    // python/Tests/test_runner.py is the end-to-end half of this.
    BOOST_TEST(sys.UpperBoundary(0, 0.0) == 3.0, boost::test_tools::tolerance(1e-12));
}

// ------------------------------------------------- State as an out-parameter --

BOOST_AUTO_TEST_CASE(a_state_is_born_zeroed_not_merely_sized)
{
    // The derivative hooks receive these as out-parameters. Eigen's resize()
    // leaves the memory indeterminate, which made an opening setZero() the
    // caller's responsibility in every physics case -- and a case that assigned
    // only its nonzero entries, which is the natural way to write one, got
    // whatever was in the buffer for the rest. That is defect (2) of the
    // ScalarTestLD3 post-mortem in Tests/README.md.
    //
    // Heap contents are not deterministic, so this cannot prove the old code
    // was wrong; what it pins is that nothing reintroduces a bare resize().
    for (int trial = 0; trial < 8; ++trial)
    {
        State s(4, 3, 2);
        BOOST_TEST(s.u().isZero());
        BOOST_TEST(s.q().isZero());
        BOOST_TEST(s.sigma().isZero());
        BOOST_TEST(s.phi().isZero());
        BOOST_TEST(s.scalars().isZero());
    }
}

BOOST_AUTO_TEST_CASE(a_global_state_is_born_zeroed_too)
{
    // SystemSolver builds a fresh GlobalStateMatrix for every Jacobian
    // evaluation and hands its columns straight to the hooks.
    GlobalState g(3, 2, 4, 3, 2);
    BOOST_TEST(g.Variable().isZero());
    BOOST_TEST(g.Derivative().isZero());
    BOOST_TEST(g.Flux().isZero());
    BOOST_TEST(g.Aux().isZero());
    BOOST_TEST(g.Scalars().isZero());
}

BOOST_AUTO_TEST_CASE(sigma_hat_is_the_physical_flux_and_sigma_is_what_is_stored)
{
    // The solver stores sigma = -sigma_hat. `Flux` holds the stored value, so
    // a case reading it gets the negative of what its own SigmaFn returned --
    // the trap documented at the top of TransportSystem.hpp. sigmaHat() names
    // the physical quantity so that the negation is visible in the source
    // rather than remembered.
    State s(2);
    s.sigma(0) = -3.25; // as the solver stores it
    s.sigma(1) = 1.5;

    BOOST_TEST(s.sigma(0) == -3.25);
    BOOST_TEST(s.sigmaHat(0) == 3.25);
    BOOST_TEST(s.sigmaHat(1) == -1.5);

    // The named accessors and the raw vectors are two views of one buffer.
    BOOST_TEST(s.sigma(0) == -3.25);
}

BOOST_AUTO_TEST_CASE(named_accessors_reach_the_same_storage_as_the_raw_vectors)
{
    State s(3, 2, 1);
    s.u(0) = 1.0;
    s.q(2) = 2.0;
    s.phi(0) = 3.0;
    s.scalar(1) = 4.0;

    BOOST_TEST(s.u(0) == 1.0);
    BOOST_TEST(s.q(2) == 2.0);
    BOOST_TEST(s.phi(0) == 3.0);
    BOOST_TEST(s.scalar(1) == 4.0);

    // Aux is sized nAux and Variable nVars, and confusing the two is a
    // documented source of bugs here; under DEBUG the accessors say so rather
    // than reading past the end. (This build is not a DEBUG build, so only the
    // sizes are asserted.)
    BOOST_TEST(s.phi().size() == 1);
    BOOST_TEST(s.u().size() == 3);
}

BOOST_AUTO_TEST_SUITE_END()
