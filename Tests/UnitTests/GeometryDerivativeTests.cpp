// The geometry derivative hooks. An absent hook is an identically zero block,
// which is the correct meaning of "this case does not read geometry" -- the
// same convention every other derivative out-parameter follows.
#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/AutodiffTransportSystem.hpp"
#include "../../TransportSystem.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

// Implements only SigmaFn and Sources, plus the five pre-existing derivative
// hooks that TransportSystem still declares pure virtual. None of the three
// new geometry hooks is overridden, so every query below is answered by
// TransportSystem's own defaults -- empty bodies, leaving the caller's zeroed
// out-parameter alone.
class MinimalCase : public TransportSystem
{
public:
    MinimalCase() : TransportSystem({.variables = numberedFields(1)}) {}

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return x; }
    Value InitialDerivative(Index, Position) const override { return 1.0; }
};

// sigma_hat = g0 * kappa * q, so d(sigma_hat)/d(g0) = kappa * q and
// d/d(g1) = 0 -- the second geometry slot never appears in the formula, which
// is what lets the test below check that an override writes only the entry it
// means to and leaves the rest at the caller's zero.
class GeometryDependentCase : public TransportSystem
{
public:
    explicit GeometryDependentCase(double kappa)
        : TransportSystem({.variables = numberedFields(1)}), kappa(kappa)
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return s.geom(0) * kappa * s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0) * kappa;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    void dSigmaFn_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v(0) = kappa * s.q(0);
    }

    Value InitialValue(Index, Position x) const override { return x; }
    Value InitialDerivative(Index, Position) const override { return 1.0; }

private:
    double kappa;
};

// A grid AutodiffTransportSystem's constructor takes and ignores, since the
// config table below is empty. Static so the reference the constructor is
// handed outlives every case built from it -- the same pattern
// CoupledResidualTests.cpp uses for its field models.
Grid const &scratchGrid()
{
    static const Grid g(0.0, 1.0, 1);
    return g;
}

// An AutodiffTransportSystem subclass whose Flux reads geometry, so the base
// class's autodiff machinery has to differentiate through it rather than
// return the identically-zero default every other autodiff-derived case gets
// today.
//
// The config table is empty, so AutodiffTransportSystem's constructor skips
// its [AutodiffTransportSystem] block entirely and never touches
// xL/xR/uL/uR/InitialHeights -- which is fine, because nothing here calls
// InitialValue, InitialDerivative or the boundary hooks.
class AutodiffGeometryCase : public AutodiffTransportSystem
{
public:
    AutodiffGeometryCase()
        // An empty *table*, not a default-constructed toml::value: the latter
        // is toml11's `empty` variant, and AutodiffTransportSystem's
        // constructor calls config.count("AutodiffTransportSystem") on it
        // unconditionally, which throws type_error::bad_cast on anything that
        // is not a table. FieldModel-derived fixtures elsewhere in this suite
        // get away with toml::value{} because FieldModel never calls count()
        // on it.
        : AutodiffTransportSystem(toml::value(toml::table{}), scratchGrid(),
                                  SystemSpec{.variables = numberedFields(1)})
    {
    }

private:
    // AutodiffTransportSystem's original Flux overload is still pure virtual --
    // widening the signature would have meant touching every existing
    // autodiff-based physics case just to add a parameter none of them read.
    // This overload is required to exist but is never actually called: SigmaFn
    // and every d.../d... hook evaluate through the geometry-aware overload
    // below instead.
    Real Flux(Index, RealVector, RealVector, Real, Time) override
    {
        throw std::logic_error("AutodiffGeometryCase: unreachable Flux overload");
    }

    // sigma_hat = q * (1 + g0^2). Nonlinear in the geometry slot so a central
    // difference actually exercises the chain rule autodiff has to get right,
    // rather than merely reproducing a constant coefficient.
    Real Flux(Index, RealVector u, RealVector q, RealVector geom, Real x, Time t) override
    {
        return q(0) * (1.0 + geom(0) * geom(0));
    }
};

} // namespace

BOOST_AUTO_TEST_SUITE(geometry_derivative_tests)

BOOST_AUTO_TEST_CASE(the_default_hooks_leave_the_block_zero)
{
    // Seeded with a nonzero sentinel and never zeroed by the test itself: the
    // point is to distinguish "the default is a genuine no-op" from "the
    // default writes zero", which a pre-zeroed `out` cannot tell apart. In
    // production `out` does arrive zeroed (State/GlobalState are born that
    // way -- see State.hpp), so the no-op default and an explicit zero write
    // are equivalent there; here they are not, and a no-op is what the
    // default virtuals in TransportSystem.hpp actually are.
    MinimalCase sys;               // implements only SigmaFn and Sources
    State s(1, 0, 0, 2);
    const Vector sentinel = Vector::Constant(2, 99.0);

    Vector out = sentinel;
    sys.dSigmaFn_dGeometry(0, out, s, 0.5, 0.0);
    BOOST_CHECK_EQUAL((out - sentinel).norm(), 0.0);

    out = sentinel;
    sys.dSources_dGeometry(0, out, s, 0.5, 0.0);
    BOOST_CHECK_EQUAL((out - sentinel).norm(), 0.0);
}

BOOST_AUTO_TEST_CASE(a_case_that_overrides_them_is_dispatched_to)
{
    // GeometryDependentCase has sigma_hat = g0 * kappa * q, so
    // d(sigma_hat)/d(g0) = kappa * q and d/d(g1) = 0.
    GeometryDependentCase sys(/*kappa=*/2.5);
    State s(1, 0, 0, 2);
    s.q(0) = 3.0;

    Vector out = Vector::Zero(2);
    sys.dSigmaFn_dGeometry(0, out, s, 0.5, 0.0);
    BOOST_CHECK_CLOSE(out(0), 2.5 * 3.0, 1e-12);
    BOOST_CHECK_EQUAL(out(1), 0.0);
}

BOOST_AUTO_TEST_CASE(the_autodiff_layer_derives_them)
{
    // AutodiffTransportSystem widens its RealVector over the geometry slots, so
    // a case that writes Flux() in terms of them gets the derivative for free.
    // Checked against a central difference of the case's own Flux.
    AutodiffGeometryCase sys;
    State s(1, 0, 0, 1);
    s.q(0) = 1.25;
    s.geom(0) = 0.8;

    Vector analytic = Vector::Zero(1);
    sys.dSigmaFn_dGeometry(0, analytic, s, 0.5, 0.0);

    const double h = std::cbrt(std::numeric_limits<double>::epsilon());
    State sp = s, sm = s;
    sp.geom(0) += h;
    sm.geom(0) -= h;
    const double fd = (sys.SigmaFn(0, sp, 0.5, 0.0) - sys.SigmaFn(0, sm, 0.5, 0.0)) / (2 * h);

    BOOST_CHECK_CLOSE(analytic(0), fd, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
