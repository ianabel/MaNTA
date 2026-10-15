// AutodiffTransportSystem with nAux > 0 -- a combination nothing else covers.
//
// The gap this closes was not a subtle one. AutodiffTransportSystem overrode
// dSources_dPhi and not its sibling dSigma_dPhi, so TransportSystem's default
// stood, and that default *throws* when nAux != 0. Every case built on the
// autodiff layer with an auxiliary variable therefore died with "nAux > 0 but no
// coupling to fluxes provided" the moment the Jacobian was assembled.
//
// Two things kept it hidden. AuxVarADTest was the only such case in the tree and
// was not in the regression suite, so nothing ran it; and no unit test had ever
// constructed an AutodiffTransportSystem with an auxiliary variable, because
// AuxVarTest -- the case that does cover the aux path -- derives from
// TransportSystem directly and hand-codes every derivative.
//
// So the probe below exists to be the thing that was missing: a case on that
// layer, with nAux = 1, whose hooks genuinely depend on phi. Its Source and
// GFunc are chosen so that every derivative block is nonzero and has a different
// closed form, which is what makes a finite-difference comparison able to fail.

#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/AutodiffTransportSystem.hpp"
#include "State.hpp"
#include "Types.hpp"
#include "gridStructures.hpp"

#include <toml.hpp>

using namespace toml::literals::toml_literals;

namespace
{

// Empty but for the section header: AutodiffTransportSystem reads its table only
// if it is present, and the probe supplies its own initial data, so there is
// nothing it needs from a configuration.
const toml::value empty_config = u8R"(
    [AutodiffAuxProbe]
)"_toml;

const double kappa = 1.3;

class AutodiffAuxProbe : public AutodiffTransportSystem
{
public:
    AutodiffAuxProbe(toml::value const &config, Grid const &grid)
        : AutodiffTransportSystem(config, grid,
                                  {.variables = {{"u", "the diffused quantity", ""}},
                                   .aux = {{"a", "an auxiliary variable", ""}}})
    {
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

protected:
    // No phi argument -- that is the signature this whole layer imposes, and the
    // reason dSigma_dPhi is structurally zero rather than merely unimplemented.
    Real Flux(Index, RealVector, RealVector q, Real, Time) override
    {
        return kappa * q(0);
    }

    // Quadratic in phi and cross-coupled to u, so dSources_dPhi is
    // 2 phi + 3 u -- a function of the state rather than a constant, which a
    // finite difference can distinguish from a wrong answer.
    Real Source(Index, RealVector u, RealVector, RealVector, RealVector phi, Real x, Time) override
    {
        return phi(0) * phi(0) + 3.0 * phi(0) * u(0) + x;
    }

    // Deliberately depends on u, q, sigma and phi, each differently, so that all
    // four blocks AuxGPrime fills are exercised and none can be confused with
    // another:  dG/dphi = 1,  dG/du = -2u,  dG/dq = -0.5,  dG/dsigma = 0.25.
    Real GFunc(Index, RealVector u, RealVector q, RealVector sigma, RealVector phi,
               Position, Time) override
    {
        return phi(0) - u(0) * u(0) - 0.5 * q(0) + 0.25 * sigma(0);
    }

    autodiff::dual2nd InitialFunction(Index, autodiff::dual2nd x, autodiff::dual2nd) const override
    {
        return 1.0 - x * x;
    }

    Value InitialAuxValue(Index, Position x) const override { return x * x; }
};

// A bare TransportSystem with an auxiliary variable, to check the base default is
// still a refusal. The override in the autodiff layer says "this derivative is
// zero"; it must not have turned the guard off for cases that simply forgot.
class BareAuxCase : public TransportSystem
{
public:
    BareAuxCase()
        : TransportSystem({.variables = {{"u", "", ""}}, .aux = {{"a", "", ""}}})
    {
    }
    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    // The rest of the pure interface, none of which this test calls. Present
    // because it is pure: what is being checked here is one default that is
    // *not*, and a case cannot exist to check it without these.
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}
    Value InitialValue(Index, Position) const override { return 0.0; }
    Value InitialDerivative(Index, Position) const override { return 0.0; }
};

// A state with every component set to something distinct and nonzero, so that a
// derivative taken with respect to the wrong one is visible.
State probeState()
{
    State s(1, 0, 1);
    s.u(0) = 0.7;
    s.q(0) = -0.4;
    s.sigma(0) = 0.3;
    s.phi(0) = 1.1;
    return s;
}

// cbrt(eps), the central-difference optimum: truncation is O(h^2 F''') against
// round-off O(eps |F| / h), and those balance at eps^(1/3). sqrt(eps) is the
// one-sided choice and costs two and a half decimal places here, exactly as
// AlgebraicDerivatives.cpp records.
const double fdStep = std::cbrt(std::numeric_limits<double>::epsilon());

} // namespace

BOOST_AUTO_TEST_SUITE(autodiff_aux_tests)

BOOST_AUTO_TEST_CASE(a_dsigma_dphi_of_zero_is_reported_rather_than_thrown)
{
    Grid grid(0.0, 1.0, 4);
    AutodiffAuxProbe probe(empty_config, grid);
    State s = probeState();

    Vector out(1);
    out.setConstant(-99.0); // so "left alone" and "written zero" are the same test

    // The whole defect, in one line. This threw for every autodiff case with an
    // auxiliary variable, which is to say it made the combination unusable.
    BOOST_CHECK_NO_THROW(probe.dSigma_dPhi(0, out, s, 0.5, 0.0));

    // And the answer is zero, because Flux takes no phi. Checked rather than
    // assumed: the contract is that a derivative out-parameter arrives zeroed and
    // a hook writes only its nonzero entries, so an override that is allowed to
    // do nothing must be one whose answer really is zero.
    out.setZero();
    probe.dSigma_dPhi(0, out, s, 0.5, 0.0);
    BOOST_TEST(out(0) == 0.0);
}

BOOST_AUTO_TEST_CASE(the_base_class_still_refuses_a_missing_flux_coupling)
{
    // The guard the override above steps around is still armed for anyone who has
    // not thought about it. Without this, "AutodiffTransportSystem defines
    // dSigma_dPhi" and "nothing checks dSigma_dPhi any more" look the same.
    BareAuxCase bare;
    State s = probeState();
    Vector out(1);
    out.setZero();

    BOOST_CHECK_THROW(bare.dSigma_dPhi(0, out, s, 0.5, 0.0), std::logic_error);
    BOOST_CHECK_THROW(bare.dSources_dPhi(0, out, s, 0.5, 0.0), std::logic_error);
}

BOOST_AUTO_TEST_CASE(dsources_dphi_matches_a_finite_difference_of_the_source)
{
    // The autodiff gradient at AutodiffTransportSystem.cpp:148, which nothing has
    // ever evaluated: the only case that could reach it could not run, and its
    // own Source ignored phi.
    Grid grid(0.0, 1.0, 4);
    AutodiffAuxProbe probe(empty_config, grid);

    const Position x = 0.37;
    const Time t = 0.0;
    State s = probeState();

    Vector analytic(1);
    analytic.setZero();
    probe.dSources_dPhi(0, analytic, s, x, t);

    State plus = s, minus = s;
    const double h = fdStep * std::max(1.0, std::abs(s.phi(0)));
    plus.phi(0) += h;
    minus.phi(0) -= h;
    const double fd =
        (probe.Sources(0, plus, x, t) - probe.Sources(0, minus, x, t)) / (2.0 * h);

    // 2 phi + 3 u = 2(1.1) + 3(0.7) = 4.3, so the closed form is checked too --
    // a finite difference of the same wrong function would agree with itself.
    BOOST_TEST(analytic(0) == fd, boost::test_tools::tolerance(1e-8));
    BOOST_TEST(analytic(0) == 4.3, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(aux_g_prime_matches_a_finite_difference_of_g_func)
{
    // AuxGPrime fills four blocks from one autodiff pass. Each is differenced
    // separately, because a pass that wrote the right value into the wrong block
    // would satisfy any check that only looked at one of them -- and that is not
    // hypothetical: dAux_Mat's column layout was exactly this defect, and
    // Tests/README.md records it.
    Grid grid(0.0, 1.0, 4);
    AutodiffAuxProbe probe(empty_config, grid);

    const Position x = 0.37;
    const Time t = 0.0;
    State s = probeState();

    State analytic(1, 0, 1);
    probe.AuxGPrime(0, analytic, s, x, t);

    struct Component
    {
        const char *name;
        double &(State::*entry)(Index);
        double expected;
    };

    // dG/dphi = 1, dG/du = -2u = -1.4, dG/dq = -0.5, dG/dsigma = 0.25.
    const Component components[] = {
        {"phi", &State::phi, 1.0},
        {"u", &State::u, -2.0 * 0.7},
        {"q", &State::q, -0.5},
        {"sigma", &State::sigma, 0.25},
    };

    for (auto const &c : components)
    {
        State plus = s, minus = s;
        const double value = (s.*(c.entry))(0);
        const double h = fdStep * std::max(1.0, std::abs(value));
        (plus.*(c.entry))(0) += h;
        (minus.*(c.entry))(0) -= h;

        const double fd =
            (probe.AuxG(0, plus, x, t) - probe.AuxG(0, minus, x, t)) / (2.0 * h);
        const double got = (analytic.*(c.entry))(0);

        BOOST_TEST_MESSAGE("dG/d" << c.name << ": autodiff " << got << ", fd " << fd
                                  << ", closed form " << c.expected);
        BOOST_TEST(got == fd, boost::test_tools::tolerance(1e-7));
        BOOST_TEST(got == c.expected, boost::test_tools::tolerance(1e-12));
    }
}

BOOST_AUTO_TEST_SUITE_END()
