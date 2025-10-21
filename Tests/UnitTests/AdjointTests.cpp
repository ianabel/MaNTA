#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/AdjointTestProblem.hpp"
#include "../../PhysicsCases/AutodiffAdjointProblem.hpp"
#include "Types.hpp"
#include <toml.hpp>

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
[AdjointTestProblem]

)"_toml;

BOOST_AUTO_TEST_SUITE(adjoint_test_suite, *boost::unit_test::tolerance(1e-6))

BOOST_AUTO_TEST_CASE(autodiff_init_tests)
{
    Grid testGrid(-1.0, 1.0, 4);
    BOOST_CHECK_NO_THROW(AdjointTestProblem problem(config_snippet, testGrid));
}

BOOST_AUTO_TEST_CASE(adjoint_init_tests)
{
    Grid testGrid(-1.0, 1.0, 4);
    std::shared_ptr<AdjointTestProblem> problem = std::make_shared<AdjointTestProblem>(config_snippet, testGrid);

    BOOST_CHECK_NO_THROW(AutodiffAdjointProblem adjoint(problem));
}

BOOST_AUTO_TEST_CASE(test_derivatives)
{

    Grid testGrid(-1.0, 1.0, 4);
    std::shared_ptr<AdjointTestProblem> problem = std::make_shared<AdjointTestProblem>(config_snippet, testGrid);

    AutodiffAdjointProblem adjoint(problem);

    auto gfun = [&](Position x, Real p, RealVector &u, RealVector &q, RealVector &sigma)
    {
        return problem->g(x, p, u, q, sigma);
    };

    BOOST_CHECK_NO_THROW(adjoint.setG(gfun));
    Value T_s = 50;
    Value SourceWidth = 0.02;
    Value SourceCentre = 0.3;

    auto dGdp = [&](Position x)
    {
        auto y = x - SourceCentre;
        return T_s * (2 * y) / SourceWidth * exp(-y * y / SourceWidth);
    };

    Values Positions(3);
    Positions << 0.2, 0.0, -0.2;
    State s(1);
    s.zero();
    Value p;
    adjoint.dSources_dp(0, p, s, Positions(0), 0.0);
    BOOST_TEST(dGdp(Positions(0)) == p);

    adjoint.dSources_dp(0, p, s, Positions(1), 0.0);
    BOOST_TEST(dGdp(Positions(1)) == p);

    adjoint.dSources_dp(0, p, s, Positions(2), 0.0);
    BOOST_TEST(dGdp(Positions(2)) == p);

    s.Variable[0] = 2.0;

    Values grad(1);
    adjoint.dgFn_du(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 4.0);

    adjoint.dgFn_dq(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 0.0);

    adjoint.dgFn_dsigma(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
