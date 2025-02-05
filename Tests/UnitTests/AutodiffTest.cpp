#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/LinearDiffSourceTest.hpp"
#include "Types.hpp"
#include <toml.hpp>

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
[LinearDiffSourceTest]
nVars = 2
useMMS = false
growth = 1.0
growth_rate = 1.0

SourceStrength = [1.0,1.0]

InitialHeight = [1.0,1.0]

Kappa = [1.0,0.0,
        0.0,1.0]

)"_toml;

const toml::value config_snippet_nc_file = u8R"(
[LinearDiffSourceTest]
nVars = 2
useNcFile = true
InitialConditionFilename = "./Tests/UnitTests/testic.nc"

SourceStrength = [1.0,1.0]

InitialHeight = [1.0,0.5]

Kappa = [1.0,0.0,
        0.0,1.0]

)"_toml;

BOOST_AUTO_TEST_SUITE(autodiff_test_suite, *boost::unit_test::tolerance(1e-3))

BOOST_AUTO_TEST_CASE(autodiff_init_tests)
{
    Grid testGrid(-1.0, 1.0, 4);
    BOOST_CHECK_NO_THROW(LinearDiffSourceTest problem(config_snippet, testGrid));
}
BOOST_AUTO_TEST_CASE(flux_values)
{
    Position x = 0.0;

    Time t = 0.0;
    double x_L = -1.0;
    double x_R = 1.0;
    double C = 0.5 * (x_R + x_L);
    double shape = 10;

    double umid = (::exp(-(x - C) * (x - C) * shape) - ::exp(-(x_L - C) * (x_L - C) * shape));

    State s(2, 0);
    s.Variable << umid, umid;
    s.Derivative << 0.0, 0.0;

    Grid testGrid(-1.0, 1.0, 4);
    LinearDiffSourceTest problem(config_snippet, testGrid);

    // BOOST_TEST(problem.InitialValue(0, x) == u(0));
    // BOOST_TEST(problem.InitialValue(1, x) == u(1));
    // BOOST_TEST(problem.InitialValue(2, x) == u(2));

    BOOST_TEST(problem.InitialDerivative(0, x) == s.Derivative(0));
    BOOST_TEST(problem.InitialDerivative(1, x) == s.Derivative(1));

    Values dGammadu(2);
    dGammadu << 0.0, 0.0;

    Values dGammadq(2);
    dGammadq << 1.0, 0.0;

    Values dQedu(2);
    dQedu << 0.0, 0.0;

    Values dQedq(2);
    dQedq << 0.0, 1.0;

    Values dQidu(2);
    dQidu << 0.0, 0.0;

    Values dQidq(2);
    dQidq << 0.0, 1.0;

    Values grad(2);
    problem.dSigmaFn_du(0, grad, s, x, t);
    BOOST_TEST(grad(0) == dGammadu(0));
    BOOST_TEST(grad(1) == dGammadu(1));
    problem.dSigmaFn_du(1, grad, s, x, t);
    BOOST_TEST(grad(0) == dQedu(0));
    BOOST_TEST(grad(1) == dQedu(1));

    problem.dSigmaFn_dq(0, grad, s, x, t);
    BOOST_TEST(grad(0) == dGammadq(0));
    BOOST_TEST(grad(1) == dGammadq(1));
    problem.dSigmaFn_dq(1, grad, s, x, t);
    BOOST_TEST(grad(0) == dQedq(0));
    BOOST_TEST(grad(1) == dQedq(1));

    double sigma1 = 0.0;
    double sigma2 = 0.0;

    BOOST_TEST(problem.SigmaFn(0, s, x, t) == sigma1);
    BOOST_TEST(problem.SigmaFn(1, s, x, t) == sigma2);

    Values dS1du(2);
    dS1du << 0.0, 0.0;

    Values dS1dq(2);
    dS1dq << 0.0, 0.0;

    Values dS1dsigma(2);
    dS1dsigma << 0.0, 0.0;

    problem.dSources_du(0, grad, s, x, t);
    BOOST_TEST(grad(0) == dS1du(0));
    BOOST_TEST(grad(1) == dS1du(1));

    problem.dSources_dq(0, grad, s, x, t);
    BOOST_TEST(grad(0) == dS1dq(0));
    BOOST_TEST(grad(1) == dS1dq(1));

    problem.dSources_dsigma(0, grad, s, x, t);
    BOOST_TEST(grad(0) == dS1dsigma(0));
    BOOST_TEST(grad(1) == dS1dsigma(1));

    Values dS2du(2);
    dS2du << 0.0, 0.0;

    Values dS2dq(2);
    dS2dq << 0.0, 0.0;

    Values dS2dsigma(2);
    dS2dsigma << 0.0, 0.0;

    problem.dSources_du(1, grad, s, x, t);
    BOOST_TEST(grad(0) == dS2du(0));
    BOOST_TEST(grad(1) == dS2du(1));

    problem.dSources_dq(1, grad, s, x, t);
    BOOST_TEST(grad(0) == dS2dq(0));
    BOOST_TEST(grad(1) == dS2dq(1));

    problem.dSources_dsigma(1, grad, s, x, t);
    BOOST_TEST(grad(0) == dS2dsigma(0));
    BOOST_TEST(grad(1) == dS2dsigma(1));
}

BOOST_AUTO_TEST_CASE(nc_file_test)
{
    Position x = 0.0;

    double x_L = -1.0;
    double x_R = 1.0;
    double C = 0.5 * (x_R + x_L);
    double shape = 10;

    Values InitialHeights(2);
    InitialHeights << 1.0, 0.5;

    auto u = [shape, C, x_L, &InitialHeights](Index i, Position x)
    {
        return InitialHeights(i) * (::exp(-(x - C) * (x - C) * shape) - ::exp(-(x_L - C) * (x_L - C) * shape));
    };

    // auto dudx = [shape, C, &InitialHeights](Index i, Position x)
    // {
    //     return InitialHeights(i) * (-2 * shape * (x - C) * ::exp(-(x - C) * (x - C) * shape));
    // };

    Grid testGrid(-1.0, 1.0, 4);
    LinearDiffSourceTest problem(config_snippet_nc_file, testGrid);

    Values positions(7);
    positions << x_L, x_R, x, -0.5, 0.5, -0.3, 0.3;

    for (auto &pos : positions)
    {
        BOOST_TEST(problem.InitialValue(0, pos) == u(0, pos));
        BOOST_TEST(problem.InitialValue(1, pos) == u(1, pos));

        // BOOST_TEST(problem.InitialDerivative(0, pos) == dudx(0, pos));
        // BOOST_TEST(problem.InitialDerivative(1, pos) == dudx(1, pos));
    }
}

BOOST_AUTO_TEST_SUITE_END()
