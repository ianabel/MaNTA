#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/AutodiffTransportSystem.hpp"
#include "Types.hpp"
#include <toml.hpp>

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
    [AutodiffTransportSystem]
    nVars = 3
    isTestProblem = true
    FluxType = "ThreeVarCylFlux"
    x_L = 0.1
    x_R = 1.0
    uL = [1.0,1.0,1.0]
    uR = [2.0,2.0,2.0]

    InitialHeights = [1.0,1.0,1.0]
    [3VarCylFlux]
)"_toml;

BOOST_AUTO_TEST_SUITE(autodiff_3varcyl_test_suite, *boost::unit_test::tolerance(1e-7))

BOOST_AUTO_TEST_CASE(autodiff_init_tests)
{
    BOOST_CHECK_NO_THROW(AutodiffTransportSystem problem(config_snippet));
}
BOOST_AUTO_TEST_CASE(flux_values)
{
    Position x = 0.4;

    Time t = 0.0;

    Values u(3);
    u << 2.14055918662440447519657027442008e+00, 2.14055918662440447519657027442008e+00, 2.14055918662440447519657027442008e+00;
    Values q(3);
    q << 2.87110199725334913622987187409308e+00, 2.87110199725334913622987187409308e+00, 2.87110199725334913622987187409308e+00;
    AutodiffTransportSystem problem(config_snippet);

    BOOST_TEST(problem.InitialValue(0, x) == u(0));
    BOOST_TEST(problem.InitialValue(1, x) == u(1));
    BOOST_TEST(problem.InitialValue(2, x) == u(2));

    BOOST_TEST(problem.InitialDerivative(0, x) == q(0));
    BOOST_TEST(problem.InitialDerivative(1, x) == q(1));
    BOOST_TEST(problem.InitialDerivative(2, x) == q(2));

    Values dGammadu(3);
    dGammadu << -2.29688159780267930898389749927446e+00, 0.00000000000000000000000000000000e+00, 0.00000000000000000000000000000000e+00;

    Values dGammadq(3);
    dGammadq << -2.56867102394928537023588432930410e+00, -8.56223674649761679056325647252379e-01, 1.71244734929952335811265129450476e+00;

    Values dQedu(3);
    dQedu << -2.45000703765619110008344705420313e-01, 8.66689989570877550306704506510869e+00, 0.00000000000000000000000000000000e+00;

    Values dQedq(3);
    dQedq << 9.60112147173932761745618336135522e+00, -2.18051629144139313964956272684503e+00, -1.14163156619968231275663583801361e+00;

    Values dQidu(3);
    dQidu << -9.27926349021486061019459157250822e+01, 0.00000000000000000000000000000000e+00, 9.66207708984864126477987156249583e+01;

    Values dQidq(3);
    dQidq << 7.34629611921026537402212852612138e+01, 1.42703945774960283543464356625918e+00, -7.20359217343530531252326909452677e+01;

    Values grad(3);
    problem.dSigmaFn_du(0, grad, u, q, x, t);
    BOOST_TEST(grad(0) == dGammadu(0));
    BOOST_TEST(grad(1) == dGammadu(1));
    BOOST_TEST(grad(2) == dGammadu(2));
    problem.dSigmaFn_du(1, grad, u, q, x, t);
    BOOST_TEST(grad(0) == dQedu(0));
    BOOST_TEST(grad(1) == dQedu(1));
    BOOST_TEST(grad(2) == dQedu(2));
    problem.dSigmaFn_du(2, grad, u, q, x, t);
    BOOST_TEST(grad(0) == dQidu(0));
    BOOST_TEST(grad(1) == dQidu(1));
    BOOST_TEST(grad(2) == dQidu(2));

    problem.dSigmaFn_dq(0, grad, u, q, x, t);
    BOOST_TEST(grad(0) == dGammadq(0));
    BOOST_TEST(grad(1) == dGammadq(1));
    BOOST_TEST(grad(2) == dGammadq(2));
    problem.dSigmaFn_dq(1, grad, u, q, x, t);
    BOOST_TEST(grad(0) == dQedq(0));
    BOOST_TEST(grad(1) == dQedq(1));
    BOOST_TEST(grad(2) == dQedq(2));
    problem.dSigmaFn_dq(2, grad, u, q, x, t);
    BOOST_TEST(grad(0) == dQidq(0));
    BOOST_TEST(grad(1) == dQidq(1));
    BOOST_TEST(grad(2) == dQidq(2));

    double Gamma = -4.91661100476506529588505145511590e+00;
    double PeFlux = 1.80275736841385700870432629017159e+01;
    double PiFlux = 8.19435167460844127162999939173460e+00;

    BOOST_TEST(problem.SigmaFn(0, u, q, x, t) == Gamma);
    BOOST_TEST(problem.SigmaFn(1, u, q, x, t) == PeFlux);
    BOOST_TEST(problem.SigmaFn(2, u, q, x, t) == PiFlux);

    double SnTest = 9.01393906661699162441436783410609e-01;
    double SPeTest =
        -1.02185418842517030668659572256729e+01;

    double SPiTest = -7.55941685470154123294150849687867e+00;
    BOOST_TEST(problem.TestSource(0, x, t) == SnTest);
    BOOST_TEST(problem.TestSource(1, x, t) == SPeTest);
    BOOST_TEST(problem.TestSource(2, x, t) == SPiTest);
}
BOOST_AUTO_TEST_SUITE_END()