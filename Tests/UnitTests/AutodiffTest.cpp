#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/AutodiffTransportSystem.hpp"
#include "../../PhysicsCases/ThreeVarMirror.hpp"
#include "Types.hpp"
#include <toml.hpp>

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
    [AutodiffTransportSystem]
    nVars = 3
    isTestProblem = false
    FluxType = "MatrixFlux"
    x_L = -1.0
    x_R = 1.0
    uL = [0.0,0.0,0.0]
    uR = [0.0,0.0,0.0]
    InitialHeights = [1.0,1.0,1.0]
    InitialProfile =["Gaussian","Gaussian","Gaussian"]

    [MatrixFlux]
    Kappa = [1.0,0.0,0.0,
            0.0,1.0,0.0,
            0.0,0.0,1.0]
)"_toml;

const toml::value config_snippet_mirror = u8R"(
[AutodiffTransportSystem]
nVars = 3
isTestProblem = false
FluxType = "ThreeVarMirror"
x_L = 0.0314
x_R = 3.14
uL = [0.05,0.0025,0.0025]
uR = [0.05,0.0025,0.0025]
InitialHeights = [0.5,0.25,0.25]
InitialProfile =["Uniform","Uniform","Uniform"]

[3VarMirror]
SourceType = "Gaussian"
SourceStrength = 10.0
SourceCenter =  0.7854
SourceWidth = 0.0314
)"_toml;

BOOST_AUTO_TEST_SUITE(autodiff_test_suite, *boost::unit_test::tolerance(1e-7))

BOOST_AUTO_TEST_CASE(autodiff_init_tests)
{
    BOOST_CHECK_NO_THROW(AutodiffTransportSystem problem(config_snippet));
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

    Values u(3);
    u << umid, umid, umid;
    Values q(3);
    q << 0.0, 0.0, 0.0;
    AutodiffTransportSystem problem(config_snippet);

    BOOST_TEST(problem.InitialValue(0, x) == u(0));
    BOOST_TEST(problem.InitialValue(1, x) == u(1));
    BOOST_TEST(problem.InitialValue(2, x) == u(2));

    BOOST_TEST(problem.InitialDerivative(0, x) == q(0));
    BOOST_TEST(problem.InitialDerivative(1, x) == q(1));
    BOOST_TEST(problem.InitialDerivative(2, x) == q(2));

    Values dGammadu(3);
    dGammadu << 0.0, 0.0, 0.0;

    Values dGammadq(3);
    dGammadq << 1.0, 0.0, 0.0;

    Values dQedu(3);
    dQedu << 0.0, 0.0, 0.0;

    Values dQedq(3);
    dQedq << 0.0, 1.0, 0.0;

    Values dQidu(3);
    dQidu << 0.0, 0.0, 0.0;

    Values dQidq(3);
    dQidq << 0.0, 0.0, 1.0;

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

    double Gamma = 0.0;
    double PeFlux = 0.0;
    double PiFlux = 0.0;

    BOOST_TEST(problem.SigmaFn(0, u, q, x, t) == Gamma);
    BOOST_TEST(problem.SigmaFn(1, u, q, x, t) == PeFlux);
    BOOST_TEST(problem.SigmaFn(2, u, q, x, t) == PiFlux);

    // double SnTest = 9.01393906661699162441436783410609e-01;
    // double SPeTest =
    //     -1.02185418842517030668659572256729e+01;

    // double SPiTest = -7.55941685470154123294150849687867e+00;
    // BOOST_TEST(problem.TestSource(0, x, t) == SnTest);
    // BOOST_TEST(problem.TestSource(1, x, t) == SPeTest);
    // BOOST_TEST(problem.TestSource(2, x, t) == SPiTest);
}

BOOST_AUTO_TEST_CASE(mirror_test)
{
    ThreeVarMirror problem(config_snippet_mirror, 3);
    double Rtest = 0.32;
    double Vtest = M_PI * Rtest * Rtest;
    double R = problem.R(Vtest, 0.0);
    BOOST_TEST(Rtest == R);
}

BOOST_AUTO_TEST_SUITE_END()