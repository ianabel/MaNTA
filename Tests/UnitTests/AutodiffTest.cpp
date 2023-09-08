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
    u << 2.08366417701714290444670041324571e+00, 2.08366417701714290444670041324571e+00, 2.08366417701714290444670041324571e+00;
    Values q(3);
    q << 2.30821216300070020110979385208338e+00, 2.30821216300070020110979385208338e+00, 2.30821216300070020110979385208338e+00;
    AutodiffTransportSystem problem(config_snippet);

    BOOST_TEST(problem.InitialValue(0, x) == u(0));
    BOOST_TEST(problem.InitialValue(1, x) == u(1));
    BOOST_TEST(problem.InitialValue(2, x) == u(2));

    BOOST_TEST(problem.InitialDerivative(0, x) == q(0));
    BOOST_TEST(problem.InitialDerivative(1, x) == q(1));
    BOOST_TEST(problem.InitialDerivative(2, x) == q(2));

    Values dGammadu(3);
    dGammadu << 1.84656973040056016088783508166671e+00, -5.51012976947947269360345251822928e-40, 0.00000000000000000000000000000000e+00;

    Values dGammadq(3);
    dGammadq << -5.51012976947947269360345251822928e-40, 2.50039701242057166297172443591990e+00, -1.66693134161371436796628131560283e+00;

    Values dQedu(3);
    dQedu << -1.96967437909393083828035742044449e-01, 6.96772311604478034041676437482238e+00, 0.00000000000000000000000000000000e+00;

    Values dQedq(3);
    dQedq << 6.96772311604478034041676437482238e+00, 9.34592838864755925953886617207900e+00, -1.11128756107580972667392416042276e+00;

    Values dQidu(3);
    dQidu << -4.06292018737533858208088588526152e-02, -3.67341984631964846240230167881952e-40, 3.11824541920802023042824657750316e+00;

    Values dQidq(3);
    dQidq << -3.67341984631964846240230167881952e-40, 4.20400505726033735243163391714916e+00, -2.81489560591557497204462379158940e+00;

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

    double Gamma = 3.84763119759985094958665285957977e+00;
    double PeFlux = 1.41079810578661195563654473517090e+01;
    double PiFlux = 6.41271866266641765719214163254946e+00;

    BOOST_TEST(problem.SigmaFn(0, u, q, x, t) == Gamma);
    BOOST_TEST(problem.SigmaFn(1, u, q, x, t) == PeFlux);
    BOOST_TEST(problem.SigmaFn(2, u, q, x, t) == PiFlux);

    double SnTest = -1206385124558814798.4391338028379;
    double SPeTest = -196.57193569210533881523949774747;
    double SPiTest = -194.25781051303646936964269715141;

    BOOST_TEST(problem.TestSource(0, x, t) == SnTest);
    BOOST_TEST(problem.TestSource(1, x, t) == SPeTest);
    BOOST_TEST(problem.TestSource(2, x, t) == SPiTest);
}
BOOST_AUTO_TEST_SUITE_END()