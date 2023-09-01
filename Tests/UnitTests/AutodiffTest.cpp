#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/Autodiff3VarCyl.hpp"
#include "Types.hpp"
#include <toml.hpp>

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
    [Autodiff3VarCyl]
	 isTestProblem = true
)"_toml;

BOOST_AUTO_TEST_SUITE(autodiff_3varcyl_test_suite, *boost::unit_test::tolerance(1e-7))

BOOST_AUTO_TEST_CASE(autodiff_init_tests)
{
    BOOST_CHECK_NO_THROW(Autodiff3VarCyl problem(config_snippet));
}
BOOST_AUTO_TEST_CASE(flux_values)
{
    Position x = 0.4;

    Time t = 0.0;

    Values u(3);
    u << 5700253868753132215.8273104052019, 913.28136977957033286570608431802, 913.2813697795703328657060843;
    Values q(3);
    q << 4929545543581451975.3046964112213, 789.80029571649370225130951430966, 789.80029571649370225130951430966;
    Autodiff3VarCyl problem(config_snippet);

    BOOST_TEST(problem.InitialValue(0, x) == u(0));
    BOOST_TEST(problem.InitialValue(1, x) == u(1));
    BOOST_TEST(problem.InitialValue(2, x) == u(2));

    BOOST_TEST(problem.InitialDerivative(0, x) == q(0));
    BOOST_TEST(problem.InitialDerivative(1, x) == q(1));
    BOOST_TEST(problem.InitialDerivative(2, x) == q(2));

    Values dGammadu(3);
    dGammadu << 0.00034353148227620134789838034737105, 30278.554317541407763560523792923, 0;

    Values dGammadq(3);
    dGammadq << 0.00059586122787358867564787989340087, 1239691067768.8920113523558615047, -2479382135537.7840227047117230094;

    Values dQedu(3);
    dQedu << -0.0000000000000000000058709164602047996083390819345235, 0.001296258829904767642096502422093, 0;

    Values dQedq(3);
    dQedq << 0.00000000000000000035683628175503932075076743407737, -0.00050581997566157977680011529964199, -0.00026482721238826160676203758388868;

    Values dQidu(3);
    dQidu << -0.0000000000000000022235754229627814046729555118989, 0.0000000000080852654983353383259999345085004, 0.014451018174426159041068021496258;

    Values dQidq(3);
    dQidq << 0.0000000000000000027303320758586042729612508935858, 0.00033103401548532704202659994077826, -0.016710358148410677840299019030735;

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

    double Gamma = 1958216688536254.3662270124432772;
    double PeFlux = 1.1503833254988804059475553499814;
    double PiFlux = 0.52290127322553370889470065111301;

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