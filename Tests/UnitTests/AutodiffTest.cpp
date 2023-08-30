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

BOOST_AUTO_TEST_SUITE(autodiff_3varcyl_test_suite, *boost::unit_test::tolerance(1e-12))

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