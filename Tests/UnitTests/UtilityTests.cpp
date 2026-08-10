// Tests for the small self-contained utilities: util/trapezoid.hpp and
// Logging.hpp.
//
// trapezoid is the quadrature MagneticField::FluxSurfaceAverage relies on, and
// it is written to work on autodiff dual numbers as well as doubles -- so both
// instantiations need checking, not just the double one.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "Logging.hpp"
#include "Types.hpp"
#include "util/trapezoid.hpp"

#include <autodiff/forward/dual.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

BOOST_AUTO_TEST_SUITE(utility_tests)

// ------------------------------------------------------------ trapezoid --

BOOST_AUTO_TEST_CASE(trapezoid_integrates_polynomials)
{
    // Refinement continues until |DN| < tol, so results are good to about tol.
    const double tol = 1e-10;

    BOOST_TEST(trapezoid<double>([](double) { return 1.0; }, 0.0, 2.0, tol) == 2.0,
               boost::test_tools::tolerance(1e-9));
    BOOST_TEST(trapezoid<double>([](double x) { return x; }, 0.0, 2.0, tol) == 2.0,
               boost::test_tools::tolerance(1e-9));
    BOOST_TEST(trapezoid<double>([](double x) { return x * x; }, 0.0, 3.0, tol) == 9.0,
               boost::test_tools::tolerance(1e-7));
}

BOOST_AUTO_TEST_CASE(trapezoid_integrates_transcendentals)
{
    const double tol = 1e-12;

    // \int_0^pi sin = 2. The default cap of 12 refinements (4096 points) is
    // reached before tol is, and second-order convergence puts the floor at
    // about 5e-8 -- so the achievable accuracy is set by max_refinements here,
    // not by tol.
    BOOST_TEST(trapezoid<double>([](double x) { return std::sin(x); }, 0.0,
                                 std::numbers::pi, tol) == 2.0,
               boost::test_tools::tolerance(1e-6));

    // \int_0^1 e^x = e - 1
    BOOST_TEST(trapezoid<double>([](double x) { return std::exp(x); }, 0.0, 1.0, tol) ==
                   std::numbers::e - 1.0,
               boost::test_tools::tolerance(1e-6));
}

BOOST_AUTO_TEST_CASE(trapezoid_returns_zero_for_an_empty_interval)
{
    BOOST_TEST(trapezoid<double>([](double x) { return std::exp(x); }, 1.5, 1.5) == 0.0);
}

BOOST_AUTO_TEST_CASE(trapezoid_rejects_reversed_endpoints)
{
    BOOST_CHECK_THROW(trapezoid<double>([](double x) { return x; }, 2.0, 1.0),
                      std::logic_error);
}

BOOST_AUTO_TEST_CASE(trapezoid_converges_at_second_order)
{
    // The header documents |DN| <= C/N^2. Cap the refinements so the loop exits
    // on the iteration count rather than the tolerance, then check that
    // quadrupling the work quarters the error.
    auto f = [](double x) { return std::sin(x); };
    const double exact = 2.0;

    const double coarse =
        std::abs(trapezoid<double>(f, 0.0, std::numbers::pi, 0.0, 4) - exact);
    const double fine =
        std::abs(trapezoid<double>(f, 0.0, std::numbers::pi, 0.0, 6) - exact);

    // Two extra refinements = 4x the points = ~16x less error at 2nd order.
    BOOST_TEST(fine < coarse / 8.0,
               "coarse err " << coarse << " fine err " << fine);
}

BOOST_AUTO_TEST_CASE(trapezoid_respects_the_refinement_cap)
{
    // With an unreachable tolerance the cap must still terminate the loop.
    auto f = [](double x) { return std::sin(50.0 * x); };
    double result = 0.0;
    BOOST_CHECK_NO_THROW(result = trapezoid<double>(f, 0.0, 1.0, 0.0, 3));
    BOOST_TEST(std::isfinite(result));
}

BOOST_AUTO_TEST_CASE(trapezoid_works_on_dual_numbers)
{
    // MagneticField integrates through autodiff, so the dual instantiation
    // must carry a correct derivative, not just a correct value.
    //
    // I(b) = \int_0^b x^2 dx = b^3/3, so dI/db = b^2.
    // NB: `Real` is not declared in Types.hpp -- each physics header defines
    // its own `using Real = autodiff::dual`. Unqualified `Real` here would
    // resolve to autodiff::Real, which is a class *template*, so
    // trapezoid<Real> would not even parse.
    using Dual = autodiff::dual;
    const double b0 = 1.7;

    Dual b = b0;
    b.grad = 1.0;
    Dual a = 0.0;

    // The `-> Dual` is load-bearing. autodiff operators return expression
    // templates holding references to their operands, so a deduced return type
    // yields an expression referring to the dead parameter `x`; trapezoid then
    // integrates garbage and silently returns 0. MagneticField's
    // FluxSurfaceAverage had exactly this shape until it was annotated.
    Dual I = trapezoid<Dual>([](Dual x) -> Dual { return x * x; }, a, b, 1e-12);

    BOOST_TEST(I.val == b0 * b0 * b0 / 3.0, boost::test_tools::tolerance(1e-5));
    BOOST_TEST(I.grad == b0 * b0, boost::test_tools::tolerance(1e-5));

    // Pin the trap itself. Only the type is asserted: actually *calling*
    // trapezoid with the deduced-return lambda is undefined behaviour (it
    // reads dangling references), and the observed result varies between zero
    // and arbitrary garbage -- so there is no value here worth asserting.
    auto deduced = [](Dual x) { return x * x; };
    static_assert(!std::is_same_v<decltype(deduced(std::declval<Dual>())), Dual>,
                  "autodiff returns an expression template from x*x; a lambda "
                  "passed to trapezoid must declare -> Dual to force "
                  "evaluation before its operands go out of scope");
}

// -------------------------------------------------------------- Logging --

BOOST_AUTO_TEST_CASE(level_to_string_covers_every_level)
{
    BOOST_TEST(levelToString(LOG_LEVEL::ERROR) == "ERROR");
    BOOST_TEST(levelToString(LOG_LEVEL::WARNING) == "WARNING");
    BOOST_TEST(levelToString(LOG_LEVEL::INFO) == "INFO");
    BOOST_TEST(levelToString(LOG_LEVEL::PDEBUG) == "DEBUG");
}

BOOST_AUTO_TEST_CASE(level_to_string_rejects_an_invalid_level)
{
    // The default branch throws; reach it with a value outside the enum.
    const auto bogus = static_cast<LOG_LEVEL>(99);
    BOOST_CHECK_THROW(levelToString(bogus), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(logmsg_writes_what_it_is_asked_to)
{
    // logmsg goes to a C FILE* via std::print, not to std::cerr, so this has to
    // capture at the descriptor level -- and capturing means the message can be
    // checked rather than merely not-thrown. The vector case exercises the
    // std::formatter shim added for libstdc++ < 15.
    std::vector<double> v{1.0, 2.5, -3.0};

    std::string out;
    {
        CapturedOutput quiet;
        logmsg<LOG_LEVEL::ERROR>("plain message");
        logmsg<LOG_LEVEL::ERROR>("value {}", 42);
        logmsg<LOG_LEVEL::ERROR>("double {} and string {}", 1.5, "s");
        logmsg<LOG_LEVEL::WARNING>("vector {}", v);
        out = quiet.text();
    }

    BOOST_TEST(out.find("ERROR: plain message") != std::string::npos, out);
    BOOST_TEST(out.find("ERROR: value 42") != std::string::npos, out);
    BOOST_TEST(out.find("ERROR: double 1.5 and string s") != std::string::npos, out);
    BOOST_TEST(out.find("WARNING: vector [1, 2.5, -3]") != std::string::npos, out);

    // One line per call: println adds the newline, so a format string should
    // not carry its own.
    BOOST_TEST(std::count(out.begin(), out.end(), '\n') == 4, out);
}

BOOST_AUTO_TEST_CASE(logmsg_filters_by_level_at_compile_time)
{
    // max_log_level is WARNING in a release build, INFO under VERBOSE and
    // PDEBUG under DEBUG. Anything below the threshold must compile to nothing
    // -- that is what makes it free to leave INFO logging in hot code.
    std::string out;
    {
        CapturedOutput quiet;
        logmsg<LOG_LEVEL::INFO>("info {}", 1);
        logmsg<LOG_LEVEL::PDEBUG>("debug {}", 1);
        out = quiet.text();
    }

#if defined(DEBUG)
    BOOST_TEST(out.find("INFO: info 1") != std::string::npos, out);
    BOOST_TEST(out.find("DEBUG: debug 1") != std::string::npos, out);
#elif defined(VERBOSE)
    BOOST_TEST(out.find("INFO: info 1") != std::string::npos, out);
    BOOST_TEST(out.find("DEBUG") == std::string::npos, out);
#else
    BOOST_TEST(out.empty(),
               "a release build should emit nothing below WARNING, got: " << out);
#endif
}

BOOST_AUTO_TEST_SUITE_END()
