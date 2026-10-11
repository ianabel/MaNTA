// Tests for the [configuration] scalar getters (Config.cpp) and for the
// validation paths in runManta (MaNTA.cpp).

#include <boost/test/unit_test.hpp>

#include "Config.hpp"

#include <stdexcept>
#include <string>

namespace
{
toml::value cfg(std::string const &body)
{
    return toml::parse_str(body);
}
} // namespace

BOOST_AUTO_TEST_SUITE(config_tests, *boost::unit_test::tolerance(1e-12))

BOOST_AUTO_TEST_CASE(get_float_reads_floating_values)
{
    auto c = cfg("tau = 2.5\nnegative = -0.75\nzero = 0.0\n");

    BOOST_TEST(getFloat("tau", c) == 2.5);
    BOOST_TEST(getFloat("negative", c) == -0.75);
    BOOST_TEST(getFloat("zero", c) == 0.0);
}

BOOST_AUTO_TEST_CASE(get_float_accepts_integer_literals)
{
    // TOML distinguishes 1 from 1.0. Writing `tau = 1` in a config is entirely
    // natural and must not be rejected.
    //
    // Regression: these getters branched on is_integer() but then called
    // as_floating(), which throws toml::type_error on an integer node -- so the
    // integer branch could never succeed.
    auto c = cfg("tau = 1\nbig = 1000\nnegative = -3\n");

    BOOST_TEST(getFloat("tau", c) == 1.0);
    BOOST_TEST(getFloat("big", c) == 1000.0);
    BOOST_TEST(getFloat("negative", c) == -3.0);

    BOOST_TEST(getFloatWithDefault("tau", c, 99.0) == 1.0);
}

BOOST_AUTO_TEST_CASE(get_float_throws_when_absent)
{
    auto c = cfg("other = 1.0\n");
    BOOST_CHECK_THROW(getFloat("tau", c), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(get_float_throws_on_wrong_type)
{
    auto c = cfg("tau = \"not a number\"\nflag = true\narr = [1.0, 2.0]\n");
    BOOST_CHECK_THROW(getFloat("tau", c), std::invalid_argument);
    BOOST_CHECK_THROW(getFloat("flag", c), std::invalid_argument);
    BOOST_CHECK_THROW(getFloat("arr", c), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(get_float_with_default_returns_default_when_absent)
{
    auto c = cfg("other = 1.0\n");
    BOOST_TEST(getFloatWithDefault("tau", c, 4.25) == 4.25);
}

BOOST_AUTO_TEST_CASE(get_float_with_default_prefers_specified_value)
{
    auto c = cfg("tau = 0.125\n");
    BOOST_TEST(getFloatWithDefault("tau", c, 4.25) == 0.125);
}

BOOST_AUTO_TEST_CASE(get_float_with_default_throws_on_wrong_type)
{
    auto c = cfg("tau = \"nope\"\n");
    BOOST_CHECK_THROW(getFloatWithDefault("tau", c, 1.0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(get_int_with_default_reads_integers)
{
    auto c = cfg("n = 42\nnegative = -7\nzero = 0\n");

    BOOST_TEST(getIntWithDefault("n", c, 1) == 42);
    BOOST_TEST(getIntWithDefault("negative", c, 1) == -7);
    BOOST_TEST(getIntWithDefault("zero", c, 1) == 0);
    BOOST_TEST(getIntWithDefault("absent", c, 301) == 301);
}

BOOST_AUTO_TEST_CASE(get_int_with_default_rejects_floats)
{
    // An integer option must not silently truncate a float.
    auto c = cfg("n = 4.5\n");
    BOOST_CHECK_THROW(getIntWithDefault("n", c, 1), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
