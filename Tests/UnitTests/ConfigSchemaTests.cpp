// The configuration schema: one declaration per option, shared by the TOML
// reader and the dict reader.
//
// These tests are about the table itself -- lookup, aliases, requiredness,
// defaults -- not about reading a config. ConfigSourceTests.cpp covers that.

#include <boost/test/unit_test.hpp>

#include "ConfigSchema.hpp"

#include <ostream>
#include <set>
#include <string>

// Boost.Test prints both operands of a failed comparison, so `BOOST_TEST(e->type
// == Type::UInt)` does not compile unless the type is streamable -- and a scoped
// enum is not. boost_test_print_type is the ADL customisation point the library
// looks for. It is here rather than in ConfigSchema.hpp so that a header linked
// into the solver does not acquire an <ostream> dependency for the sake of a
// test framework.
namespace ConfigSchema
{
std::ostream &boost_test_print_type(std::ostream &os, Type t)
{
    return os << typeName(t);
}

std::ostream &boost_test_print_type(std::ostream &os, Category c)
{
    switch (c)
    {
    case Category::Solver:
        return os << "Solver";
    case Category::ProblemSelection:
        return os << "ProblemSelection";
    case Category::Cli:
        return os << "Cli";
    }
    return os << "<unknown Category>";
}
} // namespace ConfigSchema

using namespace ConfigSchema;

BOOST_AUTO_TEST_SUITE(config_schema_tests)

BOOST_AUTO_TEST_CASE(every_canonical_name_is_unique)
{
    std::set<std::string_view> seen;
    for (auto const &e : schema())
        BOOST_TEST(seen.insert(e.name).second,
                   "duplicate schema entry: " << std::string(e.name));
}

BOOST_AUTO_TEST_CASE(no_alias_collides_with_a_name_or_another_alias)
{
    std::set<std::string_view> seen;
    for (auto const &e : schema())
        seen.insert(e.name);
    for (auto const &e : schema())
        for (auto const &a : e.aliases)
            BOOST_TEST(seen.insert(a).second,
                       "alias collides: " << std::string(a));
}

BOOST_AUTO_TEST_CASE(find_entry_resolves_canonical_names)
{
    auto const *e = findEntry("PolynomialDegree");
    BOOST_REQUIRE(e != nullptr);
    BOOST_TEST(e->type == Type::UInt);
    BOOST_TEST(isRequired(*e, Reader::Toml));
    BOOST_TEST(isRequired(*e, Reader::Dict));
}

BOOST_AUTO_TEST_CASE(find_entry_resolves_the_deprecated_aliases)
{
    // The two genuine name conflicts between the old readers. Both old
    // spellings must keep working; both must resolve to the canonical entry.
    BOOST_REQUIRE(findEntry("tZero") != nullptr);
    BOOST_TEST(findEntry("tZero")->name == "t_initial");

    BOOST_REQUIRE(findEntry("aggressiveTimesteps") != nullptr);
    BOOST_TEST(findEntry("aggressiveTimesteps")->name == "AggressiveTimesteps");
}

BOOST_AUTO_TEST_CASE(find_entry_returns_null_for_an_unknown_key)
{
    BOOST_TEST(findEntry("Superconvergnet") == nullptr);
}

BOOST_AUTO_TEST_CASE(nearest_key_suggests_the_obvious_typo)
{
    BOOST_TEST(nearestKey("Superconvergnet") == "Superconvergent");
    BOOST_TEST(nearestKey("Poly_degree") == "PolynomialDegree");
    BOOST_TEST(nearestKey("delta_T") == "delta_t");
}

BOOST_AUTO_TEST_CASE(nearest_key_gives_up_on_something_unrelated)
{
    // A suggestion that is nothing like the input is worse than none: it sends
    // the reader off to check a key they never wrote.
    BOOST_TEST(nearestKey("qqqqqqqqqqqqqqqq").empty());
}

BOOST_AUTO_TEST_CASE(transport_system_is_required_of_toml_only)
{
    // The one key whose requiredness genuinely differs by reader: PyRunner is
    // handed the physics object, so a dict has nothing to name.
    auto const *e = findEntry("TransportSystem");
    BOOST_REQUIRE(e != nullptr);
    BOOST_TEST(e->category == Category::ProblemSelection);
    BOOST_TEST(isRequired(*e, Reader::Toml));
    BOOST_TEST(!isRequired(*e, Reader::Dict));
}

BOOST_AUTO_TEST_CASE(the_cli_keys_are_recognised_but_are_not_solver_options)
{
    // manta.cli reads these; the solver never does. They are in the schema so
    // that unknown-key rejection does not fire on the eight .conf files that
    // carry them.
    for (auto const *k : {"PythonModule", "PythonModuleFile", "PythonModuleName"})
    {
        auto const *e = findEntry(k);
        BOOST_REQUIRE_MESSAGE(e != nullptr, "missing schema entry: " << k);
        BOOST_TEST(e->category == Category::Cli);
    }
}

BOOST_AUTO_TEST_CASE(every_default_matches_its_declared_type)
{
    for (auto const &e : schema())
    {
        switch (e.type)
        {
        case Type::Bool:   BOOST_TEST(std::holds_alternative<bool>(e._default),        std::string(e.name)); break;
        case Type::Int:    BOOST_TEST(std::holds_alternative<int>(e._default),         std::string(e.name)); break;
        case Type::UInt:   BOOST_TEST(std::holds_alternative<unsigned>(e._default),    std::string(e.name)); break;
        case Type::Double: BOOST_TEST(std::holds_alternative<double>(e._default),      std::string(e.name)); break;
        case Type::String: BOOST_TEST(std::holds_alternative<std::string>(e._default), std::string(e.name)); break;
        case Type::DoubleList:
            BOOST_TEST(std::holds_alternative<std::vector<double>>(e._default), std::string(e.name)); break;
        case Type::StringList:
            BOOST_TEST(std::holds_alternative<std::vector<std::string>>(e._default), std::string(e.name)); break;
        }
    }
}

BOOST_AUTO_TEST_CASE(every_entry_has_a_doc_line)
{
    // The doc string is what `manta --list-options` prints and what the
    // configuration.rst table is written from. An entry without one is an
    // option nobody can find out about.
    for (auto const &e : schema())
        BOOST_TEST(!e.doc.empty(), "no doc for " << std::string(e.name));
}

BOOST_AUTO_TEST_SUITE_END()
