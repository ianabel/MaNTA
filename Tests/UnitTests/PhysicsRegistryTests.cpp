// Tests for the physics case registry.
//
// PhysicsCases is a process-global map populated by static-initialisation side
// effects: every physics case declares REGISTER_PHYSICS_HEADER/IMPL, which
// instantiates a PhysicsCaseRegister<T> whose constructor inserts into the map
// before main() runs. The map is deliberately leaked ("never delete'ed ...
// because we can't guarantee correct destruction order") and there is no way to
// reset it between test cases.
//
// So these tests register throwaway types under names that cannot collide with
// a real physics case, and never assume the map starts empty.

#include <boost/test/unit_test.hpp>

#include "PhysicsCases.hpp"
#include "TransportSystem.hpp"
#include "Types.hpp"

#include <memory>
#include <string>
#include <toml.hpp>

using namespace toml::literals::toml_literals;

namespace
{

const toml::value empty_config = u8R"(
    [Registered]
    Value = 3
)"_toml;

// A minimal case with the (config, grid) constructor the registry requires.
class RegisteredProbe : public TransportSystem
{
public:
    RegisteredProbe(toml::value const &config, Grid const &g)
        : TransportSystem({.variables = numberedFields(1)}), lower(g.lowerBoundary())
    {
        // Read something from the config so a test can prove the value really
        // reached the constructor rather than being defaulted.
        if (config.count("Registered") == 1)
            marker = toml::find_or(config.at("Registered"), "Value", 0);
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 1.0;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    Value InitialValue(Index, Position x) const override { return x; }
    Value InitialDerivative(Index, Position) const override { return 1.0; }

    int marker = -1;
    double lower;
};

// A second, distinguishable type for the duplicate-name test. It used to
// override getVariableName to tell itself apart; names come from the spec now,
// so the dynamic_cast at the assertion is the whole discriminator.
class OtherProbe : public RegisteredProbe
{
public:
    using RegisteredProbe::RegisteredProbe;
};

// Registered the way real physics cases are: by a namespace-scope object whose
// constructor runs before main().
PhysicsCaseRegister<RegisteredProbe> autoReg("UnitTestAutoRegisteredProbe");

// getMap() is protected; deriving is how PhysicsCaseRegister reaches it, so a
// test peek can do the same. This lets registration be checked without
// constructing the case, which would need each one's own config section.
struct RegistryPeek : public PhysicsCases
{
    static bool isRegistered(std::string const &s) { return getMap()->count(s) == 1; }
    static size_t size() { return getMap()->size(); }
};

} // namespace

BOOST_AUTO_TEST_SUITE(physics_registry_tests)

BOOST_AUTO_TEST_CASE(static_registration_happens_before_main)
{
    // This is the mechanism every physics case relies on. If it stopped
    // working, MaNTA would report every model as unknown -- with no compile
    // error anywhere.
    Grid grid(0.0, 1.0, 4);
    auto p = PhysicsCases::InstantiateProblem("UnitTestAutoRegisteredProbe", empty_config,
                                              grid);
    BOOST_TEST_REQUIRE(p != nullptr);
    BOOST_TEST(p->getNumVars() == 1);
}

BOOST_AUTO_TEST_CASE(explicit_registration_round_trips_config_and_grid)
{
    // RegisterPhysicsCase is the path the Python binding uses. Both the config
    // node and the grid must reach the constructor: the grid especially, since
    // several cases size internal arrays from it.
    PhysicsCases::RegisterPhysicsCase("UnitTestExplicitProbe",
                                      &createTransportSystem<RegisteredProbe>);

    Grid grid(-2.5, 4.0, 6);
    auto p = PhysicsCases::InstantiateProblem("UnitTestExplicitProbe", empty_config, grid);
    BOOST_TEST_REQUIRE(p != nullptr);

    auto *probe = dynamic_cast<RegisteredProbe *>(p.get());
    BOOST_TEST_REQUIRE(probe != nullptr);
    BOOST_TEST(probe->marker == 3);
    BOOST_TEST(probe->lower == -2.5);
}

BOOST_AUTO_TEST_CASE(an_unknown_name_throws_and_names_what_is_available)
{
    // This used to return nullptr, leaving every caller to remember the check;
    // the one that forgot dereferenced it, so an unrecognised TransportSystem
    // segfaulted. The list of registered names travels in the message because
    // the usual cause is a case whose object file was never linked in.
    Grid grid(0.0, 1.0, 4);
    BOOST_CHECK_THROW(PhysicsCases::InstantiateProblem("NoSuchPhysicsCase", empty_config, grid),
                      std::invalid_argument);

    try
    {
        PhysicsCases::InstantiateProblem("NoSuchPhysicsCase", empty_config, grid);
    }
    catch (std::invalid_argument const &e)
    {
        const std::string what(e.what());
        BOOST_TEST(what.find("NoSuchPhysicsCase") != std::string::npos);
        // A name that is definitely registered, from the fixture above.
        BOOST_TEST(what.find("UnitTestAutoRegisteredProbe") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(a_duplicate_name_is_rejected_rather_than_silently_dropped)
{
    // map::insert is a no-op when the key exists, so the *first* registration
    // used to win and the second was dropped without a word -- leaving someone
    // whose case name collided with an existing one to wonder why their class
    // was never instantiated. It is an error now.
    PhysicsCases::RegisterPhysicsCase("UnitTestDuplicateProbe",
                                      &createTransportSystem<RegisteredProbe>);
    BOOST_CHECK_THROW(PhysicsCases::RegisterPhysicsCase("UnitTestDuplicateProbe",
                                                        &createTransportSystem<OtherProbe>),
                      std::invalid_argument);

    // The first registration is still intact and still instantiable.
    Grid grid(0.0, 1.0, 4);
    auto p = PhysicsCases::InstantiateProblem("UnitTestDuplicateProbe", empty_config, grid);
    BOOST_TEST_REQUIRE(p != nullptr);
    BOOST_TEST(dynamic_cast<OtherProbe *>(p.get()) == nullptr);
}

BOOST_AUTO_TEST_CASE(each_instantiation_is_a_fresh_object)
{
    // The registry stores a factory, not an instance. Two runs in one process
    // (which is exactly what the Python Runner does) must not share state.
    Grid grid(0.0, 1.0, 4);
    auto a = PhysicsCases::InstantiateProblem("UnitTestAutoRegisteredProbe", empty_config,
                                              grid);
    auto b = PhysicsCases::InstantiateProblem("UnitTestAutoRegisteredProbe", empty_config,
                                              grid);
    BOOST_TEST_REQUIRE(a != nullptr);
    BOOST_TEST_REQUIRE(b != nullptr);
    BOOST_TEST(a.get() != b.get());
}

BOOST_AUTO_TEST_CASE(the_real_physics_cases_are_all_reachable_by_name)
{
    // A spot check that the link line actually pulled in the physics objects.
    // These are registered from translation units nothing references directly,
    // so they only appear if the object files are linked in -- exactly the
    // failure mode that makes a model vanish from the available list, with no
    // compile or link error to point at it.
    //
    // Every physics case is linked into this binary now, as into the MaNTA
    // executable: Tests/UnitTests/CMakeLists.txt links the whole manta_objects
    // target, where the Makefile named a hand-maintained subset. The four spot
    // checks below are therefore a sample rather than the whole of what is
    // present -- kept as a sample deliberately, because naming all eighteen would
    // turn this into a list that has to be edited whenever a case is added, which
    // is the kind of maintenance the registry exists to avoid.
    //
    // Checked through the map rather than by instantiating: construction would
    // need each case's own config section, and a throw there would say nothing
    // about registration.
    for (const char *name :
         {"MatrixDiffusion", "ScalarTestLD3", "LinearDiffSourceTest", "AdjointTestProblem"})
        BOOST_TEST(RegistryPeek::isRegistered(name), name << " is not registered");

    BOOST_TEST(RegistryPeek::isRegistered("NoSuchPhysicsCase") == false);
    BOOST_TEST(RegistryPeek::size() > 3u);
}

BOOST_AUTO_TEST_SUITE_END()
