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
    RegisteredProbe(toml::value const &config, Grid const &g) : lower(g.lowerBoundary())
    {
        nVars = 1;
        // Read something from the config so a test can prove the value really
        // reached the constructor rather than being defaulted.
        if (config.count("Registered") == 1)
            marker = toml::find_or(config.at("Registered"), "Value", 0);
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return s.Derivative[0];
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

// A second, distinguishable type for the duplicate-name test.
class OtherProbe : public RegisteredProbe
{
public:
    using RegisteredProbe::RegisteredProbe;
    std::string getVariableName(Index) override { return "other"; }
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

BOOST_AUTO_TEST_CASE(an_unknown_name_returns_null_rather_than_throwing)
{
    // MaNTA.cpp relies on the null return to print the list of available
    // models. It used to dereference the result first (fixed in Stage 3), so
    // this contract is load-bearing.
    Grid grid(0.0, 1.0, 4);
    BOOST_TEST(PhysicsCases::InstantiateProblem("NoSuchPhysicsCase", empty_config, grid) ==
               nullptr);
}

BOOST_AUTO_TEST_CASE(a_duplicate_name_does_not_displace_the_first_registration)
{
    // RegisterPhysicsCase uses map::insert, which is a no-op when the key
    // exists -- so the *first* registration wins and a later one is silently
    // ignored. That is worth pinning either way: someone adding a physics case
    // whose name collides with an existing one would otherwise be baffled to
    // find their class never instantiated.
    PhysicsCases::RegisterPhysicsCase("UnitTestDuplicateProbe",
                                      &createTransportSystem<RegisteredProbe>);
    PhysicsCases::RegisterPhysicsCase("UnitTestDuplicateProbe",
                                      &createTransportSystem<OtherProbe>);

    Grid grid(0.0, 1.0, 4);
    auto p = PhysicsCases::InstantiateProblem("UnitTestDuplicateProbe", empty_config, grid);
    BOOST_TEST_REQUIRE(p != nullptr);
    BOOST_TEST(p->getVariableName(0) == "Var0");
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
    // Only the cases in REQUIRED_OBJECTS (Tests/UnitTests/Makefile) are linked
    // into this binary, which is a deliberately small subset -- LinearDiffusion
    // and the rest are absent here but present in the MaNTA executable. So this
    // asserts the mechanism works for the ones that *are* linked, not that
    // every physics case in the repo exists.
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
