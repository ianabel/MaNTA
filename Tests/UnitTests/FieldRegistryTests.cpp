// The field-model registry mirrors PhysicsCases::map, including its two
// deliberate behaviours: a duplicate name throws rather than the first
// registration quietly winning, and an unknown name throws with the list of
// what *is* registered rather than returning nullptr for callers to check.
//
// The map is never reset, so every test here uses a throwaway name.
#include <boost/test/unit_test.hpp>

#include "../../FieldModel.hpp"
#include "../../gridStructures.hpp"

#include <toml.hpp>

namespace
{
    class RegistryProbeField : public FieldModel
    {
    public:
        RegistryProbeField(toml::value const &, Grid const &) : FieldModel(makeSpec()) {}

        static FieldModelSpec makeSpec()
        {
            FieldModelSpec s;
            s.dofs = {{"p", "probe", "1", false}};
            s.geometry = {{"g", "probe", "1"}};
            s.label = "x";
            return s;
        }

        void FieldResidual(VectorRef, Vector const &, Vector const &, GlobalState const &,
                           std::vector<Position> const &, Vector const &, Time) override {}
        void Geometry(VectorRef, Vector const &, Position, Time) override {}
        void dGeometry_dpsi(MatrixRef, Vector const &, Position, Time) override {}
        void FieldResidualPrime(GlobalStateMatrix &, GlobalStateMatrix &, MatrixRef, MatrixRef,
                                Vector const &, Vector const &, GlobalState const &,
                                std::vector<Position> const &, Vector const &, Time) override {}
        void InitialFieldValue(VectorRef) override {}
    };
}

BOOST_AUTO_TEST_SUITE(field_registry_tests)

BOOST_AUTO_TEST_CASE(a_registered_model_can_be_instantiated_by_name)
{
    FieldModels::RegisterFieldModel("RegistryProbeFieldA", createFieldModel<RegistryProbeField>);

    Grid grid(0.0, 1.0, 4);
    toml::value config;
    auto model = FieldModels::InstantiateFieldModel("RegistryProbeFieldA", config, grid);

    BOOST_REQUIRE(model != nullptr);
    BOOST_CHECK_EQUAL(model->nFieldDOF(), 1);
    BOOST_CHECK_EQUAL(model->nGeometry(), 1);
}

BOOST_AUTO_TEST_CASE(a_duplicate_name_throws)
{
    FieldModels::RegisterFieldModel("RegistryProbeFieldB", createFieldModel<RegistryProbeField>);
    // std::invalid_argument, as PhysicsCases::RegisterPhysicsCase throws: the
    // exception type is part of what "mirrors PhysicsCases::map" means, and
    // MaNTA.cpp catches both registries the same way.
    BOOST_CHECK_THROW(
        FieldModels::RegisterFieldModel("RegistryProbeFieldB", createFieldModel<RegistryProbeField>),
        std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(an_unknown_name_throws_and_names_what_is_registered)
{
    FieldModels::RegisterFieldModel("RegistryProbeFieldC", createFieldModel<RegistryProbeField>);

    Grid grid(0.0, 1.0, 4);
    toml::value config;
    try
    {
        FieldModels::InstantiateFieldModel("NoSuchFieldModel", config, grid);
        BOOST_FAIL("expected InstantiateFieldModel to throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string const msg = e.what();
        BOOST_CHECK(msg.find("NoSuchFieldModel") != std::string::npos);
        BOOST_CHECK(msg.find("RegistryProbeFieldC") != std::string::npos);
    }
}

BOOST_AUTO_TEST_SUITE_END()
