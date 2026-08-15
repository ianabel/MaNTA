// A FieldModelSpec is validated once, in the FieldModel constructor, so a half-built
// field model cannot exist -- the same contract SystemSpec has. These tests pin
// the refusals, because every one of them is a configuration error that would
// otherwise surface much later as an assembly shape mismatch or an IDA failure
// code with nothing pointing back here.
#include <boost/test/unit_test.hpp>

#include "../../FieldModelSpec.hpp"
#include "../../FieldModel.hpp"

BOOST_AUTO_TEST_SUITE(field_model_spec_tests)

static FieldModelSpec twoDofOneSlot()
{
    FieldModelSpec spec;
    spec.dofs = {{"psi0", "flux at the axis", "Wb", false},
                 {"psi1", "flux at the edge", "Wb", false}};
    spec.geometry = {{"Vprime", "flux surface volume derivative", "m^3"}};
    spec.label = "V";
    return spec;
}

BOOST_AUTO_TEST_CASE(a_well_formed_spec_validates)
{
    BOOST_CHECK_NO_THROW(twoDofOneSlot().validate());
    BOOST_CHECK_EQUAL(twoDofOneSlot().nFieldDOF(), 2);
    BOOST_CHECK_EQUAL(twoDofOneSlot().nGeometry(), 1);
}

BOOST_AUTO_TEST_CASE(a_spec_with_no_dofs_is_refused)
{
    FieldModelSpec spec = twoDofOneSlot();
    spec.dofs.clear();
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_spec_with_no_geometry_slots_is_refused)
{
    // A field model that exposes no geometry cannot affect the transport at
    // all: geometry is the only channel from psi into the physics.
    FieldModelSpec spec = twoDofOneSlot();
    spec.geometry.clear();
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(duplicate_names_are_refused)
{
    FieldModelSpec spec = twoDofOneSlot();
    spec.dofs[1].name = "psi0";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);

    FieldModelSpec spec2 = twoDofOneSlot();
    spec2.geometry.push_back({"Vprime", "again", "m^3"});
    BOOST_CHECK_THROW(spec2.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(an_empty_name_is_refused)
{
    // Names become netCDF group names in Task 12, where an empty one is a
    // failure a long way from here.
    FieldModelSpec spec = twoDofOneSlot();
    spec.dofs[0].name = "";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(an_empty_label_is_refused)
{
    FieldModelSpec spec = twoDofOneSlot();
    spec.label = "";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
