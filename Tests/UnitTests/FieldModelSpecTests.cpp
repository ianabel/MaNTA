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
    // Names become netCDF variable names, where an empty one is a failure a
    // long way from here.
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

BOOST_AUTO_TEST_CASE(the_output_group_defaults_to_Field_and_may_not_be_empty)
{
    BOOST_CHECK_EQUAL(twoDofOneSlot().name, "Field");

    FieldModelSpec spec = twoDofOneSlot();
    spec.name = "";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_dof_and_a_slot_may_not_share_a_name)
{
    // New with the netCDF group, and *only* reachable because of it: a DOF and
    // a geometry slot used to be written nowhere near each other, and now they
    // are variables in the same group. checkNames compares each list with
    // itself and cannot see this, so netCDF would be the first to notice --
    // NcNameInUse, thrown from ncGroup.cpp, naming neither MaNTA nor the spec.
    FieldModelSpec spec = twoDofOneSlot();
    spec.geometry[0].name = "psi1";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);

    // ...and the message says which name and why.
    try
    {
        spec.validate();
        BOOST_FAIL("a name shared between a DOF and a slot was accepted");
    }
    catch (std::invalid_argument const &e)
    {
        const std::string what = e.what();
        BOOST_TEST(what.find("psi1") != std::string::npos, what);
        BOOST_TEST(what.find("geometry slot") != std::string::npos, what);
    }
}

BOOST_AUTO_TEST_CASE(a_name_netcdf_would_reject_is_refused_here_instead)
{
    // Each of these dies inside netCDF otherwise, as an NcBadName naming
    // ncGroup.cpp and a line number. The point of catching them here is the
    // message: it says which string, and why.
    auto refused = [](FieldModelSpec const &spec, char const *expectedFragment)
    {
        try
        {
            spec.validate();
            BOOST_FAIL("a name netCDF cannot use was accepted");
        }
        catch (std::invalid_argument const &e)
        {
            const std::string what = e.what();
            BOOST_TEST(what.find("FieldModelSpec") != std::string::npos, what);
            BOOST_TEST(what.find(expectedFragment) != std::string::npos, what);
        }
    };

    FieldModelSpec slash = twoDofOneSlot();
    slash.name = "bad/name";
    refused(slash, "path separator");

    FieldModelSpec slashDof = twoDofOneSlot();
    slashDof.dofs[0].name = "psi/0";
    refused(slashDof, "path separator");

    FieldModelSpec control = twoDofOneSlot();
    control.geometry[0].name = "V\tprime";
    refused(control, "control character");

    FieldModelSpec spaced = twoDofOneSlot();
    spaced.name = "Equilibrium ";
    refused(spaced, "whitespace");

    FieldModelSpec leading = twoDofOneSlot();
    leading.name = "-equilibrium";
    refused(leading, "letter, a digit or an underscore");

    // And the shapes that are fine stay fine: a leading underscore, a digit, and
    // a multi-byte UTF-8 lead byte are all names netCDF accepts.
    for (char const *ok : {"_psi", "2nd_field", "\xcf\x88"})
    {
        FieldModelSpec spec = twoDofOneSlot();
        spec.name = ok;
        BOOST_CHECK_NO_THROW(spec.validate());
    }
}

BOOST_AUTO_TEST_SUITE_END()
