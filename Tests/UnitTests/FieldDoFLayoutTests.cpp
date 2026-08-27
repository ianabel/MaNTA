// The field block is appended after the global scalars, so the DOF layout is
//
//     [ sigma | q | u | aux ] per cell,  then lambda,  then mu,  then psi
//
// Nothing existing shifts. That is the whole point of putting it last, and it
// is what these tests pin: every offset for the pre-existing blocks must be
// unchanged by a nonzero nField, because getting a column index wrong in this
// layout is the most common way to break the solver silently.
#include <boost/test/unit_test.hpp>

#include "../../DGSoln.hpp"
#include "../../gridStructures.hpp"

#include <vector>

BOOST_AUTO_TEST_SUITE(field_dof_layout_tests)

BOOST_AUTO_TEST_CASE(the_field_block_adds_exactly_its_own_width)
{
    Grid grid(0.0, 1.0, 5);
    const Index nVars = 2, k = 2, nScalars = 3, nAux = 1;

    DGSoln without(nVars, grid, k, nScalars, nAux);
    DGSoln with(nVars, grid, k, nScalars, nAux, 4);

    BOOST_CHECK_EQUAL(with.getDoF(), without.getDoF() + 4);
    BOOST_CHECK_EQUAL(with.getFieldDOF(), 4);
    BOOST_CHECK_EQUAL(without.getFieldDOF(), 0);
}

BOOST_AUTO_TEST_CASE(the_field_block_is_last_and_nothing_before_it_moves)
{
    Grid grid(0.0, 1.0, 3);
    const Index nVars = 1, k = 1, nScalars = 2, nAux = 0, nField = 3;

    DGSoln soln(nVars, grid, k, nScalars, nAux, nField);
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());

    // Write a recognisable value into each block and read it back out of the
    // raw memory at the offset the layout promises.
    soln.Scalar(0) = 11.0;
    soln.Scalar(1) = 12.0;
    soln.Field(0) = 21.0;
    soln.Field(2) = 23.0;

    const size_t scalarOffset = (3 * nVars + nAux) * (k + 1) * grid.getNCells() + nVars * (grid.getNCells() + 1);
    const size_t fieldOffset = scalarOffset + nScalars;

    BOOST_CHECK_EQUAL(mem[scalarOffset + 0], 11.0);
    BOOST_CHECK_EQUAL(mem[scalarOffset + 1], 12.0);
    BOOST_CHECK_EQUAL(mem[fieldOffset + 0], 21.0);
    BOOST_CHECK_EQUAL(mem[fieldOffset + 2], 23.0);
}

BOOST_AUTO_TEST_CASE(the_whole_field_vector_is_reachable)
{
    Grid grid(0.0, 1.0, 2);
    // Index(0) rather than a bare literal 0: an unadorned 0 in the fourth slot
    // is a null-pointer constant as far as overload resolution is concerned,
    // and is equally viable for the (double *memory, ...) constructor's
    // `memory` parameter, making the call ambiguous. PostprocessingTests.cpp,
    // ScalarJacobianTests.cpp and AdjointProblemTests.cpp already work around
    // the same trap the same way.
    DGSoln soln(1, grid, 1, Index(0), Index(0), 3);
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());

    soln.getField() = (Vector(3) << 1.0, 2.0, 3.0).finished();
    BOOST_CHECK_EQUAL(soln.Field(1), 2.0);
    BOOST_CHECK_EQUAL(soln.getField().sum(), 6.0);
}

BOOST_AUTO_TEST_CASE(zero_field_dofs_is_the_default_and_costs_nothing)
{
    Grid grid(0.0, 1.0, 4);
    DGSoln soln(2, grid, 2, 1, 1);
    BOOST_CHECK_EQUAL(soln.getFieldDOF(), 0);
    BOOST_CHECK_EQUAL(soln.getField().size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
