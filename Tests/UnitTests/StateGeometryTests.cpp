// Geometry is a *derived* quantity carried on State beside u, q, sigma and phi
// -- not an unknown. It reaches a physics case through s.geom(g), which is why
// SigmaFn(i, State, x, t) does not change shape and why no existing case,
// trampoline or stub had to move.
//
// The default of zero geometry slots is what keeps every existing State
// construction compiling and meaning what it did.
#include <boost/test/unit_test.hpp>

#include "../../State.hpp"

BOOST_AUTO_TEST_SUITE(state_geometry_tests)

BOOST_AUTO_TEST_CASE(geometry_defaults_to_empty)
{
    State s(3, 0, 0);
    BOOST_CHECK_EQUAL(s.geom().size(), 0);
}

BOOST_AUTO_TEST_CASE(geometry_is_born_zeroed_like_everything_else)
{
    State s(2, 1, 1, 3);
    BOOST_REQUIRE_EQUAL(s.geom().size(), 3);
    for (Index g = 0; g < 3; ++g)
        BOOST_CHECK_EQUAL(s.geom(g), 0.0);
}

BOOST_AUTO_TEST_CASE(geometry_round_trips_by_index_and_whole)
{
    State s(2, 0, 0, 2);
    s.geom(0) = 1.5;
    s.geom(1) = -2.5;
    BOOST_CHECK_EQUAL(s.geom(0), 1.5);
    BOOST_CHECK_EQUAL(s.geom(1), -2.5);
    BOOST_CHECK_EQUAL(s.geom().sum(), -1.0);
}

BOOST_AUTO_TEST_CASE(zero_clears_geometry_too)
{
    State s(1, 0, 0, 2);
    s.geom(0) = 7.0;
    s.zero();
    BOOST_CHECK_EQUAL(s.geom(0), 0.0);
}

BOOST_AUTO_TEST_CASE(clone_copies_the_geometry_width)
{
    State s(1, 0, 0, 4);
    State t;
    t.clone(s);
    BOOST_CHECK_EQUAL(t.geom().size(), 4);
}

BOOST_AUTO_TEST_CASE(a_global_state_carries_geometry_per_node)
{
    // GlobalState stores (nGeom, nNodes); the per-node accessor returns a
    // column. Orientation matters and is checked here rather than through a
    // Python round trip, because that caster transposes in both directions and
    // so cannot detect a missing transpose.
    const Index nCells = 3, k = 2, nGeom = 2;
    GlobalState gs(nCells, k, 1, 0, 0, nGeom);
    BOOST_REQUIRE_EQUAL(gs.Geometry(0).size(), nGeom);

    gs.setGeometry(4, (Vector(2) << 3.0, 4.0).finished());
    BOOST_CHECK_EQUAL(gs.Geometry(4)(0), 3.0);
    BOOST_CHECK_EQUAL(gs.Geometry(4)(1), 4.0);
    BOOST_CHECK_EQUAL(gs.Geometry(3)(0), 0.0);
}

BOOST_AUTO_TEST_CASE(a_state_extracted_from_a_global_state_carries_its_geometry)
{
    GlobalState gs(2, 1, 1, 0, 0, 2);
    gs.setGeometry(1, (Vector(2) << 5.0, 6.0).finished());
    State s = gs[1];
    BOOST_CHECK_EQUAL(s.geom(0), 5.0);
    BOOST_CHECK_EQUAL(s.geom(1), 6.0);
}

BOOST_AUTO_TEST_SUITE_END()
