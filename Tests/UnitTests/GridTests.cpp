// Tests for Interval and Grid (gridStructures.hpp).
//
// DGTests.cpp::grid_test already covers the happy path of the uniform
// constructor. This file covers the parts that were untested: the reference
// interval maps, the Chebyshev-clustered boundary-layer constructor, the
// explicit-points constructor, and every validation path.

#include <boost/test/unit_test.hpp>

#include "gridStructures.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(grid_tests, *boost::unit_test::tolerance(1e-12))

// ---------------------------------------------------------------- Interval --

BOOST_AUTO_TEST_CASE(interval_construction_orders_endpoints)
{
    // The constructor is documented to tolerate reversed endpoints.
    Interval forward(0.25, 0.75);
    Interval reversed(0.75, 0.25);

    BOOST_TEST(forward.x_l == 0.25);
    BOOST_TEST(forward.x_u == 0.75);
    BOOST_TEST(reversed.x_l == 0.25);
    BOOST_TEST(reversed.x_u == 0.75);

    bool same = (forward == reversed);
    BOOST_TEST(same);

    Interval copy(forward);
    bool copied = (copy == forward);
    BOOST_TEST(copied);
}

BOOST_AUTO_TEST_CASE(interval_h_and_contains)
{
    Interval I(-1.5, 2.5);
    BOOST_TEST(I.h() == 4.0);

    BOOST_TEST(I.contains(0.0));
    BOOST_TEST(I.contains(-1.5)); // closed at both ends
    BOOST_TEST(I.contains(2.5));
    BOOST_TEST(!I.contains(-1.5 - 1e-9));
    BOOST_TEST(!I.contains(2.5 + 1e-9));
}

BOOST_AUTO_TEST_CASE(interval_reference_maps_are_inverse)
{
    Interval I(-3.0, 7.0);

    // toRef maps [x_l, x_u] -> [-1, 1]
    BOOST_TEST(I.toRef(I.x_l) == -1.0);
    BOOST_TEST(I.toRef(I.x_u) == 1.0);
    BOOST_TEST(I.toRef(0.5 * (I.x_l + I.x_u)) == 0.0);

    // fromRef is its inverse
    BOOST_TEST(I.fromRef(-1.0) == I.x_l);
    BOOST_TEST(I.fromRef(1.0) == I.x_u);
    BOOST_TEST(I.fromRef(0.0) == 0.5 * (I.x_l + I.x_u));

    // Round-trip both directions at a scatter of points
    for (int i = 0; i <= 10; ++i)
    {
        double s = -1.0 + 0.2 * i;
        BOOST_TEST(I.toRef(I.fromRef(s)) == s);

        double x = I.x_l + 0.1 * i * I.h();
        BOOST_TEST(I.fromRef(I.toRef(x)) == x);
    }
}

BOOST_AUTO_TEST_CASE(interval_ordering_operators)
{
    Interval a(0.0, 1.0);
    Interval b(2.0, 3.0);

    // operator< compares lower bounds, operator> compares upper bounds
    bool lt = (a < b);
    bool gt = (b > a);
    BOOST_TEST(lt);
    BOOST_TEST(gt);
    BOOST_TEST(!(b < a));
    BOOST_TEST(!(a > b));
}

// -------------------------------------------------- Grid, uniform spacing --

BOOST_AUTO_TEST_CASE(grid_uniform_partitions_domain_exactly)
{
    for (Grid::Index n : {1u, 2u, 5u, 17u})
    {
        Grid g(-2.0, 3.0, n);
        BOOST_TEST(g.getNCells() == n);
        BOOST_TEST(g.lowerBoundary() == -2.0);
        BOOST_TEST(g.upperBoundary() == 3.0);

        // Cells tile the domain with no gaps or overlaps, and the last cell
        // ends exactly on the upper bound (it is set explicitly, not by
        // accumulating cellLength, to avoid drift).
        BOOST_TEST(g[0].x_l == -2.0);
        BOOST_TEST(g[n - 1].x_u == 3.0);

        double total = 0.0;
        for (Grid::Index i = 0; i < n; ++i)
        {
            BOOST_TEST(g[i].x_u > g[i].x_l);
            if (i > 0)
                BOOST_TEST(g[i].x_l == g[i - 1].x_u);
            total += g[i].h();
        }
        BOOST_TEST(total == 5.0);
    }
}

BOOST_AUTO_TEST_CASE(grid_swaps_reversed_bounds)
{
    Grid forward(0.0, 1.0, 5);
    Grid reversed(1.0, 0.0, 5);

    bool same = (forward == reversed);
    BOOST_TEST(same);
    BOOST_TEST(reversed.lowerBoundary() == 0.0);
    BOOST_TEST(reversed.upperBoundary() == 1.0);
}

BOOST_AUTO_TEST_CASE(grid_rejects_degenerate_construction)
{
    // Zero cells
    BOOST_CHECK_THROW(Grid(0.0, 1.0, 0), std::invalid_argument);

    // Bounds too close to resolve in double precision
    BOOST_CHECK_THROW(Grid(1.0, 1.0, 4), std::invalid_argument);
    BOOST_CHECK_THROW(Grid(0.0, 1e-15, 4), std::invalid_argument);

    // Just above the 1e-14 threshold is accepted
    BOOST_CHECK_NO_THROW(Grid(0.0, 1e-13, 4));
}

// --------------------------------------- Grid, clustered boundary layers --

BOOST_AUTO_TEST_CASE(grid_high_boundary_partitions_domain)
{
    // BoundaryCells = nCells/3 at each end, the rest in the bulk. Use multiples
    // of 3 so the split is unambiguous.
    for (Grid::Index n : {6u, 9u, 12u, 30u})
    {
        Grid g(0.0, 1.0, n, true, 0.2, 0.2);

        BOOST_TEST(g.getNCells() == n);
        BOOST_TEST(g[0].x_l == 0.0);
        BOOST_TEST(g[n - 1].x_u == 1.0);

        // Still a partition: contiguous, strictly increasing, summing to the
        // full domain width.
        //
        // The contiguity check is deliberately EXACT (== on the raw doubles,
        // outside BOOST_TEST's tolerance). Under the suite tolerance this
        // passed while the two sides of a shared face differed in the last
        // bits, which was enough to make Grid::operator== false and break the
        // restart round trip.
        double total = 0.0;
        for (Grid::Index i = 0; i < n; ++i)
        {
            BOOST_TEST(g[i].x_u > g[i].x_l);
            if (i > 0)
            {
                bool contiguous = (g[i].x_l == g[i - 1].x_u);
                BOOST_TEST(contiguous, "face " << i << " is not bitwise shared: "
                                               << g[i - 1].x_u << " vs " << g[i].x_l);
            }
            total += g[i].h();
        }
        BOOST_TEST(total == 1.0);
    }
}

BOOST_AUTO_TEST_CASE(grid_high_boundary_respects_layer_fractions)
{
    const Grid::Index n = 12;
    const double lowerFrac = 0.2, upperFrac = 0.3;
    Grid g(0.0, 1.0, n, true, lowerFrac, upperFrac);

    const Grid::Index boundaryCells = n / 3;

    // The lower boundary layer is exactly the first boundaryCells cells and
    // spans lowerFrac of the domain.
    BOOST_TEST(g[boundaryCells - 1].x_u == lowerFrac);
    // The upper boundary layer is the last boundaryCells cells.
    BOOST_TEST(g[n - boundaryCells].x_l == 1.0 - upperFrac);
}

BOOST_AUTO_TEST_CASE(grid_high_boundary_clusters_cells_towards_walls)
{
    // The point of the mode: cells must get smaller approaching each wall.
    const Grid::Index n = 30;
    Grid g(0.0, 1.0, n, true, 0.2, 0.2);
    const Grid::Index boundaryCells = n / 3;

    // Within the lower layer, cells shrink towards x = 0, so the first cell is
    // the smallest in that layer.
    for (Grid::Index i = 1; i < boundaryCells; ++i)
        BOOST_TEST(g[0].h() <= g[i].h());

    // Symmetrically at the upper wall.
    for (Grid::Index i = 1; i < boundaryCells; ++i)
        BOOST_TEST(g[n - 1].h() <= g[n - 1 - i].h());

    // And the boundary layers really are finer than the bulk.
    BOOST_TEST(g[0].h() < g[n / 2].h());
    BOOST_TEST(g[n - 1].h() < g[n / 2].h());
}

// ------------------------------------------- Grid from explicit points --

BOOST_AUTO_TEST_CASE(grid_from_points_reproduces_cells)
{
    std::vector<Grid::Position> points{0.0, 0.1, 0.35, 0.7, 1.0};
    Grid g(points);

    BOOST_TEST(g.getNCells() == points.size() - 1);
    BOOST_TEST(g.lowerBoundary() == 0.0);
    BOOST_TEST(g.upperBoundary() == 1.0);

    for (Grid::Index i = 0; i < g.getNCells(); ++i)
    {
        BOOST_TEST(g[i].x_l == points[i]);
        BOOST_TEST(g[i].x_u == points[i + 1]);
    }
}

BOOST_AUTO_TEST_CASE(grid_from_points_round_trips_a_uniform_grid)
{
    // This is the restart contract: NetCDFIO stores CellBoundaries and the grid
    // is rebuilt from them, so the rebuilt grid must compare equal.
    Grid original(-1.0, 1.0, 8);

    std::vector<Grid::Position> points;
    points.push_back(original[0].x_l);
    for (Grid::Index i = 0; i < original.getNCells(); ++i)
        points.push_back(original[i].x_u);

    Grid rebuilt(points);
    bool equal = (rebuilt == original);
    BOOST_TEST(equal);
}

BOOST_AUTO_TEST_CASE(grid_from_points_rejects_too_few_points)
{
    // Regression: `points.size() - 1` on a size_t underflows for an empty
    // vector, producing a nonsensical cell count, and points.front() on an
    // empty vector is undefined behaviour. A single point cannot define a cell
    // either.
    BOOST_CHECK_THROW(Grid(std::vector<Grid::Position>{}), std::invalid_argument);
    BOOST_CHECK_THROW(Grid(std::vector<Grid::Position>{0.5}), std::invalid_argument);

    // Two points is the smallest valid grid.
    BOOST_CHECK_NO_THROW(Grid(std::vector<Grid::Position>{0.0, 1.0}));
}

BOOST_AUTO_TEST_CASE(grid_from_points_rejects_a_list_that_is_not_strictly_increasing)
{
    // Interval(a, b) swaps when a > b, so none of these were reported before:
    // out-of-order points gave overlapping cells, and a repeated point gave a
    // zero-width cell whose mass matrix is (h/2) * RefMass = 0.
    using P = std::vector<Grid::Position>;

    BOOST_CHECK_THROW(Grid(P{0.0, 0.5, 0.2, 1.0}), std::invalid_argument); // out of order
    BOOST_CHECK_THROW(Grid(P{1.0, 0.5, 0.0}), std::invalid_argument);      // descending
    BOOST_CHECK_THROW(Grid(P{0.0, 0.5, 0.5, 1.0}), std::invalid_argument); // repeated
    BOOST_CHECK_THROW(Grid(P{0.0, 0.0}), std::invalid_argument);           // degenerate domain

    // A graded mesh -- the thing this constructor exists for -- is fine.
    BOOST_CHECK_NO_THROW(Grid(P{0.0, 0.01, 0.05, 0.2, 0.6, 1.0}));
}

BOOST_AUTO_TEST_CASE(grid_from_points_rejects_values_that_are_not_finite)
{
    // Checked before monotonicity, and deliberately: a NaN compares false
    // against everything, so an ordering test alone would blame the ordering.
    using P = std::vector<Grid::Position>;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    BOOST_CHECK_THROW(Grid(P{0.0, nan, 1.0}), std::invalid_argument);
    BOOST_CHECK_THROW(Grid(P{0.0, 0.5, inf}), std::invalid_argument);
    BOOST_CHECK_THROW(Grid(P{-inf, 0.0, 1.0}), std::invalid_argument);

    // The message should name the offending boundary rather than the ordering.
    try
    {
        Grid(P{0.0, nan, 1.0});
        BOOST_FAIL("expected an exception");
    }
    catch (std::invalid_argument const &e)
    {
        BOOST_TEST(std::string(e.what()).find("finite") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(grid_from_points_holds_a_narrow_domain_to_the_same_rule_as_the_other_constructor)
{
    // Grid(0.0, 1e-15, 4) throws and Grid(0.0, 1e-13, 4) does not; a config
    // supplying Grid_points instead of Grid_size should not be able to build
    // what Grid_size would reject.
    using P = std::vector<Grid::Position>;

    BOOST_CHECK_THROW(Grid(0.0, 1e-15, 4), std::invalid_argument);
    BOOST_CHECK_THROW(Grid(P{0.0, 0.5e-15, 1e-15}), std::invalid_argument);

    BOOST_CHECK_NO_THROW(Grid(0.0, 1e-13, 4));
    BOOST_CHECK_NO_THROW(Grid(P{0.0, 0.5e-13, 1e-13}));
}

// --------------------------------------------------------- Grid equality --

BOOST_AUTO_TEST_CASE(grid_equality_compares_bounds_and_cells)
{
    Grid a(0.0, 1.0, 5);
    Grid b(0.0, 1.0, 5);
    Grid c(0.0, 1.0, 6);
    Grid d(0.0, 2.0, 5);

    // Wrapped in bools: Boost.Test wants to stream operands on failure and
    // Grid has no operator<<.
    bool ab = (a == b);
    bool ac = (a != c);
    bool ad = (a != d);
    BOOST_TEST(ab);
    BOOST_TEST(ac);
    BOOST_TEST(ad);

    Grid copy(a);
    bool copied = (copy == a);
    BOOST_TEST(copied);
}

BOOST_AUTO_TEST_SUITE_END()
