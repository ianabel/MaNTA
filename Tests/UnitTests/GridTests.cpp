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

// ------------------------------------------- geometric mesh grading ----
//
// gradedMeshPoints is a pure function of six numbers, which is why it returns
// points rather than a Grid: the geometry can be pinned without constructing
// anything. MESH-REFINEMENT.md §8 measures the error on Shestakov's problem as
// 0.0487 * h0 in the width of the cell touching the graded end and in nothing
// else -- not in the cell count -- so that width is what these check hardest.

BOOST_AUTO_TEST_CASE(a_graded_mesh_puts_the_layer_cells_in_a_geometric_progression)
{
    // 9 cells over the lower 10% of [0, 1] at ratio 0.3, then one uniform cell.
    // The mesh MESH-REFINEMENT.md §9 measured at 14900x a uniform 10.
    auto p = gradedMeshPoints(0.0, 1.0, 10, 9, 0.1, 0.3, false);

    BOOST_TEST(p.size() == 11u);
    BOOST_TEST(p.front() == 0.0);
    BOOST_TEST(p.back() == 1.0);

    // Strictly increasing, which is also what Grid would demand of it.
    for (size_t i = 0; i + 1 < p.size(); ++i)
        BOOST_TEST(p[i] < p[i + 1]);

    // The far edge of the graded layer is exactly at the fraction, so the layer
    // is the width asked for rather than approximately it.
    BOOST_TEST(p[9] == 0.1);

    std::vector<double> w;
    for (size_t i = 0; i + 1 < p.size(); ++i)
        w.push_back(p[i + 1] - p[i]);

    // Not a pure geometric progression, and this is the part that is easy to get
    // wrong. The cell touching the end runs all the way to it, so it is
    // 1/(1-ratio) wider than continuing the progression would give: the first
    // width ratio is (1-r)/r = 2.3333 and every later one inside the layer is
    // 1/r = 3.3333. Getting this wrong would still produce a plausible graded
    // mesh, with a different h0 and so a different answer.
    BOOST_TEST(w[1] / w[0] == (1.0 - 0.3) / 0.3, boost::test_tools::tolerance(1e-12));
    for (size_t i = 2; i < 9; ++i)
        BOOST_TEST(w[i] / w[i - 1] == 1.0 / 0.3, boost::test_tools::tolerance(1e-12));

    // ...and therefore h0 is the closed form, which is the knob the measurements
    // are indexed on: fraction * span * ratio^(gradedCells - 1).
    BOOST_TEST(w[0] == 0.1 * std::pow(0.3, 8.0), boost::test_tools::tolerance(1e-12));

    BOOST_TEST_MESSAGE("h0 = " << w[0] << ", widest/narrowest = "
                       << *std::max_element(w.begin(), w.end()) / w[0]);
}

BOOST_AUTO_TEST_CASE(a_graded_mesh_is_uniform_outside_the_layer)
{
    // 4 graded + 6 uniform. The uniform part must be uniform: the bulk length is
    // measured from the layer's far edge rather than from the domain, and using
    // the domain would leave the first bulk cell the wrong width.
    auto p = gradedMeshPoints(0.0, 1.0, 10, 4, 0.2, 0.5, false);

    BOOST_TEST(p.size() == 11u);
    BOOST_TEST(p[4] == 0.2);

    const double bulk = (1.0 - 0.2) / 6.0;
    for (size_t i = 4; i + 1 < p.size(); ++i)
        BOOST_TEST(p[i + 1] - p[i] == bulk, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(grading_the_upper_end_is_the_exact_mirror_of_the_lower)
{
    // Built by reflection so the two ends cannot drift apart, and the endpoints
    // are pinned afterwards because Grid::operator== compares them exactly and
    // the restart round trip rebuilds a grid from the boundaries it wrote.
    auto lo = gradedMeshPoints(0.0, 1.0, 10, 9, 0.1, 0.3, false);
    auto hi = gradedMeshPoints(0.0, 1.0, 10, 9, 0.1, 0.3, true);

    BOOST_TEST(hi.size() == lo.size());
    BOOST_TEST(hi.front() == 0.0);
    BOOST_TEST(hi.back() == 1.0);

    for (size_t i = 0; i < lo.size(); ++i)
        BOOST_TEST(hi[i] == 1.0 - lo[lo.size() - 1 - i],
                   boost::test_tools::tolerance(1e-15));

    // The narrow cell is against the *upper* end now. Stated separately because a
    // reflection that forgot to reverse would satisfy nothing above but might
    // look right in a plot.
    //
    // Held to 1e-10 rather than the 1e-12 the lower end gets, and the gap is a
    // real property of the arithmetic rather than slack. The boundary next to the
    // upper end is a number near uBound, so the *width* of the cell beyond it is a
    // difference of two nearly equal numbers and carries an absolute error of
    // eps(uBound) whatever it is computed from -- a relative error of
    // eps(uBound)/h0, which is 3.4e-11 here against 6.0e-12 measured. No
    // construction avoids it: uBound - layer*ratio^j has the identical
    // cancellation, so this is not the reflection's doing. It matters at hard
    // gradings, where the narrowest cell eventually cannot be represented at all
    // -- at h0/span near eps the boundary coincides with uBound and Grid rejects
    // the zero-width cell, which is the right failure and a loud one.
    const double narrowUpper = hi[hi.size() - 1] - hi[hi.size() - 2];
    const double h0 = 0.1 * std::pow(0.3, 8.0);
    BOOST_TEST(narrowUpper == h0, boost::test_tools::tolerance(1e-10));
    BOOST_TEST_MESSAGE("upper-end h0 is out by " << std::abs(narrowUpper - h0) / h0
                       << " relative, against a floor of "
                       << std::numeric_limits<double>::epsilon() / h0);
    BOOST_TEST(hi[1] - hi[0] > 100.0 * narrowUpper);
}

BOOST_AUTO_TEST_CASE(a_graded_mesh_is_offset_and_scaled_with_the_domain)
{
    // Nothing in the construction may assume [0, 1]. Same shape on [-3, 5]:
    // widths scale with the span and the layer sits against the lower end.
    auto unit = gradedMeshPoints(0.0, 1.0, 8, 5, 0.25, 0.4, false);
    auto wide = gradedMeshPoints(-3.0, 5.0, 8, 5, 0.25, 0.4, false);

    BOOST_TEST(wide.front() == -3.0);
    BOOST_TEST(wide.back() == 5.0);
    for (size_t i = 0; i < unit.size(); ++i)
        BOOST_TEST(wide[i] == -3.0 + 8.0 * unit[i], boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(a_graded_mesh_builds_a_grid_of_the_size_asked_for)
{
    // The end to end check, and the reason the function returns points: Grid is
    // where the validation lives, so a construction that produced an out-of-order
    // or zero-width list would be caught there rather than here.
    for (Grid::Index n : {3u, 5u, 10u, 25u})
        for (Grid::Index g : {2u, 3u})
        {
            if (g >= n)
                continue;
            Grid grid(gradedMeshPoints(0.0, 1.0, n, g, 0.1, 0.5, false));
            BOOST_TEST(grid.getNCells() == n);
            BOOST_TEST(grid.lowerBoundary() == 0.0);
            BOOST_TEST(grid.upperBoundary() == 1.0);
        }
}

BOOST_AUTO_TEST_CASE(a_graded_mesh_refuses_geometry_that_would_not_grade)
{
    // ratio outside (0, 1). At 1 every boundary in the layer lands on the same
    // point -- a zero-width cell, whose MassMatrix is identically zero -- and
    // above it the cells grow towards the end being refined, which is the
    // opposite of the request.
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 0.1, 1.0, false), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 0.1, 1.5, false), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 0.1, 0.0, false), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 0.1, -0.5, false), std::invalid_argument);

    // fraction outside (0, 1): a zero-width layer, or one that swallows the domain
    // and leaves the uniform cells nothing.
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 0.0, 0.3, false), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 1.0, 0.3, false), std::invalid_argument);

    // Fewer than two cells in the layer: there is no ratio between neighbours.
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 1, 0.1, 0.3, false), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 0, 0.1, 0.3, false), std::invalid_argument);

    // ...and no cells left outside it, which would leave [layer, upper] uncovered.
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 5, 5, 0.1, 0.3, false), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 5, 6, 0.1, 0.3, false), std::invalid_argument);

    // Both of the above also hold for the reflected path, which recurses through
    // the same checks rather than repeating them.
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 10, 5, 0.1, 1.0, true), std::invalid_argument);
    BOOST_CHECK_THROW(gradedMeshPoints(0.0, 1.0, 5, 5, 0.1, 0.3, true), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
