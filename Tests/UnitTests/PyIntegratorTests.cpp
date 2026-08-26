// Tests for the memoised quadrature data in PyIntegrator.hpp.
//
// Integrator caches into process-wide mutable globals. They were only ever
// populated once -- `if (!m_basis)`, `if (size() == 0)` -- and never
// invalidated, so a second solve in the same process with a different grid or
// polynomial degree silently reused the first solve's weights. PyRunner
// explicitly supports repeated configure/run cycles for optimisation loops, so
// this is reachable from ordinary use.
//
// The header carries no pybind11 dependency, so it can be exercised directly
// here rather than through the Python layer (where it is only reached via the
// nScalars > 0 code paths).

#include <boost/test/unit_test.hpp>

#include "PyGrid.hpp"
#include "PyIntegrator.hpp"

#include <cmath>

BOOST_AUTO_TEST_SUITE(py_integrator_tests, *boost::unit_test::tolerance(1e-10))

namespace
{
// Integrate f over the whole grid using the cached global weights. getNodes()
// returns nodes on the reference interval, so map each into its cell.
double integrate(const Vector &weights, const Grid &grid, unsigned int k,
                 double (*f)(double))
{
    const BasisType basis = BasisType::getBasis(k);
    const Vector refNodes = basis.getNodes();
    double total = 0.0;
    Index n = 0;
    for (Index c = 0; c < static_cast<Index>(grid.getNCells()); ++c)
    {
        Interval const &cell = grid[c];
        for (Index i = 0; i < refNodes.size(); ++i, ++n)
            total += weights(n) * f(cell.fromRef(refNodes(i)));
    }
    return total;
}

double one(double) { return 1.0; }
double linear(double x) { return x; }
double quadratic(double x) { return x * x; }
} // namespace

BOOST_AUTO_TEST_CASE(global_weights_integrate_exactly)
{
    Integrator::Cache cache;
    const unsigned int k = 4;
    Grid grid(0.0, 1.0, 6);
    const BasisType basis = BasisType::getBasis(k);

    const Vector &w = cache.integrationWeights(basis, grid);
    BOOST_TEST(w.size() == static_cast<Index>(grid.getNCells() * (k + 1)));

    // Nodal quadrature of order k is exact for polynomials of degree <= k.
    BOOST_TEST(integrate(w, grid, k, one) == 1.0);
    BOOST_TEST(integrate(w, grid, k, linear) == 0.5);
    BOOST_TEST(integrate(w, grid, k, quadratic) == 1.0 / 3.0);
}

BOOST_AUTO_TEST_CASE(weights_are_rebuilt_when_the_grid_changes)
{
    Integrator::Cache cache;
    const unsigned int k = 4;
    const BasisType basis = BasisType::getBasis(k);

    // Prime the cache with one grid...
    Grid first(0.0, 1.0, 6);
    const Vector w1 = cache.integrationWeights(basis, first);
    BOOST_TEST(w1.size() == static_cast<Index>(6 * (k + 1)));

    // ...then ask for a different one. Regression: the cached vector was
    // returned unchanged, so this came back with the wrong length and the
    // wrong values.
    Grid second(0.0, 1.0, 11);
    const Vector &w2 = cache.integrationWeights(basis, second);
    BOOST_TEST(w2.size() == static_cast<Index>(11 * (k + 1)));
    BOOST_TEST(integrate(w2, second, k, one) == 1.0);
    BOOST_TEST(integrate(w2, second, k, quadratic) == 1.0 / 3.0);

    // A different domain, same cell count -- length alone would not catch this.
    Grid third(0.0, 2.0, 11);
    const Vector &w3 = cache.integrationWeights(basis, third);
    BOOST_TEST(integrate(w3, third, k, one) == 2.0);

    // Going back to the first grid must still give the original answer.
    const Vector &w1again = cache.integrationWeights(basis, first);
    BOOST_TEST(integrate(w1again, first, k, one) == 1.0);
}

BOOST_AUTO_TEST_CASE(weights_are_rebuilt_when_the_order_changes)
{
    Integrator::Cache cache;
    Grid grid(0.0, 1.0, 5);

    const BasisType b3 = BasisType::getBasis(3);
    const Vector w3 = cache.integrationWeights(b3, grid);
    BOOST_TEST(w3.size() == static_cast<Index>(5 * 4));

    // Regression: the per-interval cache was keyed on Interval alone, so the
    // same cells at a different polynomial order returned order-3 weights.
    const BasisType b6 = BasisType::getBasis(6);
    const Vector &w6 = cache.integrationWeights(b6, grid);
    BOOST_TEST(w6.size() == static_cast<Index>(5 * 7));
    BOOST_TEST(integrate(w6, grid, 6, one) == 1.0);
    BOOST_TEST(integrate(w6, grid, 6, quadratic) == 1.0 / 3.0);
}

BOOST_AUTO_TEST_CASE(phi_boundary_is_rebuilt_when_the_grid_changes)
{
    Integrator::Cache cache;
    const unsigned int k = 3;
    const BasisType basis = BasisType::getBasis(k);

    Grid first(0.0, 1.0, 4);
    const Matrix p1 = cache.phiBoundary(basis, first);
    BOOST_TEST(p1.rows() == static_cast<Index>(k + 1));
    BOOST_TEST(p1.cols() == 2);

    // Basis functions at the domain endpoints must sum to one (partition of
    // unity), whichever grid is in play.
    BOOST_TEST(p1.col(0).sum() == 1.0);
    BOOST_TEST(p1.col(1).sum() == 1.0);

    Grid second(-3.0, 5.0, 9);
    const Matrix &p2 = cache.phiBoundary(basis, second);
    BOOST_TEST(p2.col(0).sum() == 1.0);
    BOOST_TEST(p2.col(1).sum() == 1.0);

    // The endpoint values genuinely correspond to the new grid's end cells.
    const Interval &lo = second[0];
    const Interval &hi = second[second.getNCells() - 1];
    for (Index i = 0; i < static_cast<Index>(k + 1); ++i)
    {
        BOOST_TEST(p2(i, 0) == basis.Evaluate(lo, i, lo.x_l));
        BOOST_TEST(p2(i, 1) == basis.Evaluate(hi, i, hi.x_u));
    }
}

BOOST_AUTO_TEST_CASE(two_caches_do_not_share_state)
{
    // The reason this is a class rather than a namespace of inline variables.
    // With one process-wide cache, two owners at different orders evicted each
    // other's weights on every call -- silently, since the answers stayed right
    // and only the work was repeated. Worse, it was unsynchronised mutable state
    // reachable from a module that declares py::mod_gil_not_used().
    //
    // Interleaved deliberately: alternating between the two is what a solver and
    // a physics case at different degrees actually did.
    Grid grid(0.0, 1.0, 5);
    const BasisType b3 = BasisType::getBasis(3);
    const BasisType b6 = BasisType::getBasis(6);

    Integrator::Cache a, b;
    const Vector wa1 = a.integrationWeights(b3, grid);
    const Vector wb1 = b.integrationWeights(b6, grid);
    const Vector wa2 = a.integrationWeights(b3, grid);
    const Vector wb2 = b.integrationWeights(b6, grid);

    BOOST_TEST(wa1.size() == static_cast<Index>(5 * 4));
    BOOST_TEST(wb1.size() == static_cast<Index>(5 * 7));
    BOOST_TEST((wa1 - wa2).cwiseAbs().maxCoeff() == 0.0);
    BOOST_TEST((wb1 - wb2).cwiseAbs().maxCoeff() == 0.0);

    // And each still integrates correctly on its own order.
    BOOST_TEST(integrate(wa2, grid, 3, one) == 1.0);
    BOOST_TEST(integrate(wb2, grid, 6, one) == 1.0);
}

BOOST_AUTO_TEST_SUITE_END()
