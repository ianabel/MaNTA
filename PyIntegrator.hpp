#ifndef PYINTEGRATOR_HPP
#define PYINTEGRATOR_HPP

#include "DGSoln.hpp"
#include "PyGrid.hpp"
#include "Types.hpp"

#include <map>
#include <optional>
#include <utility>

namespace Integrator {

/// The quadrature weights and boundary basis values a scalar hook is handed,
/// memoised for the (basis, grid) they were last asked about.
///
/// **An instance, not a namespace of globals, and that is the whole point of
/// this class existing.** These were `inline` variables at namespace scope --
/// process-wide mutable state, shared by every SystemSolver, every physics case
/// and every test in the process at once. Three things follow from that, and
/// only the first had been noticed:
///
///  * They were populated once and never invalidated (`if (!m_basis)`,
///    `if (size() == 0)`), so a second configure()/run() cycle at a different
///    grid or polynomial degree silently reused the first run's weights.
///    PyRunner exists to support exactly that loop. Fixed at the time by adding
///    the staleness check below -- which made the state correct but left it
///    global.
///  * A single cache shared by every owner thrashes whenever two of them differ.
///    A solver at k = 3 and a physics case at k = 4 evict each other's weights
///    on alternate calls, and nothing says so; it is a silent recomputation, not
///    a wrong answer, which is why it could sit there indefinitely.
///  * It is unsynchronised mutable state reachable from a Python module that
///    declares `py::mod_gil_not_used()`. On a free-threaded interpreter that
///    declaration says this module does not need the GIL to be safe, and a
///    process-wide map being mutated on first touch is not.
///
/// Per-owner instances answer all three: the lifetime is the owner's, two owners
/// cannot evict each other, and nothing is shared across threads that did not
/// already share the owner.
///
/// The staleness check stays, because one owner can legitimately be reused
/// across grids -- PyRunner::configure() builds a fresh SystemSolver each time,
/// but a physics case outlives that.
class Cache
{
  public:
    /// One weight per node of the whole grid: `Int u dx` is the dot product of
    /// this with the nodal values.
    const Vector &integrationWeights(const BasisType &basis, const Grid &grid)
    {
        invalidateIfStale(basis, grid);
        if (m_globalWeights.size() == 0)
        {
            const auto k = m_basis->Order();
            m_globalWeights.resize(grid.getNCells() * (k + 1));
            for (Index i = 0; i < static_cast<Index>(grid.getNCells()); i++)
            {
                const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);
                m_globalWeights(ind) = cellWeights(grid[i]);
            }
        }
        return m_globalWeights;
    }

    /// (k+1, 2): the basis functions evaluated at the two ends of the domain.
    /// The only way to reach a boundary point value, because the nodes are
    /// Chebyshev points of the first kind and strictly interior.
    const Matrix &phiBoundary(const BasisType &basis, const Grid &grid)
    {
        invalidateIfStale(basis, grid);
        if (m_phiBoundary.size() == 0)
        {
            const Interval &I_l = grid[0];
            const Interval &I_u = grid[grid.getNCells() - 1];

            const auto k = m_basis->Order();
            m_phiBoundary.resize(k + 1, 2);
            for (Index i = 0; i < k + 1; i++)
            {
                m_phiBoundary(i, 0) = m_basis->Evaluate(I_l, i, I_l.x_l);
                m_phiBoundary(i, 1) = m_basis->Evaluate(I_u, i, I_u.x_u);
            }
        }
        return m_phiBoundary;
    }

  private:
    /// Per-cell weights, keyed on polynomial order as well as interval: the same
    /// Interval carries different weights at different orders.
    const Vector &cellWeights(Interval const &I)
    {
        const auto key = std::make_pair(m_order, I);
        if (!m_cellWeights.contains(key))
            m_cellWeights.insert({key, m_basis->getIntegrationWeights(I)});
        return m_cellWeights.at(key);
    }

    /// Discard everything cached if the basis order or the grid has changed.
    ///
    /// Keyed on order and grid, not on &basis: BasisType::getBasis() returns the
    /// flyweight *by value*, so callers may hand us a different address each
    /// time for what is the same basis. Two bases of equal order are
    /// interchangeable.
    void invalidateIfStale(const BasisType &basis, const Grid &grid)
    {
        const unsigned int order = basis.Order();
        m_basis = &basis;
        if (m_order == order && m_grid && *m_grid == grid)
            return;

        m_order = order;
        m_grid = grid;
        m_cellWeights.clear();
        m_globalWeights.resize(0);
        m_phiBoundary.resize(0, 0);
    }

    const BasisType *m_basis = nullptr; // borrowed for the duration of a call
    std::map<std::pair<unsigned int, Interval>, Vector> m_cellWeights;
    Vector m_globalWeights;
    Matrix m_phiBoundary;
    std::optional<Grid> m_grid;
    unsigned int m_order = 0;
};

} // namespace Integrator
#endif // PYINTEGRATOR_HPP
