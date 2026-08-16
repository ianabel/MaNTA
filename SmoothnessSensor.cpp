#include "SmoothnessSensor.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

namespace
{
// Coefficients below this fraction of the cell's own largest are round-off in a
// quantity of that size rather than measurements of it, and are not fitted.
//
// The size is a round-off bound on the transform and not a taste threshold:
// each uhat_j is a length-(k+1) dot product of a row of V^-1 with the nodal
// values, whose error is bounded to first order by (k+1) * eps * scale.
//
// A plain `eps` is *not* enough, and the margin matters more than it looks. On
// |x|^(4/3) at k = 6 -- a function even about the cell centre, so every odd
// coefficient is identically zero -- the computed uhat_1 came out at 2.56e-16
// of the scale, above a one-epsilon floor while uhat_3 and uhat_5 fell below
// it. That one surviving round-off mode dragged the least-squares line to
// s = -16.9, a *negative* rate, on a spectrum that plainly decays. The
// remaining modes here sit at 5.2e-2 of the scale and up, so (k+1) * eps
// separates the two populations by eleven orders; there is nothing delicate
// about where in that gap the line falls.
double relativeFloor(unsigned int k)
{
    return static_cast<double>(k + 1) * std::numeric_limits<double>::epsilon();
}
} // namespace

CellSmoothness cellSmoothness(NodalBasis const &basis,
                              Eigen::Ref<const Vector> const &nodalValues)
{
    const unsigned int k = basis.Order();

    if (k < 2)
        throw std::invalid_argument(std::format(
            "The modal decay fit needs at least two non-constant modes, so k >= 2; "
            "this basis is degree {}. MaNTA carries one global order, so this is a "
            "property of the run rather than of a cell.",
            k));

    const Vector uhat = basis.ToModal(nodalValues);

    // ||P_j||^2 = 2/(2j+1) on the reference cell, and the common factor of 2
    // cancels in the ratio, so the weights are 1/(2j+1).
    Vector energy(k + 1);
    for (Index j = 0; j <= static_cast<Index>(k); ++j)
        energy(j) = uhat(j) * uhat(j) / (2.0 * static_cast<double>(j) + 1.0);

    const double total = energy.sum();

    CellSmoothness out;

    // An identically zero cell. Nothing to measure, and the honest answer is
    // that it is as smooth as this space can represent -- which also sorts
    // correctly if a driver ranks cells by decayRate.
    if (!(total > 0.0))
    {
        out.modalEnergyFraction = 0.0;
        out.decayRate = std::numeric_limits<double>::infinity();
        return out;
    }

    out.modalEnergyFraction = energy(k) / total;

    // The scale is the largest coefficient rather than uhat_0 specifically,
    // because uhat_0 is the cell mean and vanishes on a cell where the solution
    // happens to average to zero; the largest is the cell's own size in every
    // case except the identically zero one, which is the branch above.
    const double scale = uhat.cwiseAbs().maxCoeff();
    const double floor = relativeFloor(k) * scale;
    const Index top = static_cast<Index>(k);

    // The top mode is at round-off, so the cell's solution is exactly
    // representable below degree k and there is no decay left to measure. This
    // covers the constant cell and the below-degree polynomial with one rule,
    // and it has to be covered: with the fitted coefficients all pinned at the
    // floor they are all equal, the least-squares slope is exactly zero, and a
    // zero rate is this sensor's way of saying "as rough as it gets" -- so the
    // smoothest possible field would be reported as the roughest.
    if (std::abs(uhat(top)) <= floor)
    {
        out.decayRate = std::numeric_limits<double>::infinity();
        return out;
    }

    // Least squares for the slope of log|uhat_j| against log j (Mavriplis),
    // over the modes that carry signal. j = 0 is excluded because log 0 is not
    // a thing and the constant mode says nothing about decay anyway.
    //
    // Skipping the floored modes rather than fitting them is what makes this
    // survive a *structurally* sparse spectrum, and that is not a hypothetical:
    // a function even about the cell centre has every odd coefficient
    // identically zero. Pinning those at the floor and fitting the resulting
    // alternating sequence gives a line with the wrong sign -- measured
    // s = -8.3 at k = 6 for |x|^(4/3) on a cell centred at its branch point,
    // for a spectrum that genuinely decays. A coefficient at the floor is not a
    // measurement; it was clamped precisely because it is round-off.
    auto fitted = [&](Index j) { return std::abs(uhat(j)) > floor; };

    Index n = 0;
    double sx = 0.0, sy = 0.0;
    for (Index j = 1; j <= top; ++j)
        if (fitted(j))
        {
            ++n;
            sx += std::log(static_cast<double>(j));
            sy += std::log(std::abs(uhat(j)));
        }

    // Only the top mode carries anything. The spectrum has no decay in it at
    // all, which is the roughest thing this can honestly report -- and it is a
    // different statement from the branch above, where the top mode was the one
    // that vanished.
    if (n < 2)
    {
        out.decayRate = 0.0;
        return out;
    }

    const double xbar = sx / static_cast<double>(n);
    const double ybar = sy / static_cast<double>(n);

    // n >= 2 distinct values of j, so the log j are distinct and den > 0.
    double num = 0.0, den = 0.0;
    for (Index j = 1; j <= top; ++j)
        if (fitted(j))
        {
            const double dx = std::log(static_cast<double>(j)) - xbar;
            num += dx * (std::log(std::abs(uhat(j))) - ybar);
            den += dx * dx;
        }

    // Decaying coefficients give a negative slope, and s is quoted positive.
    out.decayRate = -num / den;
    return out;
}

std::vector<CellSmoothness> cellSmoothness(DGSoln const &Y, Index var)
{
    const Grid &grid = Y.getGrid();
    NodalBasis const &basis = Y.getBasis();

    std::vector<CellSmoothness> out;
    out.reserve(grid.getNCells());

    for (Grid::Index cell = 0; cell < grid.getNCells(); ++cell)
        out.push_back(cellSmoothness(basis, Y.u(var).getCoeff(cell).second));

    return out;
}
