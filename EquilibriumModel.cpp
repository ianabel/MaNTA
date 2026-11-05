#include "EquilibriumModel.hpp"

#include <cmath>
#include <limits>

namespace
{
// Every component the input carries, as one sum of squares: the four nodal
// matrices Solve can read. Scalars are not in a model's input (the solver
// refuses a geometry source on a scalar system), and geometry rows are what
// comes out, not what goes in.
double sumOfSquares(GlobalState const &s)
{
    return s.Variable().squaredNorm() + s.Derivative().squaredNorm() + s.Flux().squaredNorm() +
           s.Aux().squaredNorm();
}

GlobalState shifted(GlobalState const &base, double h, GlobalState const &dp)
{
    GlobalState out = base;
    out.Variable() += h * dp.Variable();
    out.Derivative() += h * dp.Derivative();
    out.Flux() += h * dp.Flux();
    out.Aux() += h * dp.Aux();
    return out;
}
} // namespace

double EquilibriumModel::responseStep() const
{
    return std::sqrt(std::numeric_limits<double>::epsilon());
}

// A forward difference along dp, warm-started from the equilibrium being
// differenced, so the perturbed Solve starts a step's width from its answer.
//
// The step is responseStep() times the size of the input over the size of the
// direction, in the root-mean-square over every component, so the perturbation
// is a fixed *relative* change of the input whatever dp's scale. Floored at a
// unit input size for an input that is identically zero, where a relative step
// means nothing. The geometry at the base is taken from `e` again rather than
// passed in: it is the field-side cost, free on the metric that matters, and it
// keeps the two halves of the difference at the same points by construction.
void EquilibriumModel::applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states,
                                     GlobalState const &dp, std::vector<Position> const &inputPoints,
                                     Vector const &weights, Time t,
                                     std::vector<Position> const &geometryPoints)
{
    const Eigen::Index nPoints = static_cast<Eigen::Index>(geometryPoints.size());
    const double dpNorm = std::sqrt(sumOfSquares(dp));
    if (!(dpNorm > 0.0))
    {
        dg.setZero();
        return;
    }
    const double pNorm = std::sqrt(sumOfSquares(states));
    const double h = responseStep() * std::max(pNorm, 1.0) / dpNorm;

    const std::unique_ptr<Equilibrium> perturbed =
        Solve(shifted(states, h, dp), inputPoints, weights, t, &e);

    Matrix g0 = Matrix::Zero(nGeometry(), nPoints), g1 = Matrix::Zero(nGeometry(), nPoints);
    Geometry(g0, e, geometryPoints);
    Geometry(g1, *perturbed, geometryPoints);
    dg = (g1 - g0) / h;
}

// See the declaration. Column c of the assembled matrix is the response to a
// unit change in one input component at one point, laid out the way the dense
// G' acts: rows are the geometry at the geometry points, flattened point by
// point (slot fastest), as dg's storage is; columns run Variable, Derivative,
// Flux, Aux, and within each row by row, point by point.
void EquilibriumModel::applyResponseTranspose(GlobalState &dp, Equilibrium const &e, GlobalState const &states,
                                              Matrix const &dg, std::vector<Position> const &inputPoints,
                                              Vector const &weights, Time t,
                                              std::vector<Position> const &geometryPoints)
{
    const Eigen::Index nIn = static_cast<Eigen::Index>(inputPoints.size());
    const Eigen::Index nGeomValues = nGeometry() * static_cast<Eigen::Index>(geometryPoints.size());
    // The four blocks of the input, in the order the columns take them.
    auto block = [](GlobalState &s, int b) -> Matrix &
    {
        switch (b)
        {
        case 0:
            return s.Variable();
        case 1:
            return s.Derivative();
        case 2:
            return s.Flux();
        default:
            return s.Aux();
        }
    };

    if (e.assembledInputPoints != inputPoints || e.assembledGeometryPoints != geometryPoints ||
        e.assembledResponse.rows() != nGeomValues)
    {
        Eigen::Index nCols = 0;
        GlobalState unit = states.zeroedLike();
        for (int b = 0; b < 4; ++b)
            nCols += block(unit, b).rows() * nIn;

        Matrix G(nGeomValues, nCols);
        Matrix column(nGeometry(), static_cast<Eigen::Index>(geometryPoints.size()));
        Eigen::Index c = 0;
        for (int b = 0; b < 4; ++b)
            for (Eigen::Index row = 0; row < block(unit, b).rows(); ++row)
                for (Eigen::Index j = 0; j < nIn; ++j, ++c)
                {
                    block(unit, b)(row, j) = 1.0;
                    column.setZero();
                    applyResponse(column, e, states, unit, inputPoints, weights, t, geometryPoints);
                    G.col(c) = Eigen::Map<const Vector>(column.data(), column.size());
                    block(unit, b)(row, j) = 0.0;
                }
        e.assembledResponse = std::move(G);
        e.assembledInputPoints = inputPoints;
        e.assembledGeometryPoints = geometryPoints;
    }

    const Vector product = e.assembledResponse.transpose() * Eigen::Map<const Vector>(dg.data(), dg.size());
    Eigen::Index c = 0;
    for (int b = 0; b < 4; ++b)
    {
        Matrix &out = block(dp, b);
        for (Eigen::Index row = 0; row < out.rows(); ++row)
            for (Eigen::Index j = 0; j < nIn; ++j, ++c)
                out(row, j) = product(c);
    }
}
