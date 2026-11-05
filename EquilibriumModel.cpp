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
