#include "AutodiffAdjointProblem.hpp"

// dG/dp of the objective GFn reports, G = sum_j w_j g(Z_j, p) over the nodes the
// objective is integrated on: the k+1 basis nodes with their interpolatory
// weights, or under Superconvergent the star nodes, at u*, with b1. Taking it at
// the same nodes with the same weights is what makes it the derivative of the
// reported G rather than of some other integral of g.
//
// setPval takes the *parameter* index. `i` selects the objective, and writing
// through it would differentiate with respect to whichever parameter happens to
// share the objective's number -- and, because setPval assigns the whole Real,
// would leave that parameter holding another one's value afterwards. The physics
// case would then evaluate a different problem for the rest of the run.
//
// Writing pIndex is also what restores it: p is a copy of that parameter's own
// value with only its gradient seeded, so every assignment puts the value back.
// clearGradients() removes the seed. AutodiffTransportSystem's dSigmaFn_dp and
// dSources_dp rely on the same property.
Value AutodiffAdjointProblem::nodalParameterDerivative(Index i, Index pIndex, GlobalState const &states,
                                                       std::vector<Position> const &points,
                                                       Vector const &weights) const
{
    Real p = PhysicsProblem->getPval(pIndex);
    Value out = 0.0;
    for (size_t j = 0; j < points.size(); ++j)
    {
        const State s = states[j];
        auto g_wrapper = [&](Real pv)
        {
            RealVector u(s.u());
            RealVector q(s.q());
            RealVector sigma(s.sigma());
            RealVector phi(s.phi());
            PhysicsProblem->setPval(pIndex, pv);
            return g[i](points[j], u, q, sigma, phi);
        };
        out += weights(j) * autodiff::derivative(g_wrapper, wrt(p), at(p));
    }
    PhysicsProblem->clearGradients();
    return out;
}

Matrix AutodiffAdjointProblem::nodalParameterDerivatives(Index i, GlobalState const &states,
                                                         std::vector<Position> const &points,
                                                         Vector const &weights) const
{
    Matrix out = Matrix::Zero(1, np);
    for (Index pIndex = 0; pIndex < getNpInternal(); ++pIndex)
        out(0, pIndex) = nodalParameterDerivative(i, pIndex, states, points, weights);
    return out;
}

Value AutodiffAdjointProblem::dGFndp(Index i, Index pIndex, DGSoln &y) const
{
    return nodalParameterDerivative(i, pIndex, y.evalOnNodes(), y.getPoints(), nodalWeights(y));
}

Matrix AutodiffAdjointProblem::dGFndp(Index i, DGSoln &y) const
{
    return nodalParameterDerivatives(i, y.evalOnNodes(), y.getPoints(), nodalWeights(y));
}

// Reads the u* from the last pp.computeUStar(y), as AdjointProblem::GFn's
// superconvergent form does.
Matrix AutodiffAdjointProblem::dGFndp(Index i, DGSoln &y, Postprocessor const &pp) const
{
    const Index nCells = y.getGrid().getNCells();
    const Index nStar = pp.starDoF();
    Vector weights(nCells * nStar);
    for (Index c = 0; c < nCells; ++c)
        weights.segment(c * nStar, nStar) = pp.starWeights(c);
    return nodalParameterDerivatives(i, pp.evalOnStarNodes(y), pp.starPoints(), weights);
}

Vector AutodiffAdjointProblem::nodalWeights(DGSoln const &y)
{
    const Index nCells = y.getGrid().getNCells();
    const Index nNodes = y.getBasis().Order() + 1;
    Vector weights(nCells * nNodes);
    for (Index c = 0; c < nCells; ++c)
        weights.segment(c * nNodes, nNodes) = y.getBasis().getIntegrationWeights(y.getGrid()[c]);
    return weights;
}

Value AutodiffAdjointProblem::gFn(Index i, const State &s, Position x) const
{
    // Real p = PhysicsProblem->getPval(i);
    RealVector u(s.u());
    RealVector q(s.q());
    RealVector sigma(s.sigma());
    RealVector phi(s.phi());
    return g[i](x, u, q, sigma, phi).val;
}

void AutodiffAdjointProblem::dgFn_du(Index i, VectorRef grad, const State &s, Position x)
{
    RealVector u(s.u());
    RealVector q(s.q());
    RealVector sigma(s.sigma());
    RealVector phi(s.phi());

    Real uout;
    // Real p = PhysicsProblem->getPval(i);

    autodiff::gradient([this, i](Position X, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                       { return g[i](X, uD, qD, sigmaD, phiD); }, wrt(u), at(x, u, q, sigma, phi), uout, grad);
}

void AutodiffAdjointProblem::dgFn_dq(Index i, VectorRef grad, const State &s, Position x)
{
    RealVector u(s.u());
    RealVector q(s.q());
    RealVector sigma(s.sigma());
    RealVector phi(s.phi());

    // Real p = PhysicsProblem->getPval(i);
    Real uout;
    autodiff::gradient([this, i](Position X, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                       { return g[i](X, uD, qD, sigmaD, phiD); }, wrt(q), at(x, u, q, sigma, phi), uout, grad);
}

void AutodiffAdjointProblem::dgFn_dsigma(Index i, VectorRef grad, const State &s, Position x)
{
    RealVector u(s.u());
    RealVector q(s.q());
    RealVector sigma(s.sigma());
    RealVector phi(s.phi());

    // Real p = PhysicsProblem->getPval(i);

    Real uout;

    autodiff::gradient([this, i](Position X, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                       { return g[i](X, uD, qD, sigmaD, phiD); }, wrt(sigma), at(x, u, q, sigma, phi), uout, grad);
}

void AutodiffAdjointProblem::dgFn_dphi(Index i, VectorRef grad, const State &s, Position x)
{
    RealVector u(s.u());
    RealVector q(s.q());
    RealVector sigma(s.sigma());
    RealVector phi(s.phi());

    // Real p = PhysicsProblem->getPval(i);

    Real uout;

    autodiff::gradient([this, i](Position X, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                       { return g[i](X, uD, qD, sigmaD, phiD); }, wrt(phi), at(x, u, q, sigma, phi), uout, grad);
}

void AutodiffAdjointProblem::dSigmaFn_dp(Index i, Index pIndex, Value &grad, const State &s, Position x)
{
    return PhysicsProblem->dSigmaFn_dp(i, pIndex, grad, s, x, 0.0);
}

void AutodiffAdjointProblem::dSources_dp(Index i, Index pIndex, Value &grad, const State &s, Position x)
{
    return PhysicsProblem->dSources_dp(i, pIndex, grad, s, x, 0.0);
}

bool AutodiffAdjointProblem::computeUpperBoundarySensitivity(Index var, Index pIndex)
{
    auto it = upperBoundarySensitivities.find({var, pIndex});
    if (it != upperBoundarySensitivities.end())
        return it->second;
    else
        return false;
}

bool AutodiffAdjointProblem::computeLowerBoundarySensitivity(Index var, Index pIndex)
{
    auto it = lowerBoundarySensitivities.find({var, pIndex});
    if (it != lowerBoundarySensitivities.end())
        return it->second;
    else
        return false;
}

void AutodiffAdjointProblem::addUpperBoundarySensitivity(Index i, Index pIndex)
{
    upperBoundarySensitivities.insert(std::make_pair(std::make_tuple(i, pIndex), true));
    AdjointProblem::np_boundary++;
}

void AutodiffAdjointProblem::addLowerBoundarySensitivity(Index i, Index pIndex)
{
    lowerBoundarySensitivities.insert(std::make_pair(std::make_tuple(i, pIndex), true));
    AdjointProblem::np_boundary++;
}
