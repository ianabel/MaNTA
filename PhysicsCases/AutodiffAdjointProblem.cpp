#include "AutodiffAdjointProblem.hpp"

Value AutodiffAdjointProblem::GFn(Index i, DGSoln &y) const
{
    Real p = PhysicsProblem->getPval(i);
    auto g_wrapper = [&](Position x)
    {
        State s = y.eval(x);
        RealVector u(s.Variable);
        RealVector q(s.Derivative);
        RealVector sigma(s.Flux);
        RealVector phi(s.Aux);
        return g(x, p, u, q, sigma, phi).val;
    };

    return integrator::integrate(g_wrapper, PhysicsProblem->xL, PhysicsProblem->xR, max_depth);
}

Value AutodiffAdjointProblem::dGFndp(Index i, DGSoln &y) const
{
    Real p = PhysicsProblem->getPval(i);

    auto dgdp = [&](Position x)
    {
        State s = y.eval(x);
        RealVector u(s.Variable);
        RealVector q(s.Derivative);
        RealVector sigma(s.Flux);
        RealVector phi(s.Aux);
        auto grad = autodiff::derivative(g, wrt(p), at(x, p, u, q, sigma, phi));
        return grad;
    };
    return integrator::integrate(dgdp, PhysicsProblem->xL, PhysicsProblem->xR, max_depth);
}

Value AutodiffAdjointProblem::gFn(Index i, const State &s, Position x) const
{
    Real p = PhysicsProblem->getPval(i);
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);
    RealVector phi(s.Aux);
    return g(x, p, u, q, sigma, phi).val;
}

void AutodiffAdjointProblem::dgFn_du(Index i, Values &grad, const State &s, Position x)
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);
    RealVector phi(s.Aux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                              { return g(X, p, uD, qD, sigmaD, phiD); }, wrt(u), at(x, p, u, q, sigma, phi));
}

void AutodiffAdjointProblem::dgFn_dq(Index i, Values &grad, const State &s, Position x)
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);
    RealVector phi(s.Aux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                              { return g(X, p, uD, qD, sigmaD, phiD); }, wrt(q), at(x, p, u, q, sigma, phi));
}

void AutodiffAdjointProblem::dgFn_dsigma(Index i, Values &grad, const State &s, Position x)
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);
    RealVector phi(s.Aux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                              { return g(X, p, uD, qD, sigmaD, phiD); }, wrt(sigma), at(x, p, u, q, sigma, phi));
}

void AutodiffAdjointProblem::dgFn_dphi(Index i, Values &grad, const State &s, Position x)
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);
    RealVector phi(s.Aux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD, RealVector phiD)
                              { return g(X, p, uD, qD, sigmaD, phiD); }, wrt(phi), at(x, p, u, q, sigma, phi));
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
}

void AutodiffAdjointProblem::addLowerBoundarySensitivity(Index i, Index pIndex)
{
    lowerBoundarySensitivities.insert(std::make_pair(std::make_tuple(i, pIndex), true));
}
