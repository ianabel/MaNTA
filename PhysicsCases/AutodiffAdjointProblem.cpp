#include "AutodiffAdjointProblem.hpp"

Value AutodiffAdjointProblem::GFn(Index i, DGSoln &y) const
{
    auto g_wrapper = [this, i](const DGSoln &y, Position x)
    {
        State s = y.eval(x);
        RealVector u(s.u());
        RealVector q(s.q());
        RealVector sigma(s.sigma());
        RealVector phi(s.phi());
        return g[i](x, u, q, sigma, phi).val;
    };

    return y.EvaluateIntegral(g_wrapper);
}

Value AutodiffAdjointProblem::dGFndp(Index i, Index pIndex, DGSoln &y) const
{
    Real p = PhysicsProblem->getPval(pIndex);

    // setPval takes the *parameter* index. `i` selects the objective, and writing
    // through it would differentiate with respect to whichever parameter happens
    // to share the objective's number -- and, because setPval assigns the whole
    // Real, would leave that parameter holding another one's value once the
    // integral finished. The physics case would then evaluate a different problem
    // for the rest of the run.
    //
    // Writing pIndex is also what restores it: p is a copy of that parameter's own
    // value with only its gradient seeded, so the last assignment puts the value
    // back. clearGradients() below removes the seed. AutodiffTransportSystem's
    // dSigmaFn_dp and dSources_dp rely on the same property.
    auto g_wrapper = [&](Real p, Position x)
    {
        State s = y.eval(x);
        RealVector u(s.u());
        RealVector q(s.q());
        RealVector sigma(s.sigma());
        RealVector phi(s.phi());

        PhysicsProblem->setPval(pIndex, p);
        return g[i](x, u, q, sigma, phi);
    };
    auto I = integrator::integrate([&](Position x)
                                   { return autodiff::derivative(g_wrapper, wrt(p), at(p, x)); }, PhysicsProblem->xL, PhysicsProblem->xR, max_depth);
    PhysicsProblem->clearGradients();
    return I;
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
