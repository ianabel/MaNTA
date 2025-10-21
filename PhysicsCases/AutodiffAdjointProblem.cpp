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
        return gFn(x, p, u, q, sigma).val;
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
        auto grad = autodiff::derivative(gFn, wrt(p), at(x, p, u, q, sigma));
        return grad;
    };
    return integrator::integrate(dgdp, PhysicsProblem->xL, PhysicsProblem->xR, max_depth);
}

void AutodiffAdjointProblem::dgFn_du(Index i, Values &grad, const State &s, Position x) const
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD)
                              { return gFn(X, p, uD, qD, sigmaD); }, wrt(u), at(x, p, u, q, sigma));
}

void AutodiffAdjointProblem::dgFn_dq(Index i, Values &grad, const State &s, Position x) const
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD)
                              { return gFn(X, p, uD, qD, sigmaD); }, wrt(q), at(x, p, u, q, sigma));
}

void AutodiffAdjointProblem::dgFn_dsigma(Index i, Values &grad, const State &s, Position x) const
{
    RealVector u(s.Variable);
    RealVector q(s.Derivative);
    RealVector sigma(s.Flux);

    Real p = PhysicsProblem->getPval(i);

    grad = autodiff::gradient([this, i](Position X, Real p, RealVector uD, RealVector qD, RealVector sigmaD)
                              { return gFn(X, p, uD, qD, sigmaD); }, wrt(sigma), at(x, p, u, q, sigma));
}

void AutodiffAdjointProblem::dSigmaFn_dp(Index i, Value &grad, const State &s, Position x, Time t) const
{
    return PhysicsProblem->dSigmaFn_dp(i, grad, s, x, t);
}

void AutodiffAdjointProblem::dSources_dp(Index i, Value &grad, const State &s, Position x, Time t) const
{
    return PhysicsProblem->dSources_dp(i, grad, s, x, t);
}
