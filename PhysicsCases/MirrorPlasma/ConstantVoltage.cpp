#include "../MirrorPlasma.hpp"
#include <autodiff/forward/dual.hpp>

Value MirrorPlasma::InitialDensityTimeDerivative(RealVector u, RealVector q, Position V) const
{
    Real2nd Vval = V;

    State s(nVars, nScalars);
    Values d2udx2(nVars);
    RealVector sigma(nVars);

    for (Index j = 0; j < nVars; ++j)
    {
        auto [uval, qval, d2udx2val] = derivatives([this, j](Real2nd x)
                                                   { return InitialFunction(j, x, 0.0); }, wrt(Vval, Vval), at(Vval));

        s.Variable(j) = uval;
        s.Derivative(j) = qval;
        d2udx2(j) = d2udx2val;
    }

    Values gradu(nVars);
    Values gradq(nVars);

    Real Vreal = V;

    Real uout;

    autodiff::gradient([this](RealVector uD, RealVector qD, Real X)
                       { return Gamma(uD, qD, X, 0.0); },
                       wrt(u), at(u, q, Vreal), uout, gradu);

    autodiff::gradient([this](RealVector uD, RealVector qD, Real X)
                       { return Gamma(uD, qD, X, 0.0); },
                       wrt(q), at(u, q, Vreal), uout, gradq);

    double dSdx = autodiff::derivative([this](RealVector uD, RealVector qD, Real X)
                                       { return Gamma(uD, qD, X, 0.0); },
                                       wrt(Vreal), at(u, q, Vreal));

    double dSigma_dx = dSdx;

    for (Index j = 0; j < nVars; ++j)
    {
        dSigma_dx += s.q(j) * gradu[j] + d2udx2[j] * gradq[j];
    }

    sigma.setZero();

    RealVector phi(nAux);

    for (Index j = 0; j < nAux; ++j)
    {
        phi(j) = InitialAuxValue(j, V, 0.0);
    }

    double S = Sn(u, q, sigma, phi, Vreal, 0.0).val;
    return S + dSigma_dx;
}

Value MirrorPlasma::InitialCurrent(Time t) const
{
    if (restarting)
    {
        return 0.0;
    }
    else
    {
        return -IRadial * (1 + tanh(-t / CurrentDecay));
    }
}

Value MirrorPlasma::InitialScalarValue(Index s) const
{
    auto n = [&](Position V)
    { return uToDensity(InitialValue(Channel::Density, V)).val; };
    auto L = [&](Position V)
    { return InitialValue(Channel::AngularMomentum, V); };
    auto omega = [&](Position V)
    {
        Value R = B->R_V(V, 0.0);
        return L(V) / (n(V) * R * R);
    };

    auto Phi_V = [&](Position V)
    {
        Value phi = integrator::integrate([&](double V)
                                          { return omega(V) / B->VPrime(V); }, xL, V, max_depth);
        return phi;
    };

    switch (static_cast<Scalar>(s))
    {
    case Scalar::Error:
        return V0 - Phi_V(xR);
    case Scalar::Integral:
        return 0.0;
    case Scalar::Current:
    {
        return InitialCurrent(0);
    }
    default:
        throw std::logic_error("Initial value requested for non-existent scalar!");
    }
}
Value MirrorPlasma::InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const
{
    switch (static_cast<Scalar>(s))
    {
    case Scalar::Error:
    {
        auto domegadt = [&](Position V)
        {
            Position R = B->R_V(V, 0.0);
            Value n = uToDensity(y.u(Channel::Density)(V)).val;
            Value L = y.u(Channel::AngularMomentum)(V);
            Value ndot = dydt.u(Channel::Density)(V);
            if (evolveLogDensity)
                ndot *= n; // if evolving log, ndot actually represents d log n / dt
            Value Ldot = dydt.u(Channel::AngularMomentum)(V);
            return 1 / (R * R * B->VPrime(V)) * (Ldot / n - L * ndot / (n * n));
        };
        Value Phidot = -integrator::integrate(domegadt, xL, xR, max_depth);
        return Phidot;
    }
    case Scalar::Integral:
        return InitialScalarValue(Scalar::Error);
    default:
        throw std::logic_error("Initial derivative called for algebraic (non-differential) scalar");
    }
}

// The scalar constraints, on the nodal quadrature.
//
// These used to be handed a std::function test function P and a single cell's
// Interval, and integrated against them with nested adaptive Gauss-Kronrod --
// including Phi_V, an integral from xL to a *variable* upper limit evaluated
// inside another integral. They now use the framework's node quadrature, the
// same one the residual projects sources with:
//
//     Int f dV  ->  sum_j weights(j) * f(V_j)
//
// and the cumulative Phi at node m is the partial sum of that up to m. Two
// consequences worth stating plainly. The quadrature is no longer adaptive, so
// the value of Phi differs from the old one by the interpolation error of the
// integrand -- for a well-resolved run that is small, but it is not zero. And
// the derivative is now exactly the derivative of the quantity G actually uses,
// which the adaptive version was not: that mismatch is precisely the defect
// that cost ScalarTestLD3 a 7% error in its own Jacobian.
//
// Nothing in any test suite exercises useConstantVoltage, so this port rests on
// the algebra rather than on a measurement.
Value MirrorPlasma::ScalarG(Index s, GlobalState const &y, GlobalState const &ydot,
                            std::vector<Position> const &abscissae, Values const &weights,
                            Matrix const &phiBoundary, Time t)
{
    using namespace ScalarHooks;

    const Value dEdt = ydot.Scalars()(Scalar::Error);
    const Value E = y.Scalars()(Scalar::Error);
    const Value tfac = restarting ? 1.0 : tanh(t / CurrentDecay);

    switch (static_cast<Scalar>(s))
    {
    case Scalar::Error:
        // E = V0 - Phi(xR), with Phi(xR) = Int omega / VPrime dV over the domain.
        return E - (V0 - PhiAtUpperBoundary(y, abscissae, weights));

    case Scalar::Integral:
        return ydot.Scalars()(Scalar::Integral) - E;

    case Scalar::Current:
    {
        const Value Integral = y.Scalars()(Scalar::Integral);
        const Value Current = y.Scalars()(Scalar::Current);
        return Current - InitialCurrent(t) -
               tfac * (TotalCurrent(y, abscissae, weights, phiBoundary, t) + gamma * E +
                       gamma_d * dEdt + gamma_h * Integral);
    }
    default:
        throw std::logic_error("scalar index > nScalars");
    }
}

Value MirrorPlasma::PhiAtUpperBoundary(GlobalState const &y, std::vector<Position> const &abscissae,
                                       Values const &weights) const
{
    Value Phi = 0.0;
    for (size_t j = 0; j < y.size(); ++j)
    {
        const Position V = abscissae[j];
        const Value R = B->R_V(V, 0.0);
        const Value n = uToDensity(y.Variable()(Channel::Density, j)).val;
        const Value L = y.Variable()(Channel::AngularMomentum, j);
        Phi += weights(j) * (L / (n * R * R)) / B->VPrime(V);
    }
    return Phi;
}

Value MirrorPlasma::TotalCurrent(GlobalState const &y, std::vector<Position> const &abscissae,
                                 Values const &weights, Matrix const &phiBoundary, Time t)
{
    using namespace ScalarHooks;

    const Value FluxTerm = boundaryValue(y.Flux().row(Channel::AngularMomentum), phiBoundary, Upper) -
                           boundaryValue(y.Flux().row(Channel::AngularMomentum), phiBoundary, Lower);

    Value SourceTerm = 0.0;
    Values noScalars(nScalars);
    noScalars.setZero();
    for (size_t j = 0; j < y.size(); ++j)
    {
        State st = y[j];
        SourceTerm += weights(j) * Source(Channel::AngularMomentum, st.Variable, st.Derivative,
                                          st.Flux, st.Aux, noScalars, abscissae[j], t)
                                       .val;
    }

    return (FluxTerm - SourceTerm) / dPsi();
}

/// Int 1/VPrime dV over the domain. Pure geometry -- no solution dependence, so
/// it does not enter any derivative and keeps its adaptive rule.
Value MirrorPlasma::dPsi() const
{
    return integrator::integrate([&](double V)
                                 { return 1 / B->VPrime(V); }, xL, xR, max_depth);
}

void MirrorPlasma::ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot,
                                GlobalState const &y, GlobalState const &,
                                std::vector<Position> const &abscissae, Values const &weights,
                                Matrix const &phiBoundary, Time t)
{
    using namespace ScalarHooks;

    const Value tfac = restarting ? 1.0 : tanh(t / CurrentDecay);
    const Value invPsi = 1.0 / dPsi();

    // ---- G_Error = E - V0 + Phi(xR) -------------------------------------
    //
    // d/d(DOF j) of the nodal quadrature above is just weights(j) times the
    // integrand's derivative at node j.
    for (size_t j = 0; j < y.size(); ++j)
    {
        const Position V = abscissae[j];
        const Value R = B->R_V(V, 0.0);
        const Value n = uToDensity(y.Variable()(Channel::Density, j)).val;
        const Value L = y.Variable()(Channel::AngularMomentum, j);
        const Value w = weights(j) / B->VPrime(V);

        dG[Scalar::Error].Variable()(Channel::AngularMomentum, j) = w / (n * R * R);

        Value dPhi_dn = -w * L / (n * n * R * R);
        if (evolveLogDensity)
            dPhi_dn *= n; // u is log n, so d/du = n d/dn
        dG[Scalar::Error].Variable()(Channel::Density, j) = dPhi_dn;
    }
    dG[Scalar::Error].Scalars()(Scalar::Error) = 1.0;

    // ---- G_Integral = dI/dt - E -----------------------------------------
    dG[Scalar::Integral].Scalars()(Scalar::Error) = -1.0;
    dGdot[Scalar::Integral].Scalars()(Scalar::Integral) = 1.0;

    // ---- G_Current ------------------------------------------------------
    //
    // G = Current - InitialCurrent - tfac * ( (FluxTerm - SourceTerm)/dPsi + ... ),
    // so dG/dSourceTerm = +tfac/dPsi and dG/dFluxTerm = -tfac/dPsi.
    Values grad_u(nVars), grad_phi(nAux);
    for (size_t j = 0; j < y.size(); ++j)
    {
        // The part of the source that depends on the scalars is excluded: it is
        // accounted for by the explicit scalar entries below.
        State st = y[j];
        st.Scalars.setZero();

        dSources_du(Channel::AngularMomentum, grad_u, st, abscissae[j], t);
        for (Index v = 0; v < nVars; ++v)
            dG[Scalar::Current].Variable()(v, j) = invPsi * tfac * weights(j) * grad_u(v);

        if (nAux > 0)
        {
            dSources_dPhi(Channel::AngularMomentum, grad_phi, st, abscissae[j], t);
            for (Index a = 0; a < nAux; ++a)
                dG[Scalar::Current].Aux()(a, j) = invPsi * tfac * weights(j) * grad_phi(a);
        }
    }

    addBoundaryDerivative(dG[Scalar::Current].Flux().row(Channel::AngularMomentum), phiBoundary,
                          Upper, -invPsi * tfac);
    addBoundaryDerivative(dG[Scalar::Current].Flux().row(Channel::AngularMomentum), phiBoundary,
                          Lower, +invPsi * tfac);

    dG[Scalar::Current].Scalars()(Scalar::Error) = -gamma * tfac;
    dGdot[Scalar::Current].Scalars()(Scalar::Error) = -gamma_d * tfac;
    dG[Scalar::Current].Scalars()(Scalar::Integral) = -gamma_h * tfac;
    dG[Scalar::Current].Scalars()(Scalar::Current) = 1.0;
}

