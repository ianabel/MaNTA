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

    gradu = autodiff::gradient([this](RealVector uD, RealVector qD, Real X)
                               { return Gamma(uD, qD, X, 0.0); },
                               wrt(u), at(u, q, Vreal));

    gradq = autodiff::gradient([this](RealVector uD, RealVector qD, Real X)
                               { return Gamma(uD, qD, X, 0.0); },
                               wrt(q), at(u, q, Vreal));

    double dSdx = autodiff::derivative([this](RealVector uD, RealVector qD, Real X)
                                       { return Gamma(uD, qD, X, 0.0); },
                                       wrt(Vreal), at(u, q, Vreal));

    double dSigma_dx = dSdx;

    for (Index j = 0; j < nVars; ++j)
    {
        dSigma_dx += s.Derivative[j] * gradu[j] + d2udx2[j] * gradq[j];
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
        return -IRadial * exp(-t / CurrentDecay);
    }
}
Value MirrorPlasma::InitialScalarValue(Index s) const
{
    auto n = [&](Position V)
    { return InitialValue(Channel::Density, V); };
    auto L = [&](Position V)
    { return InitialValue(Channel::AngularMomentum, V); };
    auto omega = [&](Position V)
    {
        Value R = B->R_V(V);
        return L(V) / (n(V) * R * R);
    };

    auto Phi_V = [&](Position V)
    {
        Value VPrime = B->VPrime(V);
        Value phi = 1 / VPrime * integrator::integrate(omega, xL, V, max_depth);
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
        auto u = [&](double V)
        {
            Values U(nVars);
            for (Index i = 0; i < nVars; ++i)
                U(i) = InitialValue(i, V);

            return U;
        };

        auto q = [&](double V)
        {
            Values Q(nVars);
            for (Index i = 0; i < nVars; ++i)
                Q(i) = InitialDerivative(i, V);

            return Q;
        };

        auto aux = [&](double V)
        {
            Values Aux(nAux);
            for (Index i = 0; i < nAux; ++i)
                Aux(i) = AutodiffTransportSystem::InitialAuxValue(i, V);

            return Aux;
        };

        // auto dJdt = [&](Position V)
        // {
        //     Position R = B->R_V(V);
        //     return R * R * InitialDensityTimeDerivative(u(V), q(V), V);
        // };

        RealVector pSigma(nVars);
        pSigma.setZero();

        RealVector pScalar(nScalars);
        pScalar.setZero();

        // auto I1 = [&](Position V)
        // {
        //     return dJdt(V) * omega(V);
        // };

        // Value B1 = integrator::integrate(I1, xL, xR);

        Value TimeDerivativeTerm = 0.0; // B1;

        Value FluxTerm = Pi(u(xL), q(xL), xL, 0.0).val - Pi(u(xR), q(xR), xR, 0.0).val;
        Value SourceTerm = integrator::integrate([&](Position V)
                                                 { return Somega(u(V), q(V), pSigma, aux(V), pScalar, V, 0.0).val; }, xL, xR, max_depth);
        Value Itot = 1 / (B->Psi_V(xR) - B->Psi_V(xL)) * (TimeDerivativeTerm + FluxTerm - SourceTerm);
        return Itot + InitialCurrent(0);
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
            Position R = B->R_V(V);
            Value n = y.u(Channel::Density)(V);
            Value L = y.u(Channel::AngularMomentum)(V);
            Value ndot = dydt.u(Channel::Density)(V);
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

bool MirrorPlasma::isScalarDifferential(Index s)
{
    switch (static_cast<Scalar>(s))
    {
    case Scalar::Error:
        return true;
    case Scalar::Integral:
        return true;
    default:
        return false;
    }
}

Value MirrorPlasma::ScalarGExtended(Index s, const DGSoln &y, const DGSoln &dydt, Time t)
{
    Value dEdt = dydt.Scalar(Scalar::Error);
    Value E = y.Scalar(Scalar::Error);

    auto n = [&](Position V)
    { return y.u(Channel::Density)(V); };
    auto L = [&](Position V)
    { return y.u(Channel::AngularMomentum)(V); };
    auto omega = [&](Position V)
    {
        Value R = B->R_V(V);
        Value w = L(V) / (n(V) * R * R);
        return w;
    };
    // auto dJdt = [&](Position V)
    // {
    //     Value R = B->R_V(V);
    //     return R * R * dydt.u(Channel::Density)(V);
    // };
    // auto dJprimedt = [&](Position V)
    // {
    //     Value R = B->R_V(V);
    //     return 2 * R * dydt.u(Channel::Density)(V) + R * R * dydt.q(Channel::Density)(V);
    // };
    // auto n0 = [&](Position V)
    // {
    //     return InitialValue(Channel::Density, V);
    // };
    // Value N0 = integrator::integrate(n0, xL, xR);
    auto Phi_V = [&](Position V)
    {
        Value VPrime = B->VPrime(V);
        Value phi = 1 / VPrime * integrator::integrate(omega, xL, V, max_depth);
        return phi;
    };
    switch (static_cast<Scalar>(s))
    {
    case Scalar::Error:
    {
        Value res = E - (V0 - Phi_V(xR));
        return res;
    }
    case Scalar::Integral:
    {
        Value dIdt = dydt.Scalar(Scalar::Integral);
        Value res = dIdt - E;
        return res;
    }
    case Scalar::Current:
    {
        Value Integral = y.Scalar(Scalar::Integral);
        Value Current = y.Scalar(Scalar::Current);
        Value TimeDerivativeTerm = 0.0;
        // /integrator::integrate([&](Position V)
        //                       { return dJdt(V) * omega(V); }, xL, xR);

        Value FluxTerm = y.sigma(Channel::AngularMomentum)(xR) - y.sigma(Channel::AngularMomentum)(xL); // SigmaFn(Channel::AngularMomentum, y.eval(xL), xL, t) - SigmaFn(Channel::AngularMomentum, y.eval(xR), xR, t);

        Value SourceTerm = integrator::integrate(
            [&](Position V)
            {
                State state = y.eval(V);
                Values pScalar(nScalars);
                pScalar.setZero();
                return Somega(state.Variable, state.Derivative, state.Flux, state.Aux, pScalar, V, t).val;
            },
            xL, xR, max_depth);
        Value Itot = 1 / (B->Psi_V(xR) - B->Psi_V(xL)) * (TimeDerivativeTerm + FluxTerm - SourceTerm) + InitialCurrent(t);
        Value res = (Current - Itot) - gamma * E - gamma_d * dEdt - gamma_h * Integral;
        return res;
    }
    default:
        throw std::logic_error("scalar index > nScalars");
    }
}
void MirrorPlasma::ScalarGPrimeExtended(Index scalarIndex, State &s, State &out_dt, const DGSoln &y, const DGSoln &dydt, std::function<double(double)> P, Interval I, Time t)
{
    s.zero();
    out_dt.zero();

    auto n = [&](Position V)
    { return y.u(Channel::Density)(V); };
    // // auto dndt = [&](Position V)
    // // { return dydt.u(Channel::Density)(V); };
    auto L = [&](Position V)
    { return y.u(Channel::AngularMomentum)(V); };
    // auto omega = [&](Position V)
    // {
    //     Value R = B->R_V(V);
    //     return L(V) / (n(V) * R * R);
    // };
    // auto dJdt = [&](Position V)
    // {
    //     Value R = B->R_V(V);
    //     return R * R * dydt.u(Channel::Density)(V);
    // };
    // auto dJprimedt = [&](Position V)
    // {
    //     Value R = B->R_V(V);
    //     return 2 * R * dydt.u(Channel::Density)(V) + R * R * dydt.q(Channel::Density)(V);
    // };
    // auto Phi_V = [&](Position V)
    // {
    //     Value VPrime = B->VPrime(V);
    //     return 1 / VPrime * integrator::integrate(omega, xL, V, max_depth);
    // };

    switch (static_cast<Scalar>(scalarIndex))
    {
    case Scalar::Error:
    {
        double P_L = integrator::integrate(
            [&](Position V)
            {
                Position R = B->R_V(V);
                return (P(V) / B->VPrime(V)) / (n(V) * R * R);
            },
            I.x_l, I.x_u, max_depth);
        double P_n = integrator::integrate(
            [&](Position V)
            {
                Position R = B->R_V(V);
                Value nv = n(V);
                return -(P(V) / B->VPrime(V)) * L(V) / (nv * nv * R * R);
            },
            I.x_l, I.x_u, max_depth);
        s.Variable(Channel::AngularMomentum) = P_L;
        s.Variable(Channel::Density) = P_n;
        // double P_n = integrator::integrate(P, I.x_l, I.x_u);
        // s.Variable(Channel::Density) = P_n;
        s.Scalars(Scalar::Error) = 1.0; // dG_0/dE
        break;
    }
    case Scalar::Integral:
    {
        s.Scalars(Scalar::Error) = -1.0;
        out_dt.Scalars(Scalar::Integral) = 1.0;
        break;
    }
    case Scalar::Current:
    {
        Value dPsi = B->Psi_V(xR) - B->Psi_V(xL);
        Values grad(nVars);
        Values grad_temp(nVars);
        for (Index i = 0; i < nVars; ++i)
            grad(i) = integrator::integrate(
                [&](Position V)
                {
                    dSources_du(Channel::AngularMomentum, grad_temp, y.eval(V), V, t);
                    return grad_temp(i) * P(V);
                },
                I.x_l, I.x_u, max_depth);
        s.Variable = -1 / dPsi * grad;
        grad.resize(nAux);
        grad_temp.resize(nAux);
        for (Index i = 0; i < nAux; ++i)
            grad(i) = integrator::integrate(
                [&](Position V)
                {
                    dSources_dPhi(Channel::AngularMomentum, grad_temp, y.eval(V), V, t);
                    return grad_temp(i) * P(V);
                },
                I.x_l, I.x_u, max_depth);

        s.Aux = -1 / dPsi * grad;

        s.Scalars(Scalar::Error) = -gamma;
        out_dt.Scalars(Scalar::Error) = -gamma_d;
        s.Scalars(Scalar::Integral) = -gamma_h;
        s.Scalars(Scalar::Current) = 1.0;

        // s.Variable(Channel::AngularMomentum) += 1 / dPsi * integrator::integrate([&](Position V)
        //                                                                          {
        //     Position R = B->R_V(V);
        //     return P(V) / (R * R * n(V)) * dJdt(V); }, I.x_l, I.x_u);

        if (abs(I.x_u - xR) < 1e-9)
            s.Flux(Channel::AngularMomentum) += -1.0 / dPsi * P(I.x_u);
        if (abs(I.x_l - xL) < 1e-9)
            s.Flux(Channel::AngularMomentum) -= -1.0 / dPsi * P(I.x_l);
        break;
    }
    default:
        break;
    }

    // s.Variable(Channel::Density) += 1 / dPsi * integrator::integrate([&](Position V)
    //                                                                  { return P(V) * L(V) * dndt(V) / (n(V) * n(V)); }, I.x_l, I.x_u);

    // out_dt.Variable(Channel::Density) = 1 / dPsi * integrator::integrate([&](Position V)
    //                                                                      {
    //     Position R = B->R_V(V);
    //          return P(V) * R * R * omega(V); }, I.x_l, I.x_u);

    // if (abs(I.x_u - xR) < 1e-9)
    //     out_dt.Variable(Channel::Density) += B->VPrime(xR) * Phi_V(xR) * P(I.x_u);

    // out_dt.Derivative(Channel::Density) = -1 / dPsi * integrator::integrate([&](Position V)
    //                                                                         { Position R = B->R_V(V);
    //     return P(V) * R * R * omega(V); }, I.x_l, I.x_u);
}
