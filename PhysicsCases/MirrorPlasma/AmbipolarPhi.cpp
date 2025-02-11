#include "../MirrorPlasma.hpp"
#include <boost/math/tools/roots.hpp>

// G = Ipar = 0
Real MirrorPlasma::GFunc(Index, RealVector u, RealVector, RealVector, RealVector phi, Position V, Time t)
{
    Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real Te = floor(p_e / n, MinTemp);
    Real Ti = floor(p_i / n, MinTemp);

    Real R = B->R_V(V, 0.0);

    Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
    Real omega = u(Channel::AngularMomentum) / J;
    return ParallelCurrent<Real>(V, omega, n, Ti, Te, phi(0));
}

Value MirrorPlasma::InitialAuxValue(Index, Position V, Time t) const
{
    using boost::math::tools::eps_tolerance;
    using boost::math::tools::newton_raphson_iterate;

    Real n = uToDensity(InitialFunction(Channel::Density, V, t).val), p_e = (2. / 3.) * InitialFunction(Channel::ElectronEnergy, V, t).val, p_i = (2. / 3.) * InitialFunction(Channel::IonEnergy, V, t).val;
    Real Te = p_e / n;
    Real Ti = p_i / n;

    Real R = B->R_V(V, 0.0);

    Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
    Real omega = InitialFunction(Channel::AngularMomentum, V, t).val / J;

    auto func = [this, &n, &Te, &Ti, &omega, &V](double phi)
    {
        return ParallelCurrent<Real>(static_cast<Real>(V), omega, n, Ti, Te, static_cast<Real>(phi)).val;
    };
    auto deriv_func = [this, &n, &Te, &Ti, &omega, &V](double phi)
    {
        Real phireal = phi;
        return derivative([this](Real V, Real omega, Real n, Real Ti, Real Te, Real phi)
                          { return ParallelCurrent<Real>(V, omega, n, Ti, Te, phi); }, wrt(phireal), at(V, omega, n, Ti, Te, phireal));
    };

    const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
    int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
                                                            // just over half the digits correct.
    eps_tolerance<double> tol(get_digits);

    const boost::uintmax_t maxit = 20;
    boost::uintmax_t it = maxit;
    double phi_g = newton_raphson_iterate([&func, &deriv_func](double phi)
                                          { return std::pair<double, double>(func(phi), deriv_func(phi)); }, 0.0, -CentrifugalPotential(V, omega.val, Ti.val, Te.val), 0.01, get_digits, it);
    return phi_g;
}

template <typename T>
T MirrorPlasma::ParallelCurrent(T V, T omega, T n, T Ti, T Te, T phi) const
{
    T Xii = Xi_i<T>(V, phi, Ti, Te, omega);
    T Xie = Xi_e<T>(V, phi, Ti, Te, omega);

    T MirrorRatio = B->MirrorRatio(V, 0.0);

    double Sigma_i = 1.0;
    double Sigma_e = 1 + Z_eff;

    T a = Sigma_i * (1.0 / log(MirrorRatio * Sigma_i)) * sqrt(Plasma->IonMass() / ElectronMass) / Plasma->IonCollisionTime(n, Ti);
    T b = Sigma_e * (1.0 / log(MirrorRatio * Sigma_e)) * (Plasma->IonMass() / ElectronMass) / Plasma->ElectronCollisionTime(n, Te);

    T j = a * exp(-Xii) / Xii - b * exp(-Xie) / Xie;

    return j;
}

// Compute dphi1dV using the chain rule
// Take derivative2 of Jpar using autodiff, and then make sure to set the correct gradient values so Jacobian is correct
Real MirrorPlasma::dphi1dV(RealVector u, RealVector q, Real phi, Real V) const
{
    auto Jpar = [this](Real2ndVector u, Real2nd phi, Real2nd V)
    {
        Real2nd n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
        if (evolveLogDensity)
            n = exp(n);
        if (n < MinDensity)
            n.val.val = MinDensity;
        Real2nd Te = p_e / n;
        Real2nd Ti = p_i / n;

        Real2nd R = B->R_V(V, 0.0);

        Real2nd L = u(Channel::AngularMomentum);
        Real2nd J = n * R * R; // Normalisation of the moment of inertia includes the m_i
        Real2nd omega = L / J;

        return ParallelCurrent<Real2nd>(V, omega, n, Ti, Te, phi);
    };

    Real2ndVector u2(nVars);

    Real2nd phi2 = phi.val;

    Real2nd V2 = V.val;

    for (Index i = 0; i < nVars; ++i)
    {
        u2(i).val.val = u(i).val;
    }

    // take derivatives wrt V, phi, and u

    auto [_, dJpardV, d2JpardV2] = derivatives(Jpar, wrt(V2, V2), at(u2, phi2, V2));

    auto [__, dJpardphi1, d2Jpardphi12] = derivatives(Jpar, wrt(phi2, phi2), at(u2, phi2, V2));

    RealVector dJpardu(nVars);
    Real2nd ___;
    auto d2Jpardu2 = hessian(Jpar, wrt(u2), at(u2, phi2, V2), ___, dJpardu);

    Real n = uToDensity(u(Channel::Density)), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real Ti = p_i / n;

    Real nPrime = qToDensityGradient(q(Channel::Density), u(Channel::Density)), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

    // set all of the autodiff gradients
    // V parts
    Real dJpardV_real = dJpardV;

    if (V.grad != 0)
        dJpardV_real.grad = d2JpardV2;

    // u/q parts
    Real qdotdJdu = 0.0;
    for (Index i = 0; i < nVars; ++i)
    {
        if (u(i).grad != 0)
            dJpardu(i).grad = d2Jpardu2(i, i);
        qdotdJdu += q(i) * dJpardu(i);
    }

    // phi1 parts
    Real dJpardphi1_real = dJpardphi1;

    if (phi.grad != 0)
        dJpardphi1_real.grad = d2Jpardphi12;

    // dphi1dV computed using the chain rule derivative of the parallel current
    Real dphi1dV = -(dJpardV_real + qdotdJdu) * Ti / dJpardphi1_real + phi * Ti_prime;

    return dphi1dV;
}