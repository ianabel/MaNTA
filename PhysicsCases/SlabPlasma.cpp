#include "SlabPlasma.hpp"
#include "Constants.hpp"

REGISTER_PHYSICS_IMPL(SlabPlasma);

const double n_edge = 1.0;
const double n_mid = 1.0;
const double T_mid = 1.0, T_edge = 0.1;

SlabPlasma::SlabPlasma(toml::value const &config, Grid const &grid) : AutodiffTransportSystem(config, grid, 2, 0)
{

    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    if (config.count("SlabPlasma") == 1)
    {
        auto const &InternalConfig = config.at("SlabPlasma");

        isDensityConstant = toml::find_or(InternalConfig, "isDensityConstant", true);
        if (!isDensityConstant)
            nVars = 3;
        uL.resize(nVars);
        uR.resize(nVars);

        isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
        isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

        useMMS = toml::find_or(InternalConfig, "useMMS", false);
        growth = toml::find_or(InternalConfig, "MMSgrowth", 1.0);
        growth_rate = toml::find_or(InternalConfig, "MMSgrowth_rate", 0.5);
        EnergyExchangeFactor = toml::find_or(InternalConfig, "EnergyExchangeFactor", 1.0);
        SourceStrength = toml::find_or(InternalConfig, "ParticleSource", 0.0);

        SelectedDensityProfile = toml::find_or(InternalConfig, "DensityProfile", "Uniform");

        includePoloidalTerm = toml::find_or(InternalConfig, "includePoloidalTerm", false);

        nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
        nMid = toml::find_or(InternalConfig, "InitialDensity", n_mid);
        TMid = toml::find_or(InternalConfig, "InitialTemperature", T_mid);
        InitialWidth = toml::find_or(InternalConfig, "InitialWidth", 0.1);
        TeEdge = toml::find_or(InternalConfig, "EdgeElectronTemperature", T_edge);
        TiEdge = toml::find_or(InternalConfig, "EdgeIonTemperature", T_edge);

        uL[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
        uR[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
        uL[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
        uR[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
        if (!isDensityConstant)
        {
            uL[Channel::Density] = nEdge;
            uR[Channel::Density] = nEdge;
        }
    }
    else if (config.count("SlabPlasma") == 0)
    {
        throw std::invalid_argument("To use the Slab Plasma physics model, a [SlabPlasma] configuration section is required.");
    }
    else
    {
        throw std::invalid_argument("Unable to find unique [SlabPlasma] configuration section in configuration file.");
    }
}

Real2nd SlabPlasma::InitialFunction(Index i, Real2nd x, Real2nd t) const
{
    Real2nd tfac = 1 + growth * tanh(growth_rate * t);
    double TeMid = TMid;
    double TiMid = TMid;

    double x_mid = 0.5 * (xL + xR);

    Real2nd v = cos(pi * (x - x_mid) / (xR - xL));

    Real2nd n = DensityProfile(x, t);
    Real2nd Te = TeEdge + tfac * (TeMid - TeEdge) * v;
    Real2nd Ti = TiEdge + tfac * (TiMid - TiEdge) * v;

    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::IonEnergy:
        return (3. / 2.) * n * Ti;
        break;
    case Channel::ElectronEnergy:
        return (3. / 2.) * n * Te;
        break;
    case Channel::Density:
    {
        if (isDensityConstant)
            throw std::runtime_error("Request for initial condition for constant variable!");
        else
            return n;
    }
    default:
        throw std::runtime_error("Request for initial value for undefined variable!");
    }
}

Real SlabPlasma::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::IonEnergy:
        return qi(u, q, x, t);
    case Channel::ElectronEnergy:
        return qe(u, q, x, t);
    case Channel::Density:
    {
        if (isDensityConstant)
            throw std::runtime_error("Request for flux for constant variable!");
        else
            return Gamma(u, q, x, t);
    }
    default:
        throw std::runtime_error("Request for flux for undefined variable!");
    };
}

Real SlabPlasma::Source(Index i, RealVector u, RealVector q, RealVector sigma, Real x, Time t)
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::IonEnergy:
        return Si(u, q, sigma, x, t);
    case Channel::ElectronEnergy:
        return Se(u, q, sigma, x, t);
    case Channel::Density:
    {
        if (isDensityConstant)
            throw std::runtime_error("Request for source for constant variable!");
        else
            return Sn(u, q, sigma, x, t);
    }
    default:
        throw std::runtime_error("Request for source for undefined variable!");
    }
}

Real2nd SlabPlasma::DensityProfile(Real2nd x, Real2nd t) const
{
    DensityType d = static_cast<DensityType>(DensityMap.at(SelectedDensityProfile));
    switch (d)
    {
    case DensityType::Uniform:
        return nEdge;
    case DensityType::Gaussian:
    {
        double x_mid = 0.5 * (xL + xR);
        Real2nd v = cos(pi * (x - x_mid) / (xR - xL));
        Real2nd n = nEdge + (nMid - nEdge) * v * exp(-1.0 / InitialWidth * (x - x_mid) * (x - x_mid));
        return n;
    }
    default:
        throw std::runtime_error("Request for invalid density profile!");
    }
}

Real SlabPlasma::DensityPrime(Real x, Real t) const
{
    DensityType d = static_cast<DensityType>(DensityMap.at(SelectedDensityProfile));
    switch (d)
    {
    case DensityType::Uniform:
        return 0.0;
    case DensityType::Gaussian:
    {
        double x_mid = 0.5 * (xL + xR);
        double k = 1.0 / InitialWidth;
        Real n = -((nMid - nEdge) * exp(-k * (x - x_mid) * (x - x_mid)) * (pi * sin((pi * (x - x_mid)) / (xR - xL)) + 2 * k * (xR - xL) * (x - x_mid) * cos((pi * (x - x_mid)) / (xR - xL)))) / (xR - xL);
        return n;
    }
    default:
        throw std::runtime_error("Request for invalid density profile!");
    }
}

Real SlabPlasma::qi(RealVector u, RealVector q, Real x, Time t)
{
    Real p_i = (2. / 3.) * u(Channel::IonEnergy);

    Real n, nPrime;
    if (isDensityConstant)
    {
        nPrime = DensityPrime(x, t);
        Real2nd nreal = DensityProfile(x, t);
        if (x.grad != 0.0)
        {
            n.val = nreal.val.val;
            n.grad = nreal.grad.val;
        }
        else
        {
            n.val = nreal.val.val;
            n.grad = 0.0;
        }
    }
    else
    {
        n = u[Channel::Density];
        nPrime = q[Channel::Density];
    }
    Real Ti = p_i / n;
    Real p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

    Real HeatFlux = 2.0 * sqrt(IonMass / (2.0 * ElectronMass)) * p_i / IonCollisionTime(n, Ti) * Ti_prime;

    if (std::isfinite(HeatFlux.val))
        return HeatFlux;
    else
        throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(x.val) + " and t = " + std::to_string(t));
}
Real SlabPlasma::qe(RealVector u, RealVector q, Real x, Time t)
{
    Real p_e = (2. / 3.) * u(Channel::ElectronEnergy);

    Real n, nPrime;
    if (isDensityConstant)
    {
        nPrime = DensityPrime(x, t);
        Real2nd nreal = DensityProfile(x, t);
        if (x.grad != 0.0)
        {
            n.val = nreal.val.val;
            n.grad = nreal.grad.val;
        }
        else
        {
            n.val = nreal.val.val;
            n.grad = 0.0;
        }
    }
    else
    {
        n = u[Channel::Density];
        nPrime = q[Channel::Density];
    }
    Real Te = p_e / n;

    Real p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Te_prime = (p_e_prime - nPrime * Te) / n;

    Real PoloidalFlowTerm = 0.0;

    if (includePoloidalTerm)
        PoloidalFlowTerm = (3. / 2.) * (p_e_prime + p_i_prime) / p_e;

    Real HeatFlux = p_e * Te / ElectronCollisionTime(n, Te) * (4.66 * Te_prime / Te - PoloidalFlowTerm);

    if (std::isfinite(HeatFlux.val))
        return HeatFlux;
    else
        throw std::logic_error("Non-finite value computed for the electron heat flux at x = " + std::to_string(x.val) + " and t = " + std::to_string(t));
}

Real SlabPlasma::Gamma(RealVector u, RealVector q, Real x, Time t)
{
    Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
    Real Te = p_e / n;
    Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Te_prime = (p_e_prime - nPrime * Te) / n;
    Real PressureGradient = ((p_e_prime + p_i_prime) / p_e);
    Real TemperatureGradient = (3. / 2.) * (Te_prime / Te);

    Real Gamma = (p_e / (ElectronCollisionTime(n, Te))) * (PressureGradient - TemperatureGradient);

    if (std::isfinite(Gamma.val))
        return Gamma;
    else
        throw std::logic_error("Non-finite value computed for the particle flux at x = " + std::to_string(x.val) + " and t = " + std::to_string(t));
}

Real SlabPlasma::Si(RealVector u, RealVector q, RealVector sigma, Real x, Time t)
{
    Real n = DensityProfile(x, t).val, p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real EnergyExchange = IonElectronEnergyExchange(n, p_e, p_i, x, t);
    return EnergyExchange;
};
Real SlabPlasma::Se(RealVector u, RealVector q, RealVector sigma, Real x, Time t)
{
    Real n = DensityProfile(x, t).val, p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real EnergyExchange = -IonElectronEnergyExchange(n, p_e, p_i, x, t);
    return EnergyExchange;
}
Real SlabPlasma::Sn(RealVector, RealVector, RealVector, Real, Time)
{
    return SourceStrength;
};

inline double SlabPlasma::RhoStarRef() const
{
    return sqrt(T0 * IonMass) / (ElementaryCharge * B0);
}

// Return this normalised to log Lambda at n0,T0
inline Real SlabPlasma::LogLambda_ei(Real ne, Real Te) const
{
    // double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5;
    // Real LogLambda = 23.0 - log(2.0) - log(ne * n0) / 2.0 + log(Te * T0) * 1.5;
    return 1.0; // LogLambdaRef / LogLambda; // really needs to know Ti as well
}

// Return this normalised to log Lambda at n0,T0
inline Real SlabPlasma::LogLambda_ii(Real ni, Real Ti) const
{
    // double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5;
    // Real LogLambda = 23.0 - log(2.0) - log(ni * n0) / 2.0 + log(Ti * T0) * 1.5;
    return 1.0; // LogLambdaRef / LogLambda; // really needs to know Ti as well
}

// Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
// This is equal to tau_e as used in Braginskii
inline Real SlabPlasma::ElectronCollisionTime(Real ne, Real Te) const
{
    return pow(Te, 1.5) / (ne * LogLambda_ei(ne, Te));
}

// Return the actual value in SI units
inline double SlabPlasma::ReferenceElectronCollisionTime() const
{
    double LogLambdaRef = 24.0 - log(n0) / 2.0 + log(T0); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
    return 12.0 * pow(M_PI, 1.5) * sqrt(ElectronMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (sqrt(2) * n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}
// Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
// This is equal to tau_i as used in Braginskii
inline Real SlabPlasma::IonCollisionTime(Real ni, Real Ti) const
{
    return pow(Ti, 1.5) / (ni * LogLambda_ii(ni, Ti));
}

// Return the actual value in SI units
inline double SlabPlasma::ReferenceIonCollisionTime() const
{
    double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
    return 12.0 * pow(M_PI, 1.5) * sqrt(IonMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}

Real SlabPlasma::IonElectronEnergyExchange(Real n, Real pe, Real pi, Real x, double t) const
{
    Real Te = pe / n;
    double RhoStar = RhoStarRef();
    Real pDiff = pe - pi;
    Real IonHeating = EnergyExchangeFactor * (pDiff / (ElectronCollisionTime(n, Te))) * ((3.0 / (RhoStar * RhoStar))); //* (ElectronMass / IonMass));

    if (std::isfinite(IonHeating.val))
        return IonHeating;
    else
        throw std::logic_error("Non-finite value computed for the ion heating at x = " + std::to_string(x.val) + " and t = " + std::to_string(t));
}

void SlabPlasma::initialiseDiagnostics(NetCDFIO &nc)
{
    AutodiffTransportSystem::initialiseDiagnostics(nc);

    nc.AddGroup("Temps", "Temperature values");

    nc.AddGroup("MMS", "Manufactured solutions");
    for (Index j = 0; j < nVars; ++j)
    {
        if (!(j == Channel::Density))
        {
            nc.AddVariable("Temps", "Var" + std::to_string(j), "Temperature value", "-", [this, j](double x)
                           { 
                        Real2nd u = this->InitialFunction(j, x, 0.0);
                        Real2nd n = this->DensityProfile(x, 0.0);
                        Real2nd T = (2. / 3. * u / n);
                        return T.val.val; });
            nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [this, j](double x)
                           { return this->InitialFunction(j, x, 0.0).val.val; });
        }
    }
}

void SlabPlasma::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{
    AutodiffTransportSystem::writeDiagnostics(y, t, nc, tIndex);

    auto n = [this, &y, t](double x)
    {
        if (isDensityConstant)
            return this->DensityProfile(x, t).val.val;
        else
            return y.u(Channel::Density)(x);
    };
    auto p_i = [this, &y](double x)
    {
        return (2. / 3.) * y.u(Channel::IonEnergy)(x);
    };
    auto p_e = [this, &y](double x)
    {
        return (2. / 3.) * y.u(Channel::ElectronEnergy)(x);
    };
    Fn T_i = [this, &p_i, &n](double x)
    {
        return p_i(x) / n(x);
    };
    Fn T_e = [this, &p_e, &n](double x)
    {
        return p_e(x) / n(x);
    };

    nc.AppendToGroup<Fn>("Temps", tIndex, {{"Var0", T_i}, {"Var1", T_e}});

    Fn IonEnergySol = [this, t](double x)
    { return this->InitialFunction(Channel::IonEnergy, x, t).val.val; };
    Fn ElectronEnergySol = [this, t](double x)
    { return this->InitialFunction(Channel::ElectronEnergy, x, t).val.val; };

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", IonEnergySol}, {"Var1", ElectronEnergySol}});
}
