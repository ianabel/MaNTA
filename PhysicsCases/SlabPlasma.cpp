#include "SlabPlasma.hpp"
#include "Constants.hpp"

REGISTER_PHYSICS_IMPL(SlabPlasma);

const double n_edge = 1.0;
const double T_mid = 1.0, T_edge = 1e-2;

SlabPlasma::SlabPlasma(toml::value const &config, Grid const &grid) : AutodiffTransportSystem(config, grid, 2, 0)
{

    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    if (config.count("SlabPlasma") == 1)
    {
        auto const &InternalConfig = config.at("SlabPlasma");

        uL.resize(nVars);
        uR.resize(nVars);

        useMMS = toml::find_or(InternalConfig, "useMMS", false);
        growth = toml::find_or(InternalConfig, "MMSgrowth", 1.0);
        growth_rate = toml::find_or(InternalConfig, "MMSgrowth_rate", 0.5);
        EnergyExchangeFactor = toml::find_or(InternalConfig, "EnergyExchangeFactor", 1.0);

        nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
        InitialHeight = toml::find_or(InternalConfig, "InitialHeight", T_mid);
        InitialWidth = toml::find_or(InternalConfig, "InitialWidth", 0.1);
        TeEdge = toml::find_or(InternalConfig, "EdgeElectronTemperature", T_edge);
        TiEdge = toml::find_or(InternalConfig, "EdgeIonTemperature", T_edge);

        uL[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
        uR[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
        uL[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
        uR[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
    }
}

Real2nd SlabPlasma::InitialFunction(Index i, Real2nd x, Real2nd t) const
{
    Real2nd tfac = 1 + growth * tanh(growth_rate * t);
    double TeMid = InitialHeight;
    double TiMid = InitialHeight;

    double x_mid = 0.5 * (xL + xR);

    Real2nd v = cos(pi * (x - x_mid) / (xR - xL));

    Real2nd n = Density(x, t);
    Real2nd Te = TeEdge + tfac * (TeMid - TeEdge) * v * exp(-1 / InitialWidth * (x - x_mid) * (x - x_mid));
    Real2nd Ti = TiEdge + tfac * (TiMid - TiEdge) * v * exp(-1 / InitialWidth * (x - x_mid) * (x - x_mid));

    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::IonEnergy:
        return (3. / 2.) * n * Ti;
        break;
    case Channel::ElectronEnergy:
        return (3. / 2.) * n * Te;
        break;
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
    default:
        throw std::runtime_error("Request for source for undefined variable!");
    }
}

Real SlabPlasma::Density(Real2nd, Real2nd) const
{
    return nEdge;
}

Real SlabPlasma::DensityPrime(Real x, Real t) const
{
    return autodiff::derivative([this](Real x, Real t)
                                { return this->Density(x, t); }, wrt(x), at(x, t));
}

Real SlabPlasma::qi(RealVector u, RealVector q, Real x, Time t)
{
    Real n = Density(x, t), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real Ti = p_i / n;
    Real nPrime = DensityPrime(x, t);
    Real p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

    Real HeatFlux = 2.0 * sqrt(IonMass / (2.0 * ElectronMass)) * (p_i / (IonCollisionTime(n, Ti))) * Ti_prime;

    if (std::isfinite(HeatFlux.val))
        return HeatFlux;
    else
        throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(x.val) + " and t = " + std::to_string(t));
}
Real SlabPlasma::qe(RealVector u, RealVector q, Real x, Time t)
{
    Real n = Density(x, t), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
    Real Te = p_e / n;
    Real nPrime = DensityPrime(x, t);
    Real p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Te_prime = (p_e_prime - nPrime * Te) / n;

    Real HeatFlux = (p_e * Te / (ElectronCollisionTime(n, Te))) * (4.66 * Te_prime / Te - (3. / 2.) * (p_e_prime + p_i_prime) / p_e);

    if (std::isfinite(HeatFlux.val))
        return HeatFlux;
    else
        throw std::logic_error("Non-finite value computed for the electron heat flux at x = " + std::to_string(x.val) + " and t = " + std::to_string(t));
}

Real SlabPlasma::Si(RealVector u, RealVector q, RealVector sigma, Real x, Time t)
{
    Real n = Density(x, t), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real EnergyExchange = IonElectronEnergyExchange(n, p_e, p_i, x, t);
    return EnergyExchange;
};
Real SlabPlasma::Se(RealVector u, RealVector q, RealVector sigma, Real x, Time t)
{
    Real n = Density(x, t), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
    Real EnergyExchange = -IonElectronEnergyExchange(n, p_e, p_i, x, t);
    return EnergyExchange;
};

inline double SlabPlasma::RhoStarRef() const
{
    return sqrt(T0 * IonMass) / (ElementaryCharge * B0 * a);
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
    nc.AddGroup("MMS", "Manufactured solutions");
    for (Index j = 0; j < nVars; ++j)
        nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [this, j](double x)
                       { return this->InitialFunction(j, x, 0.0).val.val; });
}

void SlabPlasma::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{

    Fn IonEnergySol = [this, t](double x)
    { return this->InitialFunction(Channel::IonEnergy, x, t).val.val; };
    Fn ElectronEnergySol = [this, t](double x)
    { return this->InitialFunction(Channel::ElectronEnergy, x, t).val.val; };

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", IonEnergySol}, {"Var1", ElectronEnergySol}});
}
