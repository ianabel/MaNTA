#include "MirrorPlasmaLog.hpp"
#include "Constants.hpp"
#include <iostream>
#include <string>

REGISTER_PHYSICS_IMPL(MirrorPlasmaLog);
const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.2, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;
#ifdef DEBUG
const std::string B_file = "/home/eatocco/projects/MaNTA/Bfield.nc";
#else
const std::string B_file = "Bfield.nc";
#endif

MirrorPlasmaLog::MirrorPlasmaLog(toml::value const &config, Grid const &grid)
    : AutodiffTransportSystem(config, grid, 4, 0, 0)
{

    // B = new StraightMagneticField();

    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    // isLowerDirichlet = true;
    // isUpperDirichlet = true;

    if (config.count("MirrorPlasmaLog") == 1)
    {
        auto const &InternalConfig = config.at("MirrorPlasmaLog");

        std::vector<std::string> constProfiles = toml::find_or(InternalConfig, "ConstantProfiles", std::vector<std::string>());

        for (auto &p : constProfiles)
        {
            ConstantChannelMap[ChannelMap[p]] = true;
        }

        uL.resize(nVars);
        uR.resize(nVars);

        lowerBoundaryConditions.resize(nVars);
        upperBoundaryConditions.resize(nVars);

        lowerBoundaryConditions = toml::find_or(InternalConfig, "lowerBoundaryConditions", std::vector<bool>(nVars, true));
        upperBoundaryConditions = toml::find_or(InternalConfig, "upperBoundaryConditions", std::vector<bool>(nVars, true));

        useMMS = toml::find_or(InternalConfig, "useMMS", false);
        growth = toml::find_or(InternalConfig, "MMSgrowth", 1.0);
        growth_factors = toml::find_or(InternalConfig, "growth_factors", std::vector<double>(nVars, 1.0));
        growth_rate = toml::find_or(InternalConfig, "MMSgrowth_rate", 0.5);

        // test source
        EnergyExchangeFactor = toml::find_or(InternalConfig, "EnergyExchangeFactor", 1.0);
        MaxPastukhov = toml::find_or(InternalConfig, "MaxLossRate", 1.0);

        ParallelLossFactor = toml::find_or(InternalConfig, "ParallelLossFactor", 1.0);
        ViscousHeatingFactor = toml::find_or(InternalConfig, "ViscousHeatingFactor", 1.0);
        DragFactor = toml::find_or(InternalConfig, "DragFactor", 1.0);
        DragWidth = toml::find_or(InternalConfig, "DragWidth", 0.01);
        UniformHeatSource = toml::find_or(InternalConfig, "UniformHeatSource", 0.0);
        ParticlePhysicsFactor = toml::find_or(InternalConfig, "ParticlePhysicsFactor", 1.0);
        PotentialHeatingFactor = toml::find_or(InternalConfig, "PotentialHeatingFactor", 1.0);

        std::string Bfile = toml::find_or(InternalConfig, "MagneticFieldData", B_file);

        B = new StraightMagneticField();
        // B->CheckBoundaries(xL, xR);

        R_Lower = B->R_V(xL);
        R_Upper = B->R_V(xR);

        loadInitialConditionsFromFile = toml::find_or(InternalConfig, "useNcFile", false);
        if (loadInitialConditionsFromFile)
        {
            filename = toml::find_or(InternalConfig, "InitialConditionFilename", "MirrorPlasmaLog.nc");
            LoadDataToSpline(filename);

            uL[Channel::Density] = InitialValue(Channel::Density, xL);
            uR[Channel::Density] = InitialValue(Channel::Density, xR);
            uL[Channel::IonTemperature] = InitialValue(Channel::IonTemperature, xL);
            uR[Channel::IonTemperature] = InitialValue(Channel::IonTemperature, xR);
            uL[Channel::ElectronTemperature] = InitialValue(Channel::ElectronTemperature, xL);
            uR[Channel::ElectronTemperature] = InitialValue(Channel::ElectronTemperature, xR);
            uL[Channel::AngularMomentum] = InitialValue(Channel::AngularMomentum, xL);
            uR[Channel::AngularMomentum] = InitialValue(Channel::AngularMomentum, xR);
        }
        else
        {

            nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
            TeEdge = toml::find_or(InternalConfig, "EdgeElectronTemperature", T_edge);
            TiEdge = toml::find_or(InternalConfig, "EdgeIonTemperature", TeEdge);
            MEdge = toml::find_or(InternalConfig, "EdgeMachNumber", omega_edge);

            InitialPeakDensity = toml::find_or(InternalConfig, "InitialDensity", n_mid);
            InitialPeakTe = toml::find_or(InternalConfig, "InitialElectronTemperature", T_mid);
            InitialPeakTi = toml::find_or(InternalConfig, "InitialIonTemperature", T_mid);
            InitialPeakMachNumber = toml::find_or(InternalConfig, "InitialMachNumber", omega_mid);

            DensityWidth = toml::find_or(InternalConfig, "DensityWidth", 0.05);
            double Omega_Lower = sqrt(TeEdge) * MEdge / R_Lower;
            double Omega_Upper = sqrt(TeEdge) * MEdge / R_Upper;

            uL[Channel::Density] = log(nEdge);
            uR[Channel::Density] = log(nEdge);
            uL[Channel::IonTemperature] = log(TiEdge);
            uR[Channel::IonTemperature] = log(TiEdge);
            uL[Channel::ElectronTemperature] = log(TeEdge);
            uR[Channel::ElectronTemperature] = log(TeEdge);
            uL[Channel::AngularMomentum] = Omega_Lower * nEdge * R_Lower * R_Lower;
            uR[Channel::AngularMomentum] = Omega_Upper * nEdge * R_Upper * R_Upper;
        }

        jRadial = -toml::find_or(InternalConfig, "jRadial", 4.0);
        ParticleSourceStrength = toml::find_or(InternalConfig, "ParticleSource", 10.0);
        ParticleSourceWidth = toml::find_or(InternalConfig, "ParticleSourceWidth", 0.02);
        ParticleSourceCenter = toml::find_or(InternalConfig, "ParticleSourceCenter", 0.5 * (R_Lower + R_Upper));
    }
    else if (config.count("MirrorPlasmaLog") == 0)
    {
        throw std::invalid_argument("To use the Mirror Plasma physics model, a [MirrorPlasmaLog] configuration section is required.");
    }
    else
    {
        throw std::invalid_argument("Unable to find unique [MirrorPlasmaLog] configuration section in configuration file.");
    }
};

Real2nd MirrorPlasmaLog::InitialFunction(Index i, Real2nd V, Real2nd t) const
{
    auto tfac = [this, t](double growth)
    { return 1 + growth * tanh(growth_rate * t); };
    // Real2nd tfac = 1 + growth * tanh(growth_rate * t);
    Real2nd R_min = B->R_V(xL);
    Real2nd R_max = B->R_V(xR);
    Real2nd R = B->R_V(V);
    // R.grad = B->dRdV(V.val.val);
    Real2nd R_mid = (R_min + R_max) / 2.0;

    Real2nd nMid = InitialPeakDensity;
    Real2nd TeMid = InitialPeakTe;
    Real2nd TiMid = InitialPeakTi;
    Real2nd MMid = InitialPeakMachNumber;

    Real2nd v = cos(pi * (R - R_mid) / (R_max - R_min));
    double shape = 1 / DensityWidth;
    Real2nd n = nEdge + tfac(growth_factors[Channel::Density]) * (nMid - nEdge) * v * exp(-shape * (R - R_mid) * (R - R_mid));
    Real2nd Te = TeEdge + tfac(growth_factors[Channel::ElectronTemperature]) * (TeMid - TeEdge) * v * v * v;
    Real2nd Ti = TiEdge + tfac(growth_factors[Channel::IonTemperature]) * (TiMid - TiEdge) * v * v * v;
    shape = 500;
    Real2nd M = MEdge + tfac(growth_factors[Channel::AngularMomentum]) * (MMid - MEdge) * (1 - (exp(-shape * (R - R_Upper) * (R - R_Upper)) + exp(-shape * (R - R_Lower) * (R - R_Lower))));
    Real2nd omega = sqrt(Te) * M / R;

    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return log(n);
        break;
    case Channel::IonTemperature:
        return log(Ti);
        break;
    case Channel::ElectronTemperature:
        return log(Te);
        break;
    case Channel::AngularMomentum:
        return omega * n * R * R;
        break;
    default:
        throw std::runtime_error("Request for initial value for undefined variable!");
    }
}

Real MirrorPlasmaLog::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return ConstantChannelMap[Channel::Density] ? 0.0 : Gamma(u, q, x, t);
        break;
    case Channel::IonTemperature:
        return ConstantChannelMap[Channel::IonTemperature] ? 0.0 : qi(u, q, x, t);
        break;
    case Channel::ElectronTemperature:
        return ConstantChannelMap[Channel::ElectronTemperature] ? 0.0 : qe(u, q, x, t);
        break;
    case Channel::AngularMomentum:
        return ConstantChannelMap[Channel::AngularMomentum] ? 0.0 : Pi(u, q, x, t);
        break;
    default:
        throw std::runtime_error("Request for flux for undefined variable!");
    }
}

Real MirrorPlasmaLog::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t)
{
    Channel c = static_cast<Channel>(i);
    Real S;
    switch (c)
    {
    case Channel::Density:
        S = ConstantChannelMap[Channel::Density] ? 0.0 : Sn(u, q, sigma, x, t);
        break;
    case Channel::IonTemperature:
        S = ConstantChannelMap[Channel::IonTemperature] ? 0.0 : Spi(u, q, sigma, x, t);
        break;
    case Channel::ElectronTemperature:
        S = ConstantChannelMap[Channel::ElectronTemperature] ? 0.0 : Spe(u, q, sigma, x, t);
        break;
    case Channel::AngularMomentum:
        S = ConstantChannelMap[Channel::AngularMomentum] ? 0.0 : Somega(u, q, sigma, x, t);
        break;
    default:
        throw std::runtime_error("Request for flux for undefined variable!");
    }
    return S;
}

Value MirrorPlasmaLog::LowerBoundary(Index i, Time t) const
{
    return isLowerBoundaryDirichlet(i) ? uL[i] : 0.0;
}
Value MirrorPlasmaLog::UpperBoundary(Index i, Time t) const
{
    return isUpperBoundaryDirichlet(i) ? uR[i] : 0.0;
}

bool MirrorPlasmaLog::isLowerBoundaryDirichlet(Index i) const
{
    return lowerBoundaryConditions[i];
}

bool MirrorPlasmaLog::isUpperBoundaryDirichlet(Index i) const
{
    return upperBoundaryConditions[i];
}

Real2nd MirrorPlasmaLog::MMS_Solution(Index i, Real2nd x, Real2nd t)
{
    return InitialFunction(i, x, t);
}

/*
Normalisation:
   All lengths to a, densities to n0, temperatures to T0
   We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2 tau_e(n0,T0) ) ]^-1
    in effect we are normalising to the particle diffusion time across a distance 1

 */

// This is c_s / ( Omega_i * a )
// = sqrt( T0 / mi ) / ( e B0 / mi ) =  [ sqrt( T0 mi ) / ( e B0 ) ] / a
inline double MirrorPlasmaLog::RhoStarRef() const
{
    return sqrt(T0 * IonMass) / (ElementaryCharge * B0 * a);
}

// Return this normalised to log Lambda at n0,T0
inline Real MirrorPlasmaLog::LogLambda_ei(Real ne, Real Te) const
{
    // double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5;
    // Real LogLambda = 23.0 - log(2.0) - log(ne * n0) / 2.0 + log(Te * T0) * 1.5;
    return 1.0; // LogLambdaRef / LogLambda; // really needs to know Ti as well
}

// Return this normalised to log Lambda at n0,T0
inline Real MirrorPlasmaLog::LogLambda_ii(Real ni, Real Ti) const
{
    // double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5;
    // Real LogLambda = 23.0 - log(2.0) - log(ni * n0) / 2.0 + log(Ti * T0) * 1.5;
    return 1.0; // LogLambdaRef / LogLambda; // really needs to know Ti as well
}

// Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
// This is equal to tau_e as used in Braginskii
inline Real MirrorPlasmaLog::ElectronCollisionTime(Real ne, Real Te) const
{
    return pow(Te, 1.5) / (ne * LogLambda_ei(ne, Te));
}

// Return the actual value in SI units
inline double MirrorPlasmaLog::ReferenceElectronCollisionTime() const
{
    double LogLambdaRef = 24.0 - log(n0) / 2.0 + log(T0); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
    return 12.0 * pow(M_PI, 1.5) * sqrt(ElectronMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (sqrt(2) * n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}
// Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
// This is equal to tau_i as used in Braginskii
inline Real MirrorPlasmaLog::IonCollisionTime(Real ni, Real Ti) const
{
    return pow(Ti, 1.5) / (ni * LogLambda_ii(ni, Ti));
}

// Return the actual value in SI units
inline double MirrorPlasmaLog::ReferenceIonCollisionTime() const
{
    double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
    return 12.0 * pow(M_PI, 1.5) * sqrt(IonMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}

// We are in a quasineutral plasma with one ion species.
// This function returns V' * Gamma_e, and Gamma_i = Gamma_e
// c.f Helander & Sigmar -- Gamma_e = (n_e T_e / (m_e Omega_e^2 tau_e))*( (p_e' + p_i')/p_e - (3/2)(T_e'/T_e)
// Define lengths so R_ref = 1
Real MirrorPlasmaLog::Gamma(RealVector u, RealVector q, Real V, double t) const
{
    Real logTe = u(Channel::ElectronTemperature), logTi = u(Channel::IonTemperature), logn = u(Channel::Density);
    Real logTePrime = q(Channel::ElectronTemperature), logTiPrime = q(Channel::IonTemperature), lognPrime = q(Channel::Density);

    Real Te = exp(logTe), Ti = exp(logTi), n = exp(logn);

    Real R = B->R_V(V);
    Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e
    Real Gamma = GeometricFactor * GeometricFactor * (1.0 / (ElectronCollisionTime(n, Te))) * (Ti * logTiPrime - 0.5 * Te * logTePrime + (Te + Ti) * lognPrime);

    if (std::isfinite(Gamma.val))
        return Gamma;
    else
        throw std::logic_error("Non-finite value computed for the particle flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
};

/*
    Ion classical heat flux is:

    V' q_i = - 2 V'^2 ( n_i T_i / m_i Omega_i^2 tau_i ) B^2 R^2 d T_i / d V

    ( n_i T_i / m_i Omega_i^2 tau_i ) * ( m_e Omega_e_ref^2 tau_e_ref / n0 T0 ) = sqrt( m_i/2m_e ) * p_i / tau_i
*/
Real MirrorPlasmaLog::qi(RealVector u, RealVector q, Real V, double t) const
{
    Real logTi = u(Channel::IonTemperature), logn = u(Channel::Density);
    Real logTiPrime = q(Channel::IonTemperature);
    Real Ti = exp(logTi), n = exp(logn);

    Real R = B->R_V(V);
    Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
    Real HeatFlux = 2.0 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (Ti / (IonCollisionTime(n, Ti))) * logTiPrime;

    Real G = Gamma(u, q, V, t);

    if (std::isfinite(HeatFlux.val))
        return HeatFlux - G;
    else
        throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

/*
   Following Helander & Sigmar, we have
   V' q_e = n_e T_e * V'^2 B^2 R^2 * ( T_e / m_e Omega_e^2 tau_e ) * ( 4.66 T_e'/T_e - (3/2) * (p_e'+p_i')/p_e )
 */
Real MirrorPlasmaLog::qe(RealVector u, RealVector q, Real V, double t) const
{
    Real logTe = u(Channel::ElectronTemperature), logTi = u(Channel::IonTemperature), logn = u(Channel::Density);
    Real logTePrime = q(Channel::ElectronTemperature), logTiPrime = q(Channel::IonTemperature), lognPrime = q(Channel::Density);

    Real Te = exp(logTe), Ti = exp(logTi), n = exp(logn);

    Real R = B->R_V(V);
    Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
    Real HeatFlux = GeometricFactor * GeometricFactor * (1.0 / (ElectronCollisionTime(n, Te))) * (4.66 * Te * logTePrime - (3. / 2.) * (Te * logTePrime + Ti * logTiPrime + (Ti + Te) * lognPrime));

    Real G = Gamma(u, q, V, t);

    if (std::isfinite(HeatFlux.val))
        return HeatFlux - G;
    else
        throw std::logic_error("Non-finite value computed for the electron heat flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
};

/*
   Toroidal Angular Momentum Flux is given by
   Pi = Sum_s pi_cl_s + m_s omega R^2 Gamma_s
   with pi_cl_s the classical momentum flux of species s

   we only include the ions here

   The Momentum Equation is normalised by n0^2 * T0 * m_i * c_s0 / ( m_e Omega_e_ref^2 tau_e_ref )
   with c_s0 = sqrt( T0/mi )

 */
Real MirrorPlasmaLog::Pi(RealVector u, RealVector q, Real V, double t) const
{
    Real n = exp(u(Channel::Density)), Ti = exp(u(Channel::IonTemperature));
    // dOmega dV = L'/J - J' L / J^2 ; L = angular momentum / J = moment of Inertia
    Real R = B->R_V(V);

    Real J = n * R * R; // Normalisation includes the m_i
    Real nPrime = n * q(Channel::Density);
    Real dRdV = B->dRdV(V);
    Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;

    Real L = u(Channel::AngularMomentum);
    Real LPrime = q(Channel::AngularMomentum);
    Real dOmegadV = LPrime / J - JPrime * L / (J * J);

    Real Pi_v = IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) + L * Gamma(u, q, V, t);
    return Pi_v;
};

/*
   Returns V' pi_cl_i
 */
Real MirrorPlasmaLog::IonClassicalAngularMomentumFlux(Real V, Real n, Real Ti, Real dOmegadV, double t) const
{
    Real R = B->R_V(V);
    Real GeometricFactor = (B->VPrime(V) * R * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
    Real MomentumFlux = 0.3 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (n * Ti / (IonCollisionTime(n, Ti))) * dOmegadV;
    if (std::isfinite(MomentumFlux.val))
        return MomentumFlux;
    else
        return 0.0;
    // throw std::logic_error("Non-finite value computed for the ion momentum flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

// Density source normalized to the density
Real MirrorPlasmaLog::Sn(RealVector u, RealVector q, RealVector sigma, Real V, double t) const
{
    Real logTe = u(Channel::ElectronTemperature), logTi = u(Channel::IonTemperature), logn = u(Channel::Density);
    Real lognPrime = q(Channel::Density);

    Real Te = exp(logTe), Ti = exp(logTi), n = exp(logn);

    Real R = B->R_V(V);

    Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
    Real omega = u(Channel::AngularMomentum) / J;

    Real Xi = Xi_e(V, omega, n, Ti, Te);
    Real ParallelLosses = ElectronPastukhovLossRate(V, Xi, n, Te);
    Real DensitySource = ParticleSourceStrength * ParticleSource(R.val, t);

    Real p_i = 2. / 3. * n * Ti;
    Real FusionLosses = ParticlePhysicsFactor * FusionRate(n, p_i);

    Real LogFactor = sigma(Channel::Density) * lognPrime;

    Real S = log(DensitySource) - ParallelLosses - FusionLosses;
    // n

    return LogFactor + S;
};

/*
 *  Source terms in Ion heating equation
 *
 *  - pi_i * domega/dpsi + Gamma_i m_i omega^2 / B_z + Q_i
 *
 * where Q_i is the collisional heating
 */
Real MirrorPlasmaLog::Spi(RealVector u, RealVector q, RealVector sigma, Real V, double t) const
{
    Real logTe = u(Channel::ElectronTemperature), logTi = u(Channel::IonTemperature), logn = u(Channel::Density);
    Real lognPrime = q(Channel::Density), logTiPrime = q(Channel::IonTemperature);

    Real Te = exp(logTe), Ti = exp(logTi), n = exp(logn);
    // pi * d omega / d psi = (V'pi)*(d omega / d V)
    Real R = B->R_V(V);
    Real J = n * R * R; // Normalisation includes the m_i
    Real L = u(Channel::AngularMomentum);
    Real nPrime = n * lognPrime;
    Real dRdV = B->dRdV(V);
    Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;
    Real LPrime = q(Channel::AngularMomentum);
    Real dOmegadV = LPrime / J - JPrime * L / (J * J);
    Real omega = L / J;

    Real ViscousHeating = ViscousHeatingFactor * IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) * dOmegadV / (n * Ti);

    Real PotentialHeating = PotentialHeatingFactor * n * sigma(Channel::Density) * (omega * omega / B->Bz_R(R) + e_charge * B0 / sqrt(T0 * ionMass) * omega / (2 * pi) * B->Bz_R(R));

    Real p_e = 2. / 3. * n * Te, p_i = 2. / 3. * n * Ti;
    Real EnergyExchange = IonElectronEnergyExchange(n, p_e, p_i, V, t) / (n * Ti);

    Real ParticleSourceHeating = 0.0; // 0.5 * omega * omega * R * R * ParticleSource(R, t);

    Real Heating = ViscousHeating + PotentialHeating + EnergyExchange + UniformHeatSource + ParticleSourceHeating;

    Real Xi = Xi_i(V, omega, n, Ti, Te);
    Real ParticleEnergy = (1.0 + Xi);
    Real ParallelLosses = ParticleEnergy * IonPastukhovLossRate(V, Xi, n, Ti);

    Real logUiPrime = logTiPrime + lognPrime;

    Real LogFactor = sqrt(IonMass / (2 * ElectronMass)) * sigma(Channel::IonTemperature) * logUiPrime;
    Real S = Heating - ParallelLosses;
    return LogFactor + S - Sn(u, q, sigma, V, t);
}

// Energy normalisation is T0, but these return Xi_s / T_s as that is what enters the
// Pastukhov factor
inline Real MirrorPlasmaLog::Xi_i(Real V, Real omega, Real n, Real Ti, Real Te) const
{
    return CentrifugalPotential(V, omega, Ti, Te) + AmbipolarPhi(V, n, Ti, Te) / 2.0;
}

inline Real MirrorPlasmaLog::Xi_e(Real V, Real omega, Real n, Real Ti, Real Te) const
{
    return CentrifugalPotential(V, omega, Ti, Te) + AmbipolarPhi(V, n, Ti, Te) / 2.0;
}

inline Real MirrorPlasmaLog::AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const
{
    double R = B->MirrorRatio(V);
    double Sigma = 1.0;
    return log((ElectronCollisionTime(n, Te) / IonCollisionTime(n, Ti)) * (log(R * Sigma) / (Sigma * ::log(R))));
}

/*
   In SI this is

   Q_i = 3 n_e m_e ( T_e - T_i ) / ( m_i tau_e )

   Normalised for the Energy equation, whose overall normalising factor is
   n0 T0^2 / ( m_e Omega_e_ref^2 tau_e_ref )


   Q_i = 3 * (p_e - p_i) / (tau_e) * (m_e/m_i) / (rho_s/R_ref)^2

 */
Real MirrorPlasmaLog::IonElectronEnergyExchange(Real n, Real pe, Real pi, Real V, double t) const
{
    Real Te = pe / n;
    double RhoStar = RhoStarRef();
    Real pDiff = pe - pi;
    Real IonHeating = EnergyExchangeFactor * (pDiff / (ElectronCollisionTime(n, Te))) * ((3.0 / (RhoStar * RhoStar))); //* (ElectronMass / IonMass));

    if (std::isfinite(IonHeating.val))
        return IonHeating;
    else
        throw std::logic_error("Non-finite value computed for the ion heating at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

Real MirrorPlasmaLog::Spe(RealVector u, RealVector q, RealVector sigma, Real V, double t) const
{
    Real logTe = u(Channel::ElectronTemperature), logTi = u(Channel::IonTemperature), logn = u(Channel::Density);
    Real lognPrime = q(Channel::Density), logTePrime = q(Channel::ElectronTemperature);

    Real Te = exp(logTe), Ti = exp(logTi), n = exp(logn);

    Real p_e = 2. / 3. * n * Te, p_i = 2. / 3. * n * Ti;

    Real EnergyExchange = -IonElectronEnergyExchange(n, p_e, p_i, V, t) / (n * Te);

    double MirrorRatio = B->MirrorRatio(V);
    Real AlphaHeating = ParticlePhysicsFactor * sqrt(1 - 1 / MirrorRatio) * TotalAlphaPower(n, p_i) / Te;

    Real R = B->R_V(V);

    Real J = n * R * R; // Normalisation includes the m_i
    Real L = u(Channel::AngularMomentum);
    Real omega = L / J;
    // Real ParticleSourceHeating = electronMass / ionMass * .5 * omega * omega * R * R * ParticleSource(R.val, t);

    Real PotentialHeating = -PotentialHeatingFactor * e_charge * B0 / sqrt(T0 * ionMass) * sigma(Channel::Density) / Te * omega / (2 * pi) * B->Bz_R(R);

    Real Heating = EnergyExchange + AlphaHeating + PotentialHeating;

    Real Xi = Xi_e(V, omega, n, Ti, Te);
    Real ParticleEnergy = (1.0 + Xi);
    Real ParallelLosses = ParticleEnergy * ElectronPastukhovLossRate(V, Xi, n, Te);

    Real RadiationLosses = ParticlePhysicsFactor * BremsstrahlungLosses(n, p_e);

    Real logUePrime = logTePrime + lognPrime;

    Real LogFactor = sqrt(IonMass / (2 * ElectronMass)) * sigma(Channel::IonTemperature) * logUePrime;

    Real S = Heating - ParallelLosses - RadiationLosses;

    return LogFactor + S - Sn(u, q, sigma, V, t);
};

// Source of angular momentum -- this is just imposed J x B torque (we can account for the particle source being a sink later).
Real MirrorPlasmaLog::Somega(RealVector u, RealVector q, RealVector sigma, Real V, double t) const
{
    // J x B torque
    Real R = B->R_V(V);
    Real JxB = -jRadial * R * B->Bz_R(R);

    Real logTe = u(Channel::ElectronTemperature), logTi = u(Channel::IonTemperature), logn = u(Channel::Density);

    Real Te = exp(logTe), Ti = exp(logTi), n = exp(logn);
    Real L = u(Channel::AngularMomentum);
    Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
    Real omega = L / J;
    Real vtheta = omega * R;

    double shape = 1 / DragWidth;
    Real Drag = (JxB + vtheta * B->Bz_R(R) * DragFactor) * (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));

    // Neglect electron momentum
    Real Xi = Xi_i(V, omega, n, Ti, Te);
    Real ParallelLosses = L * IonPastukhovLossRate(V, Xi, n, Te);

    return JxB - ParallelLosses - Drag;
};

Real MirrorPlasmaLog::ParticleSource(double R, double t) const
{
    double shape = 1 / ParticleSourceWidth;
    // return (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));
    return ParticleSourceStrength * (exp(-shape * (R - ParticleSourceCenter) * (R - ParticleSourceCenter)));
    //   return ParticleSourceStrength;
};

Real MirrorPlasmaLog::ElectronPastukhovLossRate(Real V, Real Xi_e, Real n, Real Te) const
{
    double MirrorRatio = B->MirrorRatio(V);
    Real tau_ee = ElectronCollisionTime(n, Te);
    double Sigma = 2.0; // = 1 + Z_eff ; Include collisions with ions and impurities as well as self-collisions
    double delta = (1 - ZeroEdgeFactor) * (xR - xL);
    Real PastukhovFactor = (exp(-Xi_e) / Xi_e);
    // // Cap loss rates
    // // if ((PastukhovFactor > MaxPastukhov) && ((R < (R_Lower + delta)) || (R > (R_Upper - delta))))
    // // 	PastukhovFactor = MaxPastukhov;
    // if (PastukhovFactor > MaxPastukhov)
    // {
    //     double slope = 1.0;
    //     Real R = B->R_V(V);
    //     double rl = R_Lower + delta;
    //     double rr = R_Upper - delta;
    //     double delta2 = (1 - MaxPastukhov) / slope + delta;
    //     if ((R < rl) || (R > rr))
    //     {
    //         PastukhovFactor = MaxPastukhov;
    //     }
    //     else if (R < (rl + delta2))
    //     {
    //         PastukhovFactor = MaxPastukhov + slope * (R - rl);
    //     }
    //     else if (R > (rr - delta2))
    //     {
    //         PastukhovFactor = MaxPastukhov - slope * (R - rr);
    //     }
    // }
    // If the loss becomes a gain, flatten at zero
    if (PastukhovFactor.val < 0.0)
        return 0.0;
    double Normalization = (IonMass / ElectronMass) * (1.0 / (RhoStarRef() * RhoStarRef()));
    Real LossRate = (M_2_SQRTPI / tau_ee) * Normalization * Sigma * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;
    return ParallelLossFactor * LossRate;
}

Real MirrorPlasmaLog::IonPastukhovLossRate(Real V, Real Xi_i, Real n, Real Ti) const
{
    // For consistency, the integral in Pastukhov's paper is 1.0, as the
    // entire theory is an expansion in M^2 >> 1
    double MirrorRatio = B->MirrorRatio(V);
    Real tau_ii = IonCollisionTime(n, Ti);
    double Sigma = 1.0; // Just ion-ion collisions
    double delta = (1 - ZeroEdgeFactor) * (R_Upper - R_Lower);
    Real PastukhovFactor = (exp(-Xi_i) / Xi_i);

    // // Cap loss rates
    // if (PastukhovFactor > MaxPastukhov)
    // {
    //     double slope = 1.0;
    //     Real R = B->R_V(V);
    //     double rl = R_Lower + delta;
    //     double rr = R_Upper - delta;
    //     double delta2 = (1 - MaxPastukhov) / slope + delta;
    //     if ((R < rl) || (R > rr))
    //     {
    //         PastukhovFactor = MaxPastukhov;
    //     }
    //     else if (R < (rl + delta2))
    //     {
    //         PastukhovFactor = MaxPastukhov + slope * (R - rl);
    //     }
    //     else if (R > (rr - delta2))
    //     {
    //         PastukhovFactor = MaxPastukhov - slope * (R - rr);
    //     }
    // }
    // If the loss becomes a gain, flatten at zero
    if (PastukhovFactor.val < 0.0)
        return 0.0;

    double Normalization = sqrt(IonMass / ElectronMass) * (1.0 / (RhoStarRef() * RhoStarRef()));
    Real LossRate = (M_2_SQRTPI / tau_ii) * Normalization * Sigma * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;

    return ParallelLossFactor * LossRate;
}

// Returns (1/(1 + Tau))*(1-1/R_m)*(M^2)
Real MirrorPlasmaLog::CentrifugalPotential(Real V, Real omega, Real Ti, Real Te) const
{
    double MirrorRatio = B->MirrorRatio(V);
    Real R = B->R_V(V);
    Real tau = Ti / Te;
    Real MachNumber = omega * R / sqrt(Te); // omega is normalised to c_s0 / a
    Real Potential = (1.0 / (1.0 + tau)) * (1.0 - 1.0 / MirrorRatio) * MachNumber * MachNumber / 2.0;
    return Potential;
}
// Implements D-T fusion rate from NRL plasma formulary

// normalized to n
Real MirrorPlasmaLog::FusionRate(Real n, Real pi) const
{

    Real Normalization = n0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
    Real Factor = 1e-6 * 3.68e-12 * n0 * n0 / Normalization;
    Real Ti_kev = (pi / n) * T0 / (1000 * ElementaryCharge);
    Real R = Factor * 0.25 * n * pow(Ti_kev, -2. / 3.) * exp(-19.94 * pow(Ti_kev, -1. / 3.));

    return R;
}

Real MirrorPlasmaLog::TotalAlphaPower(Real n, Real pi) const
{
    double Factor = 5.6e-13 / (T0);
    Real AlphaPower = Factor * FusionRate(n, pi);
    return AlphaPower;
}

// Implements Bremsstrahlung radiative losses from NRL plasma formulary
Real MirrorPlasmaLog::BremsstrahlungLosses(Real n, Real pe) const
{
    double p0 = n0 * T0;
    Real Normalization = p0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
    Real Factor = 2 * 1.69e-32 * 1e-6 * n0 * n0 / Normalization;

    Real Te_eV = (pe / n) * T0 / ElementaryCharge;
    Real Pbrem = Factor * n * n * sqrt(Te_eV);

    return Pbrem;
}

// omega & n are callables
template <typename T1, typename T2>
double MirrorPlasmaLog::Voltage(T1 &L_phi, T2 &n)
{
    auto integrator = boost::math::quadrature::gauss<double, 15>();
    auto integrand = [this, &L_phi, &n](double V)
    {
        double R = B->R_V(V);
        return L_phi(V) / (n(V) * R * R * B->VPrime(V));
    };
    double cs0 = std::sqrt(T0 / IonMass);
    return cs0 * integrator.integrate(integrand, xL, xR);
}

void MirrorPlasmaLog::initialiseDiagnostics(NetCDFIO &nc)
{
    if (useMMS)
        AutodiffTransportSystem::initialiseDiagnostics(nc);
    // Add diagnostics here
    //
    double RhoStar = RhoStarRef();
    double TauNorm = (IonMass / ElectronMass) * (1.0 / (RhoStar * RhoStar)) * (ReferenceElectronCollisionTime());
    nc.AddScalarVariable("Tau", "Normalising time", "s", TauNorm);

    auto L = [this](double V)
    {
        return InitialValue(Channel::AngularMomentum, V);
    };
    auto LPrime = [this](double V)
    {
        return InitialDerivative(Channel::AngularMomentum, V);
    };
    auto n = [this](double V)
    {
        return exp(InitialValue(Channel::Density, V));
    };
    auto nPrime = [this, &n](double V)
    {
        return n(V) * InitialDerivative(Channel::Density, V);
    };
    auto p_i = [this, &n](double V)
    {
        return (2. / 3.) * n(V) * exp(InitialValue(Channel::IonTemperature, V));
    };
    auto p_e = [this, &n](double V)
    {
        return (2. / 3.) * n(V) * exp(InitialValue(Channel::ElectronTemperature, V));
    };

    double initialVoltage = Voltage(L, n);

    const std::function<double(const double &)> initialZero = [](const double &V)
    { return 0.0; };

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = B->dRdV(V);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = this->IonClassicalAngularMomentumFlux(V, n(V), Ti, dOmegadV, 0).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [this, &n, &p_i](double V)
    {
        double MirrorRatio = this->B->MirrorRatio(V);

        double Heating = sqrt(1 - 1 / MirrorRatio) * this->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [this, &n, &p_e](double V)
    {
        double Losses = this->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ParallelLosses = [this, &n, &p_i, &p_e, &L](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xe = Xi_e(V, omega, n(V), Ti, Te);

        double ParallelLosses = ElectronPastukhovLossRate(V, Xe, n(V), Te).val / ParallelLossFactor;

        return ParallelLosses;
    };

    Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        double Phi0 = Xi_i(V, omega, n(V), Ti, Te).val;

        return Phi0;
    };

    Fn EnergyExchange = [this, &n, &p_i, &p_e](double V)
    {
        return IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
    };

    nc.AddTimeSeries("Voltage", "Total voltage drop across the plasma", "Volts", initialVoltage);
    nc.AddGroup("MMS", "Manufactured solutions");
    for (int j = 0; j < nVars; ++j)
        nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [this, j](double V)
                       { return this->InitialFunction(j, V, 0.0).val.val; });
    nc.AddGroup("MomentumFlux", "Separating momentum fluxes");
    nc.AddGroup("ParallelLosses", "Separated parallel losses");
    nc.AddVariable("ParallelLosses", "ParLoss", "Parallel particle losses", "-", ParallelLosses);
    nc.AddVariable("ParallelLosses", "CentrifugalPotential", "Centrifugal potential", "-", Phi0);
    nc.AddGroup("Heating", "Separated heating sources");
    nc.AddVariable("Heating", "AlphaHeating", "Alpha heat source", "-", AlphaHeating);
    nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", ViscousHeating);
    nc.AddVariable("Heating", "RadiationLosses", "Bremsstrahlung heat losses", "-", RadiationLosses);
    nc.AddVariable("Heating", "EnergyExchange", "Collisional ion-electron energy exhange", "-", EnergyExchange);
}

void MirrorPlasmaLog::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{
    if (useMMS)
        AutodiffTransportSystem::writeDiagnostics(y, t, nc, tIndex);

    auto L = [&y](double V)
    {
        return y.u(Channel::AngularMomentum)(V);
    };
    auto LPrime = [&y](double V)
    {
        return y.q(Channel::AngularMomentum)(V);
    };
    auto n = [&y](double V)
    {
        return exp(y.u(Channel::Density)(V));
    };
    auto nPrime = [&y](double V)
    {
        return y.q(Channel::Density)(V);
    };
    auto p_i = [&y, &n](double V)
    {
        return (2. / 3.) * n(V) * exp(y.u(Channel::IonTemperature)(V));
    };
    auto p_e = [&y, &n](double V)
    {
        return (2. / 3.) * n(V) * exp(y.u(Channel::ElectronTemperature)(V));
    };

    double voltage = Voltage(L, n);
    nc.AppendToTimeSeries("Voltage", voltage, tIndex);

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i, &t](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = B->dRdV(V);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = this->IonClassicalAngularMomentumFlux(V, n(V), Ti, dOmegadV, t).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [this, &n, &p_i](double V)
    {
        double MirrorRatio = this->B->MirrorRatio(V);

        double Heating = sqrt(1 - 1 / MirrorRatio) * this->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [this, &n, &p_e](double V)
    {
        double Losses = this->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ParallelLosses = [this, &n, &p_i, &p_e, &L](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xe = Xi_e(V, omega, n(V), Ti, Te);

        double ParallelLosses = ElectronPastukhovLossRate(V, Xe, n(V), Te).val / ParallelLossFactor;

        return ParallelLosses;
    };

    Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        double Phi0 = Xi_i(V, omega, n(V), Ti, Te).val;

        return Phi0;
    };

    Fn EnergyExchange = [this, &n, &p_i, &p_e](double V)
    {
        return IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
    };

    Fn DensitySol = [this, t](double V)
    { return this->InitialFunction(Channel::Density, V, t).val.val; };
    Fn IonTemperatureSol = [this, t](double V)
    { return this->InitialFunction(Channel::IonTemperature, V, t).val.val; };
    Fn ElectronTemperatureSol = [this, t](double V)
    { return this->InitialFunction(Channel::ElectronTemperature, V, t).val.val; };
    Fn AngularMomentumSol = [this, t](double V)
    { return this->InitialFunction(Channel::AngularMomentum, V, t).val.val; };
    // std::vector<std::pair<std::string, const Fn &>> MMSsols;
    // for (int j = 0; j < nVars; ++j)
    // {
    // 	MMSsols.push_back({"MMSVar" + std::to_string(j), );
    // }
    // Add the appends for the heating stuff
    nc.AppendToGroup<Fn>("Heating", tIndex, {{"AlphaHeating", AlphaHeating}, {"ViscousHeating", ViscousHeating}, {"RadiationLosses", RadiationLosses}, {"EnergyExchange", EnergyExchange}});

    nc.AppendToGroup<Fn>("ParallelLosses", tIndex, {{"ParLoss", ParallelLosses}, {"CentrifugalPotential", Phi0}});

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", DensitySol}, {"Var1", IonTemperatureSol}, {"Var2", ElectronTemperatureSol}, {"Var3", AngularMomentumSol}});
}
