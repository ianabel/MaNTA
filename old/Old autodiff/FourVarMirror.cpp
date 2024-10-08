#include "FourVarMirror.hpp"
#include "Constants.hpp"
#include <iostream>
#include <boost/math/tools/roots.hpp>

REGISTER_PHYSICS_IMPL(FourVarMirror);

const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.2, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;

enum
{
    None = 0,
    Gaussian = 1,
    Uniform = 2,
    GaussianEdge = 3,
};

enum Channel : Index
{
    Density = 0,
    ElectronEnergy = 1,
    IonEnergy = 2,
    AngularMomentum = 3,
};

template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

FourVarMirror::FourVarMirror(toml::value const &config, Grid const &grid)
    : AutodiffTransportSystem(config, grid, 4, 0)
{
    uL.resize(nVars);
    uR.resize(nVars);

    if (config.count("4VarMirror") != 1)
        throw std::invalid_argument("There should be a [4VarMirror] section if you are using the 4VarMirror physics model.");

    auto const &DiffConfig = config.at("4VarMirror");

    std::string profile = toml::find_or(DiffConfig, "SourceType", "None");
    ParticleSource = ParticleSources[profile];

    sourceStrength = toml::find_or(DiffConfig, "sourceStrength", 0.0);
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    DragFactor = toml::find_or(DiffConfig, "DragFactor", 1.0);
    DragWidth = toml::find_or(DiffConfig, "DragWidth", 0.01);

    includeParallelLosses = toml::find_or(DiffConfig, "includeParallelLosses", false);
    includeAlphas = toml::find_or(DiffConfig, "includeAlphas", false);
    includeRadiation = toml::find_or(DiffConfig, "includeRadiation", false);

    BfieldSlope = toml::find_or(DiffConfig, "MagneticFieldSlope", 0.0);
    ParallelLossFactor = toml::find_or(DiffConfig, "ParallelLossFactor", 1.0);

    Rmin = toml::find_or(DiffConfig, "Rmin", 0.1);
    Rmax = toml::find_or(DiffConfig, "Rmax", 1.0);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 1e20);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 4.5);
    double Rm = toml::find_or(DiffConfig, "Rm", 3.3);
    Bmax = Bmid.val * Rm;
    //   E0 = toml::find_or(DiffConfig, "E0", 1e5);
    L = toml::find_or(DiffConfig, "L", 1.0);
    J0 = toml::find_or(DiffConfig, "J0", 0.01);
    // normalization values
    p0 = n0 * T0;

    Gamma0 = (p0 / L) / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;
    omega0 = 1 / L * sqrt(T0 / ionMass);
    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);

    h0 = ionMass * n0 * L * L * omega0;

    nEdge = toml::find_or(DiffConfig, "EdgeDensity", n_edge);
    TeEdge = toml::find_or(DiffConfig, "EdgeElectronTemperature", T_edge);
    TiEdge = toml::find_or(DiffConfig, "EdgeIonTemperature", T_edge);
    MEdge = toml::find_or(DiffConfig, "EdgeMachNumber", omega_edge);

    InitialPeakDensity = toml::find_or(DiffConfig, "InitialDensity", n_mid);
    InitialPeakTe = toml::find_or(DiffConfig, "InitialElectronTemperature", T_mid);
    InitialPeakTi = toml::find_or(DiffConfig, "InitialIonTemperature", T_mid);
    InitialPeakMachNumber = toml::find_or(DiffConfig, "InitialMachNumber", omega_mid);

    ParallelLossFactor = toml::find_or(DiffConfig, "ParallelLossFactor", 1.0);
    DragFactor = toml::find_or(DiffConfig, "DragFactor", 1.0);
    DragWidth = toml::find_or(DiffConfig, "DragWidth", 0.01);

    double Omega_Lower = sqrt(TeEdge) * MEdge / Rmin;
    double Omega_Upper = sqrt(TeEdge) * MEdge / Rmax;

    uL[Channel::Density] = nEdge;
    uR[Channel::Density] = nEdge;
    uL[Channel::IonEnergy] = nEdge * TiEdge;
    uR[Channel::IonEnergy] = nEdge * TiEdge;
    uL[Channel::ElectronEnergy] = nEdge * TeEdge;
    uR[Channel::ElectronEnergy] = nEdge * TeEdge;
    uL[Channel::AngularMomentum] = Omega_Lower * nEdge * Rmin * Rmin;
    uR[Channel::AngularMomentum] = Omega_Upper * nEdge * Rmax * Rmax;
};

Real2nd FourVarMirror::InitialFunction(Index i, Real2nd V, Real2nd t) const
{
    Real2nd Rval = sqrt(V / (M_PI * L));
    Real2nd R_mid = (Rmin + Rmax) / 2.0;

    Real2nd nMid = InitialPeakDensity;
    Real2nd TeMid = InitialPeakTe;
    Real2nd TiMid = InitialPeakTi;
    Real2nd MMid = InitialPeakMachNumber;

    Real2nd v = cos(pi * (Rval - R_mid) / (Rmax - Rmin));

    Real2nd n = nEdge + (nMid - nEdge) * v * v;
    Real2nd Te = TeEdge + (TeMid - TeEdge) * v * v;
    Real2nd Ti = TiEdge + (TiMid - TiEdge) * v * v;

    Real2nd M = MEdge + (MMid - MEdge) * v * v * v;
    Real2nd omega = sqrt(Te) * M / Rval;

    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return n;
        break;
    case Channel::ElectronEnergy:
        return n * Te;
        break;
    case Channel::IonEnergy:
        return n * Ti;
        break;
    case Channel::AngularMomentum:
        return omega * n * Rval * Rval;
        break;
    default:
        throw std::runtime_error("Request for initial value for undefined variable!");
    }
}

Real FourVarMirror::Flux(Index i, RealVector u, RealVector q, Position x, Time t)
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Density:
        return Gamma_hat(u, q, x, t);
        break;
    case ElectronEnergy:
        return qe_hat(u, q, x, t);
        break;
    case IonEnergy:
        return qi_hat(u, q, x, t);
        break;
    case AngularMomentum:
        return hi_hat(u, q, x, t);
        break;
    default:
        throw std::runtime_error("Request for flux for undefined variable!");
    }
}

Real FourVarMirror::Source(Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t)
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Density:
        return Sn_hat(u, q, sigma, x, t);
        break;
    case ElectronEnergy:
        return Spe_hat(u, q, sigma, x, t);
        break;
    case IonEnergy:
        return Spi_hat(u, q, sigma, x, t);
        break;
    case AngularMomentum:
        return Shi_hat(u, q, sigma, x, t);
        break;
    default:
        throw std::runtime_error("Request for source for undefined variable!");
    }
}

// Function for passing boundary conditions to the solver
Value FourVarMirror::LowerBoundary(Index i, Time t) const
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return uL[Channel::Density];
        break;
    case Channel::IonEnergy:
        return uL[Channel::IonEnergy];
        break;
    case Channel::ElectronEnergy:
        return uL[Channel::ElectronEnergy];
        break;
    case Channel::AngularMomentum:
        return 0.0;
        break;
    default:
        throw std::runtime_error("Request for boundary condition for undefined variable!");
    }
}
Value FourVarMirror::UpperBoundary(Index i, Time t) const
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return uR[Channel::Density];
        break;
    case Channel::IonEnergy:
        return uR[Channel::IonEnergy];
        break;
    case Channel::ElectronEnergy:
        return uR[Channel::ElectronEnergy];
        break;
    case Channel::AngularMomentum:
        return 0.0;
        break;
    default:
        throw std::runtime_error("Request for boundary condition for undefined variable!");
    }
}

bool FourVarMirror::isLowerBoundaryDirichlet(Index i) const
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return true;
        break;
    case Channel::IonEnergy:
        return true;
        break;
    case Channel::ElectronEnergy:
        return true;
        break;
    case Channel::AngularMomentum:
        return false;
        break;
    default:
        return true;
    }
}
bool FourVarMirror::isUpperBoundaryDirichlet(Index i) const
{
    Channel c = static_cast<Channel>(i);
    switch (c)
    {
    case Channel::Density:
        return true;
        break;
    case Channel::IonEnergy:
        return true;
        break;
    case Channel::ElectronEnergy:
        return true;
        break;
    case Channel::AngularMomentum:
        return false;
        break;
    default:
        return true;
    }
}

Real FourVarMirror::Gamma_hat(RealVector u, RealVector q, Real x, double t)
{

    // maybe add a factor of sqrt x if x = r^2/2
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    Real G = coef * u(1) / (tau_hat(u(0), u(1)) * lambda_hat(u(0), u(1), n0, p0)) * ((-q(1) / 2. + q(2)) / u(1) + 3. / 2. * q(0) / u(0));

    if (!std::isfinite(G.val))
        throw std::logic_error("Particle flux returned Inf or NaN");
    else
        return G;
};

Real FourVarMirror::hi_hat(RealVector u, RealVector q, Real x, double t)
{
    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    Real G = Gamma_hat(u, q, x, t);
    Real dV = u(3) / u(0) * (q(3) / u(3) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    Real ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 3. / 10. * u(2) * dV;
    Real H = u(3) * G / u(0) + coef * ghi;
    if (!std::isfinite(H.val))
    {
        throw std::logic_error("Ion momentum flux returned Inf or NaN");
    }
    else
    {
        return H;
    }
};
Real FourVarMirror::qi_hat(RealVector u, RealVector q, Real x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // Real G = Gamma_hat(u, q, x, t);
    Real qri = ::sqrt(ionMass / (2 * electronMass)) * 1.0 / (tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 2. * u(2) * u(2) / u(0) * (q(2) / u(2) - q(0) / u(0));
    Real Q = (2. / 3.) * (coef * qri); // + 5. / 2. * u(2) / u(0) * G);
    if (!std::isfinite(Q.val))
    {
        throw std::logic_error("Ion heat flux returned Inf or NaN");
    }
    else
    {
        return Q;
    }
    // return 0.0;
};
Real FourVarMirror::qe_hat(RealVector u, RealVector q, Real x, double t)
{

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Rval * Rval * Vpval * Vpval;
    // Real G = Gamma_hat(u, q, x, t);
    Real qre = 1.0 / (tau_hat(u(0), u(1)) * lambda_hat(u(0), u(1), n0, p0)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    Real Q = (2. / 3.) * (coef * qre); // + 5. / 2. * u(1) / u(0) * G + coef * qre);
    if (!std::isfinite(Q.val))
    {
        throw std::logic_error("Electron heat flux returned Inf or NaN");
    }
    else
    {
        return Q;
    }
};

Real FourVarMirror::Sn_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real S = 0.0;
    Real Spast = 0.0;
    Real n = u(0) * n0;
    Real T = u(2) / u(0) * T0;
    Real TeV = T / (e_charge);
    Real Sfus = 0.0;
    if (includeAlphas)
    {

        Real R = 1e-6 * 3.68e-12 * pow(TeV / 1000, -2. / 3.) * exp(-19.94 * pow(TeV / 1000, -1. / 3.));
        Sfus = L / (n0 * V0) * R * n * n;
    }

    if (includeParallelLosses)
    {
        Real Rm = Bmax / B(x.val, t);
        Real Xe = Chi_e(u, q, x, t); // phi0(u, q, x, t) * u(0) / u(1) * (1 - 1 / Rm);
        Real coef = L / (taue0 * V0);
        Real loss = PastukhovLoss(u(0), u(1), Xe, Rm);
        Spast = coef * loss;
    }
    S = ParticleSourceFn(x, t);
    return (S - Sfus + Spast);
};

Real FourVarMirror::ParticleSourceFn(Real x, double t)
{
    Real S;
    switch (ParticleSource)
    {
    case None:
        break;
    case Gaussian:
        S = sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        break;
    case Uniform:
        S = sourceStrength;
        break;
    case GaussianEdge:
    {
        Real Rval = R(x.val, t);
        S = sourceStrength * (exp(-1 / sourceWidth * (Rval - Rmin) * (Rval - Rmin)) + exp(-1 / sourceWidth * (Rval - Rmax) * (Rval - Rmax)));
        break;
    }
    default:
        S = 0.0;
        break;
    }
    return S;
}

Real FourVarMirror::Shi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real Spast = 0.0;
    if (includeParallelLosses)
    {
        Real Rm = Bmax / B(x.val, t);
        Real Xi = Chi_i(u, q, x, t);
        Spast = L / (taui0 * V0) * 0.5 * PastukhovLoss(u(0), u(2), Xi, Rm) * u(3) / u(0);
    }
    double Rval = R(x.val, t);
    Real coef = L / (h0 * V0);

    Real omega = u(Channel::AngularMomentum) / u(Channel::Density) * 1 / (Rval * Rval);
    double shape = 1 / DragWidth;
    Real Drag = Rval * Rval * Rval * omega * omega * DragFactor * (exp(-shape * (Rval - Rmin) * (Rval - Rmin)) + exp(-shape * (Rval - Rmax) * (Rval - Rmax)));

    Real S = (J0 / Rval) * coef * B(x.val, t) * Rval * Rval;
    return S + Spast - Drag; // 100 * Sn_hat(u, q, sigma, x, t); //+ u(3) / (u(0) * Rval * Rval) * Sn_hat(u, q, sigma, x, t);
};

// look at ion and electron sources again -- they should be opposite
Real FourVarMirror::Spi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real Ppot = 0;
    Real Pvis = 0;
    Real Ppast = 0.0;

    if (includeParallelLosses)
    {

        Real Rm = Bmax / B(x.val, t);
        Real Xi = Chi_i(u, q, x, t);
        Real Spast = L / (taui0 * V0) * PastukhovLoss(u(0), u(2), Xi, Rm);
        Ppast = u(2) / u(0) * (1 + Xi) * Spast;
    }

    double Rval = R(x.val, t);
    double Vpval = Vprime(Rval);
    double coef = Vpval * Vpval * Rval * Rval * Rval * Rval;
    // double shape = 1e-3;
    // Real ZeroEdge = 1 - (exp(-shape * (Rval - Rmin) * (Rval - Rmin)) + exp(-shape * (Rval - Rmax) * (Rval - Rmax)));
    Real dV = u(3) / u(0) * (q(3) / u(3) - q(0) / u(0) - 1 / (M_PI * Rval * Rval));
    Real ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2)) * lambda_hat(u(0), u(1), n0, p0)) * 3. / 10. * u(2);
    Pvis = ghi * dV * dV * coef;

    Real J = u(0) * Rval * Rval; // Normalisation includes the m_i
    Real L = u(Channel::AngularMomentum);
    Real omega = L / J;
    Real ParticleSourceHeating = .5 * omega * omega * Rval * Rval * ParticleSourceFn(Rval, t);

    Real G = sigma(0);
    Ppot = -G * dphi0dV(u, q, x, t) + 0.5 * (pow(omega, 2) / M_PI) * G;

    Real Pcol = Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real S = (2. / 3.) * (Ppot + Pcol + Pvis + Ppast + ParticleSourceHeating);

    if (S != S)
        throw std::logic_error("Error compution ion heating sources");
    else
        return S; //+ 10 * Sn_hat(u, q, sigma, x, t); //+ ::pow(ionMass / electronMass, 1. / 2.) * u(2) / u(0) * Sn_hat(u, q, sigma, x, t);
    // return 0.0;
}

Real FourVarMirror::Spe_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real Pfus = 0.0;
    Real Pbrem = 0.0;
    Real Ppast = 0.0;
    Real Rm = Bmax / B(x.val, t);
    if (includeParallelLosses)
    {
        Real Xe = Chi_e(u, q, x, t); // phi0(u, q, x, t) * u(0) / u(1) * (1 - 1 / Rm);
        Real Spast = L / (taue0 * V0) * PastukhovLoss(u(0), u(1), Xe, Rm);
        Ppast = u(1) / u(0) * (1 + Xe) * Spast;
    }

    //
    Real n = u(0) * n0;
    Real T = u(1) / u(0) * T0;
    Real TeV = T / (e_charge);

    if (includeRadiation)
    {
        Pbrem = 2 * -1e6 * 1.69e-32 * (n * n) * 1e-12 * sqrt(TeV); //(-5.34e3 * pow(n / 1e20, 2) * pow(TkeV, 0.5)) * L / (p0 * V0);
    }
    if (includeAlphas)
    {
        Real TikeV = u(2) / u(0) / 1000 * T0 / e_charge;
        if (TikeV > 25)
            TikeV = 25;
        Real R = 3.68e-12 * pow(TikeV, -2. / 3.) * exp(-19.94 * pow(TikeV, -1. / 3.));
        // 1e-6 * n0 * n0 * R * u(0) * u(0);
        Pfus = 0.25 * sqrt(1 - 1 / Rm) * 1e6 * 5.6e-13 * n * n * 1e-12 * R; // n *n * 5.6e-13
    }
    Real Rval = R(x.val, t);
    Real J = u(0) * Rval * Rval; // Normalisation includes the m_i
    Real L = u(Channel::AngularMomentum);
    Real omega = L / J;
    Real ParticleSourceHeating = electronMass / ionMass * .5 * omega * omega * Rval * Rval * ParticleSourceFn(Rval, t);
    Real Pcol = Ce(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real S = 2. / 3. * (Pcol + Ppast + ParticleSourceHeating + L / (p0 * V0) * (Pfus + Pbrem));

    if (S != S)
        throw std::logic_error("Error computing the electron heating sources");
    else
        return S; //+ u(1) / u(0) * Sn_hat(u, q, sigma, x, t);
    // return 0.0;
};

Real FourVarMirror::phi0(RealVector u, RealVector q, Real x, double t)
{
    double Rval = R(x.val, t);
    Real Rm = Bmax / Bmid.val;
    Real Romega = u(3) / (u(0) * Rval);
    Real tau = u(2) / u(1);
    Real phi0 = 1 / (1 + tau) * (1 / Rm - 1) * Romega * Romega / 2;
    // Real phi0 = u(3) * u(3) / (2 * u(2) * u(0) * u(0) * Rval * Rval) * 1 / (1 / u(2) + 1 / u(1));
    return phi0;
}

Real FourVarMirror::dphi0dV(RealVector u, RealVector q, Real x, double t)
{
    auto phi0fn = [this](RealVector u, RealVector q, Real x, double t) -> Real
    {
        return this->phi0(u, q, x, t);
    };
    Real dphi0dV = derivative(phi0fn, wrt(x), at(u, q, x, t)); // (2 * M_PI * Rval);
    auto dphi0du = gradient(phi0fn, wrt(u), at(u, q, x, t));
    auto qi = q.begin();
    for (auto &dphi0i : dphi0du)
    {
        dphi0dV += *qi * dphi0i;
        ++qi;
    }

    //  Real dphi0dV = (q.val * dphi0du).sum();
    return dphi0dV;
}

Real FourVarMirror::Chi_e(RealVector u, RealVector q, Real x, double t)
{
    Real Rval = R(x.val, t);
    Real Rm = Bmax / B(x.val, t);
    Real tau = u(2) / u(1);
    Real omega = u(3) / u(0) * 1 / (Rval * Rval);
    Real M2 = Rval * Rval * pow(omega, 2) * u(0) / u(1);

    return 0.5 * (1 - 1 / Rm) * M2 * 1 / (tau + 1);
}

Real FourVarMirror::Chi_i(RealVector u, RealVector q, Real x, double t)
{
    Real Rval = R(x.val, t);
    Real Rm = Bmax / B(x.val, t);
    Real tau = u(2) / u(1);
    Real omega = u(3) / u(0) * 1 / (Rval * Rval);
    Real M2 = Rval * Rval * pow(omega, 2) * u(0) / u(1);
    return 0.5 * tau / (1 + tau) * (1 - 1 / Rm) * M2;
}

double FourVarMirror::psi(double R)
{
    return double();
}
double FourVarMirror::V(double R)
{
    return M_PI * R * R * L;
}
double FourVarMirror::Vprime(double R)

{
    double B = (1 + BfieldSlope * (R - Rmin));

    return 2 * M_PI * L / B;
}

double FourVarMirror::B(double x, double t)
{
    double Rval = R(x, t);
    return Bmid.val * (1 + BfieldSlope * (Rval - Rmin));
}

double FourVarMirror::R(double x, double t) const
{
    return sqrt(x / (M_PI * L));
    // using boost::math::tools::bracket_and_solve_root;
    // using boost::math::tools::eps_tolerance;
    // double guess = 0.5; // Rough guess is to divide the exponent by three.
    // // double min = Rmin;                                      // Minimum possible value is half our guess.
    // // double max = Rmax;                                      // Maximum possible value is twice our guess.
    // const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
    // int get_digits = static_cast<int>(digits * 0.6);        // Accuracy doubles with each step, so stop when we have
    //                                                         // just over half the digits correct.
    // double factor = 2;
    // bool is_rising = true;
    // auto getPair = [this](double x, double R)
    // { return this->V(R) - x; }; // change to V(psi(R))

    // auto func = std::bind_front(getPair, x);
    // eps_tolerance<double> tol(get_digits);

    // const boost::uintmax_t maxit = 20;
    // boost::uintmax_t it = maxit;
    // std::pair<double, double> r = bracket_and_solve_root(func, guess, factor, is_rising, tol, it);
    // return r.first + (r.second - r.first) / 2;
};

void FourVarMirror::initialiseDiagnostics(NetCDFIO &nc)
{
    // Add diagnostics here

    const std::function<double(const double &)> initialZero = [](const double &V)
    { return 0.0; };

    nc.AddGroup("Heating", "Separated heating sources");
    nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", initialZero);
    nc.AddVariable("Heating", "ElectronParallelHeatLosses", "Electron parallel heat losses", "-", initialZero);
}

void FourVarMirror::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [this, &y, &t](double V)
    {
        Real n = y.u(0)(V), p_i = y.u(2)(V), p_e = y.u(1)(V);
        Real dn = y.q(0)(V);
        Real L = y.u(3)(V);
        Real dL = y.q(3)(V);

        double Rval = R(V, t);
        double Vpval = Vprime(Rval);
        double coef = Vpval * Vpval * Rval * Rval * Rval * Rval;

        Real dV = L / n * (dL / L - dn / n - 1 / (M_PI * Rval * Rval));
        Real ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(n, p_i) * lambda_hat(n, p_e, n0, p0)) * 3. / 10. * p_i;
        Real Pvis = ghi * dV * dV * coef;

        double Heating = Pvis.val;
        return Heating;
    };

    Fn ElectronParallelHeatLosses = [this, &y, &t](double V)
    {
        Real n = y.u(0)(V), p_i = y.u(2)(V), p_e = y.u(1)(V);
        Real L = y.u(3)(V);

        Real Rval = R(V, t);
        Real Rm = Bmax / B(V, t);
        Real tau = p_i / p_e;
        Real omega = L / n * 1 / (Rval * Rval);
        Real M2 = Rval * Rval * pow(omega, 2) * n / p_e;

        Real Xe = 0.5 * (1 - 1 / Rm) * M2 * 1 / (tau + 1);

        Real Spast = L / (taue0 * V0) * PastukhovLoss(n, p_e, Xe, Rm);
        Real Ppast = p_e / n * (1 + Xe) * Spast;

        return Ppast.val;
    };

    // Add the appends for the heating stuff
    nc.AppendToGroup<Fn>("Heating", tIndex, {{"ViscousHeating", ViscousHeating}, {"ElectronParallelHeatLosses", ElectronParallelHeatLosses}});
}
