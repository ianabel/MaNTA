#include "FourVarCylFlux.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_FLUX_IMPL(FourVarCylFlux);

enum
{
    None = 0,
    Gaussian = 1,
};

FourVarCylFlux::FourVarCylFlux(toml::value const &config, Index nVars)
{
    if (config.count("4VarCylFlux") != 1)
        throw std::invalid_argument("There should be a [4VarCylFlux] section if you are using the 4VarCylFlux physics model.");

    auto const &DiffConfig = config.at("4VarCylFlux");

    std::string profile = toml::find_or(DiffConfig, "SourceType", "None");
    ParticleSource = ParticleSources[profile];

    sourceStrength = toml::find_or(DiffConfig, "sourceStrength", 0.0);
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    E0 = toml::find_or(DiffConfig, "E0", 1e5);
    L = toml::find_or(DiffConfig, "L", 1.0);
    J0 = toml::find_or(DiffConfig, "J0", 0.01);

    p0 = n0 * T0;

    Gamma0 = p0 / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;

    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);
    h0 = ionMass * n0 * E0 / Bmid;
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    E0 = toml::find_or(DiffConfig, "E0", 1e5);
    L = toml::find_or(DiffConfig, "L", 1.0);
    J0 = toml::find_or(DiffConfig, "J0", 0.01);

    p0 = n0 * T0;

    Gamma0 = p0 / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;

    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);
    h0 = ionMass * n0 * E0 / Bmid;

    sigma.insert(std::pair<Index, sigmaptr>(0, &Gamma_hat));
    sigma.insert(std::pair<Index, sigmaptr>(1, &qe_hat));
    sigma.insert(std::pair<Index, sigmaptr>(2, &qi_hat));
    sigma.insert(std::pair<Index, sigmaptr>(3, &hi_hat));

    source.insert(std::pair<Index, sourceptr>(0, &Sn_hat));
    source.insert(std::pair<Index, sourceptr>(1, &Spe_hat));
    source.insert(std::pair<Index, sourceptr>(2, &Spi_hat));
    source.insert(std::pair<Index, sourceptr>(3, &Shi_hat));
};

int FourVarCylFlux::ParticleSource;
double FourVarCylFlux::sourceStrength;
dual FourVarCylFlux::sourceCenter;
dual FourVarCylFlux::sourceWidth;

// reference values
dual FourVarCylFlux::n0;
dual FourVarCylFlux::T0;
dual FourVarCylFlux::Bmid;
dual FourVarCylFlux::J0;
Value FourVarCylFlux::L;

dual FourVarCylFlux::E0;
dual FourVarCylFlux::p0;
dual FourVarCylFlux::Gamma0;
dual FourVarCylFlux::V0;
dual FourVarCylFlux::taue0;
dual FourVarCylFlux::taui0;
dual FourVarCylFlux::h0;
dual FourVarCylFlux::sourceCenter;
dual FourVarCylFlux::sourceWidth;

// reference values
dual FourVarCylFlux::n0;
dual FourVarCylFlux::T0;
dual FourVarCylFlux::Bmid;
dual FourVarCylFlux::J0;
Value FourVarCylFlux::L;

dual FourVarCylFlux::E0;
dual FourVarCylFlux::p0;
dual FourVarCylFlux::Gamma0;
dual FourVarCylFlux::V0;
dual FourVarCylFlux::taue0;
dual FourVarCylFlux::taui0;
dual FourVarCylFlux::h0;

dual FourVarCylFlux::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    dual G = 2 * x * u(1) / tau_hat(u(0), u(1)) * ((q(1) / 2 - q(2)) / u(1) - 3. / 2. * q(0) / u(0));

    if (G != G)
        return 0;
    else
        return -G;
};

dual FourVarCylFlux::qi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual qri = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2))) * 2. * u(2) * u(2) / u(0) * (q(2) / u(2) - q(0) / u(0));
    dual Q = (2. / 3.) * (5. / 2. * u(2) / u(0) * G + (2. * x) * qri);
    if (Q != Q)
    {

        return 0;
    }
    else

        return Q;
}
dual FourVarCylFlux::hi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2))) * 3. / 10. * u(3) * u(2) / u(1) * (q(3) / u(3) - q(0) / u(0));
    dual H = u(3) * G / u(0) + sqrt(2. * x) * ghi;
    if (H != H)
    {
        return 0.0;
    }
    else
    {
        return H;
    }
};
dual FourVarCylFlux::qe_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    dual Q = (2. / 3.) * (5. / 2. * u(1) / u(0) * G + (2. * x) * qre);
    if (Q != Q)
    {

        return 0;
    }
    else

        return Q;
};

dual FourVarCylFlux::Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual S = 0.0;
    switch (ParticleSource)
    {
    case None:
        break;
    case Gaussian:
        S = -sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        S = -sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        break;
    default:
        break;
    }
    return S;
};

// look at ion and electron sources again -- they should be opposite
dual FourVarCylFlux::Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual coef = (E0 / Bmid) * (E0 / Bmid) / (V0 * Om_i(Bmid) * Om_i(Bmid) * taui0);
    dual G = Gamma_hat(u, q, x, t) / (2. * x);
    dual V = G / u(0); //* L / (p0);
    dual dV = u(3) / u(0) * (q(3) / u(3) - q(0) / u(0));
    dual Svis = (2. * x) * coef * 3. / 10. * u(2) * 1 / tau_hat(u(0), u(2)) * dV * dV;
    // dual col = -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    dual S = /* 2. / 3. * sqrt(2. * x) * V * q(2) + col*/ -2. / 3. * Svis;
    return S;
}
dual FourVarCylFlux::Shi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t) / (2. * x);
    dual V = G / u(0);
    dual coef = e_charge * Bmid * Bmid / (ionMass * E0);
    dual S = 1. / sqrt(2. * x) * (V * u(3) - J0 * coef);
    dual coef = e_charge * Bmid * Bmid / (ionMass * E0);
    dual S = 1. / sqrt(2. * x) * (V * u(3) - J0 * coef);

    return S;
};
dual FourVarCylFlux::Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    // dual G = Gamma_hat(u, q, x, t) / (2. * x);
    //  dual V = G / u(0); //* L / (p0);

    dual S = 0; // /*2. / 3. * sqrt(2. * x) * V * q(1) */ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);

    return S;
};
