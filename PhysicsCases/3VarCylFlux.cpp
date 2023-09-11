#include "3VarCylFlux.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_FLUX_IMPL(ThreeVarCylFlux);

enum
{
    None = 0,
    Gaussian = 1,
};

ThreeVarCylFlux::ThreeVarCylFlux(toml::value const &config, Index nVars)
{

    nVars = nVars;
    if (config.count("3VarCylFlux") != 1)
        throw std::invalid_argument("There should be a [3VarCylFlux] section if you are using the 3VarCylFlux physics model.");

    auto const &DiffConfig = config.at("3VarCylFlux");

    std::string profile = toml::find_or(DiffConfig, "SourceType", "None");
    ParticleSource = ParticleSources[profile];

    sourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);

    sigma.insert(std::pair<Index, sigmaptr>(0, &Gamma_hat));
    sigma.insert(std::pair<Index, sigmaptr>(1, &qe_hat));
    sigma.insert(std::pair<Index, sigmaptr>(2, &qi_hat));

    source.insert(std::pair<Index, sourceptr>(0, &Sn_hat));
    source.insert(std::pair<Index, sourceptr>(1, &Spe_hat));
    source.insert(std::pair<Index, sourceptr>(2, &Spi_hat));
};

int ThreeVarCylFlux::ParticleSource;
double ThreeVarCylFlux::sourceStrength;

const double B_mid = 0.3; // Tesla
const double Om_i = e_charge * B_mid / ionMass;
const double Om_e = e_charge * B_mid / electronMass;
const double lambda = 15.0;

const dual n0 = 3e18;
const dual T0 = e_charge * 1e3;
const dual p0 = n0 * T0;

dual nu(dual n, dual Pe)
{
    return 3.44e-11 * pow(n, 5.0 / 2.0) * lambda / pow(Pe / e_charge, 3 / 2);
}

dual Ce(dual n, dual Pi, dual Pe)
{
    return nu(n, Pe) * (Pi - Pe) / n;
}

dual Ci(dual n, dual Pi, dual Pe)
{
    return -Ce(n, Pi, Pe);
}

dual tau_e(dual n, dual Pe)
{
    if (Pe > 0)
        return (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pe, 3.0 / 2.0));
    else
        return (1.0 / n);
}

dual tau_i(dual n, dual Pi)
{
    if (Pi > 0)

        // return ::sqrt(2.) * tau_e(n, Pi) * (::sqrt(ionMass / electronMass));
        return ::sqrt(2) * (1.0 / pow(n, 5.0 / 2.0)) * (pow(Pi, 3.0 / 2.0));

    else
    {

        return ::sqrt(2) * 1.0 / (n); // if we have a negative temp just treat it as 1eV
    }
}
const dual Gamma0 = p0 / (electronMass * Om_e * Om_e * tau_e(n0, p0));
const dual V0 = Gamma0 / n0;
const Value L = 1;

dual ThreeVarCylFlux::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    dual G = 2 * x * u(1) / tau_e(u(0), u(1)) * ((q(1) / 2 - q(2)) / u(1) - 3. / 2. * q(0) / u(0));

    if (G != G)
        return 0;

    else
        return -G;
};
dual ThreeVarCylFlux::qi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual qri = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_e(u(0), u(2))) * 2. * u(2) * u(2) / u(0) * (q(2) / u(2) - q(0) / u(0));
    dual Q = (2. / 3.) * (5. / 2. * u(2) / u(0) * G + (2. * x) * qri);
    if (Q != Q)
    {

        return 0;
    }
    else

        return Q;
};
dual ThreeVarCylFlux::qe_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual qre = 1.0 / tau_e(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    dual Q = (2. / 3.) * (5. / 2. * u(1) / u(0) * G + (2. * x) * qre);
    if (Q != Q)
    {

        return 0;
    }
    else

        return Q;
};

dual ThreeVarCylFlux::Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual S = 0;
    dual Center = 0.5;
    switch (ParticleSource)
    {
    case None:
        break;
    case Gaussian:
        S = sourceStrength * exp(-(x - Center) * (x - Center));
        break;
    default:
        break;
    }
    return S;
};

// look at ion and electron sources again -- they should be opposite
dual ThreeVarCylFlux::Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = -Gamma_hat(u, q, x, t) / (2. * x);
    dual V = G / u(0); //* L / (p0);
    dual S = 2. / 3. * sqrt(2. * x) * V * q(2);
    return S;
};
dual ThreeVarCylFlux::Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = -Gamma_hat(u, q, x, t) / (2. * x);
    dual V = G / u(0); //* L / (p0);

    dual S = 2. / 3. * sqrt(2. * x) * V * q(1);
    return S;
};