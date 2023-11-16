#include "3VarCylFlux.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_FLUX_IMPL(ThreeVarCylFlux);

enum
{
    None = 0,
    Gaussian = 1,
};

template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

ThreeVarCylFlux::ThreeVarCylFlux(toml::value const &config, Index nVars)
{

    if (config.count("3VarCylFlux") != 1)
        throw std::invalid_argument("There should be a [3VarCylFlux] section if you are using the 3VarCylFlux physics model.");

    auto const &DiffConfig = config.at("3VarCylFlux");

    std::string profile = toml::find_or(DiffConfig, "SourceType", "None");
    ParticleSource = ParticleSources[profile];

    sourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    L = toml::find_or(DiffConfig, "L", 1.0);
    p0 = n0 * T0;

    Gamma0 = p0 / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;

    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);
    sourceCenter = toml::find_or(DiffConfig, "SourceCenter", 0.25);
    sourceWidth = toml::find_or(DiffConfig, "SourceWidth", 0.01);

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = toml::find_or(DiffConfig, "T0", e_charge * 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    L = toml::find_or(DiffConfig, "L", 1.0);
    p0 = n0 * T0;

    Gamma0 = p0 / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;

    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);

    sigma.insert(std::pair<Index, sigmaptr>(0, &Gamma_hat));
    sigma.insert(std::pair<Index, sigmaptr>(1, &qe_hat));
    sigma.insert(std::pair<Index, sigmaptr>(2, &qi_hat));

    source.insert(std::pair<Index, sourceptr>(0, &Sn_hat));
    source.insert(std::pair<Index, sourceptr>(1, &Spe_hat));
    source.insert(std::pair<Index, sourceptr>(2, &Spi_hat));
};

int ThreeVarCylFlux::ParticleSource;
double ThreeVarCylFlux::sourceStrength;
dual ThreeVarCylFlux::sourceCenter;
dual ThreeVarCylFlux::sourceWidth;

// reference values
dual ThreeVarCylFlux::n0;
dual ThreeVarCylFlux::T0;
dual ThreeVarCylFlux::Bmid;
Value ThreeVarCylFlux::L;

dual ThreeVarCylFlux::p0;
dual ThreeVarCylFlux::Gamma0;
dual ThreeVarCylFlux::V0;
dual ThreeVarCylFlux::taue0;
dual ThreeVarCylFlux::taui0;

dual ThreeVarCylFlux::Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    dual G = 2. * x * u(1) / tau_hat(u(0), u(1)) * ((-q(1) / 2. + q(2)) / u(1) + 3. / 2. * q(0) / u(0));

    if (G != G)
    {
        return 0.0;
    }

    else
        return G;
};
dual ThreeVarCylFlux::qi_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual dT = q(2) / u(2) - q(0) / u(0);

    dual G = Gamma_hat(u, q, x, t);
    dual qri = ::sqrt(ionMass / (2. * electronMass)) * 1.0 / tau_hat(u(0), u(2)) * 2. * u(2) * u(2) / u(0) * dT;
    dual Q = (2. / 3.) * (5. / 2. * u(2) / u(0) * G + (2. * x) * qri);
    if ((Q != Q))
    {
        //  std::cout << Q << std::endl;
        return 0.0;
    }
    else
    {
        // std::cout << sgn(q(2).val) << std::endl;
        return Q;
    }
    // return 0.0;
};
dual ThreeVarCylFlux::qe_hat(VectorXdual u, VectorXdual q, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    dual Q = (2. / 3.) * (5. / 2. * u(1) / u(0) * G + (2. * x) * qre);
    if (Q != Q)
    {
        return 0.0;
    }
    else

        return Q;
};

dual ThreeVarCylFlux::Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
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
dual ThreeVarCylFlux::Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual V = G / u(0); //* L / (p0);
    dual S = V * q(2) - 2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S + Sn_hat(u, q, sigma, x, t);
    }
    // return 0.0;
};
dual ThreeVarCylFlux::Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual G = Gamma_hat(u, q, x, t);
    dual V = G / u(0); //* L / (p0);

    dual S = V * q(1) - 2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);

    if (S != S)
    {
        return 0.0;
    }
    else
    {
        return S;
    }
    // return 0.0;
};
