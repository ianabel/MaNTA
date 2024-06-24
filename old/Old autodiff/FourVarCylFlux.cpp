#include "FourVarCylFlux.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(FourVarCylFlux);

enum
{
    None = 0,
    Gaussian = 1,
};

FourVarCylFlux::FourVarCylFlux( toml::value const &config, Grid const& grid )
	: AutodiffTransportSystem( config, grid, 4, 0 )
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

};


Real FourVarCylFlux::Flux( Index i, RealVector u, RealVector q, Position x, Time t )
{
	Channel c = static_cast<Channel>(i);
	switch(c) {
		case Density:
			return Gamma_hat( u, q, x, t );
			break;
		case ElectronEnergy:
			return qe_hat( u, q, x, t );
			break;
		case IonEnergy:
			return qi_hat( u, q, x, t );
			break;
		case AngularMomentum:
			return hi_hat( u, q, x, t );
			break;
		default:
			throw std::runtime_error("Request for flux for undefined variable!");
	}
}

Real FourVarCylFlux::Source( Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t )
{
	Channel c = static_cast<Channel>(i);
	switch(c) {
		case Density:
			return Sn_hat( u, q, sigma, x, t );
			break;
		case ElectronEnergy:
			return Spe_hat( u, q, sigma, x, t );
			break;
		case IonEnergy:
			return Spi_hat( u, q, sigma, x, t );
			break;
		case AngularMomentum:
			return Shi_hat( u, q, sigma, x, t );
			break;
		default:
			throw std::runtime_error("Request for source for undefined variable!");
	}
}


Real FourVarCylFlux::Gamma_hat(RealVector u, RealVector q, Real x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    Real G = 2 * x * u(1) / tau_hat(u(0), u(1)) * ((q(1) / 2 - q(2)) / u(1) - 3. / 2. * q(0) / u(0));

    if (G != G)
        return 0;
    else
        return -G;
};

Real FourVarCylFlux::qi_hat(RealVector u, RealVector q, Real x, double t)
{
    Real G = Gamma_hat(u, q, x, t);
    Real qri = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2))) * 2. * u(2) * u(2) / u(0) * (q(2) / u(2) - q(0) / u(0));
    Real Q = (2. / 3.) * (5. / 2. * u(2) / u(0) * G + (2. * x) * qri);
    if (Q != Q)
    {

        return 0;
    }
    else

        return Q;
}
Real FourVarCylFlux::hi_hat(RealVector u, RealVector q, Real x, double t)
{
    Real G = Gamma_hat(u, q, x, t);
    Real ghi = ::pow(ionMass / electronMass, 1. / 2.) * 1.0 / (::sqrt(2) * tau_hat(u(0), u(2))) * 3. / 10. * u(3) * u(2) / u(1) * (q(3) / u(3) - q(0) / u(0));
    Real H = u(3) * G / u(0) + sqrt(2. * x) * ghi;
    if (H != H)
    {
        return 0.0;
    }
    else
    {
        return H;
    }
};
Real FourVarCylFlux::qe_hat(RealVector u, RealVector q, Real x, double t)
{
    Real G = Gamma_hat(u, q, x, t);
    Real qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    Real Q = (2. / 3.) * (5. / 2. * u(1) / u(0) * G + (2. * x) * qre);
    if (Q != Q)
    {

        return 0;
    }
    else

        return Q;
};

Real FourVarCylFlux::Sn_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real S = 0.0;
    switch (ParticleSource)
    {
    case None:
        break;
    case Gaussian:
        S = -sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        break;
    default:
        break;
    }
    return S;
};

// look at ion and electron sources again -- they should be opposite
Real FourVarCylFlux::Spi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real coef = (E0 / Bmid) * (E0 / Bmid) / (V0 * Om_i(Bmid) * Om_i(Bmid) * taui0);
    Real G = Gamma_hat(u, q, x, t) / (2. * x);
    Real V = G / u(0); //* L / (p0);
    Real dV = u(3) / u(0) * (q(3) / u(3) - q(0) / u(0));
    Real Svis = (2. * x) * coef * 3. / 10. * u(2) * 1 / tau_hat(u(0), u(2)) * dV * dV;
    Real col = -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real S = 2. / 3. * sqrt(2. * x) * V * q(2) + col - 2. / 3. * Svis;
    return S;
}
Real FourVarCylFlux::Shi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real G = Gamma_hat(u, q, x, t) / (2. * x);
    Real V = G / u(0);
    Real coef = e_charge * Bmid * Bmid / (ionMass * E0);
    Real S = 1. / sqrt(2. * x) * (V * u(3) - J0 * coef);

    return S;
};
Real FourVarCylFlux::Spe_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    // Real G = Gamma_hat(u, q, x, t) / (2. * x);
    //  Real V = G / u(0); //* L / (p0);

    Real S = 0; // /*2. / 3. * sqrt(2. * x) * V * q(1) */ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);

    return S;
};
