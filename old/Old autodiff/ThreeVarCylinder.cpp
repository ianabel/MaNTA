#include "ThreeVarCylinder.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(ThreeVarCylinder);


std::map<std::string, SourceType> ParticleSources = {{"None", None}, {"Gaussian", Gaussian}};

ThreeVarCylinder::ThreeVarCylinder( toml::value const &config, Grid const& grid )
	: AutodiffTransportSystem( config, grid, 3, 0 ) // Configure a blank autodiff system with three variables and no scalars
{
    if (config.count("ThreeVarCylinder") != 1)
        throw std::invalid_argument("There should be a [ThreeVarCylinder] section if you are using the 3VarCylinder physics model.");

    auto const &DiffConfig = config.at("ThreeVarCylinder");

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

};


Real ThreeVarCylinder::Flux( Index i, RealVector u, RealVector q, Position x, Time t )
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
		default:
			throw std::runtime_error("Request for flux for undefined variable!");
	}
}

Real ThreeVarCylinder::Source( Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t )
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
		default:
			throw std::runtime_error("Request for source for undefined variable!");
	}
}

Real ThreeVarCylinder::Gamma_hat(RealVector u, RealVector q, Position x, Time t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    Real G = 2. * x * u(1) / tau_hat(u(0), u(1)) * ((-q(1) / 2. + q(2)) / u(1) + 3. / 2. * q(0) / u(0));

    if ( !std::isfinite( G.val ) )
        throw std::runtime_error("Particle flux generated Inf or NaN");
    else
        return G;
};

Real ThreeVarCylinder::qi_hat(RealVector u, RealVector q, Position x, Time t)
{
    Real dT = q(2) / u(2) - q(0) / u(0);

    Real G = Gamma_hat(u, q, x, t);
    Real qri = ::sqrt(ionMass / (2. * electronMass)) * 1.0 / tau_hat(u(0), u(2)) * 2. * u(2) * u(2) / u(0) * dT;
    Real Q = (2. / 3.) * (5. / 2. * u(2) / u(0) * G + (2. * x) * qri);

	 if( !std::isfinite( Q.val ) )
        throw std::runtime_error("Ion energy flux generated Inf or NaN");
    else
        return Q;
};

Real ThreeVarCylinder::qe_hat(RealVector u, RealVector q, Position x, Time t)
{
    Real G = Gamma_hat(u, q, x, t);
    Real qre = 1.0 / tau_hat(u(0), u(1)) * (4.66 * u(1) * u(1) / u(0) * (q(1) / u(1) - q(0) / u(0)) - (3. / 2.) * u(1) / u(0) * (q(2) + q(1)));

    Real Q = (2. / 3.) * (5. / 2. * u(1) / u(0) * G + (2. * x) * qre);

	 if( !std::isfinite( Q.val ) )
        throw std::runtime_error("Electron energy flux generated Inf or NaN");
    else
        return Q;
};

Real ThreeVarCylinder::Sn_hat(RealVector u, RealVector q, RealVector sigma, Position x, double t)
{
    Real S = 0.0;
    switch (ParticleSource)
    {
    case Gaussian:
        S = sourceStrength * exp(-1 / sourceWidth * (x - sourceCenter) * (x - sourceCenter));
        break;
    case None:
    default:
        break;
    }
    return S;
};

// look at ion and electron sources again -- they should be opposite
Real ThreeVarCylinder::Spi_hat(RealVector u, RealVector q, RealVector sigma, Position x, double t)
{
    Real S = 0.0; 

    return S + Sn_hat(u, q, sigma, x, t);
    // return 0.0;
};

Real ThreeVarCylinder::Spe_hat(RealVector u, RealVector q, RealVector sigma, Position x, double t)
{
    // Real G = -Gamma_hat(u, q, x, t);
    // Real V = G / u(0); //* L / (p0);

    Real S = 0.0; // V * q(1) + 2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);

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


