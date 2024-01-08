#include "CylPlasma.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(CylPlasma);

CylPlasma::CylPlasma( toml::value const &config, Grid const& grid )
	: AutodiffTransportSystem( config, grid, 4, 0 )
{
    if (config.count("CylPlasma") != 1)
        throw std::invalid_argument("There should be a [CylPlasma] section if you are using the Cylindrical Plasma physics model.");

    auto const &DiffConfig = config.at("CylPlasma");

    // reference values
    n0 = toml::find_or(DiffConfig, "n0", 3e19);
    T0 = e_charge * toml::find_or(DiffConfig, "T0", 1e3);
    Bmid = toml::find_or(DiffConfig, "Bmid", 1.0);
    Om0 = toml::find_or(DiffConfig, "Omega0", 1e5);

    p0 = n0 * T0;

    Gamma0 = p0 / (electronMass * Om_e(Bmid) * Om_e(Bmid) * tau_e(n0, p0));
    V0 = Gamma0 / n0;

    taue0 = tau_e(n0, p0);
    taui0 = tau_i(n0, p0);
    h0 = ionMass * n0 * E0 / Bmid;

	B = new StraightMagneticField();

};

Real CylPlasma::Flux( Index i, RealVector u, RealVector q, Position x, Time t )
{
	Channel c(i);
	switch(c) {
		case Channel::Density:
			return Gamma( u, q, x, t );
			break;
		case Channel::ElectronEnergy:
			return qe( u, q, x, t );
			break;
		case Channel::IonEnergy:
			return qi( u, q, x, t );
			break;
		case Channel::AngularMomentum:
			return Pi( u, q, x, t );
			break;
		default:
			throw std::runtime_error("Request for flux for undefined variable!");
	}
}

Real CylPlasma::Source( Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t )
{
	Channel c(i);
	switch(c) {
		case Channel::Density:
			return Sn( u, q, sigma, x, t );
			break;
		case Channel::ElectronEnergy:
			return Spe( u, q, sigma, x, t );
			break;
		case Channel::IonEnergy:
			return Spi( u, q, sigma, x, t );
			break;
		case Channel::AngularMomentum:
			return Somega( u, q, sigma, x, t );
			break;
		default:
			throw std::runtime_error("Request for source for undefined variable!");
	}
}

/*
Normalisation:
   All lengths to a, densities to n0, temperatures to T0
   We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2 tau_e(n0,T0) ) ]^-1
	in effect we are normalising to the particle diffusion time across a distance 1

 */

// Return this normalised to log Lambda at n0,T0
inline Real CylPlasma::LogLambda_ei( Real, Real ) const
{
	return 1.0; // really needs to know Ti as well
}

// Return this normalised to log Lambda at n0,T0
inline Real CylPlasma::LogLambda_ii( Real ni, Real Ti ) const
{
	return 1.0; // 
}

// Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
// This is equal to tau_e as used in Braginskii
inline Real CylPlasma::ElectronCollisionTime( Real ne, Real Te ) const
{
	return pow( Te, 1.5 )/( ne * LogLambda_ei( ne, Te ) );
}

// Return the actual value in SI units
inline double CylPlasma::ReferenceElectronCollisionTime() const
{
	double LogLambdaRef = 24.0 - log( n0 )/2.0 + log( T0 ); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
	return 12.0 * pow( M_PI, 1.5 ) * sqrt( ElectronMass ) * pow( T0, 1.5 ) * VacuumPermittivity * VacuumPermittivity / ( sqrt( 2 ) * n0 * pow( ElementaryCharge, 4 ) * LogLambdaRef );
}
// Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
// This is equal to tau_i as used in Braginskii
inline Real CylPlasma::IonCollisionTime( Real ni, Real Ti ) const
{
	return pow( Ti, 1.5 )/( ni * LogLambda_ii( ni, Ti ) );
}

// Return the actual value in SI units
inline double CylPlasma::ReferenceIonCollisionTime() const
{
	double LogLambdaRef = 23.0 - log( 2.0 ) - log( n0 )/2.0 + log( T0 ) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
	return 12.0 * pow( M_PI, 1.5 ) * sqrt( IonMass ) * pow( T0, 1.5 ) * VacuumPermittivity * VacuumPermittivity / ( n0 * pow( ElementaryCharge, 4 ) * LogLambdaRef );
}

// We are in a quasineutral plasma with one ion species.
// This function returns V' * Gamma_e, and Gamma_i = Gamma_e
// c.f Helander & Sigmar -- Gamma_e = (n_e T_e / (m_e Omega_e^2 tau_e))*( (p_e' + p_i')/p_e - (3/2)(T_e'/T_e)
// Define lengths so R_ref = 1
Real CylPlasma::Gamma(RealVector u, RealVector q, Real V, double t)
{
	Real n = u( Channel::Density ), p_e = ( 2./3. )*u( Channel::ElectronEnergy ), p_i = ( 2./3. )*u( Channel::IonEnergy );
	Real Te = p_e/n;
	Real nPrime = q( Channel::Density ), p_e_prime = ( 2./3. ) * q( Channel::ElectronEnergy ), p_i_prime = ( 2./3. ) * q( Channel::IonEnergy );
	Real Te_prime = ( p_e_prime - nPrime * Te )/n;

	double R = B->R_V( V );
	double GeometricFactor = ( B->VPrime( V ) * R ); // |grad psi| = R B , cancel the B with the B in Omega_e
	Real Gamma = GeometricFactor * GeometricFactor * ( p_e / ElectronCollisionTime( n, Te ) ) * ( ( p_e_prime + p_i_prime )/p_e - ( 3./2. )*( Te_prime / Te ) );

	if( std::isfinite( Gamma.val ) )
		return Gamma;
	else
		throw std::logic_error( "Non-finite value computed for the particle flux at x = " + std::to_string(V) + " and t = " + std::to_string(t) );
};

/*
	Ion classical heat flux is:

	V' q_i = - 2 V'^2 ( n_i T_i / m_i Omega_i^2 tau_i ) B^2 R^2 d T_i / d V

	( n_i T_i / m_i Omega_i^2 tau_i ) * ( m_e Omega_e_ref^2 tau_e_ref / n0 T0 ) = sqrt( m_i/2m_e ) * p_i / tau_i
*/
Real CylPlasma::qi(RealVector u, RealVector q, Real x, double t)
{
	Real n = u( Channel::Density ), p_e = ( 2./3. )*u( Channel::ElectronEnergy ), p_i = ( 2./3. )*u( Channel::IonEnergy );
	Real Te = p_e/n, Ti = p_i/n;
	Real nPrime = q( Channel::Density ), p_i_prime = ( 2./3. ) * q( Channel::IonEnergy );
	Real Ti_prime = ( p_i_prime - nPrime * Ti )/n;

	double R = B->R_V( V );
	double GeometricFactor = ( B->VPrime( V ) * R ); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real q = 2.0 * GeometricFactor * GeometricFactor * sqrt( IonMass / (2.0*ElectronMass) )*( p_i / IonCollisionTime( n, Ti ) ) * Ti_prime;

	if( std::isfinite( q.val ) )
		return q;
	else
		throw std::logic_error( "Non-finite value computed for the ion heat flux at x = " + std::to_string(V) + " and t = " + std::to_string(t) );
}

/*
   Ion Classical Momentum Flux
 */
Real CylPlasma::Pi(RealVector u, RealVector q, Real x, double t)
{
};

/*
   Following Helander & Sigmar, we have
   V' q_e = n_e T_e * V'^2 B^2 R^2 * ( T_e / m_e Omega_e^2 tau_e ) * ( 4.66 T_e'/T_e - (3/2) * (p_e'+p_i')/p_e ) 
 */
Real CylPlasma::qe(RealVector u, RealVector q, Real x, double t)
{
	Real n = u( Channel::Density ), p_e = ( 2./3. )*u( Channel::ElectronEnergy ), p_i = ( 2./3. )*u( Channel::IonEnergy );
	Real Te = p_e/n, Ti = p_i/n;
	Real nPrime = q( Channel::Density ), p_e_prime = ( 2./3. ) * q( Channel::ElectronEnergy ), p_i_prime = ( 2./3. ) * q( Channel::IonEnergy );
	Real Te_prime = ( p_e_prime - nPrime * Te )/n;

	double R = B->R_V( V );
	double GeometricFactor = ( B->VPrime( V ) * R ); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real q = GeometricFactor * GeometricFactor * ( p_e / ElectronCollisionTime( n, Te ) ) * ( 4.66 * Te_prime/Te - (3./2.)*( P_e_prime + p_i_prime )/p_e );

	if( std::isfinite( q.val ) )
		return q;
	else
		throw std::logic_error( "Non-finite value computed for the electron heat flux at x = " + std::to_string(V) + " and t = " + std::to_string(t) );
};

Real CylPlasma::Sn(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
	// Put a particle source here
	Real Sn = 0.;
    return Sn;
};

Real CylPlasma::Spi(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real coef = (E0 / Bmid) * (E0 / Bmid) / (V0 * Om_i(Bmid) * Om_i(Bmid) * taui0);
    Real G = Gamma(u, q, x, t) / (2. * x);
    Real V = G / u(0); //* L / (p0);
    Real dV = u(3) / u(0) * (q(3) / u(3) - q(0) / u(0));
    Real Svis = (2. * x) * coef * 3. / 10. * u(2) * 1 / tau(u(0), u(2)) * dV * dV;
    Real col = -2. / 3. * Ci(u(0), u(2), u(1)) * L / (V0 * taue0);
    Real S = 2. / 3. * sqrt(2. * x) * V * q(2) + col - 2. / 3. * Svis;
    return S;
}

Real CylPlasma::Shi(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    Real G = Gamma(u, q, x, t) / (2. * x);
    Real V = G / u(0);
    Real coef = e_charge * Bmid * Bmid / (ionMass * E0);
    Real S = 1. / sqrt(2. * x) * (V * u(3) - J0 * coef);

    return S;
};
Real CylPlasma::Spe(RealVector u, RealVector q, RealVector sigma, Real x, double t)
{
    // Real G = Gamma(u, q, x, t) / (2. * x);
    //  Real V = G / u(0); //* L / (p0);

    Real S = 0; // /*2. / 3. * sqrt(2. * x) * V * q(1) */ -2. / 3. * Ce(u(0), u(2), u(1)) * L / (V0 * taue0);

    return S;
};
