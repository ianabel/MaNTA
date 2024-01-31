#include "CylPlasmaDebug.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(CylPlasmaDebug);
const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.4, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;

CylPlasmaDebug::CylPlasmaDebug(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, 4, 0)
{
	B = new StraightMagneticField();
	ParticleSourceStrength = 1.0;
	jRadial = -52.0;

	xL = grid.lowerBoundary();
	xR = grid.upperBoundary();

	R_Lower = B->R_V(xL);
	R_Upper = B->R_V(xR);

	isLowerDirichlet = true;
	isUpperDirichlet = true;

	uL.resize(nVars);
	uR.resize(nVars);
	uL[Channel::Density] = n_edge;
	uR[Channel::Density] = n_edge;
	uL[Channel::IonEnergy] = (3. / 2.) * n_edge * T_edge;
	uR[Channel::IonEnergy] = (3. / 2.) * n_edge * T_edge;
	uL[Channel::ElectronEnergy] = (3. / 2.) * n_edge * T_edge;
	uR[Channel::ElectronEnergy] = (3. / 2.) * n_edge * T_edge;
	uL[Channel::AngularMomentum] = omega_edge * n_edge * R_Lower * R_Lower;
	uR[Channel::AngularMomentum] = omega_edge * n_edge * R_Upper * R_Upper;
};

Value CylPlasmaDebug::InitialValue(Index i, Position V) const
{
	double R_min = B->R_V(xL);
	double R_max = B->R_V(xR);
	double R = B->R_V(V);
	double R_mid = (R_min + R_max) / 2.0;
	double n = n_edge + (n_mid - n_edge) * std::cos(pi * (R - R_mid) / (R_max - R_min));
	double T = T_edge + (T_mid - T_edge) * std::cos(pi * (R - R_mid) / (R_max - R_min));

	double omega = omega_edge + (omega_mid - omega_edge) * std::cos(pi * (R - R_mid) / (R_max - R_min));

	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return n;
		break;
	case Channel::IonEnergy:
	case Channel::ElectronEnergy:
		return (3. / 2.) * n * T;
		break;
	case Channel::AngularMomentum:
		return omega * n * R * R;
		break;
	default:
		throw std::runtime_error("Request for initial value for undefined variable!");
	}
}

Value CylPlasmaDebug::InitialDerivative(Index i, Position V) const
{
	double R_min = B->R_V(xL);
	double R_max = B->R_V(xR);
	double R = B->R_V(V);
	double R_mid = (R_min + R_max) / 2.0;
	double n = n_edge + (n_mid - n_edge) * std::cos(pi * (R - R_mid) / (R_max - R_min));
	double T = T_edge + (T_mid - T_edge) * std::cos(pi * (R - R_mid) / (R_max - R_min));
	double nPrime = -(pi / (R_max - R_min)) * (n_mid - n_edge) * std::sin(pi * (R - R_mid) / (R_max - R_min));
	double TPrime = -(pi / (R_max - R_min)) * (T_mid - T_edge) * std::sin(pi * (R - R_mid) / (R_max - R_min));
	double dRdV = 1. / (2.0 * pi * R);
	double omega = omega_edge + (omega_mid - omega_edge) * std::cos(pi * (R - R_mid) / (R_max - R_min));
	double omegaPrime = -(pi / (R_max - R_min)) * (omega_mid - omega_edge) * std::sin(pi * (R - R_mid) / (R_max - R_min));

	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return nPrime * dRdV;
		break;
	case Channel::IonEnergy:
	case Channel::ElectronEnergy:
		return (3. / 2.) * (nPrime * T + n * TPrime) * dRdV;
		break;
	case Channel::AngularMomentum:
		return (omegaPrime * n * R * R + omega * nPrime * R * R + 2 * omega * n * R) * dRdV;
		break;
	default:
		throw std::runtime_error("Request for initial value for undefined variable!");
	}
}

Real CylPlasmaDebug::Flux(Index i, RealVector u, RealVector q, Position x, Time t, std::vector<Position> *ExtraValues)
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return Gamma(u, q, x, t);
		break;
	case Channel::IonEnergy:
		return qi(u, q, x, t);
		break;
	case Channel::ElectronEnergy:
		return qe(u, q, x, t);
		break;
	case Channel::AngularMomentum:
		return Pi(u, q, x, t);
		break;
	default:
		throw std::runtime_error("Request for flux for undefined variable!");
	}
}

Real CylPlasmaDebug::Source(Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t, std::vector<Position> *ExtraValues)
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return Sn(u, q, sigma, x, t);
		break;
	case Channel::IonEnergy:
		return Spi(u, q, sigma, x, t);
		break;
	case Channel::ElectronEnergy:
		return Spe(u, q, sigma, x, t);
		break;
	case Channel::AngularMomentum:
		return Somega(u, q, sigma, x, t);
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

// This is c_s / ( Omega_i * a )
// = sqrt( T0 / mi ) / ( e B0 / mi ) =  [ sqrt( T0 mi ) / ( e B0 ) ] / a
inline double CylPlasmaDebug::RhoStarRef() const
{
	return sqrt(T0 * IonMass) / (ElementaryCharge * B0 * a);
}

// Return this normalised to log Lambda at n0,T0
inline Real CylPlasmaDebug::LogLambda_ei(Real, Real) const
{
	return 1.0; // really needs to know Ti as well
}

// Return this normalised to log Lambda at n0,T0
inline Real CylPlasmaDebug::LogLambda_ii(Real ni, Real Ti) const
{
	return 1.0; //
}

// Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
// This is equal to tau_e as used in Braginskii
inline Real CylPlasmaDebug::ElectronCollisionTime(Real ne, Real Te) const
{
	return pow(Te, 1.5) / (ne * LogLambda_ei(ne, Te));
}

// Return the actual value in SI units
inline double CylPlasmaDebug::ReferenceElectronCollisionTime() const
{
	double LogLambdaRef = 24.0 - log(n0) / 2.0 + log(T0); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
	return 12.0 * pow(M_PI, 1.5) * sqrt(ElectronMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (sqrt(2) * n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}
// Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
// This is equal to tau_i as used in Braginskii
inline Real CylPlasmaDebug::IonCollisionTime(Real ni, Real Ti) const
{
	return pow(Ti, 1.5) / (ni * LogLambda_ii(ni, Ti));
}

// Return the actual value in SI units
inline double CylPlasmaDebug::ReferenceIonCollisionTime() const
{
	double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
	return 12.0 * pow(M_PI, 1.5) * sqrt(IonMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}

// We are in a quasineutral plasma with one ion species.
// This function returns V' * Gamma_e, and Gamma_i = Gamma_e
// c.f Helander & Sigmar -- Gamma_e = (n_e T_e / (m_e Omega_e^2 tau_e))*( (p_e' + p_i')/p_e - (3/2)(T_e'/T_e)
// Define lengths so R_ref = 1
Real CylPlasmaDebug::Gamma(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;

	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e
	Real Gamma = GeometricFactor * GeometricFactor * (p_e / ElectronCollisionTime(n, Te)) * ((p_e_prime + p_i_prime) / p_e - (3. / 2.) * (Te_prime / Te));

	if (std::isfinite(Gamma.val))
		return Gamma;
	else
		throw std::logic_error("Non-finite value computed for the particle flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
};

/*
	Ion classical heat flux is:

	V' q_i = - 2 V'^2 ( n_i T_i / m_i Omega_i^2 tau_i ) B^2 R^2 d T_i / d V

	( n_i T_i / m_i Omega_i^2 tau_i ) * ( m_e Omega_e_ref^2 tau_e_ref / n0 T0 ) = sqrt( m_i/2m_e ) * p_i / tau_i
*/
Real CylPlasmaDebug::qi(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	Real nPrime = q(Channel::Density), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = 2.0 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (p_i / IonCollisionTime(n, Ti)) * Ti_prime;

	if (std::isfinite(HeatFlux.val))
		return HeatFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

/*
   Following Helander & Sigmar, we have
   V' q_e = n_e T_e * V'^2 B^2 R^2 * ( T_e / m_e Omega_e^2 tau_e ) * ( 4.66 T_e'/T_e - (3/2) * (p_e'+p_i')/p_e )
 */
Real CylPlasmaDebug::qe(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;

	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = GeometricFactor * GeometricFactor * (p_e / ElectronCollisionTime(n, Te)) * (4.66 * Te_prime / Te - (3. / 2.) * (p_e_prime + p_i_prime) / p_e);

	if (std::isfinite(HeatFlux.val))
		return HeatFlux;
	else
		throw std::logic_error("Non-finite value computed for the electron heat flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
};

/*
   Toroidal Angular Momentum Flux is given by
   Pi = Sum_s pi_cl_s + m_s omega R^2 Gamma_s
   with pi_cl_s the classical momentum flux of species s

   we only include the ions here

   The Momentum Equation is normalised by n0^2 * T0 * m_i * c_s0 / ( m_e Omega_e_ref^2 tau_e_ref )
   with c_s0 = sqrt( T0/mi )

 */
Real CylPlasmaDebug::Pi(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), Ti = (2. / 3.) * u(Channel::IonEnergy) / n;
	// dOmega dV = L'/J - J' L / J^2 ; L = angular momentum / J = moment of Inertia
	double R = B->R_V(V);
	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real nPrime = q(Channel::Density);
	Real JPrime = R * R * nPrime;
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;
	Real Pi_v = IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) + omega * R * R * Gamma(u, q, V, t);
	return Pi_v;
};

/*
   Returns V' pi_cl_i
 */
Real CylPlasmaDebug::IonClassicalAngularMomentumFlux(Position V, Real n, Real Ti, Real dOmegadV, double t) const
{
	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real MomentumFlux = 0.3 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (n * Ti / IonCollisionTime(n, Ti)) * dOmegadV;
	if (std::isfinite(MomentumFlux.val))
		return MomentumFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion momentum flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

Real CylPlasmaDebug::Sn(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	// See what happens with a uniform source
	return 10.0;
};

/*
 *  Source terms in Ion heating equation
 *
 *  - pi_i * domega/dpsi + Gamma_i m_i omega^2 / B_z + Q_i
 *
 * where Q_i is the collisional heating
 */
Real CylPlasmaDebug::Spi(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	// pi * d omega / d psi = (V'pi)*(d omega / d V)
	double R = B->R_V(V);
	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real nPrime = q(Channel::Density);
	Real JPrime = R * R * nPrime;
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real ViscousHeating = IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) * dOmegadV;
	Real PotentialHeating = -Gamma(u, q, V, t) * omega * omega / B->Bz_R(R);
	Real EnergyExchange = IonElectronEnergyExchange(n, p_e, p_i, V, t);

	Real Heating = ViscousHeating + PotentialHeating + EnergyExchange;
	return Heating;
}

/*
   In SI this is

   Q_i = 3 n_e m_e ( T_e - T_i ) / ( m_i tau_e )

   Normalised for the Energy equation, whose overall normalising factor is
   n0 T0^2 / ( m_e Omega_e_ref^2 tau_e_ref )


   Q_i = 3 * (p_e - p_i) / (tau_e) * (m_e/m_i) / (rho_s/R_ref)^2

 */
Real CylPlasmaDebug::IonElectronEnergyExchange(Real n, Real pe, Real pi, Position V, double t) const
{
	Real Te = pe / n;
	double RhoStar = RhoStarRef();
	Real pDiff = pe - pi;
	Real IonHeating = (pDiff / ElectronCollisionTime(n, Te)) * ((3.0 / (RhoStar * RhoStar)) * (ElectronMass / IonMass));

	if (std::isfinite(IonHeating.val))
		return IonHeating;
	else
		throw std::logic_error("Non-finite value computed for the ion heating at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

Real CylPlasmaDebug::Spe(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real EnergyExchange = -IonElectronEnergyExchange(n, p_e, p_i, V, t);
	return EnergyExchange;
};

// Source of angular momentum -- this is just imposed J x B torque (we can account for the particle source being a sink later).
Real CylPlasmaDebug::Somega(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	// J x B torque
	double R = B->R_V(V);
	return -jRadial * R * B->Bz_R(R);
};
