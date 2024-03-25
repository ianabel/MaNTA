#include "MirrorPlasma.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(MirrorPlasma);
const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.2, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;
#ifdef DEBUG
const std::string B_file = "/home/eatocco/projects/MaNTA/Bfield.nc";
#else
const std::string B_file = "Bfield.nc";
#endif

MirrorPlasma::MirrorPlasma(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, 4, 0)
{

	// B = new StraightMagneticField();

	xL = grid.lowerBoundary();
	xR = grid.upperBoundary();

	// isLowerDirichlet = true;
	// isUpperDirichlet = true;

	if (config.count("MirrorPlasma") == 1)
	{
		auto const &InternalConfig = config.at("MirrorPlasma");

		std::vector<std::string> constProfiles = toml::find_or(InternalConfig, "ConstantProfiles", std::vector<std::string>());

		for (auto &p : constProfiles)
		{
			ConstantProfiles.push_back(ChannelMap[p]);
			ConstantChannelMap[p] = true;
		}
		nConstantProfiles = ConstantProfiles.size();

		nVars -= nConstantProfiles;

		uL.resize(nVars);
		uR.resize(nVars);

		nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
		TeEdge = toml::find_or(InternalConfig, "EdgeElectronTemperature", T_edge);
		TiEdge = toml::find_or(InternalConfig, "EdgeIonTemperature", TeEdge);
		MEdge = toml::find_or(InternalConfig, "EdgeMachNumber", omega_edge);

		InitialPeakDensity = toml::find_or(InternalConfig, "InitialDensity", n_mid);
		InitialPeakTe = toml::find_or(InternalConfig, "InitialElectronTemperature", T_mid);
		InitialPeakTi = toml::find_or(InternalConfig, "InitialIonTemperature", T_mid);
		InitialPeakMachNumber = toml::find_or(InternalConfig, "InitialMachNumber", omega_mid);

		ParallelLossFactor = toml::find_or(InternalConfig, "ParallelLossFactor", 1.0);
		DragFactor = toml::find_or(InternalConfig, "DragFactor", 1.0);
		DragWidth = toml::find_or(InternalConfig, "DragWidth", 0.01);
		UniformHeatSource = toml::find_or(InternalConfig, "UniformHeatSource", 0.0);

		std::string Bfile = toml::find_or(InternalConfig, "MagneticFieldData", B_file);
		B = new CylindricalMagneticField(Bfile);
		B->CheckBoundaries(xL, xR);

		R_Lower = B->R_V(xL);
		R_Upper = B->R_V(xR);

		double Omega_Lower = sqrt(TeEdge) * MEdge / R_Lower;
		double Omega_Upper = sqrt(TeEdge) * MEdge / R_Upper;

		uL[Channel::Density] = nEdge;
		uR[Channel::Density] = nEdge;
		uL[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
		uR[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
		uL[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
		uR[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
		uL[Channel::AngularMomentum] = Omega_Lower * nEdge * R_Lower * R_Lower;
		uR[Channel::AngularMomentum] = Omega_Upper * nEdge * R_Upper * R_Upper;
		jRadial = -toml::find_or(InternalConfig, "jRadial", 4.0);
		ParticleSourceStrength = toml::find_or(InternalConfig, "ParticleSource", 10.0);
		ParticleSourceWidth = toml::find_or(InternalConfig, "ParticleSourceWidth", 0.02);
		ParticleSourceCenter = toml::find_or(InternalConfig, "ParticleSourceCenter", 0.5 * (R_Lower + R_Upper));
	}
	else if (config.count("MirrorPlasma") == 0)
	{
		throw std::invalid_argument("To use the Mirror Plasma physics model, a [MirrorPlasma] configuration section is required.");
	}
	else
	{
		throw std::invalid_argument("Unable to find unique [MirrorPlasma] configuration section in configuration file.");
	}
};

autodiff::dual2nd MirrorPlasma::InitialFunction(Index i, autodiff::dual2nd V, autodiff::dual2nd t) const
{
	autodiff::dual2nd R_min = B->R_V(xL);
	autodiff::dual2nd R_max = B->R_V(xR);
	autodiff::dual2nd R = B->R_V(V.val.val);
	R.grad = B->dRdV(V.val.val);
	autodiff::dual2nd R_mid = (R_min + R_max) / 2.0;

	autodiff::dual2nd nMid = InitialPeakDensity;
	autodiff::dual2nd TeMid = InitialPeakTe;
	autodiff::dual2nd TiMid = InitialPeakTi;
	autodiff::dual2nd MMid = InitialPeakMachNumber;

	autodiff::dual2nd v = cos(pi * (R - R_mid) / (R_max - R_min));

	autodiff::dual2nd n = nEdge + (nMid - nEdge) * v * v;
	autodiff::dual2nd Te = TeEdge + (TeMid - TeEdge) * v * v;
	autodiff::dual2nd Ti = TiEdge + (TiMid - TiEdge) * v * v;

	autodiff::dual2nd M = MEdge + (MMid - MEdge) * v * v * v;
	autodiff::dual2nd omega = sqrt(Te) * M / R;

	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return n;
		break;
	case Channel::IonEnergy:
		return (3. / 2.) * n * Ti;
		break;
	case Channel::ElectronEnergy:
		return (3. / 2.) * n * Te;
		break;
	case Channel::AngularMomentum:
		return omega * n * R * R;
		break;
	default:
		throw std::runtime_error("Request for initial value for undefined variable!");
	}
}

Real MirrorPlasma::Flux(Index i, RealVector u, RealVector q, Position x, Time t)
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

Real MirrorPlasma::Source(Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t)
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

Value MirrorPlasma::LowerBoundary(Index i, Time t) const
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return 0.0;
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
Value MirrorPlasma::UpperBoundary(Index i, Time t) const
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return 0.0;
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

bool MirrorPlasma::isLowerBoundaryDirichlet(Index i) const
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return false;
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

bool MirrorPlasma::isUpperBoundaryDirichlet(Index i) const
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return false;
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

/*
Normalisation:
   All lengths to a, densities to n0, temperatures to T0
   We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2 tau_e(n0,T0) ) ]^-1
	in effect we are normalising to the particle diffusion time across a distance 1

 */

// This is c_s / ( Omega_i * a )
// = sqrt( T0 / mi ) / ( e B0 / mi ) =  [ sqrt( T0 mi ) / ( e B0 ) ] / a
inline double MirrorPlasma::RhoStarRef() const
{
	return sqrt(T0 * IonMass) / (ElementaryCharge * B0 * a);
}

// Return this normalised to log Lambda at n0,T0
inline Real MirrorPlasma::LogLambda_ei(Real ne, Real Te) const
{
	double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5;
	Real LogLambda = 23.0 - log(2.0) - log(ne * n0) / 2.0 + log(Te * T0) * 1.5;
	return LogLambdaRef / LogLambda; // really needs to know Ti as well
}

// Return this normalised to log Lambda at n0,T0
inline Real MirrorPlasma::LogLambda_ii(Real ni, Real Ti) const
{
	double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5;
	Real LogLambda = 23.0 - log(2.0) - log(ni * n0) / 2.0 + log(Ti * T0) * 1.5;
	return LogLambdaRef / LogLambda; // really needs to know Ti as well
}

// Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
// This is equal to tau_e as used in Braginskii
inline Real MirrorPlasma::ElectronCollisionTime(Real ne, Real Te) const
{
	return pow(Te, 1.5) / (ne * LogLambda_ei(ne, Te));
}

// Return the actual value in SI units
inline double MirrorPlasma::ReferenceElectronCollisionTime() const
{
	double LogLambdaRef = 24.0 - log(n0) / 2.0 + log(T0); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
	return 12.0 * pow(M_PI, 1.5) * sqrt(ElectronMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (sqrt(2) * n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}
// Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
// This is equal to tau_i as used in Braginskii
inline Real MirrorPlasma::IonCollisionTime(Real ni, Real Ti) const
{
	return pow(Ti, 1.5) / (ni * LogLambda_ii(ni, Ti));
}

// Return the actual value in SI units
inline double MirrorPlasma::ReferenceIonCollisionTime() const
{
	double LogLambdaRef = 23.0 - log(2.0) - log(n0) / 2.0 + log(T0) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
	return 12.0 * pow(M_PI, 1.5) * sqrt(IonMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}

// We are in a quasineutral plasma with one ion species.
// This function returns V' * Gamma_e, and Gamma_i = Gamma_e
// c.f Helander & Sigmar -- Gamma_e = (n_e T_e / (m_e Omega_e^2 tau_e))*( (p_e' + p_i')/p_e - (3/2)(T_e'/T_e)
// Define lengths so R_ref = 1
Real MirrorPlasma::Gamma(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;
	Real PressureGradient = ((p_e_prime + p_i_prime) / p_e);
	Real TemperatureGradient = (3. / 2.) * (Te_prime / Te);
	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e
	Real Gamma = GeometricFactor * GeometricFactor * (p_e / (LogLambda_ei(n, Te) * ElectronCollisionTime(n, Te))) * (PressureGradient - TemperatureGradient);

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
Real MirrorPlasma::qi(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	Real nPrime = q(Channel::Density), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = 2.0 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (p_i / (LogLambda_ii(n, Ti) * IonCollisionTime(n, Ti))) * Ti_prime;

	if (std::isfinite(HeatFlux.val))
		return HeatFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

/*
   Following Helander & Sigmar, we have
   V' q_e = n_e T_e * V'^2 B^2 R^2 * ( T_e / m_e Omega_e^2 tau_e ) * ( 4.66 T_e'/T_e - (3/2) * (p_e'+p_i')/p_e )
 */
Real MirrorPlasma::qe(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;

	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = GeometricFactor * GeometricFactor * (p_e * Te / (LogLambda_ei(n, Te) * ElectronCollisionTime(n, Te))) * (4.66 * Te_prime / Te - (3. / 2.) * (p_e_prime + p_i_prime) / p_e);

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
Real MirrorPlasma::Pi(RealVector u, RealVector q, double V, double t) const
{
	Real n = u(Channel::Density), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	// dOmega dV = L'/J - J' L / J^2 ; L = angular momentum / J = moment of Inertia
	double R = B->R_V(V);

	Real J = n * R * R; // Normalisation includes the m_i
	Real nPrime = q(Channel::Density);
	double dRdV = B->dRdV(V);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;

	Real L = u(Channel::AngularMomentum);
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real Pi_v = IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) + omega * R * R * Gamma(u, q, V, t);
	return Pi_v;
};

/*
   Returns V' pi_cl_i
 */
Real MirrorPlasma::IonClassicalAngularMomentumFlux(Position V, Real n, Real Ti, Real dOmegadV, double t) const
{
	double R = B->R_V(V);
	double GeometricFactor = (B->VPrime(V) * R * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real MomentumFlux = 0.3 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (n * Ti / (LogLambda_ii(n, Ti) * IonCollisionTime(n, Ti))) * dOmegadV;
	if (std::isfinite(MomentumFlux.val))
		return MomentumFlux;
	else
		return 0.0;
	// throw std::logic_error("Non-finite value computed for the ion momentum flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

Real MirrorPlasma::Sn(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = p_e / n;
	Real Ti = p_i / n;

	double R = B->R_V(V);

	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = u(Channel::AngularMomentum) / J;

	Real Xi = Xi_e(V, omega, Ti, Te);
	Real ParallelLosses = ElectronPastukhovLossRate(V, Xi, n, Te);
	Real DensitySource = ParticleSource(R, t);
	Real FusionLosses = FusionRate(n, p_i);
	return DensitySource - ParallelLosses - FusionLosses;
};

/*
 *  Source terms in Ion heating equation
 *
 *  - pi_i * domega/dpsi + Gamma_i m_i omega^2 / B_z + Q_i
 *
 * where Q_i is the collisional heating
 */
Real MirrorPlasma::Spi(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = p_e / n;
	Real Ti = p_i / n;
	// pi * d omega / d psi = (V'pi)*(d omega / d V)
	double R = B->R_V(V);
	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real nPrime = q(Channel::Density);
	double dRdV = B->dRdV(V);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;
	// double shape = 1e-4;
	// Real ZeroEdge = 1 - (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));

	Real ViscousHeating = IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) * dOmegadV;

	auto phi0fn = [this](RealVector u, RealVector q, Real V, double t) -> Real
	{
		Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
		Real Te = p_e / n;
		Real Ti = p_i / n;
		// pi * d omega / d psi = (V'pi)*(d omega / d V)
		Real R = B->R_V(V.val);
		R.grad = B->dRdV(V.val);
		Real J = n * R * R; // Normalisation includes the m_i
		Real L = u(Channel::AngularMomentum);
		Real omega = L / J;

		return Ti * this->Xi_e(V.val, omega, Ti, Te);
	};
	Real Vreal = V;
	Real dphi0dV = autodiff::derivative(phi0fn, wrt(Vreal), at(u, q, Vreal, t)); // (2 * M_PI * Rval);
	auto dphi0du = autodiff::gradient(phi0fn, wrt(u), at(u, q, Vreal, t));
	auto qi = q.begin();
	for (auto &dphi0i : dphi0du)
	{
		dphi0dV += *qi * dphi0i;
		++qi;
	}

	Real PotentialHeating = -Gamma(u, q, V, t) * (omega * omega / B->Bz_R(R) - dphi0dV);
	Real EnergyExchange = IonElectronEnergyExchange(n, p_e, p_i, V, t);

	Real ParticleSourceHeating = 0.5 * omega * omega * R * R * ParticleSource(R, t);

	Real Heating = ViscousHeating + PotentialHeating + EnergyExchange + UniformHeatSource + ParticleSourceHeating;

	Real Xi = Xi_i(V, omega, Ti, Te);
	Real ParticleEnergy = Ti * (1.0 + Xi);
	Real ParallelLosses = ParticleEnergy * IonPastukhovLossRate(V, Xi, n, Ti);
	return Heating - ParallelLosses;
}

// Energy normalisation is T0, but these return Xi_s / T_s as that is what enters the
// Pastukhov factor
inline Real MirrorPlasma::Xi_i(Position V, Real omega, Real Ti, Real Te) const
{
	return CentrifugalPotential(V, omega, Ti, Te);
}

inline Real MirrorPlasma::Xi_e(Position V, Real omega, Real Ti, Real Te) const
{
	return CentrifugalPotential(V, omega, Ti, Te);
}

/*
   In SI this is

   Q_i = 3 n_e m_e ( T_e - T_i ) / ( m_i tau_e )

   Normalised for the Energy equation, whose overall normalising factor is
   n0 T0^2 / ( m_e Omega_e_ref^2 tau_e_ref )


   Q_i = 3 * (p_e - p_i) / (tau_e) * (m_e/m_i) / (rho_s/R_ref)^2

 */
Real MirrorPlasma::IonElectronEnergyExchange(Real n, Real pe, Real pi, Position V, double t) const
{
	Real Te = pe / n;
	double RhoStar = RhoStarRef();
	Real pDiff = pe - pi;
	Real IonHeating = (pDiff / (LogLambda_ei(n, Te) * ElectronCollisionTime(n, Te))) * ((3.0 / (RhoStar * RhoStar))); //* (ElectronMass / IonMass));

	if (std::isfinite(IonHeating.val))
		return IonHeating;
	else
		throw std::logic_error("Non-finite value computed for the ion heating at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

Real MirrorPlasma::Spe(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = p_e / n, Ti = p_i / n;
	Real EnergyExchange = -IonElectronEnergyExchange(n, p_e, p_i, V, t);

	double MirrorRatio = B->MirrorRatio(V);
	Real AlphaHeating = sqrt(1 - 1 / MirrorRatio) * TotalAlphaPower(n, p_i);

	double R = B->R_V(V);

	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real omega = L / J;
	Real ParticleSourceHeating = electronMass / ionMass * .5 * omega * omega * R * R * ParticleSource(R, t);

	Real Heating = EnergyExchange + AlphaHeating + ParticleSourceHeating;

	Real Xi = Xi_e(V, omega, Ti, Te);
	Real ParticleEnergy = Te * (1.0 + Xi);
	Real ParallelLosses = ParticleEnergy * ElectronPastukhovLossRate(V, Xi, n, Te);

	Real RadiationLosses = BremsstrahlungLosses(n, p_e);

	return Heating - ParallelLosses - RadiationLosses;
};

// Source of angular momentum -- this is just imposed J x B torque (we can account for the particle source being a sink later).
Real MirrorPlasma::Somega(RealVector u, RealVector q, RealVector sigma, Position V, double t) const
{
	// J x B torque
	double R = B->R_V(V);
	double JxB = -jRadial * R * B->Bz_R(R);

	Real n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = p_e / n, Ti = p_i / n;
	Real L = u(Channel::AngularMomentum);
	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = L / J;
	Real vtheta = omega * R;

	double shape = 1 / DragWidth;
	Real Drag = R * vtheta * vtheta * DragFactor * (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));

	// Neglect electron momentum
	Real Xi = Xi_i(V, omega, Ti, Te);
	Real AngularMomentumPerParticle = L / n;
	Real ParallelLosses = AngularMomentumPerParticle * IonPastukhovLossRate(V, Xi, n, Te);

	return JxB - ParallelLosses - Drag;
};

Real MirrorPlasma::ParticleSource(double R, double t) const
{
	double shape = 1 / ParticleSourceWidth;
	// return ParticleSourceStrength * (DelayFactor / 10 + DelayFactor * exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));
	return ParticleSourceStrength * (exp(-shape * (R - ParticleSourceCenter) * (R - ParticleSourceCenter)));
	// return ParticleSourceStrength;
};

Real MirrorPlasma::ElectronPastukhovLossRate(double V, Real Xi_e, Real n, Real Te) const
{
	double MirrorRatio = B->MirrorRatio(V);
	Real tau_ee = ElectronCollisionTime(n, Te);
	double Sigma = 2.0; // = 1 + Z_eff ; Include collisions with ions and impurities as well as self-collisions

	Real PastukhovFactor = (exp(-Xi_e) / Xi_e);
	// Cap loss rates
	// if (PastukhovFactor.val > 1.0)
	// 	PastukhovFactor.val = 1.0;
	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor.val < 0.0)
		return 0.0;
	double Normalization = (IonMass / ElectronMass) * (1.0 / (RhoStarRef() * RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ee) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;
	return ParallelLossFactor * LossRate;
}

Real MirrorPlasma::IonPastukhovLossRate(double V, Real Xi_i, Real n, Real Ti) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double MirrorRatio = B->MirrorRatio(V);
	Real tau_ii = IonCollisionTime(n, Ti);
	double Sigma = 1.0; // Just ion-ion collisions

	Real PastukhovFactor = (exp(-Xi_i) / Xi_i);
	// Cap loss rates
	// if (PastukhovFactor.val > 1.0)
	// 	PastukhovFactor.val = 1.0;
	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor.val < 0.0)
		return 0.0;

	double Normalization = sqrt(IonMass / ElectronMass) * (1.0 / (RhoStarRef() * RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ii) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;

	return ParallelLossFactor * LossRate;
}

// Returns (1/(1 + Tau))*(1-1/R_m)*(M^2)
Real MirrorPlasma::CentrifugalPotential(double V, Real omega, Real Ti, Real Te) const
{
	double MirrorRatio = B->MirrorRatio(V);
	double R = B->R_V(V);
	Real tau = Ti / Te;
	Real MachNumber = omega * R / sqrt(Te); // omega is normalised to c_s0 / a
	Real Potential = (1.0 / (1.0 + tau)) * (1.0 - 1.0 / MirrorRatio) * MachNumber * MachNumber / 2.0;
	return Potential;
}
// Implements D-T fusion rate from NRL plasma formulary
Real MirrorPlasma::FusionRate(Real n, Real pi) const
{

	Real Normalization = n0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
	Real Factor = 1e-6 * 3.68e-12 * n0 * n0 / Normalization;
	Real Ti_kev = (pi / n) * T0 / (1000 * ElementaryCharge);
	Real R = Factor * 0.25 * n * n * pow(Ti_kev, -2. / 3.) * exp(-19.94 * pow(Ti_kev, -1. / 3.));

	return R;
}

Real MirrorPlasma::TotalAlphaPower(Real n, Real pi) const
{
	double Factor = 5.6e-13 / (T0);
	Real AlphaPower = Factor * FusionRate(n, pi);
	return AlphaPower;
}

// Implements Bremsstrahlung radiative losses from NRL plasma formulary
Real MirrorPlasma::BremsstrahlungLosses(Real n, Real pe) const
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
double MirrorPlasma::Voltage(T1 &L_phi, T2 &n)
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

void MirrorPlasma::initialiseDiagnostics(NetCDFIO &nc)
{
	// Add diagnostics here
	//
	double RhoStar = RhoStarRef();
	double TauNorm = (IonMass / ElectronMass) * (1.0 / (RhoStar * RhoStar)) * (ReferenceElectronCollisionTime());
	nc.AddScalarVariable("Tau", "Normalising time", "s", TauNorm);

	auto getConstI = [this](int c)
	{
		for (auto &constI : ConstantProfiles)
		{
			if (c > constI)
				c--;
		};
		return c;
	};
	auto L = [this, &getConstI](double V)
	{
		int i = getConstI(Channel::AngularMomentum);
		return InitialValue(i, V);
	};
	auto LPrime = [this, &getConstI](double V)
	{
		int i = getConstI(Channel::AngularMomentum);
		return InitialDerivative(i, V);
	};
	auto n = [this, &getConstI](double V)
	{
		int i = getConstI(Channel::Density);
		return InitialValue(i, V);
	};
	auto nPrime = [this, &getConstI](double V)
	{
		int i = getConstI(Channel::Density);
		return InitialDerivative(i, V);
	};
	auto p_i = [this, &getConstI](double V)
	{
		int i = getConstI(Channel::IonEnergy);
		return (2. / 3.) * InitialValue(i, V);
	};
	auto p_e = [this, &getConstI](double V)
	{
		int i = getConstI(Channel::ElectronEnergy);
		return (2. / 3.) * InitialValue(i, V);
	};

	double initialVoltage = Voltage(L, n);

	const std::function<double(const double &)> initialZero = [](const double &V)
	{ return 0.0; };

	// Wrap DGApprox with lambdas for heating functions
	Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i](double V)
	{
		Real Ti = p_i(V) / n(V);
		double R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i
		double dRdV = B->dRdV(V);
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

		double R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real Xe = Xi_e(V, omega, Ti, Te);

		double ParallelLosses = ElectronPastukhovLossRate(V, Xe, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		double R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		double Phi0 = Xi_i(V, omega, Ti, Te).val;

		return Phi0;
	};

	nc.AddTimeSeries("Voltage", "Total voltage drop across the plasma", "Volts", initialVoltage);
	nc.AddGroup("MomentumFlux", "Separating momentum fluxes");
	nc.AddGroup("ParallelLosses", "Separated parallel losses");
	nc.AddVariable("ParallelLosses", "ParLoss", "Parallel particle losses", "-", ParallelLosses);
	nc.AddVariable("ParallelLosses", "CentrifugalPotential", "Centrifugal potential", "-", Phi0);
	nc.AddGroup("Heating", "Separated heating sources");
	nc.AddVariable("Heating", "AlphaHeating", "Alpha heat source", "-", AlphaHeating);
	nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", ViscousHeating);
	nc.AddVariable("Heating", "RadiationLosses", "Bremsstrahlung heat losses", "-", RadiationLosses);
}

void MirrorPlasma::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{
	auto getConstI = [this](int c)
	{
		for (auto &constI : ConstantProfiles)
		{
			if (c > constI)
				c--;
		};
		return c;
	};
	auto L = [this, &y, &getConstI](double V)
	{
		int i = getConstI(Channel::AngularMomentum);
		if (ConstantChannelMap["AngularMomentum"])
			return InitialValue(i, V);
		else
			return y.u(i)(V);
	};
	auto LPrime = [this, &y, &getConstI](double V)
	{
		int i = getConstI(Channel::AngularMomentum);
		if (ConstantChannelMap["AngularMomentum"])
			return InitialDerivative(i, V);
		else
			return y.q(i)(V);
	};
	auto n = [this, &y, &getConstI](double V)
	{
		int i = getConstI(Channel::Density);
		if (ConstantChannelMap["Density"])
			return InitialValue(i, V);
		else
			return y.u(i)(V);
	};
	auto nPrime = [this, &y, &getConstI](double V)
	{
		int i = getConstI(Channel::Density);
		if (ConstantChannelMap["Density"])
			return InitialDerivative(i, V);
		else
			return y.q(i)(V);
	};
	auto p_i = [this, &y, &getConstI](double V)
	{
		int i = getConstI(Channel::IonEnergy);
		if (ConstantChannelMap["IonEnergy"])
			return (2. / 3.) * InitialValue(i, V);
		else
			return (2. / 3.) * y.u(i)(V);
	};
	auto p_e = [this, &y, &getConstI](double V)
	{
		int i = getConstI(Channel::ElectronEnergy);
		if (ConstantChannelMap["ElectronEnergy"])
			return (2. / 3.) * InitialValue(i, V);
		else
			return (2. / 3.) * y.u(i)(V);
	};

	double voltage = Voltage(L, n);
	nc.AppendToTimeSeries("Voltage", voltage, tIndex);

	// Wrap DGApprox with lambdas for heating functions
	Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i, &t](double V)
	{
		Real Ti = p_i(V) / n(V);
		double R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i
		double dRdV = B->dRdV(V);
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

		double R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real Xe = Xi_e(V, omega, Ti, Te);

		double ParallelLosses = ElectronPastukhovLossRate(V, Xe, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		double R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		double Phi0 = Xi_i(V, omega, Ti, Te).val;

		return Phi0;
	};

	// Add the appends for the heating stuff
	nc.AppendToGroup<Fn>("Heating", tIndex, {{"AlphaHeating", AlphaHeating}, {"ViscousHeating", ViscousHeating}, {"RadiationLosses", RadiationLosses}});

	nc.AppendToGroup<Fn>("ParallelLosses", tIndex, {{"ParLoss", ParallelLosses}, {"CentrifugalPotential", Phi0}});
}