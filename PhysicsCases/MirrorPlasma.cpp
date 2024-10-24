#include "MirrorPlasma.hpp"
#include <iostream>
#include <string>
#include <boost/math/tools/roots.hpp>

REGISTER_PHYSICS_IMPL(MirrorPlasma);
const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.2, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;

template <typename T>
int sign(T x)
{
	return x >= 0 ? 1 : -1;
}

MirrorPlasma::MirrorPlasma(toml::value const &config, Grid const &grid)
{
	nVars = 4;
	nScalars = 0;
	nAux = 0;

	xL = grid.lowerBoundary();
	xR = grid.upperBoundary();

	if (config.count("MirrorPlasma") == 1)
	{
		auto const &InternalConfig = config.at("MirrorPlasma");

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

		// Compute phi to satisfy zero parallel current
		useAmbipolarPhi = toml::find_or(InternalConfig, "useAmbipolarPhi", false);
		if (useAmbipolarPhi)
			nAux = 1;

		// Add floor for computed densities and temperatures
		MinDensity = toml::find_or(InternalConfig, "MinDensity", 1e-2);
		MinTemp = toml::find_or(InternalConfig, "MinTemp", 0.1);
		RelaxFactor = toml::find_or(InternalConfig, "RelaxFactor", 1.0);

		useMMS = toml::find_or(InternalConfig, "useMMS", false);
		growth = toml::find_or(InternalConfig, "MMSgrowth", 1.0);
		growth_factors = toml::find_or(InternalConfig, "growth_factors", std::vector<double>(nVars, 1.0));
		growth_rate = toml::find_or(InternalConfig, "MMSgrowth_rate", 0.5);

		SourceCap = toml::find_or(InternalConfig, "SourceCap", 1e5);

		// std::string Bfile = toml::find_or(InternalConfig, "MagneticFieldData", B_file);
		double B_z = toml::find_or(InternalConfig, "Bz", 1.0);
		double Rm = toml::find_or(InternalConfig, "Rm", 3.0);
		double L_z = toml::find_or(InternalConfig, "Lz", 1.0);

		B = std::make_shared<StraightMagneticField>(L_z, B_z, Rm);
		// B->CheckBoundaries(xL, xR);

		R_Lower = B->R_V(xL);
		R_Upper = B->R_V(xR);

		Plasma = std::make_unique<PlasmaConstants>(PlasmaTypes::DD, B, n0, T0, Z_eff, R_Upper - R_Lower);

		loadInitialConditionsFromFile = toml::find_or(InternalConfig, "useNcFile", false);
		if (loadInitialConditionsFromFile)
		{
			filename = toml::find_or(InternalConfig, "InitialConditionFilename", "MirrorPlasmaRERUN.nc");
			LoadDataToSpline(filename);
		}

		nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
		TeEdge = toml::find_or(InternalConfig, "EdgeElectronTemperature", T_edge);
		TiEdge = toml::find_or(InternalConfig, "EdgeIonTemperature", TeEdge);
		MLower = toml::find_or(InternalConfig, "LowerMachNumber", R_Lower * omega_edge / sqrt(TeEdge));
		MUpper = toml::find_or(InternalConfig, "UpperMachNumber", R_Upper * omega_edge / sqrt(TeEdge));
		MEdge = 0.5 * (MUpper + MLower);

		InitialPeakDensity = toml::find_or(InternalConfig, "InitialDensity", n_mid);
		InitialPeakTe = toml::find_or(InternalConfig, "InitialElectronTemperature", T_mid);
		InitialPeakTi = toml::find_or(InternalConfig, "InitialIonTemperature", T_mid);
		InitialPeakMachNumber = toml::find_or(InternalConfig, "InitialMachNumber", omega_mid);

		MachWidth = toml::find_or(InternalConfig, "MachWidth", 0.05);
		double Omega_Lower = sqrt(TeEdge) * MLower / R_Lower;
		double Omega_Upper = sqrt(TeEdge) * MUpper / R_Upper;

		uL[Channel::Density] = nEdge;
		uR[Channel::Density] = nEdge;
		uL[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
		uR[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
		uL[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
		uR[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
		uL[Channel::AngularMomentum] = Omega_Lower * nEdge * R_Lower * R_Lower;
		uR[Channel::AngularMomentum] = Omega_Upper * nEdge * R_Upper * R_Upper;

		IRadial = -toml::find_or(InternalConfig, "IRadial", 4.0);
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

Real2nd MirrorPlasma::InitialFunction(Index i, Real2nd V, Real2nd t) const
{
	auto tfac = [this, t](double growth)
	{ return 1 + growth * tanh(growth_rate * t); };

	Real2nd R_min = B->R_V(xL);
	Real2nd R_max = B->R_V(xR);
	Real2nd R = B->R_V(V);

	Real2nd R_mid = (R_min + R_max) / 2.0;

	Real2nd nMid = InitialPeakDensity;
	Real2nd TeMid = InitialPeakTe;
	Real2nd TiMid = InitialPeakTi;
	Real2nd MMid = InitialPeakMachNumber;
	double shape = 1 / MachWidth;

	Real2nd v = cos(pi * (R - R_mid) / (R_max - R_min)); //* exp(-shape * (R - R_mid) * (R - R_mid));

	Real2nd Te = TeEdge + tfac(growth_factors[Channel::ElectronEnergy]) * (TeMid - TeEdge) * v * v;
	Real2nd Ti = TiEdge + tfac(growth_factors[Channel::IonEnergy]) * (TiMid - TiEdge) * v * v;
	Real2nd n = nEdge + tfac(growth_factors[Channel::Density]) * (nMid - nEdge) * v;
	auto slope = (MUpper - MLower) / (R_Upper - R_Lower);
	Real2nd M = MLower + slope * (R - R_Lower) + (MMid - 0.5 * (MUpper + MLower)) * v / exp(-shape * (R - R_mid) * (R - R_mid)); // MEdge + tfac(growth_factors[Channel::AngularMomentum]) * (MMid - MEdge) * (1 - (exp(-shape * (R - R_Upper) * (R - R_Upper)) + exp(-shape * (R - R_Lower) * (R - R_Lower))));
	Real2nd omega = sqrt(Te) * M / R;

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

Real MirrorPlasma::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
		return ConstantChannelMap[Channel::Density] ? 0.0 : Gamma(u, q, x, t);
		break;
	case Channel::IonEnergy:
		return ConstantChannelMap[Channel::IonEnergy] ? 0.0 : qi(u, q, x, t);
		break;
	case Channel::ElectronEnergy:
		return ConstantChannelMap[Channel::ElectronEnergy] ? 0.0 : qe(u, q, x, t);
		break;
	case Channel::AngularMomentum:
		return ConstantChannelMap[Channel::AngularMomentum] ? 0.0 : Pi(u, q, x, t);
		break;
	default:
		throw std::runtime_error("Request for flux for undefined variable!");
	}
}

Real MirrorPlasma::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t)
{
	Channel c = static_cast<Channel>(i);
	Real S;

	switch (c)
	{
	case Channel::Density:
		S = ConstantChannelMap[Channel::Density] ? 0.0 : Sn(u, q, sigma, phi, x, t);
		break;
	case Channel::IonEnergy:
		S = ConstantChannelMap[Channel::IonEnergy] ? 0.0 : Spi(u, q, sigma, phi, x, t);
		break;
	case Channel::ElectronEnergy:
		S = ConstantChannelMap[Channel::ElectronEnergy] ? 0.0 : Spe(u, q, sigma, phi, x, t);
		break;
	case Channel::AngularMomentum:
		S = ConstantChannelMap[Channel::AngularMomentum] ? 0.0 : Somega(u, q, sigma, phi, x, t);
		break;
	default:
		throw std::runtime_error("Request for flux for undefined variable!");
	}
	if (abs(S) > SourceCap)
		S = sign(S) * SourceCap;
	return S;
}

Real MirrorPlasma::GFunc(Index, RealVector u, RealVector, RealVector, RealVector phi, Position V, Time t)
{

	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);

	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = u(Channel::AngularMomentum) / J;
	return ParallelCurrent<Real>(V, omega, n, Ti, Te, phi(0));
}

Value MirrorPlasma::InitialAuxValue(Index, Position V, Time t) const
{
	using boost::math::tools::eps_tolerance;
	using boost::math::tools::newton_raphson_iterate;

	Real n = InitialFunction(Channel::Density, V, t).val, p_e = (2. / 3.) * InitialFunction(Channel::ElectronEnergy, V, t).val, p_i = (2. / 3.) * InitialFunction(Channel::IonEnergy, V, t).val;
	Real Te = p_e / n;
	Real Ti = p_i / n;

	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = InitialFunction(Channel::AngularMomentum, V, t).val / J;

	double M = (static_cast<Real>(omega * R / sqrt(Te))).val;
	auto func = [this, &n, &Te, &Ti, &omega, &V](double phi)
	{
		return ParallelCurrent<Real>(static_cast<Real>(V), omega, n, Ti, Te, static_cast<Real>(phi)).val;
	};
	auto deriv_func = [this, &n, &Te, &Ti, &omega, &V](double phi)
	{
		Real phireal = phi;
		return derivative([this](Real V, Real omega, Real n, Real Ti, Real Te, Real phi)
						  { return ParallelCurrent<Real>(V, omega, n, Ti, Te, phi); }, wrt(phireal), at(V, omega, n, Ti, Te, phireal));
	};

	const int digits = std::numeric_limits<double>::digits; // Maximum possible binary digits accuracy for type T.
	int get_digits = static_cast<int>(digits * 0.6);		// Accuracy doubles with each step, so stop when we have
															// just over half the digits correct.
	eps_tolerance<double> tol(get_digits);

	const boost::uintmax_t maxit = 20;
	boost::uintmax_t it = maxit;
	double phi_g = newton_raphson_iterate([&func, &deriv_func](double phi)
										  { return std::pair<double, double>(func(phi), deriv_func(phi)); }, 0.0, -CentrifugalPotential(V, omega.val, Ti.val, Te.val), 0.01, get_digits, it);
	return phi_g;
}

Value MirrorPlasma::LowerBoundary(Index i, Time t) const
{
	return isLowerBoundaryDirichlet(i) ? uL[i] : 0.0;
}
Value MirrorPlasma::UpperBoundary(Index i, Time t) const
{
	return isUpperBoundaryDirichlet(i) ? uR[i] : 0.0;
}

bool MirrorPlasma::isLowerBoundaryDirichlet(Index i) const
{
	return lowerBoundaryConditions[i];
}

bool MirrorPlasma::isUpperBoundaryDirichlet(Index i) const
{
	return upperBoundaryConditions[i];
}

Real2nd MirrorPlasma::MMS_Solution(Index i, Real2nd x, Real2nd t)
{
	return InitialFunction(i, x, t);
}

/*
Normalisation:
   All lengths to a, densities to n0, temperatures to T0
   We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2 tau_e(n0,T0) ) ]^-1
	in effect we are normalising to the particle diffusion time across a distance 1

 */

// We are in a quasineutral plasma with one ion species.
// This function returns V' * Gamma_e, and Gamma_i = Gamma_e
// c.f Helander & Sigmar -- Gamma_e = (n_e T_e / (m_e Omega_e^2 tau_e))*( (p_e' + p_i')/p_e - (3/2)(T_e'/T_e)
// Define lengths so R_ref = 1
Real MirrorPlasma::Gamma(RealVector u, RealVector q, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;

	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;
	Real PressureGradient = ((p_e_prime + p_i_prime) / p_e);
	Real TemperatureGradient = (3. / 2.) * (Te_prime / Te);

	// Real TemperatureGradient = (3. / 2.) * (p_e_prime - nPrime * Te) / p_e;
	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e
	Real Gamma = GeometricFactor * GeometricFactor * (p_e / (Plasma->ElectronCollisionTime(n, Te))) * (PressureGradient - TemperatureGradient);

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
Real MirrorPlasma::qi(RealVector u, RealVector q, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	Real nPrime = q(Channel::Density), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = 2.0 * GeometricFactor * GeometricFactor * sqrt(Plasma->IonMass() / (2.0 * ElectronMass)) * (p_i / (Plasma->IonCollisionTime(n, Ti))) * Ti_prime;

	if (std::isfinite(HeatFlux.val))
		return HeatFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

/*
   Following Helander & Sigmar, we have
   V' q_e = n_e T_e * V'^2 B^2 R^2 * ( T_e / m_e Omega_e^2 tau_e ) * ( 4.66 T_e'/T_e - (3/2) * (p_e'+p_i')/p_e )
 */
Real MirrorPlasma::qe(RealVector u, RealVector q, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;

	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = GeometricFactor * GeometricFactor * (p_e * Te / (Plasma->ElectronCollisionTime(n, Te))) * (4.66 * Te_prime / Te - (3. / 2.) * (p_e_prime + p_i_prime) / p_e);

	if (std::isfinite(HeatFlux.val))
		return HeatFlux;
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
Real MirrorPlasma::Pi(RealVector u, RealVector q, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	// dOmega dV = L'/J - J' L / J^2 ; L = angular momentum / J = moment of Inertia
	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation includes the m_i
	Real nPrime = q(Channel::Density);
	Real dRdV = B->dRdV(V);
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
Real MirrorPlasma::IonClassicalAngularMomentumFlux(Real V, Real n, Real Ti, Real dOmegadV, Time t) const
{
	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real MomentumFlux = 0.3 * GeometricFactor * GeometricFactor * sqrt(Plasma->IonMass() / (2.0 * ElectronMass)) * (n * Ti / (Plasma->IonCollisionTime(n, Ti))) * dOmegadV;
	if (std::isfinite(MomentumFlux.val))
		return MomentumFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion momentum flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

Real MirrorPlasma::Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);

	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = u(Channel::AngularMomentum) / J;
	Real Xi;
	if (useAmbipolarPhi)
	{
		Xi = Xi_e(V, phi(0), Ti, Te, omega);
	}
	else
	{
		Xi = CentrifugalPotential(V, omega, Ti, Te) + AmbipolarPhi(V, n, Ti, Te) / 2.0;
	}
	Real ParallelLosses = ElectronPastukhovLossRate(V, Xi, n, Te);
	Real DensitySource = ParticleSourceStrength * ParticleSource(R.val, t);
	Real FusionLosses = Plasma->FusionRate(n, p_i);

	Real S = DensitySource - ParallelLosses - FusionLosses;

	return S - RelaxSource(u(Channel::Density), n);
};

/*
 *  Source terms in Ion heating equation
 *
 *  - pi_i * domega/dpsi + Gamma_i m_i omega^2 / B_z + Q_i
 *
 * where Q_i is the collisional heating
 */
Real MirrorPlasma::Spi(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);
	// pi * d omega / d psi = (V'pi)*(d omega / d V)
	Real R = B->R_V(V);
	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real nPrime = q(Channel::Density);
	Real dRdV = B->dRdV(V);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real ViscousHeating = IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) * dOmegadV;

	// Use this version since we have no parallel flux in a square well
	Real PotentialHeating = -Gamma(u, q, V, t) * (omega * omega / (2 * pi * a) - dphidV(u, q, phi, V));
	Real EnergyExchange = Plasma->IonElectronEnergyExchange(n, p_e, p_i, V, t);

	Real Heating = ViscousHeating + PotentialHeating + EnergyExchange + UniformHeatSource;

	Real Xi;
	if (useAmbipolarPhi)
	{
		Xi = Xi_i(V, phi(0), Ti, Te, omega);
	}
	else
	{
		Xi = CentrifugalPotential(V, omega, Ti, Te) - AmbipolarPhi(V, n, Ti, Te) / 2.0;
	}
	Real ParticleEnergy = Ti * (1.0 + Xi);
	Real ParallelLosses = ParticleEnergy * IonPastukhovLossRate(V, Xi, n, Ti);

	Real S = Heating - ParallelLosses;

	return S + RelaxSource(u(Channel::Density) * Te, p_e) + RelaxSource(n * floor(Te, MinTemp), p_e);
}

Real MirrorPlasma::Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);
	Real EnergyExchange = -Plasma->IonElectronEnergyExchange(n, p_e, p_i, V, t);

	double MirrorRatio = B->MirrorRatio(V);
	Real AlphaHeating = sqrt(1 - 1 / MirrorRatio) * Plasma->TotalAlphaPower(n, p_i);

	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real omega = L / J;

	Real PotentialHeating = -Gamma(u, q, V, t) * dphidV(u, q, phi, V); //(dphi1dV(u, q, phi(0), V));

	Real Heating = EnergyExchange + AlphaHeating + PotentialHeating;

	Real Xi;
	if (useAmbipolarPhi)
	{
		Xi = Xi_e(V, phi(0), Ti, Te, omega);
	}
	else
	{
		Xi = CentrifugalPotential(V, omega, Ti, Te) + AmbipolarPhi(V, n, Ti, Te) / 2.0;
	}

	// Parallel Losses
	Real ParticleEnergy = Te * (1.0 + Xi);
	Real ParallelLosses = ParticleEnergy * ElectronPastukhovLossRate(V, Xi, n, Te);

	Real RadiationLosses = Plasma->BremsstrahlungLosses(n, p_e) + Plasma->CyclotronLosses(V, n, Te);

	Real S = Heating - ParallelLosses - RadiationLosses;

	return S + RelaxSource(u(Channel::Density) * Te, p_e) + RelaxSource(n * floor(Te, MinTemp), p_e);
};

// Source of angular momentum -- this is just imposed J x B torque (we can account for the particle source being a sink later).
Real MirrorPlasma::Somega(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) const
{
	// J x B torque
	Real R = B->R_V(V);

	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);

	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);

	Real JxB = -IRadial / B->VPrime(V); //-jRadial * R * B->Bz_R(R);
	Real L = u(Channel::AngularMomentum);
	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = L / J;

	// Neglect electron momentum
	Real Xi;
	if (useAmbipolarPhi)
	{
		Xi = Xi_i(V, phi(0), Ti, Te, omega);
	}
	else
	{
		Xi = CentrifugalPotential(V, omega, Ti, Te) - AmbipolarPhi(V, n, Ti, Te) / 2.0;
	}
	Real AngularMomentumPerParticle = L / n;
	Real ParallelLosses = AngularMomentumPerParticle * IonPastukhovLossRate(V, Xi, n, Te);

	return JxB - ParallelLosses + RelaxSource(omega * R * R * u(Channel::Density), L);
};

// Use the chain rule to calculate dphi0/dV, making sure to set gradient values correctly
Real MirrorPlasma::dphi0dV(RealVector u, RealVector q, Real V) const
{
	auto phi0fn = [this](Real2ndVector u, Real2nd V)
	{
		return this->phi0(u, V);
	};
	Real2ndVector u2(nVars);

	Real2nd V2 = V.val;

	for (Index i = 0; i < nVars; ++i)
	{
		u2(i).val.val = u(i).val;
	}
	Real2nd dphi0dV_2 = phi0(u2, V2) / (pi * V2);
	Real dphi0dV;
	dphi0dV.val = dphi0dV_2.val.val;
	dphi0dV.grad = dphi0dV_2.grad.val;
	RealVector dphi0du(nVars);
	Real2nd phi0;
	auto d2phi0du2 = hessian(phi0fn, wrt(u2), at(u2, V), phi0, dphi0du);
	for (Index i = 0; i < nVars; ++i)
	{
		if (u(i).grad != 0)
		{
			dphi0du(i).grad = d2phi0du2(i, i);
		}
		dphi0dV += q(i) * dphi0du(i);
	}

	return dphi0dV;
}

// Compute dphi1dV using the chain rule
// Take derivative2 of Jpar using autodiff, and then make sure to set the correct gradient values so Jacobian is correct
Real MirrorPlasma::dphi1dV(RealVector u, RealVector q, Real phi, Real V) const
{
	auto Jpar = [this](Real2ndVector u, Real2nd phi, Real2nd V)
	{
		Real2nd n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
		if (n < MinDensity)
			n.val.val = MinDensity;
		Real2nd Te = p_e / n;
		Real2nd Ti = p_i / n;

		Real2nd R = B->R_V(V);

		Real2nd L = u(Channel::AngularMomentum);
		Real2nd J = n * R * R; // Normalisation of the moment of inertia includes the m_i
		Real2nd omega = L / J;

		return ParallelCurrent<Real2nd>(V, omega, n, Ti, Te, phi);
	};

	Real2ndVector u2(nVars);

	Real2nd phi2 = phi.val;

	Real2nd V2 = V.val;

	for (Index i = 0; i < nVars; ++i)
	{
		u2(i).val.val = u(i).val;
	}

	// take derivatives wrt V, phi, and u

	auto [_, dJpardV, d2JpardV2] = derivatives(Jpar, wrt(V2, V2), at(u2, phi2, V2));

	auto [__, dJpardphi1, d2Jpardphi12] = derivatives(Jpar, wrt(phi2, phi2), at(u2, phi2, V2));

	RealVector dJpardu(nVars);
	Real2nd ___;
	auto d2Jpardu2 = hessian(Jpar, wrt(u2), at(u2, phi2, V2), ___, dJpardu);

	Real n = floor(u(Channel::Density), MinDensity), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;

	Real nPrime = q(Channel::Density), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

	// set all of the autodiff gradients
	// V parts
	Real dJpardV_real = dJpardV;

	if (V.grad != 0)
		dJpardV_real.grad = d2JpardV2;

	// u/q parts
	Real qdotdJdu = 0.0;
	for (Index i = 0; i < nVars; ++i)
	{
		if (u(i).grad != 0)
			dJpardu(i).grad = d2Jpardu2(i, i);
		qdotdJdu += q(i) * dJpardu(i);
	}

	// phi1 parts
	Real dJpardphi1_real = dJpardphi1;

	if (phi.grad != 0)
		dJpardphi1_real.grad = d2Jpardphi12;

	// dphi1dV computed using the chain rule derivative of the parallel current
	Real dphi1dV = -(dJpardV_real + qdotdJdu) * Ti / dJpardphi1_real + phi * Ti_prime;

	return dphi1dV;
}

Real MirrorPlasma::dphidV(RealVector u, RealVector q, RealVector phi, Real V) const
{
	Real dphidV = dphi0dV(u, q, V);

	if (nAux > 0)
		dphidV += dphi1dV(u, q, phi(0), V);

	return dphidV;
}

// Energy normalisation is T0, but these return Xi_s / T_s as that is what enters the
// Pastukhov factor
template <typename T>
T MirrorPlasma::Xi_i(T V, T phi, T Ti, T Te, T omega) const
{
	return CentrifugalPotential<T>(V, omega, Ti, Te) + phi;
}
template <typename T>
T MirrorPlasma::Xi_e(T V, T phi, T Ti, T Te, T omega) const
{
	return CentrifugalPotential<T>(V, omega, Ti, Te) - Ti / Te * phi;
}

inline Real MirrorPlasma::AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const
{
	double R = B->MirrorRatio(V);
	double Sigma = 1.0;
	return log((Plasma->ElectronCollisionTime(n, Te) / Plasma->IonCollisionTime(n, Ti)) * (log(R * Sigma) / (Sigma * ::log(R))));
}

template <typename T>
T MirrorPlasma::ParallelCurrent(T V, T omega, T n, T Ti, T Te, T phi) const
{
	T Xii = Xi_i<T>(V, phi, Ti, Te, omega);
	T Xie = Xi_e<T>(V, phi, Ti, Te, omega);

	double MirrorRatio = B->MirrorRatio(V);

	double Sigma_i = 1.0;
	double Sigma_e = 1 + Z_eff;

	T a = Sigma_i * (1.0 / log(MirrorRatio * Sigma_i)) * sqrt(Plasma->IonMass() / ElectronMass) / Plasma->IonCollisionTime(n, Ti);
	T b = Sigma_e * (1.0 / log(MirrorRatio * Sigma_e)) * (Plasma->IonMass() / ElectronMass) / Plasma->ElectronCollisionTime(n, Te);

	T j = a * exp(-Xii) / Xii - b * exp(-Xie) / Xie;

	return j;
}

Real MirrorPlasma::ParticleSource(double R, double t) const
{
	double shape = 1 / ParticleSourceWidth;
	return (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));
	// return exp(-shape * (R - ParticleSourceCenter) * (R - ParticleSourceCenter));
	//   return ParticleSourceStrength;
};

Real MirrorPlasma::ElectronPastukhovLossRate(Real V, Real Xi_e, Real n, Real Te) const
{
	double MirrorRatio = B->MirrorRatio(V);
	Real tau_ee = Plasma->ElectronCollisionTime(n, Te);
	double Sigma = 1 + Z_eff; // Include collisions with ions and impurities as well as self-collisions
	Real PastukhovFactor = (exp(-Xi_e) / Xi_e);

	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor.val < 0.0)
		return 0.0;
	double Normalization = (Plasma->IonMass() / ElectronMass) * (1.0 / (Plasma->RhoStarRef() * Plasma->RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ee) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;
	return LossRate;
}

Real MirrorPlasma::IonPastukhovLossRate(Real V, Real Xi_i, Real n, Real Ti) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double MirrorRatio = B->MirrorRatio(V);
	Real tau_ii = Plasma->IonCollisionTime(n, Ti);
	double Sigma = 1.0; // Just ion-ion collisions
	Real PastukhovFactor = (exp(-Xi_i) / Xi_i);

	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor.val < 0.0)
		return 0.0;

	double Normalization = sqrt(Plasma->IonMass() / ElectronMass) * (1.0 / (Plasma->RhoStarRef() * Plasma->RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ii) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;

	return LossRate;
}

// Returns (1/(1 + Tau))*(1-1/R_m)*(M^2)
template <typename T>
T MirrorPlasma::CentrifugalPotential(T V, T omega, T Ti, T Te) const
{
	double MirrorRatio = B->MirrorRatio(V);
	T R = B->R_V(V);
	T tau = Ti / Te;
	T MachNumber = omega * R / sqrt(Te); // omega is normalised to c_s0 / a
	T Potential = (1.0 / (1.0 + tau)) * (1.0 - 1.0 / MirrorRatio) * MachNumber * MachNumber / 2.0;
	return Potential;
}
