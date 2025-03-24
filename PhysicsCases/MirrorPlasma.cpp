#include "MirrorPlasma.hpp"
#include <iostream>
#include <string>

REGISTER_PHYSICS_IMPL(MirrorPlasma);
const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.2, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;

const std::string B_file = "./Bfield.nc";

template <typename T>
T sign(T x)
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

		evolveLogDensity = toml::find_or(InternalConfig, "LogDensity", false);

		std::string Bfile = toml::find_or(InternalConfig, "MagneticFieldData", B_file);
		bool useNcBField = toml::find_or(InternalConfig, "useNcBField", false);
		double B_z = toml::find_or(InternalConfig, "Bz", 1.0);
		double Rm = toml::find_or(InternalConfig, "Rm", 3.0);
		double L_z = toml::find_or(InternalConfig, "Lz", 1.0) / a;

		double m = toml::find_or(InternalConfig, "FieldSlope", 0.0);
		if (useNcBField)
			B = createMagneticField<CylindricalMagneticField>(Bfile);
		else
			B = createMagneticField<StraightMagneticField>(L_z, B_z, Rm, xL, m); // std::make_shared<StraightMagneticField>(L_z, B_z, Rm, xL, m); //  //

		R_Lower = B->R_V(xL, 0.0);
		R_Upper = B->R_V(xR, 0.0);
		// B->CheckBoundaries(xL, xR);

		std::string IonType = toml::find_or(InternalConfig, "IonSpecies", "Deuterium");

		Plasma = std::make_unique<PlasmaConstants>(IonType, B, n0, T0, Z_eff, a * (R_Upper - R_Lower));

		// Compute phi to satisfy zero parallel current
		useAmbipolarPhi = toml::find_or(InternalConfig, "useAmbipolarPhi", false);
		if (useAmbipolarPhi)
			nAux = 1;

		// Integrate over momentum equation for constant voltage
		useConstantVoltage = toml::find_or(InternalConfig, "useConstantVoltage", false);
		if (useConstantVoltage)
		{
			double cs0 = std::sqrt(T0 / Plasma->IonMass());
			V0 = toml::find_or(InternalConfig, "PlasmaVoltage", 50e3); // default 50 kV
			V0 /= cs0;
			CurrentDecay = toml::find_or(InternalConfig, "CurrentDecay", 1e-5);
			gamma = toml::find_or(InternalConfig, "gamma", 1.0);
			gamma_d = toml::find_or(InternalConfig, "gamma_d", 0.0);
			gamma_h = toml::find_or(InternalConfig, "gamma_h", 0.0);
			growth = 0.0;
			nScalars = 3;
		}

		// Add floor for computed densities and temperatures
		MinDensity = toml::find_or(InternalConfig, "MinDensity", 1e-2);
		MinTemp = toml::find_or(InternalConfig, "MinTemp", 0.1);
		RelaxFactor = toml::find_or(InternalConfig, "RelaxFactor", 1.0);

		useMMS = toml::find_or(InternalConfig, "useMMS", false);
		growth = toml::find_or(InternalConfig, "MMSgrowth", 1.0);
		growth_factors = toml::find_or(InternalConfig, "growth_factors", std::vector<double>(nVars, 1.0));
		growth_rate = toml::find_or(InternalConfig, "MMSgrowth_rate", 0.5);

		SourceCap = toml::find_or(InternalConfig, "SourceCap", 1e5);

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

		uL[Channel::Density] = evolveLogDensity ? log(nEdge) : nEdge;
		uR[Channel::Density] = evolveLogDensity ? log(nEdge) : nEdge;
		uL[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
		uR[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
		uL[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
		uR[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;
		uL[Channel::AngularMomentum] = Omega_Lower * nEdge * R_Lower * R_Lower;
		uR[Channel::AngularMomentum] = Omega_Upper * nEdge * R_Upper * R_Upper;

		IRadial = -toml::find_or(InternalConfig, "IRadial", 0.1);											// Current in amps
		I0 = Plasma->IonMass() * n0 * sqrt(T0 / Plasma->IonMass()) * a * a * a / Plasma->NormalizingTime(); // Current normalizing factor
		IRadial /= I0;

		useNeutralModel = toml::find_or(InternalConfig, "useNeutralsModel", false);

		nNeutrals = toml::find_or(InternalConfig, "NeutralDensity", 1e18) / n0;

		LowerParticleSourceStrength = toml::find_or(InternalConfig, "LowerPS", 10.0);

		UpperParticleSourceStrength = toml::find_or(InternalConfig, "UpperPS", 0.0);

		ParticleSourceWidth = toml::find_or(InternalConfig, "ParticleSourceWidth", 0.02);
		ParticleSourceCenter = toml::find_or(InternalConfig, "ParticleSourceCenter", 0.5 * (R_Lower + R_Upper));

		// Adding diffusion for low values of density or pressure
		lowNDiffusivity = toml::find_or(InternalConfig, "lowNDiffusivity", 0.0);
		lowNThreshold = toml::find_or(InternalConfig, "lowNThreshold", 1.0);

		lowPDiffusivity = toml::find_or(InternalConfig, "lowPDiffusivity", 0.0);
		lowPThreshold = toml::find_or(InternalConfig, "lowPThreshold", 1.0);

		lowLDiffusivity = toml::find_or(InternalConfig, "lowLDiffusivity", 0.0);
		lowLThreshold = toml::find_or(InternalConfig, "lowLThreshold", 1.0);

		TeDiffusivity = toml::find_or(InternalConfig, "TeDiffusivity", 0.0);

		transitionLength = toml::find_or(InternalConfig, "transitionLength", 1.0);

		for (Index i = 0; i < nVars; ++i)
		{
			if (!isUpperBoundaryDirichlet(i))
				uL[i] = InitialDerivative(i, xL);
			if (!isLowerBoundaryDirichlet(i))
				uR[i] = InitialDerivative(i, xR);
		}
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

	Real2nd R_min = B->R_V(xL, 0.0);
	Real2nd R_max = B->R_V(xR, 0.0);
	Real2nd R = B->R_V(V, 0.0);

	Real2nd R_mid = (R_min + R_max) / 2.0;

	Real2nd nMid = InitialPeakDensity;
	Real2nd TeMid = InitialPeakTe;
	Real2nd TiMid = InitialPeakTi;

	Real2nd MMid = InitialPeakMachNumber;
	Real2nd OmegaMid = sqrt(TeMid) * MMid / R_mid;
	Real2nd Omega_Lower = sqrt(TeEdge) * MLower / R_Lower;
	Real2nd Omega_Upper = sqrt(TeEdge) * MUpper / R_Upper;

	double xmid = 0.5 * (xR + xL);
	Real2nd m = (Omega_Upper - Omega_Lower) / (xR - xL);
	double shape = 1 / MachWidth;

	Real2nd v = cos(pi * (R - R_mid) / (R_max - R_min)); //* exp(-shape * (R - R_mid) * (R - R_mid));

	/// Real2nd Te = TeEdge + tfac(growth_factors[Channel::ElectronEnergy]) * (TeMid - TeEdge) * v * v;
	Real2nd omega = Omega_Lower + m * (V - xL) + OmegaMid * cos(pi * (V - xmid) / (xR - xL));
	// TeEdge + tfac(growth_factors[Channel::ElectronEnergy]) * (TeMid - TeEdge) * v * v;
	Real2nd Ti = TiEdge + tfac(growth_factors[Channel::IonEnergy]) * (TiMid - TiEdge) * v * v;
	Real2nd n = nEdge + tfac(growth_factors[Channel::Density]) * (nMid - nEdge) * cos(pi * (V - xmid) / (xR - xL));
	auto slope = (MUpper - MLower) / (R_Upper - R_Lower);
	Real2nd M = MLower + slope * (R - R_Lower) + (MMid - 0.5 * (MUpper + MLower)) * v / exp(-shape * (R - R_mid) * (R - R_mid)); // MEdge + tfac(growth_factors[Channel::AngularMomentum]) * (MMid - MEdge) * (1 - (exp(-shape * (R - R_Upper) * (R - R_Upper)) + exp(-shape * (R - R_Lower) * (R - R_Lower))));
	Real2nd Te = R * R * omega * omega / (M * M);

	Channel c = static_cast<Channel>(i);
	switch (c)
	{
	case Channel::Density:
	{
		if (evolveLogDensity)
			return log(n);
		else
			return n;
		break;
	}
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
	{
		if (evolveLogDensity)
			return ConstantChannelMap[Channel::Density] ? 0.0 : static_cast<Real>(Gamma(u, q, x, t) / uToDensity(u(Channel::Density)));
		else
			return ConstantChannelMap[Channel::Density] ? 0.0 : Gamma(u, q, x, t);
		break;
	}
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

Real MirrorPlasma::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars, Real x, Time t)
{
	Channel c = static_cast<Channel>(i);
	Real S;

	switch (c)
	{
	case Channel::Density:
	{
		if (evolveLogDensity)
		{
			S = ConstantChannelMap[Channel::Density] ? 0.0 : static_cast<Real>(1 / uToDensity(u(Channel::Density)) * (Sn(u, q, sigma, phi, x, t) + Gamma(u, q, x, t) * q(Channel::Density)));
			return S; // Avoid source cap when evolving log
		}
		else
			S = ConstantChannelMap[Channel::Density] ? 0.0 : Sn(u, q, sigma, phi, x, t);
		break;
	}
	case Channel::IonEnergy:
		S = ConstantChannelMap[Channel::IonEnergy] ? 0.0 : Spi(u, q, sigma, phi, x, t);
		break;
	case Channel::ElectronEnergy:
		S = ConstantChannelMap[Channel::ElectronEnergy] ? 0.0 : Spe(u, q, sigma, phi, x, t);
		break;
	case Channel::AngularMomentum:
		S = ConstantChannelMap[Channel::AngularMomentum] ? 0.0 : Somega(u, q, sigma, phi, Scalars, x, t);
		break;
	default:
		throw std::runtime_error("Request for flux for undefined variable!");
	}
	if (abs(S) > SourceCap)
		S = sign(S) * SourceCap;
	return S;
}

Value MirrorPlasma::LowerBoundary(Index i, Time t) const
{
	return uL[i];
}
Value MirrorPlasma::UpperBoundary(Index i, Time t) const
{
	return uR[i];
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
	Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = p_e / n;
	Real Ti = p_i / n;

	Real nPrime = qToDensityGradient(q(Channel::Density), u(Channel::Density)), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);

	Real ThermalForce = (3. / 2.) * (p_e_prime - nPrime * Te); // p_e  * (1/T) * dT/dpsi

	Real R = B->R_V(V, 0.0);
	Real J = n * R * R; // Normalisation includes the m_i
	Real dRdV = B->dRdV(V, 0.0);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;

	Real L = u(Channel::AngularMomentum);
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real U = (p_e_prime + p_i_prime) + n * omega * R * R * dOmegadV; // -n*(U_e - U_i) dot grad phi

	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e
	Real Gamma = GeometricFactor * GeometricFactor * (1 / (Plasma->ElectronCollisionTime(n, Te))) * (U - ThermalForce);

	// If gradient is too steep for MaNTA to handle, add diffusion

	Real rhon = abs(sqrt(Ti) * Plasma->RhoStarRef() * nPrime / B->dRdV(V, 0.0) / n);
	// Real lambda_n = (B->L_V(V) * (R_Upper - R_Lower)) * abs(nPrime);

	Real x = (rhon - lowNThreshold) / lowNThreshold;

	// Real Chi_n = 1.0; // sqrt(Plasma->mu()) * pow(Ti, 3. / 2.) * lambda_n;

	if (x >= 0)
		Gamma += x * lowNDiffusivity * nPrime * GeometricFactor * R; // SmoothTransition(x, transitionLength, lowNDiffusivity) * Chi_n * GeometricFactor * R * nPrime * x;

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
	Real n = uToDensity(u(Channel::Density)), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	Real nPrime = qToDensityGradient(q(Channel::Density), u(Channel::Density)), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

	Real R = B->R_V(V, 0.0);
	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = 2.0 * GeometricFactor * GeometricFactor * sqrt(Plasma->mu() / 2.0) * (p_i / (Plasma->IonCollisionTime(n, Ti))) * Ti_prime;

	// Real lambda_p = abs(p_i_prime / p_i);

	// Real x = lambda_p - lowPThreshold;

	// if (lambda_p > lowPThreshold)
	// 	HeatFlux += SmoothTransition(x, transitionLength, lowPDiffusivity) * p_i_prime / p_i; // Ti_prime / Ti;

	// Real lambda_T = (B->L_V(V) * (R_Upper - R_Lower)) * abs(Ti_prime);

	Real rhoTi = abs(sqrt(Ti) * Plasma->RhoStarRef() * Ti_prime / B->dRdV(V, 0.0) / Ti);

	// Real x = lambda_T * Plasma->RhoStarRef() - lowPThreshold;
	Real x = (rhoTi - lowPThreshold) / lowPThreshold;

	// Real lambda_n = 1 / B->dRdV(V 0.0) * abs(nPrime / n);

	// Real x = sqrt(Ti) * lambda_n / (lowNThreshold / Plasma->RhoStarRef()) - 1.0;

	Real Chi_i = 1.0; // sqrt(Plasma->mu()) * pow(Ti, 3. / 2.) * lambda_T;

	if (x >= 0)
		HeatFlux += x * lowPDiffusivity * Chi_i * GeometricFactor * R * Ti_prime; // / Ti;

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
	Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = qToDensityGradient(q(Channel::Density), u(Channel::Density)), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;

	Real R = B->R_V(V, 0.0);
	Real J = n * R * R; // Normalisation includes the m_i
	Real dRdV = B->dRdV(V, 0.0);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;

	Real L = u(Channel::AngularMomentum);
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real U = Te * (p_e_prime + p_i_prime) + p_e * omega * R * R * dOmegadV; // -p_e*(U_e-U_i)

	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = GeometricFactor * GeometricFactor * (1 / (Plasma->ElectronCollisionTime(n, Te))) * (4.66 * p_e * Te_prime - (3. / 2.) * U);

	// Real Chi_e = pow(Te, 3. / 2.) * abs(1 / dRdV * Te_prime / Te);
	HeatFlux += TeDiffusivity * GeometricFactor * Te_prime;

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
	Real n = uToDensity(u(Channel::Density)), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	// dOmega dV = L'/J - J' L / J^2 ; L = angular momentum / J = moment of Inertia
	Real R = B->R_V(V, 0.0);

	Real J = n * R * R; // Normalisation includes the m_i
	Real nPrime = qToDensityGradient(q(Channel::Density), u(Channel::Density));
	Real dRdV = B->dRdV(V, 0.0);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;

	Real L = u(Channel::AngularMomentum);
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real Pi_v = IonClassicalAngularMomentumFlux(V, n, Ti, omega, dOmegadV, t) + omega * R * R * Gamma(u, q, V, t);

	// Real lambda_L = abs(LPrime / L);

	// Real x = lambda_L - lowLThreshold;

	// if (lambda_L > lowLThreshold)
	// 	Pi_v += SmoothTransition(x, transitionLength, lowLDiffusivity) * LPrime / L;
	Real GeometricFactor = (B->VPrime(V) * R * R);
	// Real x = sqrt(Ti) * lambda_omega / (lowLThreshold / Plasma->RhoStarRef()) - 1.0;
	// // Real GeometricFactor = (B->VPrime(V) * R);
	// // Real lambda_n = 1 / dRdV * abs(nPrime / n);

	// // Real x = sqrt(Ti) * lambda_n / (lowNThreshold / Plasma->RhoStarRef()) - 1.0;

	// Real Chi_L = 1.0; // sqrt(Plasma->mu()) * pow(Ti, 3. / 2.) * lambda_omega;
	Real Chi_L = 1.0; // sqrt(Plasma->mu()) * pow(Ti, 3. / 2.) * lambda_omega;

	Real rho_omega = abs(sqrt(Ti) * Plasma->RhoStarRef() * dOmegadV / B->dRdV(V, 0.0) / omega);
	// Real lambda_omega = (B->L_V(V) * (R_Upper - R_Lower)) * abs(dOmegadV);
	// Real x = lambda_omega * Plasma->RhoStarRef() - lowLThreshold;
	Real x = (rho_omega - lowLThreshold) / lowLThreshold;
	if (x >= 0)
		Pi_v += x * lowLDiffusivity * GeometricFactor * Chi_L * (R * R * dOmegadV);
	// if (x > 0)
	// 	Pi_v += SmoothTransition(x, transitionLength, lowLDiffusivity) * GeometricFactor * GeometricFactor * Chi_L * (R * R * dOmegadV); /// (R * R * omega);

	return Pi_v;
};

/*
   Returns V' pi_cl_i
 */
Real MirrorPlasma::IonClassicalAngularMomentumFlux(Real V, Real n, Real Ti, Real omega, Real dOmegadV, Time t) const
{
	Real R = B->R_V(V, 0.0);
	Real GeometricFactor = (B->VPrime(V) * R * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real MomentumFlux = 0.3 * GeometricFactor * GeometricFactor * sqrt(Plasma->mu() / 2.0) * (n * Ti / (Plasma->IonCollisionTime(n, Ti))) * dOmegadV;

	if (std::isfinite(MomentumFlux.val))
		return MomentumFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion momentum flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

Real MirrorPlasma::Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) const
{
	Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);

	Real R = B->R_V(V, 0.0);

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
	Real DensitySource;
	if (useNeutralModel)
		DensitySource = ParticleSource(R.val, t) + Plasma->NormalizingTime() / n0 * (Plasma->IonizationRate(n, NeutralDensity(R, t), R * omega, Te, Ti));
	else
		DensitySource = ParticleSource(R.val, t);
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
	Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);
	// pi * d omega / d psi = (V'pi)*(d omega / d V)
	Real R = B->R_V(V, 0.0);
	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real nPrime = qToDensityGradient(q(Channel::Density), u(Channel::Density));
	Real dRdV = B->dRdV(V, 0.0);
	Real JPrime = R * R * nPrime + 2.0 * dRdV * R * n;
	Real LPrime = q(Channel::AngularMomentum);
	Real dOmegadV = LPrime / J - JPrime * L / (J * J);
	Real omega = L / J;

	Real ViscousHeating = IonClassicalAngularMomentumFlux(V, n, Ti, omega, dOmegadV, t) * dOmegadV;

	// Use this version since we have no parallel flux in a square well
	Real PotentialHeating = 0.0; //-G * (omega * omega / (2 * pi * a) - dphidV(u, q, phi, V));
	Real EnergyExchange = Plasma->IonElectronEnergyExchange(n, p_e, p_i, V, t);

	Real Heating = ViscousHeating + PotentialHeating + EnergyExchange;

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

	Real ChargeExchangeHeatLosses = 0;
	Real ParticleSourceHeating = 0.0;
	// if (useNeutralModel)
	// {
	ChargeExchangeHeatLosses = Ti * Plasma->NormalizingTime() / n0 * Plasma->ChargeExchangeLossRate(n, NeutralDensity(R, t), R * omega, Ti);
	// 	// ParticleSourceHeating = 0.5 * omega * omega * R * R * (ParticleSource(R.val, t) + Plasma->NormalizingTime() / n0 * (Plasma->IonizationRate(n, NeutralDensity(R, t), R * omega, Te, Ti) - Plasma->ChargeExchangeLossRate(n, NeutralDensity(R, t), R * omega, Ti)));
	// }

	Real S = Heating - ParallelLosses - ChargeExchangeHeatLosses + ParticleSourceHeating;

	return S; //+ RelaxSource(u(Channel::Density) * Te, p_e) + RelaxSource(n * floor(Te, MinTemp), p_e);
}

Real MirrorPlasma::Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) const
{
	Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);
	Real EnergyExchange = -Plasma->IonElectronEnergyExchange(n, p_e, p_i, V, t);

	Real MirrorRatio = B->MirrorRatio(V, 0.0);
	Real AlphaHeating = sqrt(1 - 1 / MirrorRatio) * Plasma->TotalAlphaPower(n, p_i);

	Real R = B->R_V(V, 0.0);

	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real omega = L / J;

	Real PotentialHeating = 0.0; // ElectronPotentialHeating(u, q, phi, V); //(dphi1dV(u, q, phi(0), V));

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

	Real RadiationLosses = Plasma->BremsstrahlungLosses(n, p_e); //+ Plasma->CyclotronLosses(V, n, Te);

	Real S = Heating - ParallelLosses - RadiationLosses;

	return S; //+ RelaxSource(u(Channel::Density) * Te, p_e) + RelaxSource(n * floor(Te, MinTemp), p_e);
};

// Source of angular momentum -- this is just imposed J x B torque (we can account for the particle source being a sink later).
Real MirrorPlasma::Somega(RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars, Real V, Time t) const
{
	// J x B torque
	Real R = B->R_V(V, 0.0);

	Real n = uToDensity(u(Channel::Density)), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);

	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);

	Real RadialCurrent;
	if (useConstantVoltage)
		RadialCurrent = -Scalars(Scalar::Current);
	else
		RadialCurrent = IRadial;

	Real JxB = -RadialCurrent / B->VPrime(V); //-jRadial * R * B->Bz_R(R);
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

	Real ChargeExchangeMomentumLosses = 0;
	// if (useNeutralModel)
	ChargeExchangeMomentumLosses = AngularMomentumPerParticle * Plasma->NormalizingTime() / n0 * Plasma->ChargeExchangeLossRate(n, NeutralDensity(R, t), R * omega, Ti);

	return JxB - ParallelLosses - ChargeExchangeMomentumLosses; //+ RelaxSource(omega * R * R * u(Channel::Density), L);
};

Real MirrorPlasma::uToDensity(Real un) const
{
	if (evolveLogDensity)
		return exp(un);
	else
		return floor(un, MinDensity);
}

Real MirrorPlasma::qToDensityGradient(Real qn, Real un) const
{
	if (evolveLogDensity)
		return uToDensity(un) * qn;
	else
		return qn;
}

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

Real MirrorPlasma::dphidV(RealVector u, RealVector q, RealVector phi, Real V) const
{
	Real dphidV = 0.0; // dphi0dV(u, q, V);

	// if (nAux > 0)
	// 	dphidV += dphi1dV(u, q, phi(0), V);

	return dphidV;
}

inline Real MirrorPlasma::AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const
{
	Real R = B->MirrorRatio(V, 0.0);
	double Sigma = 1.0;
	return log((Plasma->ElectronCollisionTime(n, Te) / Plasma->IonCollisionTime(n, Ti)) * (log(R * Sigma) / (Sigma * log(R))));
}

Real MirrorPlasma::ParticleSource(double R, double t) const
{

	// // return (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));
	Real S = LowerParticleSourceStrength * exp(-(R - R_Lower) / ParticleSourceWidth) + UpperParticleSourceStrength * exp((R - R_Upper) / ParticleSourceWidth);
	// // return LowerParticleSourceStrength; //* exp(-t / 5e-2);

	// return
	// return LowerParticleSourceStrength +
	// 	   UpperParticleSourceStrength * exp(-shape * (R - ParticleSourceCenter) * (R - ParticleSourceCenter));
	return S;

	//   return ParticleSourceStrength;
};

Real MirrorPlasma::ElectronPastukhovLossRate(Real V, Real Xi_e, Real n, Real Te) const
{
	Real MirrorRatio = B->MirrorRatio(V, 0.0);
	Real tau_ee = Plasma->ElectronCollisionTime(n, Te);
	double Sigma = 1 + Z_eff; // Include collisions with ions and impurities as well as self-collisions
	Real PastukhovFactor = (exp(-Xi_e) / Xi_e);

	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor < 0.0)
		return 0.0;
	double Normalization = Plasma->mu() * (1.0 / (Plasma->RhoStarRef() * Plasma->RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ee) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;
	return LossRate;
}

Real MirrorPlasma::NeutralDensity(Real R, Time t) const
{
	return nNeutrals; //* exp(-(R - R_Lower) / ParticleSourceWidth);
}

Real MirrorPlasma::IonPastukhovLossRate(Real V, Real Xi_i, Real n, Real Ti) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	Real MirrorRatio = B->MirrorRatio(V, 0.0);
	Real tau_ii = Plasma->IonCollisionTime(n, Ti);
	double Sigma = 1.0; // Just ion-ion collisions
	Real PastukhovFactor = (exp(-Xi_i) / Xi_i);

	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor < 0.0)
		return 0.0;

	double Normalization = sqrt(Plasma->mu()) * (1.0 / (Plasma->RhoStarRef() * Plasma->RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ii) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;

	return LossRate;
}

Real MirrorPlasma::IonPotentialHeating(RealVector u, RealVector q, RealVector phi, Real V) const
{
	Real n = floor(u(Channel::Density), MinDensity);

	Real R = B->R_V(V, 0.0);

	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real omega = L / J;
	return -Gamma(u, q, V, 0.0) * (omega * omega / (2 * pi * a) - dphidV(u, q, phi, V));
}

Real MirrorPlasma::ElectronPotentialHeating(RealVector u, RealVector q, RealVector phi, Real V) const
{
	return -Gamma(u, q, V, 0.0) * dphidV(u, q, phi, V);
}
