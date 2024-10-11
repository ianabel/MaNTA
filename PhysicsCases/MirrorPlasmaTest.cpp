#include "MirrorPlasmaTest.hpp"
#include "Constants.hpp"
#include <iostream>
#include <string>
#include <boost/math/tools/roots.hpp>

REGISTER_PHYSICS_IMPL(MirrorPlasmaTest);
const double n_mid = 0.25;
const double n_edge = 0.05;
const double T_mid = 0.2, T_edge = 0.1;

const double omega_edge = 0.1, omega_mid = 1.0;
#ifdef DEBUG
const std::string B_file = "/home/eatocco/projects/MaNTA/Bfield.nc";
#else
const std::string B_file = "Bfield.nc";
#endif

template <typename T>
int sign(T x)
{
	return x >= 0 ? 1 : -1;
}

MirrorPlasmaTest::MirrorPlasmaTest(toml::value const &config, Grid const &grid)
{
	nVars = 4;
	nScalars = 0;
	nAux = 0;

	// B = new StraightMagneticField();

	xL = grid.lowerBoundary();
	xR = grid.upperBoundary();

	// isLowerDirichlet = true;
	// isUpperDirichlet = true;

	if (config.count("MirrorPlasmaTest") == 1)
	{
		auto const &InternalConfig = config.at("MirrorPlasmaTest");

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

		MinDensity = toml::find_or(InternalConfig, "MinDensity", 1e-2);
		MinTemp = toml::find_or(InternalConfig, "MinTemp", 0.1);
		RelaxFactor = toml::find_or(InternalConfig, "RelaxFactor", 1.0);

		useMMS = toml::find_or(InternalConfig, "useMMS", false);
		growth = toml::find_or(InternalConfig, "MMSgrowth", 1.0);
		growth_factors = toml::find_or(InternalConfig, "growth_factors", std::vector<double>(nVars, 1.0));
		growth_rate = toml::find_or(InternalConfig, "MMSgrowth_rate", 0.5);
		SourceCap = toml::find_or(InternalConfig, "SourceCap", 1e5);
		// test source
		EdgeSourceSize = toml::find_or(InternalConfig, "EdgeSourceSize", 0.0);
		EdgeSourceWidth = toml::find_or(InternalConfig, "EdgeSourceWidth", 1e-5);
		ZeroEdgeSources = toml::find_or(InternalConfig, "ZeroEdgeSources", false);
		ZeroEdgeFactor = toml::find_or(InternalConfig, "ZeroEdgeFactor", 0.9);
		EnergyExchangeFactor = toml::find_or(InternalConfig, "EnergyExchangeFactor", 1.0);
		MaxPastukhov = toml::find_or(InternalConfig, "MaxLossRate", 1.0);

		ParallelLossFactor = toml::find_or(InternalConfig, "ParallelLossFactor", 1.0);
		ViscousHeatingFactor = toml::find_or(InternalConfig, "ViscousHeatingFactor", 1.0);
		DragFactor = toml::find_or(InternalConfig, "DragFactor", 1.0);
		DragWidth = toml::find_or(InternalConfig, "DragWidth", 0.01);
		UniformHeatSource = toml::find_or(InternalConfig, "UniformHeatSource", 0.0);
		ParticlePhysicsFactor = toml::find_or(InternalConfig, "ParticlePhysicsFactor", 1.0);
		PotentialHeatingFactor = toml::find_or(InternalConfig, "PotentialHeatingFactor", 1.0);

		std::string Bfile = toml::find_or(InternalConfig, "MagneticFieldData", B_file);
		double B_z = toml::find_or(InternalConfig, "Bz", 1.0);
		double Rm = toml::find_or(InternalConfig, "Rm", 3.0);
		double L_z = toml::find_or(InternalConfig, "Lz", 1.0);

		B = new StraightMagneticField(L_z, B_z, Rm);
		// B->CheckBoundaries(xL, xR);

		R_Lower = B->R_V(xL);
		R_Upper = B->R_V(xR);

		loadInitialConditionsFromFile = toml::find_or(InternalConfig, "useNcFile", false);
		if (loadInitialConditionsFromFile)
		{
			filename = toml::find_or(InternalConfig, "InitialConditionFilename", "MirrorPlasmaTestRERUN.nc");
			LoadDataToSpline(filename);

			// uL[Channel::Density] = InitialValue(Channel::Density, xL);
			// uR[Channel::Density] = InitialValue(Channel::Density, xR);
			// uL[Channel::IonEnergy] = InitialValue(Channel::IonEnergy, xL);
			// uR[Channel::IonEnergy] = InitialValue(Channel::IonEnergy, xR);
			// uL[Channel::ElectronEnergy] = InitialValue(Channel::ElectronEnergy, xL);
			// uR[Channel::ElectronEnergy] = InitialValue(Channel::ElectronEnergy, xR);
			// uL[Channel::AngularMomentum] = InitialValue(Channel::AngularMomentum, xL);
			// uR[Channel::AngularMomentum] = InitialValue(Channel::AngularMomentum, xR);
		}
		// else
		// {

		nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
		TeEdge = toml::find_or(InternalConfig, "EdgeElectronTemperature", T_edge);
		TiEdge = toml::find_or(InternalConfig, "EdgeIonTemperature", TeEdge);
		MLower = toml::find_or(InternalConfig, "LowerMachNumber", omega_edge);
		MUpper = toml::find_or(InternalConfig, "UpperMachNumber", omega_edge);
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
		// }

		IRadial = -toml::find_or(InternalConfig, "IRadial", 4.0);
		ParticleSourceStrength = toml::find_or(InternalConfig, "ParticleSource", 10.0);
		ParticleSourceWidth = toml::find_or(InternalConfig, "ParticleSourceWidth", 0.02);
		ParticleSourceCenter = toml::find_or(InternalConfig, "ParticleSourceCenter", 0.5 * (R_Lower + R_Upper));
	}
	else if (config.count("MirrorPlasmaTest") == 0)
	{
		throw std::invalid_argument("To use the Mirror Plasma physics model, a [MirrorPlasmaTest] configuration section is required.");
	}
	else
	{
		throw std::invalid_argument("Unable to find unique [MirrorPlasmaTest] configuration section in configuration file.");
	}
};

Real2nd MirrorPlasmaTest::InitialFunction(Index i, Real2nd V, Real2nd t) const
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

Real MirrorPlasmaTest::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
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

Real MirrorPlasmaTest::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t)
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

Real MirrorPlasmaTest::GFunc(Index, RealVector u, RealVector, RealVector, RealVector phi, Position V, Time t)
{

	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);

	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	Real omega = u(Channel::AngularMomentum) / J;
	return ParallelCurrent<Real>(V, omega, n, Ti, Te, phi(0));
}

Value MirrorPlasmaTest::InitialAuxValue(Index, Position V) const
{
	using boost::math::tools::eps_tolerance;
	using boost::math::tools::newton_raphson_iterate;

	if (loadInitialConditionsFromFile)
	{
		return (*NcFileInitialAuxValue[0])(V);
	}
	else
	{
		Real n = InitialValue(Channel::Density, V), p_e = (2. / 3.) * InitialValue(Channel::ElectronEnergy, V), p_i = (2. / 3.) * InitialValue(Channel::IonEnergy, V);
		Real Te = p_e / n;
		Real Ti = p_i / n;

		Real R = B->R_V(V);

		Real J = n * R * R; // Normalisation of the moment of inertia includes the m_i
		Real omega = InitialValue(Channel::AngularMomentum, V) / J;

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
		// std::pair<double, double> phi_g = bisect(func, -0.9 * (M - 1), 0.0, tol, it);
		double phi_g = newton_raphson_iterate([&func, &deriv_func](double phi)
											  { return std::pair<double, double>(func(phi), deriv_func(phi)); }, 0.0, -CentrifugalPotential(V, omega.val, Ti.val, Te.val), 0.01, get_digits, it);
		return phi_g;
		// return phi_g.first + (phi_g.second - phi_g.first) / 2;
	}
}

Value MirrorPlasmaTest::LowerBoundary(Index i, Time t) const
{
	return isLowerBoundaryDirichlet(i) ? uL[i] : 0.0;
}
Value MirrorPlasmaTest::UpperBoundary(Index i, Time t) const
{
	return isUpperBoundaryDirichlet(i) ? uR[i] : 0.0;
}

bool MirrorPlasmaTest::isLowerBoundaryDirichlet(Index i) const
{
	return lowerBoundaryConditions[i];
}

bool MirrorPlasmaTest::isUpperBoundaryDirichlet(Index i) const
{
	return upperBoundaryConditions[i];
}

Real2nd MirrorPlasmaTest::MMS_Solution(Index i, Real2nd x, Real2nd t)
{
	return InitialFunction(i, x, t);
}

/*
Normalisation:
   All lengths to a, densities to n0, temperatures to T0
   We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2 tau_e(n0,T0) ) ]^-1
	in effect we are normalising to the particle diffusion time across a distance 1

 */

// This is c_s / ( Omega_i * a )
// = sqrt( T0 / mi ) / ( e B0 / mi ) =  [ sqrt( T0 mi ) / ( e B0 ) ] / a
inline double MirrorPlasmaTest::RhoStarRef() const
{
	return sqrt(T0 * IonMass) / (ElementaryCharge * B0 * a);
}

// Return this normalised to log Lambda at n0,T0
template <typename T>
T MirrorPlasmaTest::LogLambda_ei(T ne, T Te) const
{
	double LogLambdaRef = 23.0 - log(2.0) - log(n0cgs) / 2.0 + log(T0eV) * 1.5;
	T LogLambda = 23.0 - log(2.0) - log(ne * n0cgs) / 2.0 + log(Te * T0eV) * 1.5;
	return LogLambda / LogLambdaRef; // really needs to know Ti as well
}

// Return this normalised to log Lambda at n0,T0
template <typename T>
T MirrorPlasmaTest::LogLambda_ii(T ni, T Ti) const
{
	double LogLambdaRef = 24.0 - log(n0cgs) / 2.0 + log(T0eV);
	T LogLambda = 24.0 - log(n0cgs * ni) / 2.0 + log(T0eV * Ti);
	return LogLambda / LogLambdaRef; // really needs to know Ti as well
}

// Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
// This is equal to tau_e as used in Braginskii

template <typename T>
T MirrorPlasmaTest::ElectronCollisionTime(T ne, T Te) const
{
	return pow(Te, 1.5) / (ne * LogLambda_ei(ne, Te));
}

// Return the actual value in SI units
inline double MirrorPlasmaTest::ReferenceElectronCollisionTime() const
{
	double LogLambdaRef = 24.0 - log(n0cgs) / 2.0 + log(T0eV); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
	return 12.0 * pow(M_PI, 1.5) * sqrt(ElectronMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (sqrt(2) * n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}
// Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
// This is equal to tau_i as used in Braginskii
template <typename T>
inline T MirrorPlasmaTest::IonCollisionTime(T ni, T Ti) const
{
	return pow(Ti, 1.5) / (ni * LogLambda_ii(ni, Ti));
}

// Return the actual value in SI units
inline double MirrorPlasmaTest::ReferenceIonCollisionTime() const
{
	double LogLambdaRef = 23.0 - log(2.0) - log(n0cgs) / 2.0 + log(T0eV) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
	return 12.0 * pow(M_PI, 1.5) * sqrt(IonMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
}

// We are in a quasineutral plasma with one ion species.
// This function returns V' * Gamma_e, and Gamma_i = Gamma_e
// c.f Helander & Sigmar -- Gamma_e = (n_e T_e / (m_e Omega_e^2 tau_e))*( (p_e' + p_i')/p_e - (3/2)(T_e'/T_e)
// Define lengths so R_ref = 1
Real MirrorPlasmaTest::Gamma(RealVector u, RealVector q, Real V, double t) const
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
	Real Gamma = GeometricFactor * GeometricFactor * (p_e / (ElectronCollisionTime(n, Te))) * (PressureGradient - TemperatureGradient);

	// if ((u(Channel::Density) <= MinDensity))
	// 	Gamma.val = 0.0;

	// Gamma *= floor((1 - 5 * (n - u(Channel::Density)) / MinDensity), 0) + 5 * (n - u(Channel::Density)) / MinDensity * nPrime;

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
Real MirrorPlasmaTest::qi(RealVector u, RealVector q, Real V, double t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Ti = p_i / n;
	Real nPrime = q(Channel::Density), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Ti_prime = (p_i_prime - nPrime * Ti) / n;

	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = 2.0 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (p_i / (IonCollisionTime(n, Ti))) * Ti_prime;

	if (std::isfinite(HeatFlux.val))
		return HeatFlux;
	else
		throw std::logic_error("Non-finite value computed for the ion heat flux at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

/*
   Following Helander & Sigmar, we have
   V' q_e = n_e T_e * V'^2 B^2 R^2 * ( T_e / m_e Omega_e^2 tau_e ) * ( 4.66 T_e'/T_e - (3/2) * (p_e'+p_i')/p_e )
 */
Real MirrorPlasmaTest::qe(RealVector u, RealVector q, Real V, double t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy);
	Real Te = p_e / n;
	Real nPrime = q(Channel::Density), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy), p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
	Real Te_prime = (p_e_prime - nPrime * Te) / n;

	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real HeatFlux = GeometricFactor * GeometricFactor * (p_e * Te / (ElectronCollisionTime(n, Te))) * (4.66 * Te_prime / Te - (3. / 2.) * (p_e_prime + p_i_prime) / p_e);

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
Real MirrorPlasmaTest::Pi(RealVector u, RealVector q, Real V, double t) const
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
Real MirrorPlasmaTest::IonClassicalAngularMomentumFlux(Real V, Real n, Real Ti, Real dOmegadV, double t) const
{
	Real R = B->R_V(V);
	Real GeometricFactor = (B->VPrime(V) * R * R); // |grad psi| = R B , cancel the B with the B in Omega_e, leaving (V'R)^2
	Real MomentumFlux = 0.3 * GeometricFactor * GeometricFactor * sqrt(IonMass / (2.0 * ElectronMass)) * (n * Ti / (IonCollisionTime(n, Ti))) * dOmegadV;
	if (std::isfinite(MomentumFlux.val))
		return MomentumFlux;
	else
		return 0.0;
	// throw std::logic_error("Non-finite value computed for the ion momentum flux at x = " + std::to_string(V) + " and t = " + std::to_string(t));
}

Real MirrorPlasmaTest::Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, double t) const
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
	Real FusionLosses = ParticlePhysicsFactor * FusionRate(n, p_i);

	Real S = DensitySource - ParallelLosses - FusionLosses;

	Real RelaxDensity = 0.0;
	if (!isUpperBoundaryDirichlet(Channel::Density) || !isLowerBoundaryDirichlet(Channel::Density))
		RelaxDensity = RelaxEdge(V, u(Channel::Density), MinDensity);

	return S - RelaxSource(u(Channel::Density), n) + R * RelaxDensity;
};

/*
 *  Source terms in Ion heating equation
 *
 *  - pi_i * domega/dpsi + Gamma_i m_i omega^2 / B_z + Q_i
 *
 * where Q_i is the collisional heating
 */
Real MirrorPlasmaTest::Spi(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, double t) const
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

	Real ViscousHeating = ViscousHeatingFactor * IonClassicalAngularMomentumFlux(V, n, Ti, dOmegadV, t) * dOmegadV;

	// Use this version since we have no parallel flux in a square well
	Real PotentialHeating = -PotentialHeatingFactor * Gamma(u, q, V, t) * (omega * omega / (2 * pi * a) - dphidV(u, q, phi, V));
	Real EnergyExchange = IonElectronEnergyExchange(n, p_e, p_i, V, t);

	// Real ParticleSourceHeating = 0.5 * omega * omega * R * R * ParticleSource(R.val, t);

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

Real MirrorPlasmaTest::Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, double t) const
{
	Real n = floor(u(Channel::Density), MinDensity), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);
	Real Te = floor(p_e / n, MinTemp);
	Real Ti = floor(p_i / n, MinTemp);
	Real EnergyExchange = -IonElectronEnergyExchange(n, p_e, p_i, V, t);

	double MirrorRatio = B->MirrorRatio(V);
	Real AlphaHeating = ParticlePhysicsFactor * sqrt(1 - 1 / MirrorRatio) * TotalAlphaPower(n, p_i);

	Real R = B->R_V(V);

	Real J = n * R * R; // Normalisation includes the m_i
	Real L = u(Channel::AngularMomentum);
	Real omega = L / J;
	// Real ParticleSourceHeating = electronMass / ionMass * .5 * omega * omega * R * R * ParticleSource(R.val, t);

	Real PotentialHeating = -PotentialHeatingFactor * Gamma(u, q, V, t) * dphidV(u, q, phi, V); //(dphi1dV(u, q, phi(0), V));

	Real Heating = EnergyExchange + AlphaHeating + PotentialHeating;

	Real Xi;
	if (useAmbipolarPhi)
	{
		Xi = Xi_e(V, phi(0), Ti, Te, omega); //+ AmbipolarPhi(V, n, Ti, Te) / 2.0;
	}
	else
	{
		Xi = CentrifugalPotential(V, omega, Ti, Te) + AmbipolarPhi(V, n, Ti, Te) / 2.0;
	}
	Real ParticleEnergy = Te * (1.0 + Xi);
	Real ParallelLosses = ParticleEnergy * ElectronPastukhovLossRate(V, Xi, n, Te);

	Real RadiationLosses = ParticlePhysicsFactor * BremsstrahlungLosses(n, p_e) + CyclotronLosses(V, n, Te);

	Real S = Heating - ParallelLosses - RadiationLosses;

	return S + RelaxSource(u(Channel::Density) * Te, p_e) + RelaxSource(n * floor(Te, MinTemp), p_e);
};

// Source of angular momentum -- this is just imposed J x B torque (we can account for the particle source being a sink later).
Real MirrorPlasmaTest::Somega(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, double t) const
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
	Real vtheta = omega * R;

	// double shape = 1 / DragWidth;
	// Real Drag = (omega * B->Bz_R(R) * DragFactor) * (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));

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
	Real RelaxMach = 0.0;
	Real M = vtheta / sqrt(Te);
	if (!isUpperBoundaryDirichlet(Channel::AngularMomentum) || !isLowerBoundaryDirichlet(Channel::AngularMomentum))
		RelaxMach = RelaxEdge(V, M, MEdge);

	return JxB - ParallelLosses + R * RelaxMach + RelaxSource(omega * R * R * u(Channel::Density), L);
};

// Template function to avoid annoying dual vs. dual2nd behavior
template <typename T>
T MirrorPlasmaTest::phi0(Eigen::Matrix<T, -1, 1, 0, -1, 1> u, T V) const
{
	T n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);

	T Te = p_e / n, Ti = p_i / n;
	T L = u(Channel::AngularMomentum);
	T R = B->R_V(V);
	T J = n * R * R; // Normalisation of the moment of inertia includes the m_i
	T omega = L / J;
	T phi = 0.5 / (1 / Ti + 1 / Te) * omega * omega * R * R / Ti * (1 / B->MirrorRatio(V) - 1);

	return phi;
}

// Use the chain rule to calculate dphi0/dV, making sure to set gradient values correctly
Real MirrorPlasmaTest::dphi0dV(RealVector u, RealVector q, Real V) const
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
Real MirrorPlasmaTest::dphi1dV(RealVector u, RealVector q, Real phi, Real V) const
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

Real MirrorPlasmaTest::dphidV(RealVector u, RealVector q, RealVector phi, Real V) const
{
	Real dphidV = dphi0dV(u, q, V);

	if (nAux > 0)
		dphidV += dphi1dV(u, q, phi(0), V);

	return dphidV;
}

// Energy normalisation is T0, but these return Xi_s / T_s as that is what enters the
// Pastukhov factor
template <typename T>
T MirrorPlasmaTest::Xi_i(T V, T phi, T Ti, T Te, T omega) const
{
	return CentrifugalPotential<T>(V, omega, Ti, Te) + phi;
}
template <typename T>
T MirrorPlasmaTest::Xi_e(T V, T phi, T Ti, T Te, T omega) const
{
	return CentrifugalPotential<T>(V, omega, Ti, Te) - Ti / Te * phi;
}

inline Real MirrorPlasmaTest::AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const
{
	double R = B->MirrorRatio(V);
	double Sigma = 1.0;
	return log((ElectronCollisionTime(n, Te) / IonCollisionTime(n, Ti)) * (log(R * Sigma) / (Sigma * ::log(R))));
}

template <typename T>
T MirrorPlasmaTest::ParallelCurrent(T V, T omega, T n, T Ti, T Te, T phi) const
{
	// T phi0 = CentrifugalPotential(V, omega, Ti, Te);
	// T a = sqrt(ElectronMass / IonMass) * ElectronCollisionTime(n, Te) / IonCollisionTime(n, Ti);
	// T j = 1 / a * exp(2 * phi) * (1 + phi / phi0) - (1 - phi / phi0);
	T Xii = Xi_i<T>(V, phi, Ti, Te, omega);
	T Xie = Xi_e<T>(V, phi, Ti, Te, omega);

	double MirrorRatio = B->MirrorRatio(V);

	double Sigma_i = 1.0;
	double Sigma_e = 1 + Z_eff;

	T a = Sigma_i * (1.0 / log(MirrorRatio * Sigma_i)) * sqrt(IonMass / ElectronMass) / IonCollisionTime(n, Ti);
	T b = Sigma_e * (1.0 / log(MirrorRatio * Sigma_e)) * (IonMass / ElectronMass) / ElectronCollisionTime(n, Te);

	T j = a * exp(-Xii) / Xii - b * exp(-Xie) / Xie;

	// Real ji = IonPastukhovLossRate(V, Xii, n, Ti);
	// Real je = ElectronPastukhovLossRate(V, Xie, n, Te);
	// Real j = ji - je;
	return j;
}

/*
   In SI this is

   Q_i = 3 n_e m_e ( T_e - T_i ) / ( m_i tau_e )

   Normalised for the Energy equation, whose overall normalising factor is
   n0 T0^2 / ( m_e Omega_e_ref^2 tau_e_ref )


   Q_i = 3 * (p_e - p_i) / (tau_e) * (m_e/m_i) / (rho_s/R_ref)^2

 */
Real MirrorPlasmaTest::IonElectronEnergyExchange(Real n, Real pe, Real pi, Real V, double t) const
{
	Real Te = pe / n;
	double RhoStar = RhoStarRef();
	Real pDiff = pe - pi;
	Real IonHeating = EnergyExchangeFactor * (pDiff / (ElectronCollisionTime(n, Te))) * ((3.0 / (RhoStar * RhoStar))); //* (ElectronMass / IonMass));

	if (std::isfinite(IonHeating.val))
		return IonHeating;
	else
		throw std::logic_error("Non-finite value computed for the ion heating at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
}

Real MirrorPlasmaTest::ParticleSource(double R, double t) const
{
	double shape = 1 / ParticleSourceWidth;
	return (exp(-shape * (R - R_Lower) * (R - R_Lower)) + exp(-shape * (R - R_Upper) * (R - R_Upper)));
	// return exp(-shape * (R - ParticleSourceCenter) * (R - ParticleSourceCenter));
	//   return ParticleSourceStrength;
};

Real MirrorPlasmaTest::ElectronPastukhovLossRate(Real V, Real Xi_e, Real n, Real Te) const
{
	double MirrorRatio = B->MirrorRatio(V);
	Real tau_ee = ElectronCollisionTime(n, Te);
	double Sigma = 1 + Z_eff; // Include collisions with ions and impurities as well as self-collisions
	Real PastukhovFactor = (exp(-Xi_e) / Xi_e);

	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor.val < 0.0)
		return 0.0;
	double Normalization = (IonMass / ElectronMass) * (1.0 / (RhoStarRef() * RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ee) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;
	return ParallelLossFactor * LossRate;
}

Real MirrorPlasmaTest::IonPastukhovLossRate(Real V, Real Xi_i, Real n, Real Ti) const
{
	// For consistency, the integral in Pastukhov's paper is 1.0, as the
	// entire theory is an expansion in M^2 >> 1
	double MirrorRatio = B->MirrorRatio(V);
	Real tau_ii = IonCollisionTime(n, Ti);
	double Sigma = 1.0; // Just ion-ion collisions
	Real PastukhovFactor = (exp(-Xi_i) / Xi_i);

	// If the loss becomes a gain, flatten at zero
	if (PastukhovFactor.val < 0.0)
		return 0.0;

	double Normalization = sqrt(IonMass / ElectronMass) * (1.0 / (RhoStarRef() * RhoStarRef()));
	Real LossRate = (M_2_SQRTPI / tau_ii) * Normalization * Sigma * n * (1.0 / log(MirrorRatio * Sigma)) * PastukhovFactor;

	return ParallelLossFactor * LossRate;
}

// Returns (1/(1 + Tau))*(1-1/R_m)*(M^2)
template <typename T>
T MirrorPlasmaTest::CentrifugalPotential(T V, T omega, T Ti, T Te) const
{
	double MirrorRatio = B->MirrorRatio(V);
	T R = B->R_V(V);
	T tau = Ti / Te;
	T MachNumber = omega * R / sqrt(Te); // omega is normalised to c_s0 / a
	T Potential = (1.0 / (1.0 + tau)) * (1.0 - 1.0 / MirrorRatio) * MachNumber * MachNumber / 2.0;
	return Potential;
}
// Implements D-T fusion rate from NRL plasma formulary
Real MirrorPlasmaTest::FusionRate(Real n, Real pi) const
{

	Real Normalization = n0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
	Real Factor = 1e-6 * 3.68e-12;
	Real Ti_kev = (pi / n) * T0 / (1000 * ElementaryCharge);
	Real sigmav;
	if (Ti_kev < 25)
		sigmav = Factor * pow(Ti_kev, -2. / 3.) * exp(-19.94 * pow(Ti_kev, -1. / 3.));
	else
		sigmav = 1e-6 * 2.7e-16; // Formula is only valid for Ti < 25 keV, just set to a constant after

	Real R = n0 * n0 / Normalization * 0.25 * n * n * sigmav;

	return R;
}

Real MirrorPlasmaTest::TotalAlphaPower(Real n, Real pi) const
{
	double Factor = 5.6e-13 / (T0);
	Real AlphaPower = Factor * FusionRate(n, pi);
	return AlphaPower;
}

// Implements Bremsstrahlung radiative losses from NRL plasma formulary
Real MirrorPlasmaTest::BremsstrahlungLosses(Real n, Real pe) const
{
	double p0 = n0 * T0;
	Real Normalization = p0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
	Real Factor = (1 + Z_eff) * 1.69e-32 * 1e-6 * n0 * n0 / Normalization;

	Real Te_eV = (pe / n) * T0 / ElementaryCharge;
	Real Pbrem = Factor * n * n * sqrt(Te_eV);

	return Pbrem;
}

Real MirrorPlasmaTest::CyclotronLosses(Real V, Real n, Real Te) const
{
	// NRL formulary with reference values factored out
	// Return units are W/m^3
	Real Te_eV = T0 / ElementaryCharge * Te;
	Real n_e20 = n * n0 / 1e20;
	Real B_z = B->Bz_R(B->R_V(V)) * B0; // in Tesla
	Real P_vacuum = 6.21 * n_e20 * Te_eV * B_z * B_z;

	// Characteristic absorption length
	// lambda_0 = (Electron Inertial Lenght) / ( Plasma Frequency / Cyclotron Frequency )  ; Eq (4) of Tamor
	//				= (5.31 * 10^-4 / (n_e20)^1/2) / ( 3.21 * (n_e20)^1/2 / B ) ; From NRL Formulary, converted to our units (Tesla for B & 10^20 /m^3 for n_e)
	Real LambdaZero = (5.31e-4 / 3.21) * (B_z / n_e20);
	double WallReflectivity = 0.95;
	Real OpticalThickness = ((R_Upper - R_Lower) / (1.0 - WallReflectivity)) / LambdaZero;
	// This is the Phi introduced by Trubnikov and later approximated by Tamor
	Real TransparencyFactor = pow(Te_eV, 1.5) / (200.0 * sqrt(OpticalThickness));
	// Moderate the vacuum emission by the transparency factor
	Real Normalization = n0 * T0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));

	Real P_cy = P_vacuum * TransparencyFactor / Normalization;
	return P_cy;
}

// omega & n are callables
template <typename T1, typename T2>
double MirrorPlasmaTest::Voltage(T1 &L_phi, T2 &n)
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

Real MirrorPlasmaTest::RelaxEdge(Real x, Real y, Real EdgeVal) const
{
	double shape = 1 / EdgeSourceWidth;
	Real g = (exp(-shape * (x - xL) * (x - xL)) + exp(-shape * (x - xR) * (x - xR)));
	return EdgeSourceSize * g * (EdgeVal - y);
}

void MirrorPlasmaTest::initialiseDiagnostics(NetCDFIO &nc)
{

	AutodiffTransportSystem::initialiseDiagnostics(nc);
	// Add diagnostics here
	//
	double RhoStar = RhoStarRef();
	double TauNorm = (IonMass / ElectronMass) * (1.0 / (RhoStar * RhoStar)) * (ReferenceElectronCollisionTime());
	nc.AddScalarVariable("Tau", "Normalising time", "s", TauNorm);

	// lambda wrappers for DGSoln object

	auto L = [this](double V)
	{
		return InitialValue(Channel::AngularMomentum, V);
	};
	auto LPrime = [this](double V)
	{
		return InitialDerivative(Channel::AngularMomentum, V);
	};
	auto n = [this](double V)
	{
		return InitialValue(Channel::Density, V);
	};
	auto nPrime = [this](double V)
	{
		return InitialDerivative(Channel::Density, V);
	};
	auto p_i = [this](double V)
	{
		return (2. / 3.) * InitialValue(Channel::IonEnergy, V);
	};
	auto p_e = [this](double V)
	{
		return (2. / 3.) * InitialValue(Channel::ElectronEnergy, V);
	};

	auto u = [this](double V)
	{
		RealVector U(nVars);
		for (Index i = 0; i < nVars; ++i)
			U(i) = InitialValue(i, V);

		return U;
	};

	auto q = [this](double V)
	{
		RealVector Q(nVars);
		for (Index i = 0; i < nVars; ++i)
			Q(i) = InitialDerivative(i, V);

		return Q;
	};

	auto aux = [this](double V)
	{
		RealVector Aux(nAux);
		for (Index i = 0; i < nAux; ++i)
			Aux(i) = InitialAuxValue(i, V);

		return Aux;
	};

	Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real p0 = CentrifugalPotential((Real)V, omega, Ti, Te);

		return p0.val;
	};

	Fn phi = [this, &Phi0](double V)
	{
		if (useAmbipolarPhi)
		{
			return Phi0(V) + InitialAuxValue(0, V);
		}
		else
		{
			return Phi0(V);
		}
	};

	auto ShearingRate = [this, &u, &q](double V)
	{
		auto qV = q(V);
		auto uV = u(V);
		Real Ti = 2. / 3. * uV(Channel::IonEnergy) / uV(Channel::Density);
		double R = B->R_V(V);
		Real vtheta = 1 / R * uV(Channel::AngularMomentum) / uV(Channel::Density);

		double dVdR = 1 / B->dRdV(V);
		Real SR = 1.0 / sqrt(Ti) * (dVdR / uV(Channel::Density) * (qV(Channel::AngularMomentum) - R * vtheta * qV(Channel::Density)) - vtheta);
		return SR.val;
	};

	auto ElectrostaticPotential = [this, &u, &aux, &p_i, &n](double V)
	{
		if (useAmbipolarPhi)
		{
			return phi0(u(V), (Real)V).val + p_i(V) / n(V) * aux(V)[0].val;
		}
		else
		{
			return phi0(u(V), (Real)V).val;
		}
	};

	double initialVoltage = Voltage(L, n);

	const std::function<double(const double &)> initialZero = [](const double &V)
	{ return 0.0; };

	// Wrap DGApprox with lambdas for heating functions
	Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i
		Real dRdV = B->dRdV(V);
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

	Fn ElectronParallelLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real Xe = Xi_e((Real)V, aux(V)[0], Ti, Te, omega);

		double ParallelLosses = Te.val * (1 + Xe.val) * ElectronPastukhovLossRate(V, Xe, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn IonParallelLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real Xi = Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

		double ParallelLosses = Ti.val * (1 + Xi.val) * IonPastukhovLossRate(V, Xi, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn AngularMomentumLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real AngularMomentumPerParticle = L(V) / n(V);

		Real Xi = Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

		double ParallelLosses = AngularMomentumPerParticle.val * IonPastukhovLossRate(V, Xi, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn EnergyExchange = [this, &n, &p_i, &p_e](double V)
	{
		return IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
	};

	Fn IonPotentialHeating = [this, &u, &q, &n, &L, &aux](double V)
	{
		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;
		Real S = -PotentialHeatingFactor * Gamma(u(V), q(V), V, 0.0) * (omega * omega / (2 * pi * a) - dphidV(u(V), q(V), aux(V), V));

		return S.val;
	};

	Fn ElectronPotentialHeating = [this, &u, &q, &aux](double V)
	{
		Real S = -PotentialHeatingFactor * Gamma(u(V), q(V), V, 0.0) * (dphidV(u(V), q(V), aux(V), V));

		return S.val;
	};

	Fn ParticleSourceHeating = [this, &n, &L](double V)
	{
		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;
		Real ParticleSourceHeating = 0.5 * omega * omega * R * R * ParticleSource(R.val, 0.0);
		return ParticleSourceHeating.val;
	};

	Real tnorm = n0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
	nc.AddScalarVariable("tnorm", "time normalization", "s", 1 / tnorm.val);
	nc.AddScalarVariable("Lnorm", "Length normalization", "m", 1 / a);
	nc.AddScalarVariable("n0", "Density normalization", "m^-3", n0);
	nc.AddScalarVariable("T0", "Temperature normalization", "J", T0);
	nc.AddScalarVariable("B0", "Reference magnetic field", "T", B0);
	nc.AddScalarVariable("IRadial", "Radial current", "A", IRadial * (IonMass * sqrt(T0 / IonMass) * a) * tnorm.val / B0);
	nc.AddTimeSeries("Voltage", "Total voltage drop across the plasma", "Volts", initialVoltage);
	nc.AddGroup("MMS", "Manufactured solutions");
	for (int j = 0; j < nVars; ++j)
		nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [this, j](double V)
					   { return this->InitialFunction(j, V, 0.0).val.val; });
	nc.AddVariable("ShearingRate", "Plasma shearing rate", "-", ShearingRate);
	nc.AddVariable("ElectrostaticPotential", "electrostatic potential (phi0+phi1)", "-", ElectrostaticPotential);

	nc.AddGroup("MomentumFlux", "Separating momentum fluxes");
	nc.AddGroup("ParallelLosses", "Separated parallel losses");
	nc.AddVariable("ParallelLosses", "ElectronParLoss", "Parallel particle losses", "-", ElectronParallelLosses);
	nc.AddVariable("ParallelLosses", "IonParLoss", "Parallel particle losses", "-", IonParallelLosses);
	nc.AddVariable("ParallelLosses", "CentrifugalPotential", "Centrifugal potential", "-", phi);
	nc.AddVariable("ParallelLosses", "AngularMomentumLosses", "Angular momentum loss rate", "-", AngularMomentumLosses);

	// Heat Sources
	nc.AddGroup("Heating", "Separated heating sources");
	nc.AddVariable("Heating", "AlphaHeating", "Alpha heat source", "-", AlphaHeating);
	nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", ViscousHeating);
	nc.AddVariable("Heating", "RadiationLosses", "Bremsstrahlung heat losses", "-", RadiationLosses);
	nc.AddVariable("Heating", "EnergyExchange", "Collisional ion-electron energy exhange", "-", EnergyExchange);
	nc.AddVariable("Heating", "IonPotentialHeating", "Ion potential heating", "-", IonPotentialHeating);
	nc.AddVariable("Heating", "ElectronPotentialHeating", "Ion potential heating", "-", ElectronPotentialHeating);
	nc.AddVariable("Heating", "ParticleSourceHeating", "Heating due to particle source", "-", ParticleSourceHeating);
	nc.AddVariable("dPhi0dV", "Phi0 derivative", "-", [this, &u, &q](double V)
				   { return dphi0dV(u(V), q(V), V).val; });
	nc.AddVariable("dPhi1dV", "Phi1 derivative", "-", [this, &u, &q, &aux](double V)
				   {
		if (nAux > 0)
			return dphi1dV(u(V), q(V), aux(V)[0], V).val;
		else
			return 0.0; });
}

void MirrorPlasmaTest::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{

	AutodiffTransportSystem::writeDiagnostics(y, t, nc, tIndex);

	// lambda wrappers for DGSoln object
	auto L = [&y](double V)
	{
		return y.u(Channel::AngularMomentum)(V);
	};
	auto LPrime = [&y](double V)
	{
		return y.q(Channel::AngularMomentum)(V);
	};
	auto n = [&y](double V)
	{
		return y.u(Channel::Density)(V);
	};
	auto nPrime = [&y](double V)
	{
		return y.q(Channel::Density)(V);
	};
	auto p_i = [&y](double V)
	{
		return (2. / 3.) * y.u(Channel::IonEnergy)(V);
	};
	auto p_e = [&y](double V)
	{
		return (2. / 3.) * y.u(Channel::ElectronEnergy)(V);
	};

	auto u = [this, &y](double V)
	{
		RealVector U(nVars);
		for (Index i = 0; i < nVars; ++i)
			U(i) = y.u(i)(V);

		return U;
	};

	auto q = [this, &y](double V)
	{
		RealVector Q(nVars);
		for (Index i = 0; i < nVars; ++i)
			Q(i) = y.q(i)(V);

		return Q;
	};

	auto aux = [this, &y](double V)
	{
		RealVector Aux(nAux);
		for (Index i = 0; i < nAux; ++i)
			Aux(i) = y.Aux(i)(V);

		return Aux;
	};

	Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real p0 = CentrifugalPotential((Real)V, omega, Ti, Te) + AmbipolarPhi(V, n(V), Ti, Te) / 2.0;

		return p0.val;
	};

	Fn phi = [this, &Phi0, &y](double V)
	{
		if (useAmbipolarPhi)
		{
			return Phi0(V) + y.Aux(0)(V);
		}
		else
		{
			return Phi0(V);
		}
	};

	auto ShearingRate = [this, &u, &q](double V)
	{
		auto qV = q(V);
		auto uV = u(V);
		Real Ti = 2. / 3. * uV(Channel::IonEnergy) / uV(Channel::Density);
		double R = B->R_V(V);
		Real vtheta = 1 / R * uV(Channel::AngularMomentum) / uV(Channel::Density);

		double dVdR = 1 / B->dRdV(V);
		Real SR = 1.0 / sqrt(Ti) * (dVdR / uV(Channel::Density) * (qV(Channel::AngularMomentum) - R * vtheta * qV(Channel::Density)) - vtheta);
		return SR.val;
	};

	auto ElectrostaticPotential = [this, &u, &aux, &p_i, &n](double V)
	{
		if (useAmbipolarPhi)
		{
			return phi0(u(V), (Real)V).val + p_i(V) / n(V) * aux(V)[0].val;
		}
		else
		{
			return phi0(u(V), (Real)V).val;
		}
	};

	double voltage = Voltage(L, n);
	nc.AppendToTimeSeries("Voltage", voltage, tIndex);

	// Wrap DGApprox with lambdas for heating functions
	Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i, &t](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i
		Real dRdV = B->dRdV(V);
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

	Fn ElectronParallelLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real Xe = Xi_e((Real)V, aux(V)[0], Ti, Te, omega);

		double ParallelLosses = Te.val * (1 + Xe.val) * ElectronPastukhovLossRate(V, Xe, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn IonParallelLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real Xi = Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

		double ParallelLosses = Ti.val * (1 + Xi.val) * IonPastukhovLossRate(V, Xi, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn AngularMomentumLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
	{
		Real Ti = p_i(V) / n(V);
		Real Te = p_e(V) / n(V);

		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;

		Real AngularMomentumPerParticle = L(V) / n(V);

		Real Xi = Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

		double ParallelLosses = AngularMomentumPerParticle.val * IonPastukhovLossRate(V, Xi, n(V), Te).val / ParallelLossFactor;

		return ParallelLosses;
	};

	Fn EnergyExchange = [this, &n, &p_i, &p_e](double V)
	{
		return IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
	};

	Fn dPhi0dV = [this, &u, &q](double V)
	{
		return dphi0dV(u(V), q(V), V).val;
	};

	Fn IonPotentialHeating = [this, &u, &q, &n, &L, &aux](double V)
	{
		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;
		Real S = -PotentialHeatingFactor * Gamma(u(V), q(V), V, 0.0) * (omega * omega / (2 * pi * a) - dphidV(u(V), q(V), aux(V), V));

		return S.val;
	};

	Fn ElectronPotentialHeating = [this, &u, &q, &aux](double V)
	{
		Real S = -PotentialHeatingFactor * Gamma(u(V), q(V), V, 0.0) * (dphidV(u(V), q(V), aux(V), V));

		return S.val;
	};

	Fn ParticleSourceHeating = [this, &n, &L, t](double V)
	{
		Real R = this->B->R_V(V);
		Real J = n(V) * R * R; // Normalisation includes the m_i

		Real omega = L(V) / J;
		Real ParticleSourceHeating = 0.5 * omega * omega * R * R * ParticleSource(R.val, t);
		return ParticleSourceHeating.val;
	};

	Fn DensitySol = [this, t](double V)
	{ return this->InitialFunction(Channel::Density, V, t).val.val; };
	Fn IonEnergySol = [this, t](double V)
	{ return this->InitialFunction(Channel::IonEnergy, V, t).val.val; };
	Fn ElectronEnergySol = [this, t](double V)
	{ return this->InitialFunction(Channel::ElectronEnergy, V, t).val.val; };
	Fn AngularMomentumSol = [this, t](double V)
	{ return this->InitialFunction(Channel::AngularMomentum, V, t).val.val; };

	// Add the appends for the heating stuff
	nc.AppendToGroup<Fn>("Heating", tIndex, {{"AlphaHeating", AlphaHeating}, {"ViscousHeating", ViscousHeating}, {"RadiationLosses", RadiationLosses}, {"EnergyExchange", EnergyExchange}, {"IonPotentialHeating", IonPotentialHeating}, {"ElectronPotentialHeating", ElectronPotentialHeating}, {"ParticleSourceHeating", ParticleSourceHeating}});

	nc.AppendToGroup<Fn>("ParallelLosses", tIndex, {{"ElectronParLoss", ElectronParallelLosses}, {"IonParLoss", IonParallelLosses}, {"CentrifugalPotential", phi}, {"AngularMomentumLosses", AngularMomentumLosses}});

	nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", DensitySol}, {"Var1", IonEnergySol}, {"Var2", ElectronEnergySol}, {"Var3", AngularMomentumSol}});

	nc.AppendToVariable("dPhi0dV", [this, &u, &q](double V)
						{ return dphi0dV(u(V), q(V), V).val; }, tIndex);

	nc.AppendToVariable("dPhi1dV", [this, &u, &q, &aux](double V)
						{
		if (nAux > 0)
			return dphi1dV(u(V), q(V), aux(V)[0], V).val;
		else
			return 0.0; }, tIndex);
	nc.AppendToVariable("ElectrostaticPotential", ElectrostaticPotential, tIndex);
	nc.AppendToVariable("ShearingRate", ShearingRate, tIndex);
}
