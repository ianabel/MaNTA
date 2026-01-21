#ifndef MIRRORPLASMA_HPP
#define MIRRORPLASMA_HPP

#include "MagneticFields.hpp"
#include "AutodiffTransportSystem.hpp"
// #include "Constants.hpp"
#include "MirrorPlasma/PlasmaConstants.hpp"
#include "MirrorPlasma/MirrorPlasmaDiagnostics.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>

/*
	Ground-up reimplementation of a collisional cylindrical plasma, with a single ion species
	we use V( psi ), the volume enclosed by a flux surface as the radial coordinate.

	The equations to be solved are then

	d_dt n_e + d_dV ( V' Gamma_e ) = Sn
	d_dt p_e + d_dV ( V' q_e ) = Spe + Cei
	d_dt p_i + d_dV ( V' q_i ) = Spi - pi_i_cl * domega_dpsi + Gamma m_i omega^2/B_z + Cie
	d_t L_phi + d_dV ( V' pi ) = Somega - j_R R B_z

	pi = Sum_s { pi_s_cl + m_s omega R^2 Gamma_s }

 */

#include <cmath>

inline Real
floor(Real x, double val)
{
	if (x >= val)
		return x;
	else
	{
		x.val = val;
		return x;
	}
}

class MirrorPlasma : public AutodiffTransportSystem
{
public:
	MirrorPlasma(toml::value const &config, Grid const &grid);
	~MirrorPlasma() = default;

	virtual Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

	// Add scalars to allow for constant voltage runs
	Value InitialScalarValue(Index) const override;
	Value InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const override;

	bool isScalarDifferential(Index) override;
	Value ScalarGExtended(Index, const DGSoln &, const DGSoln &, Time) override;
	void ScalarGPrimeExtended(Index, State &, State &, const DGSoln &, const DGSoln &, std::function<double(double)>, Interval, Time) override;

private:
	using integrator = boost::math::quadrature::gauss_kronrod<double, 15>;
	constexpr static int max_depth = 2;

	Real Flux(Index, RealVector, RealVector, Real, Time) override;
	Real Source(Index, RealVector, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

	Real GFunc(Index, RealVector, RealVector, RealVector, RealVector, Position, Time) override;
	Value InitialAuxValue(Index, Position, Time) const override;

	Value LowerBoundary(Index i, Time t) const override;
	Value UpperBoundary(Index i, Time t) const override;

	virtual bool isLowerBoundaryDirichlet(Index i) const override;
	virtual bool isUpperBoundaryDirichlet(Index i) const override;

	Real2nd MMS_Solution(Index i, Real2nd V, Real2nd t) override;

	Real Gamma(RealVector u, RealVector q, Real x, Time t) const;
	Real qe(RealVector u, RealVector q, Real x, Time t) const;
	Real qi(RealVector u, RealVector q, Real x, Time t) const;
	Real Pi(RealVector u, RealVector q, Real x, Time t) const;
	Real Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
	Real Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
	Real Spi(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
	Real Somega(RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars, Real x, Time t) const;

	// Underlying functions

	Real uToDensity(Real) const;
	Real qToDensityGradient(Real, Real) const;

	Value InitialDensityTimeDerivative(RealVector u, RealVector q, Position V) const;
	Value InitialCurrent(Time t) const;

	Real IonClassicalAngularMomentumFlux(Real V, Real n, Real Ti, Real omega, Real dOmegadV, Time t) const;

	Real ParticleSource(double R, double t) const;
	Real NeutralDensity(Real R, Time t) const;

	Real ElectronPastukhovLossRate(Real V, Real Xi_e, Real n, Real Te) const;
	Real IonPastukhovLossRate(Real V, Real Xi_i, Real n, Real Ti) const;

	Real ViscousHeating(RealVector, RealVector, Real, Time) const;
	Real IonPotentialHeating(RealVector, RealVector, RealVector, Real) const;
	Real ElectronPotentialHeating(RealVector, RealVector, RealVector, Real) const;
	// Template function to avoid annoying dual vs. dual2nd behavior
	template <typename T>
	T phi0(Eigen::Matrix<T, -1, 1, 0, -1, 1> u, T V) const
	{
		T n = u(Channel::Density), p_e = (2. / 3.) * u(Channel::ElectronEnergy), p_i = (2. / 3.) * u(Channel::IonEnergy);

		T Te = p_e / n, Ti = p_i / n;
		T L = u(Channel::AngularMomentum);
		T R = B->R_V(V, 0.0);
		T J = n * R * R; // Normalisation of the moment of inertia includes the m_i
		T omega = L / J;
		T phi = 0.5 / (1 / Ti + 1 / Te) * omega * omega * R * R / Ti * (1 / B->MirrorRatio(V, 0.0) - 1);

		return phi;
	}
	Real dphi0dV(RealVector u, RealVector q, Real V) const;
	Real dphi1dV(RealVector u, RealVector q, Real phi, Real V) const;
	Real dphidV(RealVector u, RealVector q, RealVector phi, Real V) const;

	// Returns (1/(1 + Tau))*(1-1/R_m)*(M^2)
	template <typename T>
	T CentrifugalPotential(T V, T omega, T Ti, T Te) const
	{
		T R = B->R_V(V, 0.0);
		T MirrorRatio = B->MirrorRatio(V, 0.0);
		T tau = Ti / Te;
		T MachNumber = omega * R / sqrt(Te); // omega is normalised to c_s0 / a
		T Potential = (1.0 / (1.0 + tau)) * (1.0 - 1.0 / MirrorRatio) * MachNumber * MachNumber / 2.0;
		return Potential;
	}
	// Energy normalisation is T0, but these return Xi_s / T_s as that is what enters the
	// Pastukhov factor
	template <typename T>
	T Xi_i(T V, T phi, T Ti, T Te, T omega) const
	{
		return CentrifugalPotential<T>(V, omega, Ti, Te) + phi;
	}
	template <typename T>
	T Xi_e(T V, T phi, T Ti, T Te, T omega) const
	{
		return CentrifugalPotential<T>(V, omega, Ti, Te) - Ti / Te * phi;
	}

	// First order correction to phi, only used if not calculating auxiliary phi1
	Real AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const;

	template <typename T>
	T ParallelCurrent(T V, T omega, T n, T Ti, T Te, T phi) const;

	double R_Lower, R_Upper;

	Real RelaxSource(Real A, Real B) const
	{
		return RelaxFactor * (A - B);
	};

	// Transitions smoothly from 0 to yf in x = L
	template <typename T>
	T SmoothTransition(T x, double L, double yf) const
	{
		T xl = yf / 2.0 * (1 - cos(M_PI * x / L));
		T xr = yf;

		return x < L ? xl : xr;
	}

	Value TotalCurrent(DGSoln const &y, Time t);

	template <typename T1, typename T2>
	double Voltage(T1 &L_phi, T2 &n);

	void initialiseDiagnostics(NetCDFIO &) override;
	void writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex) override;

private:
	// // Reference Values

	// // All reference values should be in SI (non SI variants are explicitly noted)
	constexpr static double n0 = 1.e20, n0cgs = n0 / 1.e6;
	constexpr static double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
	constexpr static double B0 = 1.0; // Reference field in T
	constexpr static double a = 1.0;  // Reference length in m
	constexpr static double Z_eff = 3.0;

	//
	std::unique_ptr<PlasmaConstants> Plasma;
	std::shared_ptr<MagneticField> B;

	bool evolveLogDensity;

	// Desired voltage to keep constant
	double V0;
	double gamma;
	double gamma_d;
	double gamma_h;

	enum Scalar : Index
	{
		Error = 0,
		Integral = 1,
		Current = 2
	};

	double RelaxFactor;
	double MinDensity;
	double MinTemp;

	enum Channel : Index
	{
		Density = 0,
		IonEnergy = 1,
		ElectronEnergy = 2,
		AngularMomentum = 3
	};
	std::map<std::string, Index> ChannelMap = {{"Density", Channel::Density}, {"IonEnergy", Channel::IonEnergy}, {"ElectronEnergy", Channel::ElectronEnergy}, {"AngularMomentum", Channel::AngularMomentum}};
	std::map<Index, bool> ConstantChannelMap = {{Channel::Density, false}, {Channel::IonEnergy, false}, {Channel::ElectronEnergy, false}, {Channel::AngularMomentum, false}};

	enum ParticleSourceType
	{
		None = 0,
		Gaussian = 1,
		Distributed = 2,
		Ionization = 3,
	};

	std::vector<bool> upperBoundaryConditions;
	std::vector<bool> lowerBoundaryConditions;

	double nEdge, TeEdge, TiEdge, MUpper, MLower, MEdge;

	double lowNDiffusivity, lowNThreshold;
	double lowPDiffusivity, lowPThreshold;
	double lowLDiffusivity, lowLThreshold;

	double TeDiffusivity;

	double transitionLength;

	double InitialPeakDensity, InitialPeakTe, InitialPeakTi, InitialPeakMachNumber;
	double nNeutrals;
	bool useNeutralModel;
	double MachWidth;
	bool useAmbipolarPhi;
	bool useConstantVoltage;

	std::vector<double> growth_factors;

	double LowerParticleSourceStrength, UpperParticleSourceStrength, ParticleSourceCenter,
		ParticleSourceWidth, UniformHeatSource;

	double SourceRampup;

	double IRadial, I0, CurrentDecay;

	// Add a cap to sources to prevent ridiculous values
	double SourceCap;

	REGISTER_PHYSICS_HEADER(MirrorPlasma)
};

#endif
