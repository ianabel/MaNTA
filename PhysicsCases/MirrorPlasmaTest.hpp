#ifndef MIRRORPLASMATEST
#define MIRRORPLASMATEST

#include "MagneticFields.hpp"
#include "AutodiffTransportSystem.hpp"

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

class MirrorPlasmaTest : public AutodiffTransportSystem
{
public:
	MirrorPlasmaTest(toml::value const &config, Grid const &grid);
	virtual ~MirrorPlasmaTest() { delete B; };

	virtual autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t) const override;

private:
	double RelaxFactor;
	double MinDensity;
	double MinTemp;
	Real floor(Real x, double val) const
	{
		if (x >= val)
			return x;
		else
		{
			x.val = val;
			return x;
		}
	}

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
	double InitialPeakDensity, InitialPeakTe, InitialPeakTi, InitialPeakMachNumber, ParallelLossFactor, DragFactor, DragWidth, ParticlePhysicsFactor, PotentialHeatingFactor, ViscousHeatingFactor, EnergyExchangeFactor;
	double MaxPastukhov;
	double MachWidth;
	bool ZeroEdgeSources;
	double ZeroEdgeFactor;
	bool useAmbipolarPhi;

	Real Flux(Index, RealVector, RealVector, Real, Time) override;
	Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

	Real Phi(Index, RealVector, RealVector, RealVector, RealVector, Position, Time) override;
	Value InitialAuxValue(Index, Position) const override;

	Value LowerBoundary(Index i, Time t) const override;
	Value UpperBoundary(Index i, Time t) const override;

	virtual bool isLowerBoundaryDirichlet(Index i) const override;
	virtual bool isUpperBoundaryDirichlet(Index i) const override;

	Real2nd MMS_Solution(Index i, Real2nd V, Real2nd t) override;

	std::vector<double> growth_factors;

	double ParticleSourceStrength, ParticleSourceCenter,
		ParticleSourceWidth, UniformHeatSource;

	mutable double IRadial;

	// Reference Values
	constexpr static double ElectronMass = 9.1094e-31;		   // Electron Mass, kg
	constexpr static double IonMass = 2.0 * 1.6726e-27;		   // 2.0* Ion Mass ( =2 * proton mass) kg (D-D plasma)
	constexpr static double ElementaryCharge = 1.60217663e-19; // Coulombs
	constexpr static double VacuumPermittivity = 8.8541878128e-12;

	// All reference values should be in SI (non SI variants are explicitly noted)
	constexpr static double n0 = 1.e20, n0cgs = n0 / 1.e6;
	constexpr static double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
	constexpr static double B0 = 1.0; // Reference field in T
	constexpr static double a = 1.0;  // Reference length in m
	constexpr static double Z_eff = 3.0;

	Real Gamma(RealVector u, RealVector q, Real x, Time t) const;
	Real qe(RealVector u, RealVector q, Real x, Time t) const;
	Real qi(RealVector u, RealVector q, Real x, Time t) const;
	Real Pi(RealVector u, RealVector q, Real x, Time t) const;
	Real Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
	Real Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
	Real Spi(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
	Real Somega(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;

	// Underlying functions

	template <typename T>
	T LogLambda_ii(T, T) const;

	template <typename T>
	T LogLambda_ei(T, T) const;

	template <typename T>
	T ElectronCollisionTime(T, T) const;

	template <typename T>
	T IonCollisionTime(T, T) const;

	double ReferenceElectronCollisionTime() const;
	double ReferenceIonCollisionTime() const;

	Real IonElectronEnergyExchange(Real n, Real pe, Real pi, Real V, double t) const;
	Real IonClassicalAngularMomentumFlux(Real V, Real n, Real Ti, Real dOmegadV, double t) const;
	double RhoStarRef() const;

	StraightMagneticField *B;

	// test source
	double EdgeSourceSize, EdgeSourceWidth;

	double SourceCap;

	Real ParticleSource(double R, double t) const;

	Real ElectronPastukhovLossRate(Real V, Real Xi_e, Real n, Real Te) const;
	Real IonPastukhovLossRate(Real V, Real Xi_i, Real n, Real Ti) const;

	template <typename T>
	T CentrifugalPotential(T V, T omega, T Ti, T Te) const;

	Real FusionRate(Real n, Real pi) const;
	Real TotalAlphaPower(Real n, Real pi) const;
	Real BremsstrahlungLosses(Real n, Real pe) const;
	Real CyclotronLosses(Real V, Real n, Real Te) const;

	// Real Xi_i(Real V, Real omega, Real n, Real Ti, Real Te) const;
	// Real Xi_e(Real V, Real omega, Real n, Real Ti, Real Te) const;
	template <typename T>
	T phi0(Eigen::Matrix<T, -1, 1, 0, -1, 1> u, T V) const;
	Real dphi0dV(RealVector u, RealVector q, Real V) const;
	Real dphi1dV(RealVector u, RealVector q, Real phi, Real V) const;
	Real dphidV(RealVector u, RealVector q, RealVector phi, Real V) const;

	template <typename T>
	T Xi_i(T V, T phi, T Ti, T Te, T omega) const;

	template <typename T>
	T Xi_e(T V, T phi, T Ti, T Te, T omegae) const;
	Real AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const;

	template <typename T>
	T ParallelCurrent(T V, T omega, T n, T Ti, T Te, T phi) const;
	double R_Lower, R_Upper;

	Real RelaxSource(Real A, Real B) const
	{
		return RelaxFactor * (A - B);
	};

	Real RelaxEdge(Real x, Real y, Real EdgeVal) const;

	template <typename T1, typename T2>
	double Voltage(T1 &L_phi, T2 &n);
	void initialiseDiagnostics(NetCDFIO &nc) override;
	void writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex) override;

	REGISTER_PHYSICS_HEADER(MirrorPlasmaTest)
};

#endif
