#ifndef MIRRORPLASMA
#define MIRRORPLASMA

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

class MirrorPlasma : public AutodiffTransportSystem
{
public:
	MirrorPlasma(toml::value const &config, Grid const &grid);
	virtual ~MirrorPlasma() { delete B; };

	virtual autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t, double u_R, double u_L, double x_L, double x_R) const override;

private:
	enum Channel : Index
	{
		Density = 0,
		IonEnergy = 1,
		ElectronEnergy = 2,
		AngularMomentum = 3
	};

	enum ParticleSourceType
	{
		None = 0,
		Gaussian = 1,
		Distributed = 2,
		Ionization = 3,
	};

	double nEdge, TeEdge, TiEdge, MEdge;
	double InitialPeakDensity, InitialPeakTe, InitialPeakTi, InitialPeakMachNumber, ParallelLossFactor, DelayFactor;

	Real Flux(Index, RealVector, RealVector, Position, Time) override;
	Real Source(Index, RealVector, RealVector, RealVector, Position, Time) override;

	double ParticleSourceStrength, jRadial, ParticleSourceWidth, UniformHeatSource;

	// Reference Values
	constexpr static double ElectronMass = 9.1094e-31;		   // Electron Mass, kg
	constexpr static double IonMass = 2.5 * 1.6726e-27;		   // 2.5* Ion Mass ( = proton mass) kg (DT fusion)
	constexpr static double ElementaryCharge = 1.60217663e-19; // Coulombs
	constexpr static double VacuumPermittivity = 8.8541878128e-12;

	// All reference values should be in SI (non SI variants are explicitly noted)
	constexpr static double n0 = 1.e20, n0cgs = n0 / 1.e6;
	constexpr static double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
	constexpr static double B0 = 1.0; // Reference field in T
	constexpr static double a = 1.0;  // Reference length in m

	Real Gamma(RealVector u, RealVector q, Position x, Time t) const;
	Real qe(RealVector u, RealVector q, Position x, Time t) const;
	Real qi(RealVector u, RealVector q, Position x, Time t) const;
	Real Pi(RealVector u, RealVector q, Position x, Time t) const;
	Real Sn(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;
	Real Spe(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;
	Real Spi(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;
	Real Somega(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;

	// Underlying functions
	Real LogLambda_ii(Real, Real) const;
	Real LogLambda_ei(Real, Real) const;
	Real ElectronCollisionTime(Real, Real) const;
	Real IonCollisionTime(Real, Real) const;
	double ReferenceElectronCollisionTime() const;
	double ReferenceIonCollisionTime() const;

	Real IonElectronEnergyExchange(Real n, Real pe, Real pi, Position V, double t) const;
	Real IonClassicalAngularMomentumFlux(Position V, Real n, Real Ti, Real dOmegadV, double t) const;
	double RhoStarRef() const;

	CylindricalMagneticField *B;

	Real ParticleSource(double R, double t) const;

	Real ElectronPastukhovLossRate(double V, Real Xi_e, Real n, Real Te) const;
	Real IonPastukhovLossRate(double V, Real Xi_i, Real n, Real Ti) const;
	Real CentrifugalPotential(double V, Real omega, Real Ti, Real Te) const;

	Real FusionRate(Real n, Real pi) const;
	Real TotalAlphaPower(Real n, Real pi) const;
	Real BremsstrahlungLosses(Real n, Real pe) const;

	Real Xi_i(Position V, Real omega, Real Ti, Real Te) const;
	Real Xi_e(Position V, Real omega, Real Ti, Real Te) const;
	double R_Lower, R_Upper;

	template <typename T1, typename T2>
	double Voltage(T1 &L_phi, T2 &n);
	void initialiseDiagnostics(NetCDFIO &nc) override;
	void writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex) override;

	REGISTER_PHYSICS_HEADER(MirrorPlasma)
};

#endif
