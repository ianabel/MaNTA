#ifndef CYLPLASMA_DEBUG
#define CYLPLASMA_DEBUG

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

#include <numbers>
#include <cmath>
using std::numbers::pi;

// Magnetic field
class StraightMagneticField
{
	public:
		double Bz_R( double R ) { return B_z; };
		double V( double Psi ) { 
			return 2 * pi * Psi * L_z / B_z;
		};
		double Psi( double R ) {
			return R*R*B_z/2.0;
		};
		double Psi_V( double V ) {
			return B_z * V / ( 2 * pi * L_z );
		};
		double VPrime( double V ) {
			return 2*pi*L_z/B_z;
		};
		double R( double Psi ) {
			return sqrt( 2*Psi/B_z );
		};
		double R_V( double V ) {
			return sqrt( V / ( pi * L_z ) );
		};
	private:
		double L_z = 1.0;
		double B_z = 1.0;
};


class CylPlasmaDebug : public AutodiffTransportSystem
{
	public:
		CylPlasmaDebug(toml::value const &config, Grid const& grid );
		virtual ~CylPlasmaDebug() { delete B; };

	private:

		enum Channel : Index {
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

		Real Flux( Index, RealVector, RealVector, Position, Time ) override;
		Real Source( Index, RealVector, RealVector, RealVector, Position, Time ) override;

		double ParticleSourceStrength, jRadial;

		// Reference Values
		constexpr static double ElectronMass = 9.1094e-31; // Electron Mass, kg
		constexpr static double IonMass = 1.6726e-27;      // Ion Mass ( = proton mass) kg
		constexpr static double ElementaryCharge = 1.60217663e-19; // Coulombs
		constexpr static double VacuumPermittivity = 8.8541878128e-12;

		// All reference values should be in SI (non SI variants are explicitly noted)
		constexpr static double n0 = 1.e20, n0cgs = n0/1.e6;
		constexpr static double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
		constexpr static double B0 = 1.0; // Reference field in T
		constexpr static double a = 1.0; // Reference length in m

		Real Gamma(RealVector u, RealVector q, Position x, Time t) const;
		Real qe(RealVector u, RealVector q, Position x, Time t) const;
		Real qi(RealVector u, RealVector q, Position x, Time t) const;
		Real Pi(RealVector u, RealVector q, Position x, Time t) const;
		Real Sn(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;
		Real Spe(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;
		Real Spi(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;
		Real Somega(RealVector u, RealVector q, RealVector sigma, Position x, Time t) const;

		// Underlying functions
		Real LogLambda_ii( Real, Real ) const;
		Real LogLambda_ei( Real, Real ) const;
		Real ElectronCollisionTime( Real, Real ) const;
		Real IonCollisionTime( Real, Real ) const;
		double ReferenceElectronCollisionTime() const;
		double ReferenceIonCollisionTime() const;

		Real IonElectronEnergyExchange( Real n, Real pe, Real pi, Position V, double t ) const;
		Real IonClassicalAngularMomentumFlux( Position V, Real n, Real Ti, Real dOmegadV, double t ) const;
		double RhoStarRef() const;

		StraightMagneticField *B;

		REGISTER_PHYSICS_HEADER(CylPlasmaDebug)
};

#endif
