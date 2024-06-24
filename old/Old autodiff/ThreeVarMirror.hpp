#ifndef THREEVARMIRROR
#define THREEVARMIRROR

#include "AutodiffTransportSystem.hpp"

class ThreeVarMirror : public AutodiffTransportSystem
{
	public:
		ThreeVarMirror( toml::value const &config, Grid const& grid );
		double R(double x, double t);

	private:
		Real Flux( Index, RealVector, RealVector, Position, Time ) override;
		Real Source( Index, RealVector, RealVector, RealVector, Position, Time ) override;

		std::map<std::string, int> ParticleSources = {{"None", 0}, {"Gaussian", 1}};
		int ParticleSource;
		double sourceStrength;
		Real sourceWidth;
		Real sourceCenter;

		// reference values
		Real n0;
		Real Bmid;
		Real T0;
		Value L;
		Real p0;
		Real V0;
		Real Gamma0;
		Real taue0;
		Real taui0;

		Vector InitialHeights;
		Real
			Gamma_hat(RealVector u, RealVector q, Real x, double t);
		Real qe_hat(RealVector u, RealVector q, Real x, double t);
		Real qi_hat(RealVector u, RealVector q, Real x, double t);
		Real Sn_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
		Real Spe_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
		Real Spi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);

		Real omega(Real R, double t);
		double domegadV(Real x, double t);
		Real omegaOffset;
		bool useConstantOmega;
		bool includeParallelLosses;

		Real phi0(RealVector u, RealVector q, Real x, double t);
		Real dphi0dV(RealVector u, RealVector q, Real x, double t);
		Real Chi_e(RealVector u, RealVector q, Real x, double t);
		Real Chi_i(RealVector u, RealVector q, Real x, double t);

		double Rmin;
		double Rmax;
		Real M0;
		double psi(double R);
		double V(double R);
		double Vprime(double R);
		double B(double x, double t);
		double Bmax;

		// const double Zi = 1;
		enum class Channel : Index {
			Density = 0,
			ElectronEnergy = 1,
			IonEnergy = 2,
		};


		REGISTER_PHYSICS_HEADER(ThreeVarMirror)
};

#endif
