#ifndef CYLPLASMA
#define CYLPLASMA

#include "AutodiffTransportSystem.hpp"

enum ParticleSourceType
{
    None = 0,
    Gaussian = 1,
	 Distributed = 2,
	 Ionization = 3,
};


class CylPlasma : public AutodiffTransportSystem
{
public:
    CylPlasma(toml::value const &config, Grid const& grid );

private:

	 Real Flux( Index, RealVector, RealVector, Position, Time ) override;
	 Real Source( Index, RealVector, RealVector, RealVector, Position, Time ) override;

    double ParticleSource, Torque;
	 double Centre;

    // reference values
    Real n0;
    Real Bmid;
    Real T0;
	 Real Omega0;
    Value L;
    Real p0;
    Real V0;
    Real Gamma0;
    Real taue0;
    Real taui0;

    Vector InitialHeights;
    Real Gamma_hat(RealVector u, RealVector q, Position x, Time t);
    Real qe_hat(RealVector u, RealVector q, Position x, Time t);
    Real qi_hat(RealVector u, RealVector q, Position x, Time t);
	 Real pi_hat(RealVector u, RealVector q, Position x, Time t);
    Real Sn_hat(RealVector u, RealVector q, RealVector sigma, Position x, Time t);
    Real Spe_hat(RealVector u, RealVector q, RealVector sigma, Position x, Time t);
    Real Spi_hat(RealVector u, RealVector q, RealVector sigma, Position x, Time t);
	 Real S_omega(RealVector u, RealVector q, RealVector sigma, Position x, Time t);

    REGISTER_PHYSICS_HEADER(CylPlasma)
};

#endif
