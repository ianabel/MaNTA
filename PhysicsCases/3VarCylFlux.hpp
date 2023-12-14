#ifndef THREEVARCYLFLUX
#define THREEVARCYLFLUX

#include "AutodiffTransportSystem.hpp"

class ThreeVarCylFlux : public AutodiffTransportSystem
{
public:
    ThreeVarCylFlux(toml::value const &config, Grid const& grid );

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
    Real Gamma_hat(RealVector u, RealVector q, Position x, Time t);
    Real qe_hat(RealVector u, RealVector q, Position x, Time t);
    Real qi_hat(RealVector u, RealVector q, Position x, Time t);
    Real Sn_hat(RealVector u, RealVector q, RealVector sigma, Position x, Time t);
    Real Spe_hat(RealVector u, RealVector q, RealVector sigma, Position x, Time t);
    Real Spi_hat(RealVector u, RealVector q, RealVector sigma, Position x, Time t);

    REGISTER_PHYSICS_HEADER(ThreeVarCylFlux)
};

#endif
