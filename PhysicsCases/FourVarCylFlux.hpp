#ifndef FOURVARCYLFLUX
#define FOURVARCYLFLUX

#include "AutodiffTransportSystem.hpp"

class FourVarCylFlux : public AutodiffTransportSystem
{
public:
    FourVarCylFlux(toml::value const &, Grid const &);

private:
    Real Flux(Index, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;
    Real Source(Index, RealVector, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;

    Real Postprocessor(const FluxWrapper &f, std::vector<Position> *ExtraValues = nullptr) override { return f(ExtraValues); };
    Values Postprocessor(const GradWrapper &f, std::vector<Position> *ExtraValues = nullptr) override { return f(ExtraValues); };

    std::map<std::string, int> ParticleSources = {{"None", 0}, {"Gaussian", 1}};
    int ParticleSource;
    double sourceStrength;
    Vector InitialHeights;
    Real sourceWidth;
    Real sourceCenter;

    // reference values
    Real n0;
    Real Bmid;
    Real T0;
    Real E0;
    Real J0;
    Value L;
    Real p0;
    Real V0;
    Real Gamma0;
    Real taue0;
    Real taui0;
    Real h0;

    Real Gamma_hat(RealVector u, RealVector q, Real x, double t);
    Real qe_hat(RealVector u, RealVector q, Real x, double t);
    Real qi_hat(RealVector u, RealVector q, Real x, double t);
    Real hi_hat(RealVector u, RealVector q, Real x, double t);
    Real Sn_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
    Real Spe_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
    Real Spi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
    Real Shi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);

    enum Channel : Index
    {
        Density = 0,
        ElectronEnergy = 1,
        IonEnergy = 2,
        AngularMomentum = 3
    };
    REGISTER_PHYSICS_HEADER(FourVarCylFlux)
};

#endif
