#ifndef FOURVARMirror
#define FOURVARMirror

#include "AutodiffTransportSystem.hpp"

class FourVarMirror : public AutodiffTransportSystem
{
public:
    FourVarMirror(toml::value const &, Grid const &);

private:
    Real Flux(Index, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;
    Real Source(Index, RealVector, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;

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
    Real omega0;

    Real Gamma_hat(RealVector u, RealVector q, Real x, double t);
    Real qe_hat(RealVector u, RealVector q, Real x, double t);
    Real qi_hat(RealVector u, RealVector q, Real x, double t);
    Real hi_hat(RealVector u, RealVector q, Real x, double t);
    Real Sn_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
    Real Spe_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
    Real Spi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);
    Real Shi_hat(RealVector u, RealVector q, RealVector sigma, Real x, double t);

    Real phi0(RealVector u, RealVector q, Real x, double t);
    Real dphi0dV(RealVector u, RealVector q, Real x, double t);
    Real Chi_e(RealVector u, RealVector q, Real x, double t);
    Real Chi_i(RealVector u, RealVector q, Real x, double t);
    bool includeParallelLosses;

    double Rmin;
    double Rmax;

    double R(double x, double t);
    double psi(double R);
    double V(double R);
    double Vprime(double R);
    double B(double x, double t);
    double Bmax;

    REGISTER_PHYSICS_HEADER(FourVarMirror)
};

#endif
