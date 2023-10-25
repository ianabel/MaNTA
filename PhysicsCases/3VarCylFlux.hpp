#include "AutodiffFlux.hpp"

#ifndef THREEVARCYLFLUX
#define THREEVARCYLFLUX

class ThreeVarCylFlux : public FluxObject
{
public:
    ThreeVarCylFlux(toml::value const &config, Index nVars);

private:
    std::map<std::string, int> ParticleSources = {{"None", 0}, {"Gaussian", 1}};
    static int ParticleSource;
    static double sourceStrength;
    static dual sourceWidth;
    static dual sourceCenter;

    // reference values
    static dual n0;
    static dual Bmid;
    static dual T0;
    static Value L;
    static dual p0;
    static dual V0;
    static dual Gamma0;
    static dual taue0;
    static dual taui0;

    Vector InitialHeights;
    static dual
    Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual qe_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual qi_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);

    REGISTER_FLUX_HEADER(ThreeVarCylFlux)
};

#endif
