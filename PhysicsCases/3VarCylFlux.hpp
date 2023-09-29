#include "AutodiffFlux.hpp"

#ifndef THREEVARCYLFLUX
#define THREEVARCYLFLUX

class ThreeVarCylFlux : public FluxObject
{
public:
    ThreeVarCylFlux(toml::value const &config, Index nVars);

private:
    double InitialWidth, Centre;

    std::map<std::string, int> ParticleSources = {{"None", 0}, {"Gaussian", 1}};
    static int ParticleSource;
    static double sourceStrength;
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