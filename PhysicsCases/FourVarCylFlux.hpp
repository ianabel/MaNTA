#include "AutodiffFlux.hpp"

#ifndef FOURVARCYLFLUX
#define FOURVARCYLFLUX

class FourVarCylFlux : public FluxObject
{
public:
    FourVarCylFlux(toml::value const &config, Index nVars);

private:
    std::map<std::string, int> ParticleSources = {{"None", 0}, {"Gaussian", 1}};
    static int ParticleSource;
    static double sourceStrength;
    Vector InitialHeights;
    static dual Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual qe_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual qi_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual hi_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Shi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);

    REGISTER_FLUX_HEADER(FourVarCylFlux)
};

#endif
