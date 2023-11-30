#include "AutodiffFlux.hpp"

#ifndef THREEVARMIRROR
#define THREEVARMIRROR

class ThreeVarMirror : public FluxObject
{
public:
    ThreeVarMirror(toml::value const &config, Index nVars);
    static double R(double x, double t);

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

    static dual omega(dual R, double t);
    static double domegadV(dual x, double t);
    static dual omegaOffset;
    static bool useConstantOmega;
    static bool includeParallelLosses;

    static dual phi0(VectorXdual u, VectorXdual q, dual x, double t);
    static dual dphi0dV(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Chi_e(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Chi_i(VectorXdual u, VectorXdual q, dual x, double t);

    static double Rmin;
    static double Rmax;
    static dual M0;
    static double psi(double R);
    static double V(double R);
    static double Vprime(double R);
    static double B(double x, double t);
    static double Bmax;

    const double Zi = 1;

    REGISTER_FLUX_HEADER(ThreeVarMirror)
};

#endif
