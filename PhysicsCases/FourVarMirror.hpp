#include "AutodiffFlux.hpp"

#ifndef FOURVARMirror
#define FOURVARMirror

class FourVarMirror : public FluxObject
{
public:
    FourVarMirror(toml::value const &config, Index nVars);

private:
    std::map<std::string, int> ParticleSources = {{"None", 0}, {"Gaussian", 1}};
    static int ParticleSource;
    static double sourceStrength;
    Vector InitialHeights;
    static dual sourceWidth;
    static dual sourceCenter;

    // reference values
    static dual n0;
    static dual Bmid;
    static dual T0;
    static dual E0;
    static dual J0;
    static Value L;
    static dual p0;
    static dual V0;
    static dual Gamma0;
    static dual taue0;
    static dual taui0;
    static dual h0;
    static dual omega0;

    static dual Gamma_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual qe_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual qi_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual hi_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Spe_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Spi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Shi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);

    static dual phi0(VectorXdual u, VectorXdual q, dual x, double t);
    static dual dphi0dV(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Chi_e(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Chi_i(VectorXdual u, VectorXdual q, dual x, double t);
    static bool includeParallelLosses;

    static double Rmin;
    static double Rmax;

    static double R(double x, double t);
    static double psi(double R);
    static double V(double R);
    static double Vprime(double R);
    static double B(double x, double t);
    static double Bmax;

    REGISTER_FLUX_HEADER(FourVarMirror)
};

#endif
