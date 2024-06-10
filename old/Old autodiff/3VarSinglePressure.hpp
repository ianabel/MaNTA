#include "AutodiffFlux.hpp"

#ifndef THREEVARSINGLEPRESSURE
#define THREEVARSINGLEPRESSURE

class ThreeVarSinglePressure : public FluxObject
{
public:
    ThreeVarSinglePressure(toml::value const &config, Index nVars);

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
    static dual q_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual hi_hat(VectorXdual u, VectorXdual q, dual x, double t);
    static dual Sn_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Sp_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual Shi_hat(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);

    static dual phi0(VectorXdual u, VectorXdual q, dual x, double t);
    static dual dphi0dV(VectorXdual u, VectorXdual q, dual x, double t);

    static double Rmin;
    static double Rmax;

    static double R(double x, double t);
    static double psi(double R);
    static double V(double R);
    static double Vprime(double R);
    static double B(double x, double t);

    REGISTER_FLUX_HEADER(ThreeVarSinglePressure)
};

#endif
