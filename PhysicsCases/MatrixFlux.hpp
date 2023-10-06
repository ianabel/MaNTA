#ifndef MATRIXFLUX
#define MATRIXFLUX

#include "AutodiffFlux.hpp"

class MatrixFlux : public FluxObject
{
public:
    MatrixFlux(toml::value const &config, Index nVars);

private:
    double InitialWidth, Centre;
    static Matrix Kappa;
    Vector InitialHeights;

    static dual F1(VectorXdual u, VectorXdual q, dual x, double t);
    static dual F2(VectorXdual u, VectorXdual q, dual x, double t);
    static dual F3(VectorXdual u, VectorXdual q, dual x, double t);
    static dual S1(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual S2(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);
    static dual S3(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t);

    REGISTER_FLUX_HEADER(MatrixFlux)
};
#endif