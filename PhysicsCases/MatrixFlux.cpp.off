#include "MatrixFlux.hpp"
#include "Constants.hpp"

REGISTER_FLUX_IMPL(MatrixFlux);

MatrixFlux::MatrixFlux(toml::value const &config, Index nVars)
{
    if (config.count("MatrixFlux") != 1)
        throw std::invalid_argument("There should be a [MatrixFlux] section if you are using the MatrixFlux physics model.");

    auto const &DiffConfig = config.at("MatrixFlux");
    std::vector<double> Kappa_v = toml::find<std::vector<double>>(DiffConfig, "Kappa");

    Kappa = MatrixWrapper(Kappa_v.data(), nVars, nVars);

    sigma.insert(std::pair<Index, sigmaptr>(0, &F1));
    sigma.insert(std::pair<Index, sigmaptr>(1, &F2));
    sigma.insert(std::pair<Index, sigmaptr>(2, &F3));

    source.insert(std::pair<Index, sourceptr>(0, &S1));
    source.insert(std::pair<Index, sourceptr>(1, &S2));
    source.insert(std::pair<Index, sourceptr>(2, &S3));
}

Matrix MatrixFlux::Kappa;

dual MatrixFlux::F1(VectorXdual u, VectorXdual q, dual x, double t)
{
    auto sigma = Kappa * q;

    return sigma(0);
};

dual MatrixFlux::F2(VectorXdual u, VectorXdual q, dual x, double t)
{

    auto sigma = Kappa * q;

    return sigma(1);
};
dual MatrixFlux::F3(VectorXdual u, VectorXdual q, dual x, double t)
{

    auto sigma = Kappa * q;

    return sigma(2);
};

dual MatrixFlux::S1(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return u(0) + q(0) + sigma(0);
};

// look at ion and electron sources again -- they should be opposite
dual MatrixFlux::S2(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return u(1) + q(1) + sigma(1);
};
dual MatrixFlux::S3(VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return u(2) + q(2) + sigma(2);
};
