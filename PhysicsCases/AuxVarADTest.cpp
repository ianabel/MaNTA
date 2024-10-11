
#include "AuxVarADTest.hpp"
#include <iostream>

/*
 * Simple reaction-diffusion test case
 *
 * d_t u - kappa * d_xx u = u*u + f(x)
 *
 * where we set f(x) = - kappa d_xx U(x) - U(x) * U(x) to push the system towards u(t->inf,x) = U(x)
 *
 * We artificially introduce a = u * u as an auxiliary variable and solve
 *
 * d_t u - kappa * d_xx u = a + f(x)  ; a = u * u
 *
 * with the auxiliary variable system
 *
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(AuxVarADTest);

AuxVarADTest::AuxVarADTest(toml::value const &config, Grid const &grid) : AutodiffTransportSystem(config, grid, 1, 0, 1)
{
    // Construst your problem from user-specified config
    // throw an exception if you can't. NEVER leave a part-constructed object around
    // here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

    if (config.count("DiffusionProblem") != 1)
        throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the AuxVarADTest physics model.");

    auto const &DiffConfig = config.at("DiffusionProblem");

    kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
    InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
    InitialHeight = toml::find_or(DiffConfig, "InitialHeight", 1.0);
    Centre = toml::find_or(DiffConfig, "Centre", 0.0);
}

// Dirichlet Boundary Conditon
Value AuxVarADTest::LowerBoundary(Index, Time t) const
{
    return 0.0;
}

Value AuxVarADTest::UpperBoundary(Index, Time t) const
{
    return 0.0;
}

bool AuxVarADTest::isLowerBoundaryDirichlet(Index) const { return true; };
bool AuxVarADTest::isUpperBoundaryDirichlet(Index) const { return true; };

Real AuxVarADTest::Flux(Index i, RealVector, RealVector q, Real, Time)
{
    return kappa * q(i);
}

Real AuxVarADTest::Source(Index, RealVector u, RealVector, RealVector, RealVector phi, Real x, Time)
{
    Real U = cos(M_PI_2 * x);
    return kappa * M_PI_2 * M_PI_2 * U + phi(0) - U * U;
}

Real AuxVarADTest::GFunc(Index, RealVector u, RealVector, RealVector, RealVector phi, Position, Time)
{
    return phi(0) - u(0) * u(0);
}

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0

autodiff::dual2nd AuxVarADTest::InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t) const
{
    autodiff::dual2nd y = (x - Centre);
    return exp(-25 * y * y);
}

Value AuxVarADTest::InitialAuxValue(Index, Position x) const
{
    double u0 = InitialValue(0, x);
    return u0 * u0;
}
