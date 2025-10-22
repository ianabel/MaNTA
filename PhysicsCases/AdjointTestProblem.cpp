#include "AdjointTestProblem.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(AdjointTestProblem);

AdjointTestProblem::AdjointTestProblem(toml::value const &config, Grid const &grid)
    : AutodiffTransportSystem(config, grid, 1, 0, 0) // Configure a blank autodiff system with three variables and no scalars
{
    if (config.count("AdjointTestProblem") != 1)
    {
        throw std::invalid_argument("There should be a [AdjointTestProblem] section.");
    }

    T_s = 50;
    D = 1.0;
    SourceWidth = 0.02;
    SourceCentre = 0.3;

    addP(std::ref(SourceCentre));
}

Real AdjointTestProblem::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
    return D * q(0);
}

Real AdjointTestProblem::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t)

{
    Real y = (x - SourceCentre);
    return T_s * exp(-y * y / SourceWidth);
}

Real AdjointTestProblem::g(Position, Real, RealVector u, RealVector, RealVector, RealVector)
{
    return 0.5 * u(0) * u(0);
}
