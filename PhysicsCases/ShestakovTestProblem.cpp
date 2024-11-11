#include "ShestakovTestProblem.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(ShestakovTestProblem);

ShestakovTestProblem::ShestakovTestProblem(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, 1, 0, 0) // Configure a blank autodiff system with one variable and no scalars
{
    uL={0.0};
    uR={0.0};
    isUpperDirichlet = true;  // u(1) = 0
    isLowerDirichlet = false; // u'(0) = 0
}

Real ShestakovTestProblem::Flux(Index, RealVector u, RealVector q, Real, Time)
{
    // Gamma = - n'^3/ n^2
    return q(0) * q(0) * q(0) / (u(0)*u(0));
}

Real ShestakovTestProblem::Source(Index, RealVector, RealVector, RealVector, RealVector, Real x, Time)
{
    if( x < 0.1 )
        return 1.0;
    else
        return 0.0;
}

Value ShestakovTestProblem::InitialValue(Index, Position x) const
{
    if( x > 0.9 )
        return 10.0*(1.0 - x);
    else
        return 1.0;

}

Value ShestakovTestProblem::InitialDerivative(Index, Position x) const
{
    if( x > 0.9 )
        return -10.0;
    else
        return 0.0;

}
