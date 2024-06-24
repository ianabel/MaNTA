#include "ADTestProblem.hpp"
#include "Constants.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(ADTestProblem);

ADTestProblem::ADTestProblem(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, 1, 0) // Configure a blank autodiff system with three variables and no scalars
{
	if (config.count("ADTestProblem") != 1)
	{
		throw std::invalid_argument("There should be a [ADTestProblem] section if you are using the 3VarCylinder physics model.");
	}

	T_s = 50;
	a = 6.0;
	SourceWidth = 0.02;
	SourceCentre = 0.3;
};

Real ADTestProblem::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
	return (a / pow(u(0), 1.5)) * q(0);
}

Real ADTestProblem::Source(Index i, RealVector u, RealVector q, RealVector sigma, Real x, Time t)
{
	Real y = (x - SourceCentre);
	return T_s * exp(-y * y / SourceWidth);
}

Value ADTestProblem::InitialValue(Index, Position) const
{
	return 0.3;
}

Value ADTestProblem::InitialDerivative(Index, Position) const
{
	return 0.0;
}
