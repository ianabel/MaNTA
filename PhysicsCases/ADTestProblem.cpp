#include "ADTestProblem.hpp"
#include <iostream>

REGISTER_PHYSICS_IMPL(ADTestProblem);

// RegridPolicy::InPlace: the domain ends are all the constructor takes from its
// grid, and AutodiffTransportSystem::prepareEvaluation takes them again from
// each plan.
ADTestProblem::ADTestProblem(toml::value const &config, Grid const &grid)
	: AutodiffTransportSystem(config, grid, {.variables = {{"u", "the diffused quantity", ""}},
	                                         .regrid = RegridPolicy::InPlace})
{
	if (config.count("ADTestProblem") != 1)
	{
		throw std::invalid_argument("There should be a [ADTestProblem] section.");
	}

	auto const &InternalConfig = config.at("ADTestProblem");
	T_s = 50;
	a = 6.0;
	SourceWidth = 0.02;
	SourceCentre = 0.3;
	afn_test = toml::find_or(InternalConfig, "afn", 1.0);
}

Value ADTestProblem::aFn(Index i, Position x)
{
	return afn_test * x;
}

Real ADTestProblem::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
	return afn_test * x * (a / pow(u(0), 1.5)) * q(0);
}

Real ADTestProblem::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t)

{
	Real y = (x - SourceCentre);
	return T_s * exp(-y * y / SourceWidth) * afn_test * x;
}
