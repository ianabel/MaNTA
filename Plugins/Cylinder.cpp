
#include "Cylinder.hpp"

Cylinder::Cylinder(toml::value const &config)
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the Cylinder physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);

}

// Dirichlet Boundary Conditon
Value Cylinder::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value Cylinder::UpperBoundary(Index, Time) const
{
	return 0.0;
}

bool Cylinder::isLowerBoundaryDirichlet(Index) const { return true; };
bool Cylinder::isUpperBoundaryDirichlet(Index) const { return true; };

Value Cylinder::SigmaFn(Index, const Values &, const Values &q, Position x, Time)
{

	return kappa * q[0];
}

Value Cylinder::Sources(Index, const Values &, const Values &, const Values &, Position, Time)
{
	return 0.0;
}

void Cylinder::dSigmaFn_dq(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v[0] = kappa;
};

void Cylinder::dSigmaFn_du(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v[0] = 0.0;
};

void Cylinder::dSources_du(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v[0] = 0.0;
};

void Cylinder::dSources_dq(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v[0] = 0.0;
};

void Cylinder::dSources_dsigma(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value Cylinder::InitialValue(Index, Position x) const
{
	return ::exp(-x*x);
}

Value Cylinder::InitialDerivative(Index, Position x) const
{
	return (-2.0 * x) * ::exp(-x * x);
}
