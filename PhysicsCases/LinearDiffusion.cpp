
#include "LinearDiffusion.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(LinearDiffusion);

LinearDiffusion::LinearDiffusion(toml::value const &config, Grid const &)
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construct your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the LinearDiffusion physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
	InitialHeight = toml::find_or(DiffConfig, "InitialHeight", 1.0);
	Centre = toml::find_or(DiffConfig, "Centre", 0.5);
	SourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);

	lowerNeumann = toml::find_or(DiffConfig, "LowerNeumann", false);
}

// Dirichlet Boundary Condition
Value LinearDiffusion::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value LinearDiffusion::UpperBoundary(Index, Time) const
{
	return 0.0;
}

bool LinearDiffusion::isLowerBoundaryDirichlet(Index) const { return !lowerNeumann; };
bool LinearDiffusion::isUpperBoundaryDirichlet(Index) const { return true; };

Value LinearDiffusion::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value LinearDiffusion::Sources(Index, const State &s, Position x, Time t)
{
	double u = s.Variable[0];
	return SourceStrength * u;
}

void LinearDiffusion::dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time)
{
	v[0] = kappa;
};

void LinearDiffusion::dSigmaFn_du(Index, VectorRef v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LinearDiffusion::dSources_du(Index, VectorRef v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LinearDiffusion::dSources_dq(Index, VectorRef v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LinearDiffusion::dSources_dsigma(Index, VectorRef v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value LinearDiffusion::InitialValue(Index, Position x) const
{
	double alpha = 1 / InitialWidth;
	double y = (x - Centre);
	return InitialHeight * ::exp(-alpha * y * y);
}

Value LinearDiffusion::InitialDerivative(Index, Position x) const
{
	double y = (x - Centre);
	double alpha = 1 / InitialWidth;
	return -InitialHeight * (2.0 * y) * ::exp(-alpha * y * y) * alpha;
}

