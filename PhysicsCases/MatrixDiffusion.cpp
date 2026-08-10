
#include "MatrixDiffusion.hpp"
/*
	Implementation of the Matrix Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(MatrixDiffusion);

SystemSpec MatrixDiffusion::buildSpec(toml::value const &config)
{
	Index nVars = 1;
	if (config.count("DiffusionProblem") == 1)
		nVars = toml::find_or(config.at("DiffusionProblem"), "nVars", 1);
	return {.variables = numberedFields(nVars)};
}

MatrixDiffusion::MatrixDiffusion(toml::value const &config, Grid const &)
	: TransportSystem(buildSpec(config))
{

	// Construct your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the MatrixDiffusion physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
	Centre = toml::find_or(DiffConfig, "Centre", 0.0);

	std::vector<double> InitialHeight_v = toml::find<std::vector<double>>(DiffConfig, "InitialHeights");
	std::vector<double> Kappa_v = toml::find<std::vector<double>>(DiffConfig, "Kappa");

	Kappa = MatrixWrapper(Kappa_v.data(), nVars, nVars);

	if (static_cast<Index>(InitialHeight_v.size()) != nVars)
	{
		throw std::invalid_argument("Initial height vector must have " + std::to_string(nVars) + " elements");
	}

	InitialHeights.resize(nVars);
	for (Index i = 0; i < nVars; ++i)
		InitialHeights[i] = InitialHeight_v[i];

	//	Kappa = Matrix::Identity(nVars, nVars);
}

// Dirichlet Boundary Condition
Value MatrixDiffusion::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value MatrixDiffusion::UpperBoundary(Index, Time) const
{
	return 0.0;
}

Value MatrixDiffusion::SigmaFn(Index i, const State &s, Position, Time)
{
	auto sigma = Kappa * s.q();

	return sigma(i);
}

Value MatrixDiffusion::Sources(Index, const State &, Position, Time)
{
	return 0.0;
}

void MatrixDiffusion::dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time)
{
	for (Index j = 0; j < nVars; ++j)
		v[j] = Kappa(i, j);
};

void MatrixDiffusion::dSigmaFn_du(Index, VectorRef v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusion::dSources_du(Index, VectorRef v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusion::dSources_dq(Index, VectorRef v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusion::dSources_dsigma(Index, VectorRef v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value MatrixDiffusion::InitialValue(Index i, Position x) const
{
	double y = (x - Centre) / InitialWidth;
	return InitialHeights[i] * ::exp(-y * y);
}

Value MatrixDiffusion::InitialDerivative(Index i, Position x) const
{
	double y = (x - Centre) / InitialWidth;
	return InitialHeights[i] * (-2.0 * y) * ::exp(-y * y) * (1.0 / InitialWidth);
}
