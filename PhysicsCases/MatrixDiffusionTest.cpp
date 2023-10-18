
#include "MatrixDiffusionTest.hpp"

/*
	Implementation of the Matrix Diffusion Test case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(MatrixDiffusionTest);

MatrixDiffusionTest::MatrixDiffusionTest(toml::value const &config)
{

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the MatrixDiffusion physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	nVars = toml::find_or(DiffConfig, "nVars", 2 );
	Centre = toml::find_or(DiffConfig, "Centre", 0.0);
	alpha = toml::find_or(DiffConfig, "alpha", 0.0);

	std::vector<double> InitialHeight_v = toml::find<std::vector<double>>(DiffConfig, "InitialHeights");

	if (static_cast<Index>(InitialHeight_v.size()) != nVars)
	{
		throw std::invalid_argument("Initial height vector must have " + std::to_string( nVars ) + " elements");
	}

	InitialHeights.resize(nVars);
	for (Index i = 0; i < nVars; ++i)
		InitialHeights[i] = InitialHeight_v[i];

	Kappa = Matrix::Identity(nVars, nVars);
}

// Dirichlet Boundary Conditon
Value MatrixDiffusionTest::LowerBoundary(Index i, Time t) const
{
	return InitialHeights[i] * ::exp( -( t*M_PI_2*M_PI_2 ) );
}

Value MatrixDiffusionTest::UpperBoundary(Index, Time) const
{
	return 0.0;
}

bool MatrixDiffusionTest::isLowerBoundaryDirichlet(Index) const { return true; };
bool MatrixDiffusionTest::isUpperBoundaryDirichlet(Index) const { return true; };

Value MatrixDiffusionTest::SigmaFn(Index i, const Values &, const Values &q, Position, Time)
{
	auto sigma = Kappa * q;

	return sigma( i );
}

Value MatrixDiffusionTest::Sources(Index, const Values &, const Values &, const Values &, Position, Time)
{
	return 0.0;
}

void MatrixDiffusionTest::dSigmaFn_dq(Index i, Values &v, const Values &, const Values &, Position, Time)
{
	for (Index j = 0; j < nVars; ++j)
		v[j] = Kappa(i, j);
};

void MatrixDiffusionTest::dSigmaFn_du(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusionTest::dSources_du(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusionTest::dSources_dq(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusionTest::dSources_dsigma(Index, Values &v, const Values &, const Values &, Position, Time)
{
	v = Vector::Zero(nVars);
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value MatrixDiffusionTest::InitialValue(Index i, Position x) const
{
	double y = (x - Centre);
	return InitialHeights[i] * ::cos( M_PI_2 * y );
}

Value MatrixDiffusionTest::InitialDerivative(Index i, Position x) const
{
	double y = (x - Centre);
	return -1.0 * M_PI_2 * InitialHeights[i] * ::sin( M_PI_2 * y );
}
