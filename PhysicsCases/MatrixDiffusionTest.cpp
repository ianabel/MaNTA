
#include "MatrixDiffusionTest.hpp"
#include <iostream>

/*
	Implementation of the Matrix Diffusion Test case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(MatrixDiffusionTest);

MatrixDiffusionTest::MatrixDiffusionTest( toml::value const &config, Grid const& )
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
	Kappa( 0, 1 ) = alpha;
	Kappa( 1, 0 ) = alpha;

	Lambda1 = 1 + alpha;
	Lambda2 = 1 - alpha;

	// Orthonormal Eigenvectors 
	// are [ 1 , +/-1 ] / Sqrt(2)

	// u0 = Sqrt(2) * (a1 * e1 + a2 * e2)
	// => u(0) = a1 + a2
	//    u(1) = a1 - a2
	a1 = ( InitialHeights[ 0 ] + InitialHeights[ 1 ] )/( 2.0 );
	a2 = ( InitialHeights[ 0 ] - InitialHeights[ 1 ] )/( 2.0 );

}

// Dirichlet Boundary Conditon
Value MatrixDiffusionTest::LowerBoundary(Index i, Time t) const
{
	if ( i == 0 )
		return a1 * ::exp( -( t*Lambda1*M_PI_2*M_PI_2 ) ) + a2 * ::exp( -( t*Lambda2*M_PI_2*M_PI_2 ) );
	else if ( i == 1 )
		return a1 * ::exp( -( t*Lambda1*M_PI_2*M_PI_2 ) ) - a2 * ::exp( -( t*Lambda2*M_PI_2*M_PI_2 ) );
	else {
		throw std::runtime_error("i > nVars in LowerBoundary");
		return 0.0;
	}
}


Value MatrixDiffusionTest::UpperBoundary(Index, Time) const
{
	return 0.0;
}

bool MatrixDiffusionTest::isLowerBoundaryDirichlet(Index) const { return true; };
bool MatrixDiffusionTest::isUpperBoundaryDirichlet(Index) const { return true; };

Value MatrixDiffusionTest::SigmaFn(Index i, const State &s, Position, Time)
{
	Eigen::Map<const Vector> qVec( s.Derivative.data(), s.Derivative.size() );
	auto sigma = Kappa * qVec;

	return sigma( i );
}

Value MatrixDiffusionTest::Sources(Index, const State &, Position, Time)
{
	return 0.0;
}

void MatrixDiffusionTest::dSigmaFn_dq(Index i, Values &v, const State &, Position, Time)
{
	for (Index j = 0; j < nVars; ++j)
		v[j] = Kappa(i, j);
};

void MatrixDiffusionTest::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusionTest::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusionTest::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v = Vector::Zero(nVars);
};

void MatrixDiffusionTest::dSources_dsigma(Index, Values &v, const State &, Position, Time)
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
