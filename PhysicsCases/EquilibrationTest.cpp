
#include "EquilibrationTest.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(EquilibrationTest);

EquilibrationTest::EquilibrationTest(toml::value const &config)
{
	// Always set nVars in a derived constructor
	nVars = 2;

	auto const& DiffConfig = config.at( "DiffusionProblem" );
	kappa1 = toml::find_or(DiffConfig, "Kappa1", 2.0);
	kappa2 = toml::find_or(DiffConfig, "Kappa2", 1.0);
	S1 = toml::find_or(DiffConfig, "S1", 1.0);
	S2 = toml::find_or(DiffConfig, "S2", 2.0);
	Q = toml::find_or(DiffConfig, "Q", 10.0);

	EdgeValue = 0.1;
}

// Dirichlet Boundary Conditon
Value EquilibrationTest::LowerBoundary(Index, Time) const
{
	return EdgeValue;
}

Value EquilibrationTest::UpperBoundary(Index, Time) const
{
	return EdgeValue;
}

bool EquilibrationTest::isLowerBoundaryDirichlet(Index) const { return true; };
bool EquilibrationTest::isUpperBoundaryDirichlet(Index) const { return true; };

Value EquilibrationTest::SigmaFn(Index i, const State &s, Position x, Time)
{
	if ( i == 0 ) {
		return kappa1 * s.Derivative[ 0 ];
	} else if ( i == 1 ) {
		return kappa2 * s.Derivative[ 1 ];
	} else {
		throw std::runtime_error( "Index out of bounds" );
	}
}

Value EquilibrationTest::Sources(Index i, const State &s, Position x, Time)
{
	double Gaussian = ::exp( -25*x*x );
	double u0,u1;
	u0 = s.Variable[ 0 ];
	u1 = s.Variable[ 1 ];
	if ( i == 0 ) {
		return S1*Gaussian - Q*( u0 - u1 );
	} else if ( i == 1 ) {
		return S2*Gaussian - Q*( u1 - u0 );
	} else {
		throw std::runtime_error( "Index out of bounds" );
	}
}

void EquilibrationTest::dSigmaFn_dq(Index i, Values &v, const State &, Position, Time)
{
	if ( i == 0 ) {
		v[ 0 ] = kappa1;
		v[ 1 ] = 0.0;
	} else if ( i == 1 ) {
		v[ 0 ] = 0.0;
		v[ 1 ] = kappa2;
	} else {
		throw std::runtime_error( "Index out of bounds" );
	}
};

void EquilibrationTest::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
	v[ 1 ] = 0.0;
};

void EquilibrationTest::dSources_du(Index i, Values &v, const State &, Position, Time)
{
	if ( i == 0 ) {
		v[ 0 ] = -Q;
		v[ 1 ] = Q;
	} else if ( i == 1 ) {
		v[ 0 ] = Q;
		v[ 1 ] = -Q;
	} else {
		throw std::runtime_error( "Index out of bounds" );
	}
};

void EquilibrationTest::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
	v[ 1 ] = 0.0;
};

void EquilibrationTest::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
	v[1] = 0.0;
};

Value EquilibrationTest::InitialValue(Index, Position) const
{
	return EdgeValue;
}

Value EquilibrationTest::InitialDerivative(Index, Position) const
{
	return 0;
}
