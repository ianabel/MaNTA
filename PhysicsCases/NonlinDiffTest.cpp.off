
#include "NonlinDiffTest.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( NonlinDiffTest );

NonlinDiffTest::NonlinDiffTest( toml::value const& config )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around

	if ( config.count( "DiffusionProblem" ) != 1 ) {
		n = 2;
	} else {
		auto const& DiffConfig = config.at( "DiffusionProblem" );
		n = toml::find_or( DiffConfig, "n", 2 );
	}

	t0 = 1.1;

}

// Dirichlet Boundary Conditon at x = 0
Value NonlinDiffTest::LowerBoundary( Index, Time ) const
{
	return 1.0;
}

// Exact solution at x = 1
Value NonlinDiffTest::UpperBoundary( Index, Time t ) const
{
	return ExactSolution( 1, t0 + t );
}

bool NonlinDiffTest::isLowerBoundaryDirichlet( Index ) const { return true; };
bool NonlinDiffTest::isUpperBoundaryDirichlet( Index ) const { return true; };


Value NonlinDiffTest::SigmaFn( Index, const Values &uV, const Values & qV, Position, Time )
{
	double u = uV[ 0 ],q = qV[ 0 ];

	double NonlinearKappa = ( n/2.0 )*::pow( u, n )*( 1.0 - ::pow( u, n )/( n + 1.0 ) );
	return NonlinearKappa * q;
}

Value NonlinDiffTest::Sources( Index, const Values &, const Values &, const Values &, Position, Time )
{
	return 0.0;
}

void NonlinDiffTest::dSigmaFn_dq( Index, Values& v, const Values& uV, const Values&, Position, Time )
{
	double u = uV[ 0 ];
	double NonlinearKappa = ( n/2.0 )*::pow( u, n )*( 1.0 - ::pow( u, n )/( n + 1.0 ) );

	v[ 0 ] = NonlinearKappa;
};

void NonlinDiffTest::dSigmaFn_du( Index, Values& v, const Values& uV, const Values& qV, Position, Time )
{
	double u = uV[ 0 ], q = qV[ 0 ];
	v[ 0 ] = ( ( n*n )/( 2.0*( n + 1.0 ) ) ) * ::pow( u, n - 1.0 ) * ( 1 + n - 2*::pow( u, n ) ) * q;
};

void NonlinDiffTest::dSources_du( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void NonlinDiffTest::dSources_dq( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void NonlinDiffTest::dSources_dsigma( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};



// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise at t=1
Value NonlinDiffTest::InitialValue( Index, Position x ) const
{
	return ExactSolution( x, t0 );
}

Value NonlinDiffTest::InitialDerivative( Index, Position x ) const
{
	return ExactSolution( x, t0 ) / ( n*( x/::sqrt( t0 ) - 1.0 ) );
}

Value NonlinDiffTest::ExactSolution( Position x, Time t ) const
{
	double eta = x/::sqrt( t );
	if ( eta >= 1.0 )
		return 0.0;
	return ::pow( 1.0 - eta, 1.0/n );
}

