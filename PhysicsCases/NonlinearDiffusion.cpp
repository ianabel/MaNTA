
#include "NonlinearDiffusion.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( NonlinearDiffusion );

NonlinearDiffusion::NonlinearDiffusion( toml::value const& config )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 ) {
		n = 2;
	} else {
		auto const& DiffConfig = config.at( "DiffusionProblem" );
		n = toml::find_or( DiffConfig, "n", 2 );
	}

}

// Dirichlet Boundary Conditon at x = 0
Value NonlinearDiffusion::LowerBoundary( Index, Time ) const
{
	return 1.0;
}

// Exact solution at x = 1
Value NonlinearDiffusion::UpperBoundary( Index, Time t ) const
{
	return ExactSolution( 1, 1 + t );
}

bool NonlinearDiffusion::isLowerBoundaryDirichlet( Index ) const { return true; };
bool NonlinearDiffusion::isUpperBoundaryDirichlet( Index ) const { return true; };


Value NonlinearDiffusion::SigmaFn( Index, const Values &uV, const Values & qV, Position, Time )
{
	double u = uV[ 0 ],q = qV[ 0 ];

	double NonlinearKappa = ( n/2.0 )*::pow( u, n )*( 1.0 - ::pow( u, n )/( n + 1.0 ) );
	return NonlinearKappa * q;
}

Value NonlinearDiffusion::Sources( Index, const Values &, const Values &, const Values &, Position, Time )
{
	return 0.0;
}

void NonlinearDiffusion::dSigmaFn_dq( Index, Values& v, const Values& uV, const Values&, Position, Time )
{
	double u = uV[ 0 ];
	double NonlinearKappa = ( n/2.0 )*::pow( u, n )*( 1.0 - ::pow( u, n )/( n + 1.0 ) );

	v[ 0 ] = NonlinearKappa;
};

void NonlinearDiffusion::dSigmaFn_du( Index, Values& v, const Values& uV, const Values& qV, Position, Time )
{
	double u = uV[ 0 ], q = qV[ 0 ];
	v[ 0 ] = ( ( n*n )/( 2.0*( n + 1.0 ) ) ) * ::pow( u, n - 1.0 ) * ( 1 + n - 2*::pow( u, n ) ) * q;
};

void NonlinearDiffusion::dSources_du( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void NonlinearDiffusion::dSources_dq( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void NonlinearDiffusion::dSources_dsigma( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};



// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise at t=1
Value NonlinearDiffusion::InitialValue( Index, Position x ) const
{
	return ExactSolution( x, 0.0 );
}

Value NonlinearDiffusion::InitialDerivative( Index, Position x ) const
{
	return ExactSolution( x, 0.0 ) / ( n*( x - 1.0 ) );
}

Value NonlinearDiffusion::ExactSolution( Position x, Time t ) const
{
	double eta = x/::sqrt( 1 + t );
	if ( eta >= 1.0 )
		return 0.0;
	return ::pow( 1.0 - eta, 1.0/n );
}

