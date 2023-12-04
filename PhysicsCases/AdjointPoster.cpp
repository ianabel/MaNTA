
#include "AdjointPoster.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( AdjointPoster );

AdjointPoster::AdjointPoster( toml::value const& config )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around

	T_s = 50;
	a = 6.0;
	SourceWidth = 0.02;
	SourceCentre = 0.3;
}

// Dirichlet Boundary Conditon at x = 0
Value AdjointPoster::LowerBoundary( Index, Time ) const
{
	return 0.0;
}

// Exact solution at x = 1
Value AdjointPoster::UpperBoundary( Index, Time t ) const
{
	return 0.3;
}

bool AdjointPoster::isLowerBoundaryDirichlet( Index ) const { return false; };
bool AdjointPoster::isUpperBoundaryDirichlet( Index ) const { return true; };


Value AdjointPoster::SigmaFn( Index, const State &s, Position, Time )
{
	double u = s.Variable[ 0 ],q = s.Derivative[ 0 ];

	double NonlinearKappa = a / ::pow( u, 1.5 );
	return NonlinearKappa * q;
}

Value AdjointPoster::Sources( Index, const State &, Position x, Time )
{
	double y = ( x - SourceCentre );
	return T_s*::exp( -y*y/SourceWidth );
}

void AdjointPoster::dSigmaFn_dq( Index, Values& v, const State &s, Position, Time )
{
	double u = s.Variable[ 0 ];
	double NonlinearKappa = a / ::pow( u, 1.5 );
	v[ 0 ] = NonlinearKappa;
};

void AdjointPoster::dSigmaFn_du( Index, Values& v, const State& s, Position, Time )
{
	double u = s.Variable[ 0 ], q = s.Derivative[ 0 ];
	v[ 0 ] = -( 3.0/2.0 )*( a / ::pow( u, 2.5 ) )*q;
};

void AdjointPoster::dSources_du( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void AdjointPoster::dSources_dq( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void AdjointPoster::dSources_dsigma( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};



// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise at t=1
Value AdjointPoster::InitialValue( Index, Position x ) const
{
	return 0.3;
}

Value AdjointPoster::InitialDerivative( Index, Position x ) const
{
	return 0.0;
}



