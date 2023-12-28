
#include "TwoChannelNonlinear.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( TwoChannelNonlinear );

TwoChannelNonlinear::TwoChannelNonlinear( toml::value const& config, Grid const & )
{
	// Always set nVars in a derived constructor
	nVars = 2;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around

	T_s = 50;
	a = 6.0;
	SourceWidth = 0.02;
	SourceCentre = 0.3;
}

// Dirichlet Boundary Conditon at x = 0
Value TwoChannelNonlinear::LowerBoundary( Index, Time ) const
{
	return 0.0;
}

// Exact solution at x = 1
Value TwoChannelNonlinear::UpperBoundary( Index, Time t ) const
{
	return 0.3;
}

bool TwoChannelNonlinear::isLowerBoundaryDirichlet( Index ) const { return false; };
bool TwoChannelNonlinear::isUpperBoundaryDirichlet( Index ) const { return true; };


Value TwoChannelNonlinear::SigmaFn( Index i, const State &s, Position, Time )
{
	double u,q;
	if( i == 0 ) {
		u = s.Variable[ 0 ]; q = s.Derivative[ 0 ];
	} else {
		u = s.Variable[ 1 ]; q = s.Derivative[ 1 ];
	}

	double NonlinearKappa = a / ::pow( u, 1.5 );
	return NonlinearKappa * q;
}

Value TwoChannelNonlinear::Sources( Index, const State &, Position x, Time )
{
	double y = ( x - SourceCentre );
	return T_s*::exp( -y*y/SourceWidth );
}

void TwoChannelNonlinear::dSigmaFn_dq( Index i, Values& v, const State &s, Position, Time )
{
	if( i == 0 ) {
		double u = s.Variable[ 0 ];
		double NonlinearKappa = a / ::pow( u, 1.5 );
		v[ 0 ] = NonlinearKappa;
		v[ 1 ] = 0.0;
	} else {
		double u = s.Variable[ 1 ];
		double NonlinearKappa = a / ::pow( u, 1.5 );
		v[ 0 ] = 0.0;
		v[ 1 ] = NonlinearKappa;
	}
};

void TwoChannelNonlinear::dSigmaFn_du( Index i, Values& v, const State& s, Position, Time )
{
	double u,q;
	if( i == 0 ) {
		u = s.Variable[ 0 ]; q = s.Derivative[ 0 ];
		v[ 0 ] = -( 3.0/2.0 )*( a / ::pow( u, 2.5 ) )*q;
		v[ 1 ] = 0.0;
	} else {
		u = s.Variable[ 1 ]; q = s.Derivative[ 1 ];
		v[ 0 ] = 0.0;
		v[ 1 ] = -( 3.0/2.0 )*( a / ::pow( u, 2.5 ) )*q;
	}
};

void TwoChannelNonlinear::dSources_du( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
	v[ 1 ] = 0.0;
};

void TwoChannelNonlinear::dSources_dq( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
	v[ 1 ] = 0.0;
};

void TwoChannelNonlinear::dSources_dsigma( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
	v[ 1 ] = 0.0;
};



// We don't need the index variables as both channels are identical

// Initialise at t=1
Value TwoChannelNonlinear::InitialValue( Index, Position x ) const
{
	return 0.3;
}

Value TwoChannelNonlinear::InitialDerivative( Index, Position x ) const
{
	return 0.0;
}



