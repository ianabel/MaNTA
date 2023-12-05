
#include "GlobalConstraintTest.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>

#include <iostream>


/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( GCT );

GCT::GCT( toml::value const& config )
{
	// Always set nVars in a derived constructor
	nVars = 1;
	nScalars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 )
		throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the GCT physics model." );

	auto const& DiffConfig = config.at( "DiffusionProblem" );

	kappa = toml::find_or( DiffConfig, "Kappa", 1.0 );
	alpha = toml::find_or( DiffConfig, "Alpha", 0.2 );
	beta  = toml::find_or( DiffConfig, "Beta", 1.0 );
	u0    = toml::find_or( DiffConfig, "u0", 0.1 );

}

// Neumann Boundary Conditon
Value GCT::LowerBoundary( Index, Time t ) const
{
	return 0.0;
}

Value GCT::UpperBoundary( Index, Time t ) const
{
	return u0;
}

bool GCT::isLowerBoundaryDirichlet( Index ) const { return false; };
bool GCT::isUpperBoundaryDirichlet( Index ) const { return true; };


Value GCT::SigmaFn( Index, const State &s, Position, Time )
{
	return kappa * s.Derivative[ 0 ];
}

Value GCT::Sources( Index, const State &, Position, Time )
{
	return 0.0;
}

void GCT::dSigmaFn_dq( Index, Values& v, const State &, Position, Time )
{
	v[ 0 ] = kappa;
};

void GCT::dSigmaFn_du( Index, Values& v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void GCT::dSources_du( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void GCT::dSources_dq( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void GCT::dSources_dsigma( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

// Initialise with a constant
Value GCT::InitialValue( Index, Position ) const
{
	return u0;
}

Value GCT::InitialDerivative( Index, Position ) const
{
	return 0.0;
}

Value GCT::ScalarG( Index i, const DGSoln& soln, Time t )
{
	// J = Int_0^1 u
	// So G is J - Int_0^1 u 
	return soln.Scalar( 0 ) - boost::math::quadrature::gauss_kronrod<double, 15>::integrate( soln.u( 0 ), 0, 1 );
}

// Provides Int_I ( delta G / delta {u,q,sigma} * p(x) )
void GCT::ScalarGPrime( Index i, State &v, const DGSoln &soln, std::function<double( double )> p, Interval I, Time t )
{
	// For our problem delta G/delta u = -1 ; dG/dJ = 1; and all other derivatives vanish
	new ( &v ) State( 1, 1 ); // reinit and zero
	v.Variable[ 0 ] = -1.0 * boost::math::quadrature::gauss_kronrod<double, 15>::integrate( p, I.x_l, I.x_u );
	v.Scalars[ 0 ] = 1.0; // Scalar derivatives are plain derivatives.
}

