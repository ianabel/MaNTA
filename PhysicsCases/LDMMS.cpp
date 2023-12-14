
#include "LDMMS.hpp"
#include <iostream>

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( LDMMS );

LDMMS::LDMMS( toml::value const& config, Grid const& )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 )
		throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the LDMMS physics model." );

	auto const& DiffConfig = config.at( "DiffusionProblem" );

	kappa =         toml::find_or( DiffConfig, "Kappa", 1.0 );
	InitialWidth  = toml::find_or( DiffConfig, "InitialWidth", 0.2 );
	InitialHeight = toml::find_or( DiffConfig, "InitialHeight", 1.0 );
	Centre =        toml::find_or( DiffConfig, "Centre", 0.0 );

	t0 = InitialWidth * InitialWidth / ( 4.0 * kappa );
	std::cerr << "t0 = " << t0 << std::endl;

}

// Dirichlet Boundary Conditon
Value LDMMS::LowerBoundary( Index, Time t ) const
{
	return 0.0;
}

Value LDMMS::UpperBoundary( Index, Time t ) const
{
	return 0.0;
}

bool LDMMS::isLowerBoundaryDirichlet( Index ) const { return true; };
bool LDMMS::isUpperBoundaryDirichlet( Index ) const { return true; };


Value LDMMS::SigmaFn( Index, const State & s, Position, Time )
{
	return kappa * s.Derivative[ 0 ];
}

// forcing such that u -> cos(pi*x/2) => f(x) = -kappa * u'' = kappa * (pi^2/4) cos(pi*x/2) 
Value LDMMS::Sources( Index, const State &, Position x, Time )
{
	return kappa * ( M_PI_2 * M_PI_2 * ::cos( M_PI_2*x ) );
}

void LDMMS::dSigmaFn_dq( Index, Values& v, const State&, Position, Time )
{
	v[ 0 ] = kappa;
};

void LDMMS::dSigmaFn_du( Index, Values& v, const State&, Position, Time )
{
	v[ 0 ] = 0.0;
};

void LDMMS::dSources_du( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void LDMMS::dSources_dq( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void LDMMS::dSources_dsigma( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};



// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value LDMMS::InitialValue( Index, Position x ) const
{
	double y = (x - Centre);
	return ::exp( -25*y*y );
}

Value LDMMS::InitialDerivative( Index, Position x ) const
{
	double y = (x - Centre);
	return -50*y*::exp( -25*y*y );
}


