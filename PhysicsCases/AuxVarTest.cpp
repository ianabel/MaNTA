
#include "AuxVarTest.hpp"
#include <iostream>

/*
 * Simple reaction-diffusion test case
 *
 * d_t u - kappa * d_xx u = u*u + f(x)
 *
 * where we set f(x) = - kappa d_xx U(x) - U(x) * U(x) to push the system towards u(t->inf,x) = U(x)
 *
 * We artificially introduce a = u * u as an auxiliary variable and solve
 *
 * d_t u - kappa * d_xx u = a + f(x)  ; a = u * u
 *
 * with the auxiliary variable system
 *
 */


// Needed to register the class
REGISTER_PHYSICS_IMPL( AuxVarTest );

AuxVarTest::AuxVarTest( toml::value const& config, Grid const& )
{
	// Always set nVars in a derived constructor
	nVars = 1;
    nAux = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 )
		throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the AuxVarTest physics model." );

	auto const& DiffConfig = config.at( "DiffusionProblem" );

	kappa =         toml::find_or( DiffConfig, "Kappa", 1.0 );
	InitialWidth  = toml::find_or( DiffConfig, "InitialWidth", 0.2 );
	InitialHeight = toml::find_or( DiffConfig, "InitialHeight", 1.0 );
	Centre =        toml::find_or( DiffConfig, "Centre", 0.0 );

}

// Dirichlet Boundary Conditon
Value AuxVarTest::LowerBoundary( Index, Time t ) const
{
	return 1.0;
}

Value AuxVarTest::UpperBoundary( Index, Time t ) const
{
	return 0.0;
}

bool AuxVarTest::isLowerBoundaryDirichlet( Index ) const { return true; };
bool AuxVarTest::isUpperBoundaryDirichlet( Index ) const { return true; };


Value AuxVarTest::SigmaFn( Index, const State & s, Position, Time )
{
	return kappa * s.Derivative[ 0 ];
}

// 
Value AuxVarTest::Sources( Index, const State &st, Position x, Time )
{
    double U = ::cos( M_PI_2 * x );
    double a = st.Aux[ 0 ];
	return kappa * M_PI_2 * M_PI_2 * U + a - U * U;
}

void AuxVarTest::dSigmaFn_dq( Index, Values& v, const State&, Position, Time )
{
	v[ 0 ] = kappa;
};

void AuxVarTest::dSigmaFn_du( Index, Values& v, const State&, Position, Time )
{
	v[ 0 ] = 0.0;
};

void AuxVarTest::dSources_du( Index, Values&v , const State &st, Position, Time )
{
	v[ 0 ] = 0.0;
};

void AuxVarTest::dSources_dq( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void AuxVarTest::dSources_dsigma( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

Value AuxVarTest::InitialAuxValue( Index i, Position x ) const
{
    double u0 = InitialValue( i, x );
    return u0*u0;
}

Value AuxVarTest::AuxG( Index, const State & st, Position x, Time )
{
    double a = st.Aux[0];
    double u = st.Variable[0];
    return a - u*u;
}

void AuxVarTest::AuxGPrime( Index, State &out, const State &st, Position, Time ) {
    double u = st.Variable[ 0 ];

    // most derivatives are zero
    out.zero();
    // dG/du = -2.0 * u
    out.Variable[ 0 ] = -2.0 * u;
    // dG/da = 1.0
    out.Aux[ 0 ] = 1.0;

    return;
}


void AuxVarTest::dSources_dPhi( Index, Values &v, const State &st, Position, Time ) {
    v[ 0 ] = 1.0;
    return;
}

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value AuxVarTest::InitialValue( Index, Position x ) const
{
	double y = (x - Centre);
	return ::exp( -25*y*y );
}

Value AuxVarTest::InitialDerivative( Index, Position x ) const
{
	double y = (x - Centre);
	return -50*y*::exp( -25*y*y );
}


