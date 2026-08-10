#ifndef TESTDIFFUSION_HPP
#define TESTDIFFUSION_HPP

#include "TransportSystem.hpp"

// The constructor takes a toml::value. Without this include the header only
// compiles when the translation unit happens to have included <toml.hpp>
// first, which is why it worked from SystemSolverTests.cpp but nowhere else.
#include <toml.hpp>

/*
	Header-only TransportSystem for the Unit tests to
	enable testing of construction of a SystemSolver
 */

class TestDiffusion : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit TestDiffusion( toml::value const& config )
			: TransportSystem({.variables = numberedFields(1)})
		{
			// Construct your problem from user-specified config
			// throw an exception if you can't. NEVER leave a part-constructed object around
			// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

			if ( config.count( "DiffusionProblem" ) != 1 )
				throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the LinearDiffusion physics model." );

			auto const& DiffConfig = config.at( "DiffusionProblem" );

			kappa  =         toml::find_or( DiffConfig, "Kappa", 1.0 );
			Centre =         toml::find_or( DiffConfig, "Centre", 0.5 );
			InitialHeight = 1.0;

		};

		// You must provide implementations of both, these are your boundary condition functions
		Value LowerBoundary( Index, Time t ) const override { return ExactSolution( 0.0, t );};
		Value UpperBoundary( Index, Time t ) const override { return ExactSolution( 1.0, t );};

		// The guts of the physics problem (these are non-const as they
		// are allowed to alter internal state such as to store computations
		// for future calls)
		Value SigmaFn( Index, const State &s, Position, Time ) override {
			return kappa * s.Derivative[ 0 ];
		};
		Value Sources( Index, const State&, Position, Time ) override {
			return 0.0;
		};

		void dSigmaFn_dq( Index, VectorRef  v, const State &, Position, Time ) override
		{
			v[ 0 ] = kappa;
		};

		void dSigmaFn_du( Index, VectorRef  v, const State&, Position, Time ) override
		{
			v[ 0 ] = 0.0;
		};

		void dSources_du( Index, VectorRef v , const State &, Position, Time ) override
		{
			v[ 0 ] = 0.0;
		};

		void dSources_dq( Index, VectorRef v , const State &, Position, Time ) override
		{
			v[ 0 ] = 0.0;
		};

		void dSources_dsigma( Index, VectorRef v , const State &, Position, Time ) override 
		{
			v[ 0 ] = 0.0;
		};

		// Finally one has to provide initial conditions for u & q
		Value InitialValue( Index, Position x ) const override {
			double y = ( x - Centre );
			return InitialHeight * ::cos( y* M_PI_2 );
		};
		Value InitialDerivative( Index, Position x ) const override {
			double y = ( x - Centre );
			return -M_PI_2 * InitialHeight * ::sin( y*M_PI_2 );
		};
		Value ExactSolution( Position x, Time t ) const
		{
			return InitialValue( 0, x ) * ::exp( -( t*M_PI_2*M_PI_2 ) );
		}

	// Put class-specific data here
	double kappa, InitialHeight, Centre;

	// Doesn't include class registration, this is header-only for internal testing
};

#endif // TESTDIFFUSION_HPP
