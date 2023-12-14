#ifndef VonNeumannTest_HPP
#define VonNeumannTest_HPP

#include "PhysicsCases.hpp"

/*
	Linear Diffusion Test Case, showcasing how to write a physics case that is compiled
	at the same time as the 
 */

// Always inherit from TransportSystem
class VonNeumannTest : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit VonNeumannTest( toml::value const& config, Grid const& );

		// You must provide implementations of both, these are your boundary condition functions
		Value LowerBoundary( Index, Time ) const override;
		Value UpperBoundary( Index, Time ) const override;

		bool isLowerBoundaryDirichlet( Index ) const override;
		bool isUpperBoundaryDirichlet( Index ) const override;

		// The guts of the physics problem (these are non-const as they
		// are allowed to alter internal state such as to store computations
		// for future calls)
		Value SigmaFn( Index, const State &, Position, Time ) override;
		Value Sources( Index, const State &, Position, Time ) override;

		void dSigmaFn_du( Index, Values &, const State &, Position, Time ) override;
		void dSigmaFn_dq( Index, Values &, const State &, Position, Time ) override;

		void dSources_du( Index, Values&v , const State &, Position, Time ) override;
		void dSources_dq( Index, Values&v , const State &, Position, Time ) override;
		void dSources_dsigma( Index, Values&v , const State &, Position, Time ) override;

		// Finally one has to provide initial conditions for u & q
		Value      InitialValue( Index, Position ) const override;
		Value InitialDerivative( Index, Position ) const override;

private:
	// Put class-specific data here
	double kappa, InitialWidth, InitialHeight, Centre;
	double t0;
	Value ExactSolution( Position, Time ) const;

	// Without this (and the implementation line in VonNeumannTest.cpp)
	// ManTA won't know how to relate the string 'VonNeumannTest' to the class.
	REGISTER_PHYSICS_HEADER( VonNeumannTest )
};

#endif // VonNeumannTest_HPP
