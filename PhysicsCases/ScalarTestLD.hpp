#ifndef SCALARTESTLD_HPP
#define SCALARTESTLD_HPP

#include "PhysicsCases.hpp"

/*
	Linear Diffusion Test Case with a trivial scalar
 */

// Always inherit from TransportSystem
class ScalarTestLD : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit ScalarTestLD( toml::value const& config, Grid const& );

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


		virtual Value ScalarGExtended( Index, const DGSoln&, const DGSoln &, Time ) override;
		virtual void ScalarGPrimeExtended( Index, State &, State &, const DGSoln &, std::function<double( double )>, Interval, Time ) override;
		virtual void dSources_dScalars( Index, Values &, const State &, Position, Time ) override;

		// Finally one has to provide initial conditions for u & q
		Value      InitialValue( Index, Position ) const override;
		Value InitialDerivative( Index, Position ) const override;
<<<<<<< HEAD
		Value InitialScalarValue( Index ) const override;
=======
		Value InitialScalarValue( Index ) const override { return 0.0; };
>>>>>>> relax-sources

private:
	// Put class-specific data here
	double kappa, InitialWidth, InitialHeight, Centre;

	// Without this (and the implementation line in ScalarTestLD.cpp)
	// ManTA won't know how to relate the string 'ScalarTestLD' to the class.
	REGISTER_PHYSICS_HEADER( ScalarTestLD )
};

#endif // SCALARTESTLD_HPP
