#ifndef GCT_HPP
#define GCT_HPP

#include "PhysicsCases.hpp"

/*
	Linear Diffusion Test Case, showcasing how to write a physics case that is compiled
	at the same time as the 
 */

// Always inherit from TransportSystem
class GCT : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit GCT( toml::value const& config );

		// You must provide implementations of both, these are your boundary condition functions
		Value LowerBoundary( Index, Time ) const override;
		Value UpperBoundary( Index, Time ) const override;

		bool isLowerBoundaryDirichlet( Index ) const override;
		bool isUpperBoundaryDirichlet( Index ) const override;

		// The guts of the physics problem (these are non-const as they
		// are allowed to alter internal state such as to store computations
		// for future calls)
		virtual Value SigmaFn(Index i, const State &s, Position x, Time t) override;
		virtual Value Sources(Index i, const State &s, Position x, Time t) override;

		// We need derivatives of the flux functions
		virtual void dSigmaFn_du(Index i, Values &, const State &s, Position x, Time t) override;
		virtual void dSigmaFn_dq(Index i, Values &, const State &s, Position x, Time t) override;

		// and for the sources
		virtual void dSources_du(Index i, Values &, const State &, Position x, Time t) override;
		virtual void dSources_dq(Index i, Values &, const State &, Position x, Time t) override;
		virtual void dSources_dsigma(Index i, Values &, const State &, Position x, Time t) override;

		// Finally one has to provide initial conditions for u & q
		Value      InitialValue( Index, Position ) const override;
		Value InitialDerivative( Index, Position ) const override;

		// This problem has Scalars...

		virtual Value ScalarG( Index i, const DGSoln& soln, Time t ) override;
		virtual void ScalarGPrime( Index i, State &v, const DGSoln &soln, std::function<double( double )> p, Interval I, Time t ) override;


private:
	// Put class-specific data here
	double kappa, alpha, beta, u0;

	// Without this (and the implementation line in GCT.cpp)
	// ManTA won't know how to relate the string 'GCT' to the class.
	REGISTER_PHYSICS_HEADER( GCT )
};

#endif // GCT_HPP
