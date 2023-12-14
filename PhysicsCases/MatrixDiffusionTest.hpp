#ifndef MATRIXDIFFUSIONTEST_HPP
#define MATRIXDIFFUSIONTEST_HPP

#include "PhysicsCases.hpp"

/*
	Matrix Diffusion Test Case, showcasing how to write a physics case that is compiled
	at the same time as the 
 */

// Always inherit from TransportSystem
class MatrixDiffusionTest : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit MatrixDiffusionTest( toml::value const& config, Grid const& );

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
	double Centre,alpha;
	double Lambda1,Lambda2,a1,a2;
	Matrix Kappa;
	Vector InitialHeights;

	// Without this (and the implementation line in MatrixDiffusionTest.cpp)
	// ManTA won't know how to relate the string 'MatrixDiffusionTest' to the class.
	REGISTER_PHYSICS_HEADER( MatrixDiffusionTest )
};

#endif // MATRIXDIFFUSIONTEST_HPP
