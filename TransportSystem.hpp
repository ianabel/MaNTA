#ifndef TRANSPORTSYSTEM_HPP
#define TRANSPORTSYSTEM_HPP

#include "Types.hpp"

/*
	Pure interface class
	defines a problem in the form
		a_i d_t u_i + d_x ( sigma_i ) = S_i( u(x), q(x), x, t ) ; S_i can depend on the entire u & q vector, but only locally.
		sigma_i = sigma_hat_i( u( x ), q( x ), x, t ) ; so can sigma_hat_i

 */

class TransportSystem {
	using 
	public:
		virtual ~TransportSystem() = default;

		Index getNumVars() const { return nVars; };

		// Function for passing boundary conditions to the solver
		virtual Value  LowerBoundary( Index i, Time t ) const = 0;
		virtual Value  UpperBoundary( Index i, Time t ) const = 0;

		virtual bool isLowerBoundaryDirichlet( Index i ) const = 0;
		virtual bool isUpperBoundaryDirichlet( Index i ) const = 0;

		// The same for the flux and source functions -- the vectors have length nVars
		virtual Value SigmaFn( Index i, const Values &u, const Values &q, Position x, Time t ) = 0;
		virtual Value Sources( Index i, const Values &u, const Values &q, Position x, Time t ) = 0;

		// We need derivatives of the flux functions
		virtual Value dSigmaFn_du( Index i, Values&, const Values &u, const Values &q, Position x, Time t ) = 0;
		virtual Value dSigmaFn_dq( Index i, Values&, const Values &u, const Values &q, Position x, Time t ) = 0;

		// and for the sources
		virtual Value dSources_du( Index i, Values&, const Values &u, const Values &q, Position x, Time t ) = 0;
		virtual Value dSources_dq( Index i, Values&, const Values &u, const Values &q, Position x, Time t ) = 0;
		virtual Value dSources_dsigma( Index i, Values&, const Values &u, const Values &q, Position x, Time t ) = 0;

		// and initial conditions for u & q
		virtual Value      InitialValue( Index i, Position x ) const = 0;
		virtual Value InitialDerivative( Index i, Position x ) const = 0;

	protected:
		Index nVars;

};

#endif // TRANSPORTSYSTEM_HPP
