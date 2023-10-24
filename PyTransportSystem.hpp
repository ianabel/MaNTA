#ifndef PYTRANSPORTSYSTEM_HPP
#define PYTRANSPORTSYSTEM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "TransportSystem.hpp"

class PyTransportSystem : public TransportSystem
{
	using TransportSystem::TransportSystem;
	Value LowerBoundary( Index i, Time t ) const override { PYBIND11_OVERRIDE_PURE( Value, TransportSystem, LowerBoundary, i, t ); };
	Value UpperBoundary( Index i, Time t ) const override { PYBIND11_OVERRIDE_PURE( Value, TransportSystem, UpperBoundary, i, t ); };
	bool isLowerBoundaryDirichlet( Index i ) const override { PYBIND11_OVERRIDE_PURE( Value, TransportSystem, isLowerBoundaryDirichlet, i ); };
	bool isUpperBoundaryDirichlet( Index i ) const override { PYBIND11_OVERRIDE_PURE( Value, TransportSystem, isUpperBoundaryDirichlet, i ); };

	// The guts of the physics problem (these are non-const as they
	// are allowed to alter internal state such as to store computations
	// for future calls)
	Value SigmaFn( Index i, const Values &u, const Values &q, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( Value, SigmaFn, TransportSystem, i, u, q, x, t );
	};
	Value Sources( Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( Value, Sources, TransportSystem, i, u, q, sigma, x, t );
	};
	void dSigmaFn_du( Index, Values &, const Values &, const Values &, Position, Time ) override;
	void dSigmaFn_dq( Index, Values &, const Values &, const Values &, Position, Time ) override;

	void dSources_du( Index, Values&v , const Values &, const Values &, Position, Time ) override;
	void dSources_dq( Index, Values&v , const Values &, const Values &, Position, Time ) override;
	void dSources_dsigma( Index, Values&v , const Values &, const Values &, Position, Time ) override;

	// Finally one has to provide initial conditions for u & q
	Value      InitialValue( Index, Position ) const override;
	Value InitialDerivative( Index, Position ) const override;
};


#endif // PYTRANSPORTSYSTEM_HPP
