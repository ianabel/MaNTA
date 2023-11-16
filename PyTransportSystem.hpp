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
		PYBIND11_OVERRIDE_PURE( Value, TransportSystem, SigmaFn, i, u, q, x, t );
	};
	Value Sources( Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( Value, TransportSystem, Sources, i, u, q, sigma, x, t );
	};
	void dSigmaFn_du( Index i, Values &out, const Values &u, const Values &q, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( void, TransportSystem, dSigmaFn_du, i, out, u, q, x, t );
	};
	
	void dSigmaFn_dq( Index i, Values &out, const Values &u, const Values &q, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( void, TransportSystem, dSigmaFn_dq, i, out, u, q, x, t );
	};

	void dSources_du( Index i, Values &v, const Values &u, const Values &q, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( void, TransportSystem, dSources_du, i, v, u, q, x, t );
	};

	void dSources_dq( Index i, Values &v, const Values &u, const Values &q, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( void, TransportSystem, dSources_dq, i, v, u, q, x, t );
	};

	void dSources_dsigma( Index i, Values &v, const Values &u, const Values &q, Position x, Time t ) override {
		PYBIND11_OVERRIDE_PURE( void, TransportSystem, dSources_dsigma, i, v, u, q, x, t );
	};

	// Finally one has to provide initial conditions for u & q
	Value InitialValue( Index i, Position x ) const override {
		PYBIND11_OVERRIDE_PURE( Value, TransportSystem, InitialValue, i, x );
	};

	Value InitialDerivative( Index i, Position x ) const override {
		PYBIND11_OVERRIDE_PURE( Value, TransportSystem, InitialDerivative, i, x );
	};
};

#endif // PYTRANSPORTSYSTEM_HPP
