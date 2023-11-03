#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>

#include "PhysicsCases.hpp"
#include "PyTransportSystem.hpp"

namespace py = pybind11;

int runManta( std::string const& );

PYBIND11_MODULE( MaNTA, m ) 
{
	m.doc() = "Python bindings for MaNTA";
	m.def( "run", runManta, "Runs the MaNTA suite using given configuration file" );
	m.def( "registerPhysicsCase", PhysicsCases::RegisterPhysicsCase, "Register a Physics Case" );

	py::class_<TransportSystem,PyTransportSystem>( m, "TransportSystem" )
		.def( py::init<>() )
		.def( "LowerBoundary", &TransportSystem::LowerBoundary )
		.def( "UpperBoundary", &TransportSystem::UpperBoundary )
		.def( "isLowerBoundaryDirichlet", &TransportSystem::isLowerBoundaryDirichlet )
		.def( "isUpperBoundaryDirichlet", &TransportSystem::isUpperBoundaryDirichlet )
		.def( "SigmaFn", &TransportSystem::SigmaFn )
		.def( "Sources", &TransportSystem::Sources )
		.def( "dSigmaFn_du", &TransportSystem::dSigmaFn_du )
		.def( "dSigmaFn_dq", &TransportSystem::dSigmaFn_dq )
		.def( "dSources_du", &TransportSystem::dSources_du )
		.def( "dSources_dq",  &TransportSystem::dSources_dq )
		.def( "dSources_dsigma",  &TransportSystem::dSources_dsigma )
		.def( "InitialValue", &TransportSystem::InitialValue )
		.def( "InitialDerivative", &TransportSystem::InitialDerivative );
}
