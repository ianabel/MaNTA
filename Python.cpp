#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <string>
#include <toml.hpp>

#include "PhysicsCases.hpp"
#include "PyTransportSystem.hpp"
#include "PyAdjointProblem.hpp"
#include "PyRunner.hpp"
#include "PyGrid.hpp"
// #include "xla/ffi/api/c_api.h"
// #include "xla/ffi/api/api.h"
// #include <type_traits>

namespace py = pybind11;

int runManta(std::string const &);

// // Needed to be able to handle FFI in JAX
// template <typename T, typename... Args>
// static py::capsule EncapsulateFfiCall(T (PyRunner::*fn)(Args...))
// {
// 	// This check is optional, but it can be helpful for avoiding invalid handlers.
// 	static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
// 				  "Encapsulated function must be an XLA FFI handler");
// 	return py::capsule(reinterpret_cast<void *&>(fn));
// }

// This allows one to use a python dict as a state variable,
// if the python dict has the right keys in it
namespace pybind11
{
	namespace detail
	{
		template <>
		struct type_caster<State>
		{
		public:
			PYBIND11_TYPE_CASTER(State, const_name("dict[Sequence[float]]"));

			bool load(handle src, bool)
			{
				py::dict d = py::cast<py::dict>(src);
				value.Variable = py::cast<Vector>(d["Variable"]);
				value.Derivative = py::cast<Vector>(d["Derivative"]);
				value.Flux = py::cast<Vector>(d["Flux"]);
				value.Aux = py::cast<Vector>(d["Aux"]);
				value.Scalars = py::cast<Vector>(d["Scalars"]);
				return true;
			}

			static handle cast(const State &src, return_value_policy /* policy */, handle /* parent */)
			{
				py::dict d;
				d["Variable"] = src.Variable;
				d["Derivative"] = src.Derivative;
				d["Flux"] = src.Flux;
				d["Aux"] = src.Aux;
				d["Scalars"] = src.Scalars;
				return d.release();
			}
		};
		template <>
		struct type_caster<GlobalState>
		{
		public:
			PYBIND11_TYPE_CASTER(GlobalState, const_name("dict[Sequence[float]]"));

			bool load(handle src, bool)
			{
				py::dict d = py::cast<py::dict>(src);

				value.Variable() = py::cast<Matrix>(d["Variable"]);
				value.Derivative() = py::cast<Matrix>(d["Derivative"]);
				value.Flux() = py::cast<Matrix>(d["Flux"]);
				value.Aux() = py::cast<Matrix>(d["Aux"]);
				value.Scalars() = py::cast<Vector>(d["Scalars"]);
				return true;
			}

			static handle cast(const GlobalState &src, return_value_policy /* policy */, handle /* parent */)
			{
				py::dict d;
				d["Variable"] = src.Variable().transpose();
				d["Derivative"] = src.Derivative().transpose();
				d["Flux"] = src.Flux().transpose();
				d["Aux"] = src.Aux().transpose();
				d["Scalars"] = src.Scalars();
				return d.release();
			}
		};
	}
};

// TODO: Check if we can just use pytoml instead and
// remove this extra cast
py::object cast_toml(toml::value v)
{
	if (v.is_boolean())
		return py::bool_(v.as_boolean());
	else if (v.is_integer())
		return py::int_(v.as_integer());
	else if (v.is_floating())
		return py::float_(v.as_floating());
	else if (v.is_string())
		return py::str(v.as_string());
	else if (v.is_array())
	{
		py::list lst;
		for (const auto &elem : v.as_array())
		{
			lst.append(cast_toml(elem));
		}
		return lst;
	}
	else if (v.is_table())
	{
		py::dict d;
		for (const auto &[key, val] : v.as_table())
		{
			d[py::str(key)] = cast_toml(val);
		}
		return d;
	}
	else
	{
		return py::none();
	}
}

// Defines the MaNTA module and what can be called
PYBIND11_MODULE(MaNTA, m, py::mod_gil_not_used())
{
	m.doc() = "Python bindings for MaNTA";

	m.def("run", runManta, py::return_value_policy::reference, "Runs the MaNTA suite using given configuration file");
	m.def("registerPhysicsCase", &PhysicsCases::RegisterPhysicsCase, py::return_value_policy::reference, "Register a Physics Case");

	m.def("getNodes", py::overload_cast<const std::vector<double> &, unsigned int>(&getNodes), py::return_value_policy::reference, "Get the points of a grid");
	m.def("getNodes", py::overload_cast<Position, Position, Index, unsigned int>(&getNodes), py::return_value_policy::reference, "Get the points of a grid");

	// List all interfaces of the main TransportSystem class which is what has to be derived from in python
	py::class_<TransportSystem, PyTransportSystem, py::smart_holder>(m, "TransportSystem")
		.def(py::init<>())
		.def("LowerBoundary", &TransportSystem::LowerBoundary)
		.def("UpperBoundary", &TransportSystem::UpperBoundary)
		.def("isLowerBoundaryDirichlet", &TransportSystem::isLowerBoundaryDirichlet)
		.def("isUpperBoundaryDirichlet", &TransportSystem::isUpperBoundaryDirichlet)
		.def("SigmaFn", py::overload_cast<Index, const State &, Position, Time>(&TransportSystem::SigmaFn))
		.def("SigmaFn_v", py::overload_cast<Index, GlobalState const &, std::vector<Position> const &, Time>(&TransportSystem::SigmaFn))
		.def("Sources", py::overload_cast<Index, const State &, Position, Time>(&TransportSystem::Sources))
		.def("Sources_v", py::overload_cast<Index, GlobalState const &, std::vector<Position> const &, Time>(&TransportSystem::Sources))
		.def("dSigmaFn_du", &TransportSystem::dSigmaFn_du)
		.def("dSigmaFn_dq", &TransportSystem::dSigmaFn_dq)
		.def("dSources_du", &TransportSystem::dSources_du)
		.def("dSources_dq", &TransportSystem::dSources_dq)
		.def("dSources_dsigma", &TransportSystem::dSources_dsigma)
		.def("dSigma", &TransportSystem::dSigma)
		.def("dSources", &TransportSystem::dSources)
		.def("InitialValue", &TransportSystem::InitialValue)
		.def("InitialDerivative", &TransportSystem::InitialDerivative)
		.def("InitialAuxValue", &TransportSystem::InitialAuxValue)
		.def("AuxG", &TransportSystem::AuxG)
		.def("AuxGPrime", &TransportSystem::AuxGPrime)
		.def("dSources_dPhi", &TransportSystem::dSources_dPhi)
		.def("dSigma_dPhi", &TransportSystem::dSigma_dPhi)
		.def("createAdjointProblem", &TransportSystem::createAdjointProblem)
		.def_readwrite("isUpperDirichlet", &PyTransportSystem::isUpperDirichlet)
		.def_readwrite("isLowerDirichlet", &PyTransportSystem::isLowerDirichlet)
		.def_readwrite("nVars", &PyTransportSystem::nVars)
		.def_readwrite("nAux", &PyTransportSystem::nAux);

	py::class_<AdjointProblem, PyAdjointProblem, py::smart_holder>(m, "AdjointProblem")
		.def(py::init<>())
		.def("gFn", py::overload_cast<Index, const GlobalState &, std::vector<Position> const &>(&AdjointProblem::gFn, py::const_))
		.def("dgFndp", &AdjointProblem::dgFndp)
		.def("dgFn_dphi", &AdjointProblem::dgFn_dphi)
		.def("dg", &AdjointProblem::dg)
		.def("dSigma", &AdjointProblem::dSigma)
		.def("dSources", &AdjointProblem::dSources)
		.def("dAux_dp", &AdjointProblem::dAux_dp)
		.def("computeUpperBoundarySensitivity", &AdjointProblem::computeUpperBoundarySensitivity)
		.def("computeLowerBoundarySensitivity", &AdjointProblem::computeLowerBoundarySensitivity)
		.def("getName", &AdjointProblem::getName)
		.def_readwrite("np", &PyAdjointProblem::np)
		.def_readwrite("np_boundary", &PyAdjointProblem::np_boundary)
		.def_readwrite("ng", &PyAdjointProblem::ng)
		.def_readwrite("spatialParameters", &PyAdjointProblem::spatialParameters);

	py::class_<Grid>(m, "Grid")
		.def(py::init<>(), py::return_value_policy::reference)
		.def(py::init<Grid::Position, Grid::Position, Grid::Index, bool, double, double>(), py::return_value_policy::reference)
		.def("getNCells", &Grid::getNCells);

	py::class_<toml::value>(m, "TomlValue")
		.def(py::init<>())
		.def("__getitem__",
			 [](const toml::value &v, const std::string &key)
			 {
				 auto temp = v;
				 py::object result = py::none();
				 if (!v.contains(key))
				 {
					 for (auto &[k, val] : temp.as_table())
					 {
						 result = cast_toml(val[key]);
						 if (!result.is_none())
							 break;
					 }
				 }
				 else
				 {
					 result = cast_toml(temp[key]);
				 }

				 if (result.is_none())
				 {
					 throw std::out_of_range("Key " + key + " not found in TOML value.");
				 }
				 else
				 {
					 return result;
				 }
			 });

	py::class_<PyRunner, py::smart_holder>(m, "Runner")
		.def(py::init<std::shared_ptr<TransportSystem>>())
		.def("configure", &PyRunner::configure)
		.def("run", &PyRunner::run)
		.def("run_ss", &PyRunner::run_ss)
		.def("setAdjointProblem", &PyRunner::setAdjointProblem)
		.def("runAdjointSolve", &PyRunner::runAdjointSolve);
}
