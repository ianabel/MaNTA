#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <string>
#include <toml.hpp>

#include "PhysicsCases.hpp"
#include "PyTransportSystem.hpp"
#include "PyAdjointProblem.hpp"

namespace py = pybind11;

int runManta(std::string const &);

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
				auto value = State();
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

	// List all interfaces of the main TransportSystem class which is what has to be derived from in python
	py::class_<TransportSystem, PyTransportSystem, py::smart_holder>(m, "TransportSystem")
		.def(py::init<>())
		.def("LowerBoundary", &TransportSystem::LowerBoundary)
		.def("UpperBoundary", &TransportSystem::UpperBoundary)
		.def("isLowerBoundaryDirichlet", &TransportSystem::isLowerBoundaryDirichlet)
		.def("isUpperBoundaryDirichlet", &TransportSystem::isUpperBoundaryDirichlet)
		.def("SigmaFn", py::overload_cast<Index, const State &, Position, Time>(&TransportSystem::SigmaFn))
		.def("Sources", py::overload_cast<Index, const State &, Position, Time>(&TransportSystem::Sources))
		.def("dSigmaFn_du", &TransportSystem::dSigmaFn_du)
		.def("dSigmaFn_dq", &TransportSystem::dSigmaFn_dq)
		.def("dSources_du", &TransportSystem::dSources_du)
		.def("dSources_dq", &TransportSystem::dSources_dq)
		.def("dSources_dsigma", &TransportSystem::dSources_dsigma)
		.def("InitialValue", &TransportSystem::InitialValue)
		.def("InitialDerivative", &TransportSystem::InitialDerivative)
		.def("createAdjointProblem", &TransportSystem::createAdjointProblem)
		.def_readwrite("isUpperDirichlet", &PyTransportSystem::isUpperDirichlet)
		.def_readwrite("isLowerDirichlet", &PyTransportSystem::isLowerDirichlet)
		.def_readwrite("nVars", &PyTransportSystem::nVars);

	py::class_<AdjointProblem, PyAdjointProblem, py::smart_holder>(m, "AdjointProblem")
		.def(py::init<>())
		.def("GFn", &AdjointProblem::GFn)
		.def("dGFndp", &AdjointProblem::dGFndp)
		.def("gFn", &AdjointProblem::gFn)
		.def("dgFn_du", &AdjointProblem::dgFn_du)
		.def("dgFn_dq", &AdjointProblem::dgFn_dq)
		.def("dgFn_dsigma", &AdjointProblem::dgFn_dsigma)
		.def("dgFn_dphi", &AdjointProblem::dgFn_dphi)
		.def("dSigmaFn_dp", py::overload_cast<Index, Index, Value &, const State &, Position>(&AdjointProblem::dSigmaFn_dp))
		.def("dSources_dp", py::overload_cast<Index, Index, Value &, const State &, Position>(&AdjointProblem::dSources_dp))
		.def("computeUpperBoundarySensitivity", &AdjointProblem::computeUpperBoundarySensitivity)
		.def("computeLowerBoundarySensitivity", &AdjointProblem::computeLowerBoundarySensitivity)
		.def("getName", &AdjointProblem::getName)
		.def_readwrite("np", &PyAdjointProblem::np)
		.def_readwrite("np_boundary", &PyAdjointProblem::np_boundary);

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
}
