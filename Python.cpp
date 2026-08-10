#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <toml.hpp>

#include "AdjointProblem.hpp"
#include "PhysicsCases.hpp"
#include "PyAdjointProblem.hpp"
#include "PyGrid.hpp"
#include "PyIntegrator.hpp"
#include "PyRunner.hpp"
#include "PyTransportSystem.hpp"
#include "State.hpp"
#include "TransportSystem.hpp"

namespace py = pybind11;

#ifdef XLA_FFI
#include "ffi.hpp"

namespace ffi = xla::ffi;

template <typename T> py::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
};
#endif

int runManta(std::string const &);
// This allows one to use a python dict as a state variable,
// if the python dict has the right keys in it
namespace pybind11 {
namespace detail {
template <> struct type_caster<State> {
public:
  PYBIND11_TYPE_CASTER(State, const_name("dict[Sequence[float]]"));

  bool load(handle src, bool) {
    py::dict d = py::cast<py::dict>(src);
    value.Variable = py::cast<Vector>(d["Variable"]);
    value.Derivative = py::cast<Vector>(d["Derivative"]);
    value.Flux = py::cast<Vector>(d["Flux"]);
    value.Aux = py::cast<Vector>(d["Aux"]);
    value.Scalars = py::cast<Vector>(d["Scalars"]);
    return true;
  }

  static handle cast(const State &src, return_value_policy /* policy */,
                     handle /* parent */) {
    py::dict d;
    d["Variable"] = src.Variable;
    d["Derivative"] = src.Derivative;
    d["Flux"] = src.Flux;
    d["Aux"] = src.Aux;
    d["Scalars"] = src.Scalars;
    return d.release();
  }
};
template <> struct type_caster<GlobalState> {
public:
  PYBIND11_TYPE_CASTER(GlobalState, const_name("dict[Sequence[float]]"));

  bool load(handle src, bool) {
    py::dict d = py::cast<py::dict>(src);
    value.Variable() = py::cast<Matrix>(d["Variable"]).transpose();
    value.Derivative() = py::cast<Matrix>(d["Derivative"]).transpose();
    value.Flux() = py::cast<Matrix>(d["Flux"]).transpose();
    value.Aux() = py::cast<Matrix>(d["Aux"]).transpose();

    auto scalars = py::cast<py::array_t<double>>(d["Scalars"]);
    py::buffer_info info = scalars.request();
    double *data = static_cast<double *>(info.ptr);
    value.Scalars() = Eigen::Map<Vector>(data, info.size);

    // PYBIND11_TYPE_CASTER default-constructs `value`, so its size members are
    // whatever the default constructor left them. Derive them from the arrays
    // just assigned; without this, size() and operator[] on a GlobalState that
    // came from Python read uninitialised state.
    value.setShapeFromData();

    return true;
  }

  static handle cast(const GlobalState &src, return_value_policy /* policy */,
                     handle /* parent */) {
    py::dict d;
    d["Variable"] = src.Variable().transpose();
    d["Derivative"] = src.Derivative().transpose();
    d["Flux"] = src.Flux().transpose();
    d["Aux"] = src.Aux().transpose();
    d["Scalars"] = src.Scalars();
    return d.release();
  }
};
} // namespace detail
}; // namespace pybind11

// TODO: Check if we can just use pytoml instead and
// remove this extra cast
py::object cast_toml(toml::value v) {
  if (v.is_boolean())
    return py::bool_(v.as_boolean());
  else if (v.is_integer())
    return py::int_(v.as_integer());
  else if (v.is_floating())
    return py::float_(v.as_floating());
  else if (v.is_string())
    return py::str(v.as_string());
  else if (v.is_array()) {
    py::list lst;
    for (const auto &elem : v.as_array()) {
      lst.append(cast_toml(elem));
    }
    return lst;
  } else if (v.is_table()) {
    py::dict d;
    for (const auto &[key, val] : v.as_table()) {
      d[py::str(key)] = cast_toml(val);
    }
    return d;
  } else {
    return py::none();
  }
}

// Defines the MaNTA module and what can be called
// The extension is private to the `manta` package: python/manta/__init__.py
// re-exports it and adds the parts that are more naturally written in Python
// (the declarative class-attribute spec, chiefly). Users import `manta`.
PYBIND11_MODULE(_manta, m, py::mod_gil_not_used()) {
  m.doc() = "Compiled core of the MaNTA Python package; import `manta` instead.";

  m.def("run", runManta, py::return_value_policy::reference,
        "Runs the MaNTA suite using given configuration file");
  m.def("registerPhysicsCase", &PhysicsCases::RegisterPhysicsCase,
        py::return_value_policy::reference, "Register a Physics Case");

  m.def("getNodes",
        py::overload_cast<const std::vector<double> &, unsigned int>(&getNodes),
        py::return_value_policy::reference, "Get the points of a grid");
  m.def("getNodes",
        py::overload_cast<Position, Position, Index, unsigned int>(&getNodes),
        py::return_value_policy::reference, "Get the points of a grid");

  // A physics case describes itself as data, in Python exactly as in C++. This
  // replaces setting self.nVars / self.isUpperDirichlet after construction,
  // which could not be validated and left the boundary flags indeterminate if
  // a case forgot them.
  py::enum_<BoundaryKind>(m, "BoundaryKind")
      .value("Dirichlet", BoundaryKind::Dirichlet)
      .value("Neumann", BoundaryKind::Neumann);
  // Also as bare module attributes, so a case reads `lower=manta.Neumann`.
  m.attr("Dirichlet") = BoundaryKind::Dirichlet;
  m.attr("Neumann") = BoundaryKind::Neumann;

  py::class_<FieldSpec>(m, "Field")
      .def(py::init([](std::string name, std::string description, std::string units,
                       BoundaryKind lower, BoundaryKind upper)
                    { return FieldSpec{std::move(name), std::move(description),
                                       std::move(units), lower, upper}; }),
           py::arg("name"), py::arg("description") = "", py::arg("units") = "",
           py::arg("lower") = BoundaryKind::Dirichlet,
           py::arg("upper") = BoundaryKind::Dirichlet)
      .def_readwrite("name", &FieldSpec::name)
      .def_readwrite("description", &FieldSpec::description)
      .def_readwrite("units", &FieldSpec::units)
      .def_readwrite("lower", &FieldSpec::lower)
      .def_readwrite("upper", &FieldSpec::upper);

  py::class_<ScalarSpec>(m, "Scalar")
      .def(py::init([](std::string name, std::string description, std::string units,
                       bool differential)
                    { return ScalarSpec{std::move(name), std::move(description),
                                        std::move(units), differential}; }),
           py::arg("name"), py::arg("description") = "", py::arg("units") = "",
           py::arg("differential") = false)
      .def_readwrite("name", &ScalarSpec::name)
      .def_readwrite("description", &ScalarSpec::description)
      .def_readwrite("units", &ScalarSpec::units)
      .def_readwrite("differential", &ScalarSpec::differential);

  py::class_<AuxSpec>(m, "Aux")
      .def(py::init([](std::string name, std::string description, std::string units)
                    { return AuxSpec{std::move(name), std::move(description),
                                     std::move(units)}; }),
           py::arg("name"), py::arg("description") = "", py::arg("units") = "")
      .def_readwrite("name", &AuxSpec::name)
      .def_readwrite("description", &AuxSpec::description)
      .def_readwrite("units", &AuxSpec::units);

  // The Python half of SystemSpec.hpp's numberedFields/Scalars/Aux: a spec
  // whose entries are called Var0, Scalar0, AuxVariable0 and so on. Those are
  // the names baked into the checked-in .ref.nc files, so a case ported from
  // the old `self.nVars = 1` form keeps them. A case written from scratch
  // should name its variables instead.
  m.def(
      "numbered_spec",
      [](Index nVars, Index nScalars, Index nAux, BoundaryKind lower, BoundaryKind upper,
         bool differential)
      {
        return SystemSpec{numberedFields(nVars, lower, upper), numberedScalars(nScalars, differential),
                          numberedAux(nAux)};
      },
      py::arg("nVars"), py::arg("nScalars") = 0, py::arg("nAux") = 0,
      py::arg("lower") = BoundaryKind::Dirichlet, py::arg("upper") = BoundaryKind::Dirichlet,
      py::arg("differential") = false,
      "A SystemSpec using the historical placeholder names (Var0, Scalar0, AuxVariable0).");

  py::class_<SystemSpec>(m, "SystemSpec")
      .def(py::init([](std::vector<FieldSpec> variables, std::vector<ScalarSpec> scalars,
                       std::vector<AuxSpec> aux)
                    { return SystemSpec{std::move(variables), std::move(scalars),
                                        std::move(aux)}; }),
           py::arg("variables"), py::arg("scalars") = std::vector<ScalarSpec>{},
           py::arg("aux") = std::vector<AuxSpec>{})
      .def_readwrite("variables", &SystemSpec::variables)
      .def_readwrite("scalars", &SystemSpec::scalars)
      .def_readwrite("aux", &SystemSpec::aux)
      .def("validate", &SystemSpec::validate);

  // List all interfaces of the main TransportSystem class which is what has to
  // be derived from in python
  py::class_<TransportSystem, PyTransportSystem, py::smart_holder>(
      m, "TransportSystem")
      .def(py::init<SystemSpec>(), py::arg("spec"))
      // The same thing spelled out, so a case can say what it is at the point
      // it calls up without building a SystemSpec first:
      //     super().__init__(variables=[manta.Field("n", lower=manta.Neumann)])
      .def(py::init([](std::vector<FieldSpec> variables, std::vector<ScalarSpec> scalars,
                       std::vector<AuxSpec> aux)
                    { return new PyTransportSystem(SystemSpec{
                          std::move(variables), std::move(scalars), std::move(aux)}); }),
           py::arg("variables"), py::arg("scalars") = std::vector<ScalarSpec>{},
           py::arg("aux") = std::vector<AuxSpec>{})
      .def("LowerBoundary", &TransportSystem::LowerBoundary)
      .def("UpperBoundary", &TransportSystem::UpperBoundary)
      .def("isLowerBoundaryDirichlet",
           &TransportSystem::isLowerBoundaryDirichlet)
      .def("isUpperBoundaryDirichlet",
           &TransportSystem::isUpperBoundaryDirichlet)
      .def("ComputePhysics", &TransportSystem::ComputePhysics)
      .def("ComputePhysicsDerivatives",
           &TransportSystem::ComputePhysicsDerivatives)
      .def("SigmaFn", py::overload_cast<Index, const State &, Position, Time>(
                          &TransportSystem::SigmaFn))
      .def("SigmaFn_v", py::overload_cast<Index, GlobalState const &,
                                          std::vector<Position> const &, Time>(
                            &TransportSystem::SigmaFn))
      .def("Sources", py::overload_cast<Index, const State &, Position, Time>(
                          &TransportSystem::Sources))
      .def("Sources_v", py::overload_cast<Index, GlobalState const &,
                                          std::vector<Position> const &, Time>(
                            &TransportSystem::Sources))
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
      .def("AuxG", py::overload_cast<Index, const State &, Position, Time>(
                       &TransportSystem::AuxG))
      .def("AuxG_v", py::overload_cast<Index, const State &, Position, Time>(
                         &TransportSystem::AuxG))
      .def("AuxGPrime",
           py::overload_cast<Index, State &, const State &, Position, Time>(
               &TransportSystem::AuxGPrime))
      .def("AuxGPrime_v",
           py::overload_cast<Index, GlobalState &, GlobalState const &,
                             std::vector<Position> const &, Time>(
               &TransportSystem::AuxGPrime))
      .def("dSources_dPhi", &TransportSystem::dSources_dPhi)
      .def("dSigma_dPhi", &TransportSystem::dSigma_dPhi)
      .def("ScalarG", &TransportSystem::ScalarG)
      .def("ScalarGPrime", &TransportSystem::ScalarGPrime)
      .def("InitialScalarValue", &TransportSystem::InitialScalarValue)
      .def("dSources_dScalars", &TransportSystem::dSources_dScalars)
      .def("createAdjointProblem", &TransportSystem::createAdjointProblem)
      .def("isScalarDifferential", &TransportSystem::isScalarDifferential)
      .def_property_readonly("spec", &TransportSystem::spec)
      // Read-only now: these are derived from the spec. Assigning self.nVars
      // in a subclass __init__ is the pattern this replaces, and leaving it
      // writable would mean the count and the spec could disagree.
      .def_property_readonly("nVars", &TransportSystem::getNumVars)
      .def_property_readonly("nAux", &TransportSystem::getNumAux)
      .def_property_readonly("nScalars", &TransportSystem::getNumScalars);

  py::class_<AdjointProblem, PyAdjointProblem, py::smart_holder>(
      m, "AdjointProblem")
      .def(py::init<>())
      .def("gFn", py::overload_cast<Index, const GlobalState &,
                                    std::vector<Position> const &>(
                      &AdjointProblem::gFn, py::const_))
      .def("dgFndp", &AdjointProblem::dgFndp)
      .def("dgFn_dphi", &AdjointProblem::dgFn_dphi)
      .def("dg", &AdjointProblem::dg)
      .def("ComputePhysicsDerivatives",
           &AdjointProblem::ComputePhysicsDerivatives)
      .def("dSigma", &AdjointProblem::dSigma)
      .def("dSources", &AdjointProblem::dSources)
      .def("dAux", &AdjointProblem::dAux)
      .def("dAux_dp", &AdjointProblem::dAux_dp)
      .def("computeUpperBoundarySensitivity",
           &AdjointProblem::computeUpperBoundarySensitivity)
      .def("computeLowerBoundarySensitivity",
           &AdjointProblem::computeLowerBoundarySensitivity)
      .def("getName", &AdjointProblem::getName)
      .def_readwrite("np", &PyAdjointProblem::np)
      .def_readwrite("np_boundary", &PyAdjointProblem::np_boundary)
      .def_readwrite("ng", &PyAdjointProblem::ng)
      .def_readwrite("spatialParameters", &PyAdjointProblem::spatialParameters);

  py::class_<Grid>(m, "Grid")
      .def(py::init<>(), py::return_value_policy::reference)
      .def(py::init<Grid::Position, Grid::Position, Grid::Index, bool, double,
                    double>(),
           py::return_value_policy::reference)
      .def("getNCells", &Grid::getNCells);

  py::class_<toml::value>(m, "TomlValue")
      .def(py::init<>())
      .def("__getitem__", [](const toml::value &v, const std::string &key) {
        auto temp = v;
        py::object result = py::none();
        if (!v.contains(key)) {
          for (auto &[k, val] : temp.as_table()) {
            result = cast_toml(val[key]);
            if (!result.is_none())
              break;
          }
        } else {
          result = cast_toml(temp[key]);
        }

        if (result.is_none()) {
          throw std::out_of_range("Key " + key + " not found in TOML value.");
        } else {
          return result;
        }
      });
  py::class_<PyRunner, py::smart_holder>(m, "Runner")
      .def(py::init<std::shared_ptr<TransportSystem>>())
      .def("configure", &PyRunner::configure)
      .def("run", &PyRunner::run)
      .def("run_ss", &PyRunner::run_ss)
      .def("G", &PyRunner::G)
      .def("getAdjointGradients", &PyRunner::getAdjointGradients)
      .def("getSolution", &PyRunner::getSolution)
      .def("getPostprocessedSolution", &PyRunner::getPostprocessedSolution)
      .def("get_address", [](const PyRunner &runner) // needed for xla interface
           { return reinterpret_cast<std::uint64_t>(&runner); });
#ifdef XLA_FFI
  m.def("runner_ffi_ops", []() {
    py::dict ffi_ops;
    ffi_ops["get_solution_ffi"] = EncapsulateFfiCall(get_solution_ffi_ops);
    ffi_ops["get_adjoint_gradients_ffi"] =
        EncapsulateFfiCall(get_adjoint_gradients_ffi_ops);
    ffi_ops["run_ffi"] = EncapsulateFfiCall(run_ffi_ops);
    ffi_ops["run_ss_ffi"] = EncapsulateFfiCall(run_ss_ffi_ops);
    return ffi_ops;
  });
#ifdef CUDA
  m.def("runner_ffi_ops_cuda", []() {
    py::dict ffi_ops;
    ffi_ops["get_solution_ffi_cuda"] =
        EncapsulateFfiCall(get_solution_ffi_ops_cuda);
    ffi_ops["get_adjoint_gradients_ffi_cuda"] =
        EncapsulateFfiCall(get_adjoint_gradients_ffi_ops_cuda);
    return ffi_ops;
  });
#endif
#endif
};
