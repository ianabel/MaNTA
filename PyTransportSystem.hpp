#ifndef PYTRANSPORTSYSTEM_HPP
#define PYTRANSPORTSYSTEM_HPP

#include "PyIntegrator.hpp"
#include "TransportSystem.hpp"
#include "extern/pybind11/include/pybind11/pybind11.h"
#include "pybind11/gil.h"
#include <functional>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string_view>

constexpr std::array<std::string_view, 7> required_method_names = {
    "SigmaFn",     "Sources",     "dSigmaFn_du",    "dSigmaFn_dq",
    "dSources_du", "dSources_dq", "dSources_dsigma"};

constexpr std::array<std::string_view, 2> required_method_names_vectorized = {
    "ComputePhysics", "ComputePhysicsDerivatives"};

constexpr std::array<std::string_view, 4> required_scalar_methods = {
    "ScalarG", "ScalarGPrime", "InitialScalarDerivative", "dSources_dScalars"};

namespace py = pybind11;

class PyTransportSystem : public TransportSystem,
                          public py::trampoline_self_life_support {
public:
  using TransportSystem::TransportSystem;

  void initializeOverrides() {
    auto make_override = [this](const char *method_name) {
      py::gil_scoped_acquire gil;
      py::function _override = py::get_override(this, method_name);
      return _override;
    };

    std::vector<std::string_view> missing_methods;
    std::vector<std::string_view> missing_vectorized_methods;
    std::vector<std::string_view> missing_scalar_method;

    for (const auto &method_name : required_method_names) {
      auto _override = make_override(method_name.data());
      if (!_override) {
        missing_methods.push_back(method_name);
      } else {
        method_overrides.insert(
            std::make_pair(method_name, std::move(_override)));
      }
    }

    for (const auto &method_name : required_method_names_vectorized) {
      auto _override = make_override(method_name.data());
      if (!_override) {
        missing_vectorized_methods.push_back(method_name);
      } else {
        method_overrides.insert(
            std::make_pair(method_name, std::move(_override)));
      }
    }

    if (nScalars > 0) {
      for (const auto &method_name : required_scalar_methods) {
        auto _override = make_override(method_name.data());
        if (!_override) {
          missing_scalar_method.push_back(method_name);
        } else {
          method_overrides.insert(
              std::make_pair(method_name, std::move(_override)));
        }
      }
    }

    bool non_vectorized = missing_methods.empty();
    bool has_scalar = missing_scalar_method.empty();
    vectorized = missing_vectorized_methods.empty();

    if (!vectorized || !non_vectorized) {
      if (vectorized || non_vectorized) {
        // do nothing
      } else {
        std::string error_message = "The following required methods are "
                                    "missing in the Python subclass:\n";
        error_message += "Non-vectorized methods:\n";
        for (const auto &method_name : missing_methods) {
          error_message += std::string(method_name) + "\n";
        }
        error_message += "Vectorized methods:\n";
        for (const auto &method_name : missing_vectorized_methods) {
          error_message += std::string(method_name) + "\n";
        }
        error_message += "MaNTA requires either all vectorized or all "
                         "non-vectorized methods to be implemented. Please "
                         "implement the missing methods and try again.\n";
        throw std::runtime_error(error_message);
      }
    }

    if (!has_scalar) {
      std::string error_message = "The following required scalar methods are "
                                  "missing in the Python subclass:\n";
      for (const auto &method_name : missing_scalar_method) {
        error_message += std::string(method_name) + "\n";
      }
      throw std::runtime_error(error_message);
    }

    if (nAux > 0) {
      method_overrides.insert(
          std::make_pair("AuxGPrime", make_override("AuxGPrime")));
      method_overrides.insert(
          std::make_pair("dSources_dPhi", make_override("dSources_dPhi")));
      method_overrides.insert(
          std::make_pair("dSigma_dPhi", make_override("dSigma_dPhi")));
    }

    initialized = true;
  }

  Value LowerBoundary(Index i, Time t) const override {
    PYBIND11_OVERRIDE(Value, TransportSystem, LowerBoundary, i, t);
  };
  Value UpperBoundary(Index i, Time t) const override {
    PYBIND11_OVERRIDE(Value, TransportSystem, UpperBoundary, i, t);
  };
  bool isLowerBoundaryDirichlet(Index i) const override {
    PYBIND11_OVERRIDE(Value, TransportSystem, isLowerBoundaryDirichlet, i);
  };
  bool isUpperBoundaryDirichlet(Index i) const override {
    PYBIND11_OVERRIDE(Value, TransportSystem, isUpperBoundaryDirichlet, i);
  };

  PhysicsOutput ComputePhysics(GlobalState const &states,
                               std::vector<Position> const &abscissae,
                               Time time) override {
    py::gil_scoped_acquire gil;
    py::function _override = py::get_override(this, "ComputePhysics");

    if (!_override)
      return TransportSystem::ComputePhysics(states, abscissae, time);
    PhysicsOutput out =
        _override(states, abscissae, time).cast<PhysicsOutput>();
    m_sourceCache.resize(nVars);
    for (Index var = 0; var < nVars; ++var)
      m_sourceCache[var] = out[1][var];
    return out;
  }
  void ComputePhysicsDerivatives(
      std::array<std::reference_wrapper<GlobalStateMatrix>, NPHYSICS_FUNCTIONS>
          &&out,
      GlobalState const &states, std::vector<Position> const &abscissae,
      Time time) override {
    py::gil_scoped_acquire gil;
    py::function _override =
        py::get_override(this, "ComputePhysicsDerivatives");

    if (!_override)
      TransportSystem::ComputePhysicsDerivatives(std::move(out), states,
                                                 abscissae, time);
    std::array<std::vector<GlobalState>, NPHYSICS_FUNCTIONS> temp =
        _override(states, abscissae, time)
            .cast<std::array<std::vector<GlobalState>, NPHYSICS_FUNCTIONS>>();

    GlobalStateMatrix &dflux = out[0];
    GlobalStateMatrix &dsource = out[1];
    GlobalStateMatrix &daux = out[2];

    for (Index var = 0; var < nVars; ++var) {
      dflux[var] = temp[0][var];
      dsource[var] = temp[1][var];
    }
    for (Index aux = 0; aux < nAux; ++aux) {
      daux[aux] = temp[2][aux];
    }
  }
  Value SigmaFn(Index i, const State &s, Position x, Time t) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      return method_overrides["SigmaFn"](i, s, x, t).cast<Value>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate SigmaFn: ") +
          e.what());
    }
  };
  Values SigmaFn(Index i, GlobalState const &states,
                 std::vector<Position> const &abscissae, Time time) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      if (!vectorized)
        return TransportSystem::SigmaFn(
            i, states, abscissae, time); // Call base class version which will
                                         // loop over non-vectorized method

      return method_overrides["SigmaFn_v"](i, states, abscissae, time)
          .cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate SigmaFn: ") +
          e.what());
    }
  };

  Value Sources(Index i, const State &s, Position x, Time t) override {
    if (!initialized)
      initializeOverrides();

    try {
      py::gil_scoped_acquire gil;
      return method_overrides["Sources"](i, s, x, t).cast<Value>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate Sources: ") +
          e.what());
    }
  };

  Values Sources(Index i, GlobalState const &states,
                 std::vector<Position> const &abscissae, Time time) override {
    if (!initialized)
      initializeOverrides();

    try {
      py::gil_scoped_acquire gil;
      if (!vectorized)
        return TransportSystem::Sources(
            i, states, abscissae, time); // Call base class version which will
                                         // loop over non-vectorized method

      return method_overrides["Sources_v"](i, states, abscissae, time)
          .cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate Sources: ") +
          e.what());
    }
  };

  void dSigmaFn_du(Index i, VectorRef out, const State &s, Position x,
                   Time t) override {
    if (!initialized)
      initializeOverrides();

    try {
      py::gil_scoped_acquire gil;
      out = method_overrides["dSigmaFn_du"](i, s, x, t).cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate dSigmaFn_du: ") +
          e.what());
    }
  };
  void dSigmaFn_dq(Index i, VectorRef out, const State &s, Position x,
                   Time t) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      out = method_overrides["dSigmaFn_dq"](i, s, x, t).cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate dSources_dq: ") +
          e.what());
    }
  };

  void dSources_du(Index i, VectorRef v, const State &s, Position x,
                   Time t) override {
    if (!initialized)
      initializeOverrides();

    try {
      py::gil_scoped_acquire gil;
      v = method_overrides["dSources_du"](i, s, x, t).cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate dSources_du: ") +
          e.what());
    }
  };

  void dSources_dq(Index i, VectorRef v, const State &s, Position x,
                   Time t) override {
    if (!initialized)
      initializeOverrides();

    try {
      py::gil_scoped_acquire gil;
      v = method_overrides["dSources_dq"](i, s, x, t).cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate dSources_dq: ") +
          e.what());
    }
  };

  void dSources_dsigma(Index i, VectorRef v, const State &s, Position x,
                       Time t) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      v = method_overrides["dSources_dsigma"](i, s, x, t).cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string(
              "Error occurred when trying to calculate dSources_dsigma: ") +
          e.what());
    }
  };

  void dSigma(Index i, GlobalState &out, GlobalState const &states,
              std::vector<Position> const &abscissae, Time time) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      if (!vectorized) {
        TransportSystem::dSigma(i, out, states, abscissae,
                                time); // Call base class version which will
                                       // loop over non-vectorized method
        return;
      }

      out = method_overrides["dSigma"](i, states, abscissae, time)
                .cast<GlobalState>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate dSigma: ") +
          e.what());
    }
  };

  void dSources(Index i, GlobalState &out, GlobalState const &states,
                std::vector<Position> const &abscissae, Time time) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      if (!vectorized) {
        TransportSystem::dSources(i, out, states, abscissae,
                                  time); // Call base class version which will
                                         // loop over non-vectorized method
        return;
      }

      out = method_overrides["dSources"](i, states, abscissae, time)
                .cast<GlobalState>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate dSources: ") +
          e.what());
    }
  };

  // Finally one has to provide initial conditions for u & q
  Value InitialValue(Index i, Position x) const override {
    PYBIND11_OVERRIDE_PURE(Value, TransportSystem, InitialValue, i, x);
  };

  Value InitialDerivative(Index i, Position x) const override {
    PYBIND11_OVERRIDE_PURE(Value, TransportSystem, InitialDerivative, i, x);
  };

  Value InitialAuxValue(Index i, Position x) const override {
    PYBIND11_OVERRIDE(Value, TransportSystem, InitialAuxValue, i, x);
  }

  Values AuxG(Index i, GlobalState const &states,
              std::vector<Position> const &abscissae, Time time) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      if (!vectorized)
        return TransportSystem::AuxG(
            i, states, abscissae, time); // Call base class version which will
                                         // loop over non-vectorized method

      return method_overrides["AuxG_v"](i, states, abscissae, time)
          .cast<Values>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate AuxG_v: ") +
          e.what());
    }
  }
  Value AuxG(Index i, const State &s, Position x, Time t) override {
    if (nAux > 0) {
      PYBIND11_OVERRIDE_PURE(Value, TransportSystem, AuxG, i, s, x, t);
    }
  }

  void AuxGPrime(Index i, GlobalState &out, GlobalState const &states,
                 std::vector<Position> const &abscissae, Time time) override {
    if (!initialized)
      initializeOverrides();
    try {
      py::gil_scoped_acquire gil;
      if (!vectorized) {
        TransportSystem::AuxGPrime(i, out, states, abscissae,
                                   time); // Call base class version which will
                                          // loop over non-vectorized method
        return;
      }

      out = method_overrides["AuxGPrime_v"](i, states, abscissae, time)
                .cast<GlobalState>();
    } catch (const std::exception &e) {
      throw std::runtime_error(
          std::string("Error occurred when trying to calculate AuxGPrime: ") +
          e.what());
    }
  }
  void AuxGPrime(Index i, State &out, const State &s, Position x,
                 Time t) override {
    if (!initialized)
      initializeOverrides();
    py::gil_scoped_acquire gil;
    out = method_overrides["AuxGPrime"](i, s, x, t).cast<State>();
  }

  void dSources_dPhi(Index i, VectorRef v, const State &s, Position x,
                     Time t) override {
    if (nAux == 0) {
      v.setZero();
      return;
    }
    if (!initialized)
      initializeOverrides();
    py::gil_scoped_acquire gil;
    v = method_overrides["dSources_dPhi"](i, s, x, t).cast<Values>();
  }

  void dSigma_dPhi(Index i, Values &v, const State &s, Position x,
                   Time t) override {
    if (nAux == 0) {
      v.setZero();
      return;
    }
    if (!initialized)
      initializeOverrides();
    py::gil_scoped_acquire gil;
    v = method_overrides["dSigma_dPhi"](i, s, x, t).cast<Values>();
  }

  Value InitialScalarValue(Index s) const override {
    if (nScalars > 0) {
      PYBIND11_OVERRIDE_PURE(Value, TransportSystem, InitialScalarValue, s);
    }
  }
  Value InitialScalarDerivative(Index s, const DGSoln &y,
                                const DGSoln &dydt) const override {
    py::gil_scoped_acquire gil;
    py::function _override = py::get_override(this, "InitialScalarDerivative");

    GlobalState state = y.evalOnNodes();
    GlobalState state_dot = dydt.evalOnNodes();

    Value out = _override(s, state, state_dot, Integrator::getIntegrationWeights(y.getBasis(), y.getGrid())).cast<Value>();
    return out;
  }
  Value ScalarGExtended(Index s, const DGSoln &y, const DGSoln &dydt,
                        Time t) override {
    if (!initialized)
      initializeOverrides();

    GlobalState state = y.evalOnNodes();
    GlobalState state_dot = dydt.evalOnNodes();

    Value out = method_overrides["ScalarG"](s, state, state_dot, Integrator::getIntegrationWeights(y.getBasis(), y.getGrid()), t)
                    .cast<Value>();
    return out;
  }

  virtual void ScalarGPrimeExtended(GlobalStateMatrix &out,
                                    GlobalStateMatrix &out_dt, const DGSoln &y,
                                    const DGSoln &dydt, Time t) override {
    if (!initialized)
      initializeOverrides();

    GlobalState state = y.evalOnNodes();
    GlobalState state_dot = dydt.evalOnNodes();

    const auto& basis = y.getBasis();
    const auto& grid = y.getGrid();

    auto temp =
        method_overrides["ScalarGPrime"](state, state_dot, Integrator::getIntegrationWeights(basis, grid), Integrator::getPhiCell(basis, grid), Integrator::getPhiBoundary(basis, grid), t)
            .cast<std::array<std::vector<py::dict>, 2>>();

    for (Index i = 0; i < nScalars; i++) {
      out[i] = temp[0][i].cast<GlobalState>();
      out_dt[i] = temp[1][i].cast<GlobalState>();
    }
  }

  bool isScalarDifferential(Index i) override {
    if (nScalars > 0) {
      PYBIND11_OVERRIDE_PURE(bool, TransportSystem, isScalarDifferential, i);
    }
  }
  void dSources_dScalars(Index s, VectorRef v, const State &state, Position x,
                         Time t) override {

    if (!initialized)
      initializeOverrides();

    v = method_overrides["dSources_dScalars"](s, state, x, t).cast<Vector>();
  }
  std::unique_ptr<AdjointProblem> createAdjointProblem() override {
    PYBIND11_OVERRIDE(std::unique_ptr<AdjointProblem>, TransportSystem,
                      createAdjointProblem);
  }

public:
  using TransportSystem::isLowerDirichlet;
  using TransportSystem::isUpperDirichlet;
  using TransportSystem::nAux;
  using TransportSystem::nScalars;
  using TransportSystem::nVars;

private:
  bool initialized = false;
  bool vectorized = false;
  std::map<std::string_view, py::function> method_overrides;
};

#endif // PYTRANSPORTSYSTEM_HPP
