#ifndef PYADJOINTRPOBLEM_HPP
#define PYADJOINTRPOBLEM_HPP

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/math/quadrature/gauss.hpp>

#include "AdjointProblem.hpp"
#include "PyState.hpp"

namespace py = pybind11;

class PyAdjointProblem : public AdjointProblem,
                         public py::trampoline_self_life_support {
public:
  using AdjointProblem::AdjointProblem;

  void initializeOverrides() {
    auto make_override = [this](const char *method_name) {
      py::gil_scoped_acquire gil;
      py::function _override = py::get_override(this, method_name);

      if (_override) {
        return _override;
      } else {
        throw std::runtime_error(std::string("Pure virtual method ") +
                                 method_name +
                                 " not overridden in Python subclass");
      }
    };
    method_overrides.insert(
        std::make_pair("dgFn_dphi", make_override("dgFn_dphi")));
    method_overrides.insert(
        std::make_pair("dAux_dp", make_override("dAux_dp")));

    initialized = true;
  }
  // PyTransportSystem(TransportSystem &&base) : Tr

  using IntegratorType = boost::math::quadrature::gauss<double, 30>;
  static IntegratorType integrator;

  // We don't have the DGSoln object in Python, so we implement GFn and
  // dGFndp here
  Value GFn(Index gIndex, DGSoln &y) const override {
    const auto states = y.evalOnNodes();
    const auto points = y.getPoints();
    Value out = 0.0;

    Values g = gFn(gIndex, states, points);

    for (size_t i = 0; i < y.getGrid().getNCells(); i++) {
      const Interval &I = y.getGrid()[i];

      // https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas
      // integrate interpolation to get weights
      // compute integral as sum g * weights
      const auto k = y.getBasis().Order();

      const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);

      const auto weights = y.getBasis().getIntegrationWeights(I);
      const Vector g_cellwise = g(ind);
      out += g_cellwise.dot(weights);
    }

    return out;
  };

  Value dGFndp(Index i, Index pIndex, DGSoln &y) const override {
    throw std::runtime_error("Non-vectorized version of function \"dGFndp\" "
                             "deprecated.");
  };
  Matrix dGFndp(Index gIndex, DGSoln &y) const override {
    const auto states = y.evalOnNodes();
    const auto points = y.getPoints();
    Matrix out;

    // If parameters are spatial, int dg/dp dx = int dg/dp_cell
    // delta(x - x_cell) dx, so we just return dg/dp evaluated at
    // the nodes. Otherwise, we need to integrate dg/dp over the
    // domain to get the total sensitivity with respect to that
    // parameter.
    if (areParametersSpatial()) {
      // Transposed on the way out. dgFndp is (np, nPoints) -- that is the
      // orientation the non-spatial branch below indexes as dgdp(p, ind) --
      // while computeAdjointGradients assigns this into a
      // (nCells * (k + 1), np) block of G_p. Returned raw it aborted inside
      // Eigen's assignment, and would have silently transposed the gradient
      // rather than aborting had np ever equalled the node count.
      Matrix retval = dgFndp(gIndex, states, points);
      return retval.transpose();
    } else {
      out.resize(1, np);
    }

    out.setZero();

    Matrix dgdp = dgFndp(gIndex, states, points);

    for (size_t i = 0; i < y.getGrid().getNCells(); i++) {
      const Interval &I = y.getGrid()[i];

      // interpolate dgFndp onto the quadrature points
      // integrate interpolation to get weights
      // compute integral as sum dgFndp * weights
      const auto k = y.getBasis().Order();

      const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);

      const auto weights = y.getBasis().getIntegrationWeights(I);
      for (Index p = 0; p < np; p++) {
        const Vector dgdp_cellwise = dgdp(p, ind);
        out(p) += dgdp_cellwise.dot(weights);
      }
    }

    return out;
  }
  Value gFn(Index gIndex, const State &s, Position x) const override {
    throw std::runtime_error(
        "Non-vectorized version of function \"gFn\" deprecated.");
  };
  Values gFn(Index gIndex, const GlobalState &s,
             std::vector<Position> const &abscissae) const override {
    PYBIND11_OVERRIDE(Values, AdjointProblem, gFn, gIndex, s, abscissae);
  };

  Matrix dgFndp(Index gIndex, const GlobalState &states,
                std::vector<Position> const &abscissae) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, AdjointProblem, dgFndp, gIndex, states,
                           abscissae);
  };
  void dgFn_du(Index i, VectorRef out, const State &s, Position x) override {
    throw std::runtime_error(
        "Individual derivative function \"dgFn_du\" deprecated; "
        "use vectorized version dg instead.");
    // if (!initialized)
    //     initializeOverrides();
    // out = method_overrides["dgFn_du"](i, StateView(const_cast<State &>(s)), x).cast<Values>();
  };
  void dgFn_dq(Index i, VectorRef out, const State &s, Position x) override {
    throw std::runtime_error(
        "Individual derivative function \"dgFn_dq\" deprecated; "
        "use vectorized version dg instead.");
    // if (!initialized)
    //     initializeOverrides();
    // out = method_overrides["dgFn_dq"](i, StateView(const_cast<State &>(s)), x).cast<Values>();
  };
  void dgFn_dsigma(Index i, VectorRef out, const State &s,
                   Position x) override {
    throw std::runtime_error("Individual derivative function \"dgFn_dsigma\" "
                             "deprecated; use vectorized version dg instead.");
    // if (!initialized)
    //     initializeOverrides();
    // out = method_overrides["dgFn_dsigma"](i, s,
    // x).cast<Values>();
  };
  void dgFn_dphi(Index i, VectorRef out, const State &s, Position x) override {
    if (!initialized)
      initializeOverrides();
    out = method_overrides["dgFn_dphi"](i, StateView(const_cast<State &>(s)), x).cast<Values>();
  };

  void dg(Index gIndex, GlobalState &out, GlobalState const &states,
          std::vector<Position> const &abscissae) override {
    constexpr const char *method_name = "dg";
    py::gil_scoped_acquire gil;
    py::function _override = py::get_override(this, method_name);

    if (!_override) {
      throw std::runtime_error("Vectorized function \"dg\" not found in Python "
                               "subclass");
      // std::cerr << "WARNING: Vectorized function \"dSigma\"
      // not found in Python subclass" << std::endl;
      // TransportSystem::dSigma(i, out, states, abscissae,
      // time); return;
    }

    out = _override(gIndex, states, abscissae).cast<GlobalState>();
  };

  void ComputePhysicsDerivatives(
      std::array<std::reference_wrapper<GlobalStateMatrix>, NPHYSICS_FUNCTIONS>
          &&out,
      GlobalState const &states,
      std::vector<Position> const &abscissae) override {
    py::gil_scoped_acquire gil;
    py::function _override =
        py::get_override(this, "ComputePhysicsDerivatives");

    if (!_override) {
      AdjointProblem::ComputePhysicsDerivatives(std::move(out), states,
                                                abscissae);
      return;
    }

    std::array<std::vector<Matrix>, NPHYSICS_FUNCTIONS> temp =
        _override(states, abscissae)
            .cast<std::array<std::vector<Matrix>, NPHYSICS_FUNCTIONS>>();

    GlobalStateMatrix &dflux = out[0];
    GlobalStateMatrix &dsource = out[1];
    GlobalStateMatrix &daux = out[2];

    for (Index var = 0; var < dflux.size(); ++var) {
      dflux[var].Variable() = temp[0][var];
      dsource[var].Variable() = temp[1][var];
    }
    for (Index aux = 0; aux < daux.size(); ++aux) {
      daux[aux].Variable() = temp[2][aux];
    }
  };

  void dAux(Index i, GlobalState &out, GlobalState const &states,
            std::vector<Position> const &abscissae) override {
    std::string method_name = "dAux";
    py::gil_scoped_acquire gil;
    py::function _override = py::get_override(this, method_name.c_str());

    if (!_override) {
      throw std::runtime_error("Vectorized function \"dAux\" not found in "
                               "Python subclass");
      // std::cerr << "WARNING: Vectorized function \"dAux\"
      // not found in Python subclass" << std::endl;
      // TransportSystem::dAux(i, out, states, abscissae,
      // time); return;
    }

    checkShapeAndSet(out.Variable(),
                     _override(i, states, abscissae).cast<Matrix>(),
                     "dAux in PyAdjointProblem");
  }
  void dSigmaFn_dp(Index i, Index pIndex, Value &out, const State &s,
                   Position x) override {
    throw std::runtime_error("Individual derivative functions deprecated; use "
                             "vectorized version dSigma instead.");
    // if (!initialized)
    //     initializeOverrides();
    // out = method_overrides["dSigmaFn_dp"](i, pIndex, s,
    // x).cast<Value>();
  };

  void dSources_dp(Index i, Index pIndex, Value &out, const State &s,
                   Position x) override {
    throw std::runtime_error("Individual derivative functions deprecated; use "
                             "vectorized version dSources instead.");
    // if (!initialized)
    //     initializeOverrides();
    // out = method_overrides["dSources_dp"](i, pIndex, s,
    // x).cast<Value>();
  };

  void dSigma(Index i, GlobalState &out, GlobalState const &states,
              std::vector<Position> const &abscissae) override {
    std::string method_name = "dSigma";
    py::gil_scoped_acquire gil;
    py::function _override = py::get_override(this, method_name.c_str());

    if (!_override) {
      throw std::runtime_error("Vectorized function \"dSigma\" not found in "
                               "Python subclass");
      // std::cerr << "WARNING: Vectorized function \"dSigma\"
      // not found in Python subclass" << std::endl;
      // TransportSystem::dSigma(i, out, states, abscissae,
      // time); return;
    }

    checkShapeAndSet(out.Variable(),
                     _override(i, states, abscissae).cast<Matrix>(),
                     "dSigma in PyAdjointProblem");
  };

  void dSources(Index i, GlobalState &out, GlobalState const &states,
                std::vector<Position> const &abscissae) override {
    std::string method_name = "dSources";
    py::gil_scoped_acquire gil;
    py::function _override = py::get_override(this, method_name.c_str());

    if (!_override) {
      throw std::runtime_error("Vectorized function \"dSources\" not found in "
                               "Python subclass");
      // std::cerr << "WARNING: Vectorized function \"dSigma\"
      // not found in Python subclass" << std::endl;
      // TransportSystem::dSigma(i, out, states, abscissae,
      // time); return;
    }

    checkShapeAndSet(out.Variable(),
                     _override(i, states, abscissae).cast<Matrix>(),
                     "dSource in PyAdjointProblem");
  };

  void dAux_dp(Index i, Index pIndex, Value &out, const State &s,
               Position x) override {
    if (!initialized)
      initializeOverrides();
    out = method_overrides["dAux_dp"](i, pIndex, StateView(const_cast<State &>(s)), x)
              .cast<Value>();
  }

  std::string getName(Index pIndex) const override {
    PYBIND11_OVERRIDE(std::string, AdjointProblem, getName, pIndex);
  };

  bool computeUpperBoundarySensitivity(Index i, Index pIndex) override {
    PYBIND11_OVERRIDE(bool, AdjointProblem, computeUpperBoundarySensitivity, i,
                      pIndex);
  };
  bool computeLowerBoundarySensitivity(Index i, Index pIndex) override {
    PYBIND11_OVERRIDE(bool, AdjointProblem, computeLowerBoundarySensitivity, i,
                      pIndex);
  };

public:
  using AdjointProblem::ng;
  using AdjointProblem::np;
  using AdjointProblem::np_boundary;

  using AdjointProblem::spatialParameters;

private:
  bool initialized = false;
  std::map<std::string, py::function> method_overrides;
};

#endif // PYADJOINTPROBLEM_HPP
