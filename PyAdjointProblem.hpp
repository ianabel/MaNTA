#ifndef PYADJOINTRPOBLEM_HPP
#define PYADJOINTRPOBLEM_HPP

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AdjointProblem.hpp"
#include "PyState.hpp"

namespace py = pybind11;

class PyAdjointProblem : public AdjointProblem,
                         public py::trampoline_self_life_support {
public:
  using AdjointProblem::AdjointProblem;

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
  // dgFndp must report (np, nPoints), which is what the non-spatial branch
  // indexes as dgdp(p, ind) and what the spatial branch transposes into G_p's
  // (nPoints, np) block.
  //
  // Checked rather than assumed, because neither wrong shape announces itself.
  // checkShapeAndSet is a plain assignment outside a DEBUG build, so a mismatch
  // reaches Eigen and aborts the process naming
  // Block<Matrix<double,-1,-1>,-1,-1,false> and nothing about MaNTA -- and where
  // np happens to equal the node count nothing aborts at all: the gradient is
  // silently transposed and the run returns a plausible wrong answer.
  void checkDgFndpShape(Matrix const &dgdp, std::size_t nPoints) const {
    if (dgdp.rows() == np && dgdp.cols() == static_cast<Eigen::Index>(nPoints))
      return;

    std::string hint;
    if (dgdp.rows() == static_cast<Eigen::Index>(nPoints) && dgdp.cols() == np)
      hint = " It is the transpose of what is wanted: return dgdp.T, or build it "
             "with the parameter as the first axis.";

    throw std::runtime_error(
        "Adjoint hook \"dgFndp\" returned a (" + std::to_string(dgdp.rows()) +
        ", " + std::to_string(dgdp.cols()) + ") array; it must be (np, nPoints) = (" +
        std::to_string(np) + ", " + std::to_string(nPoints) + ")." + hint);
  }

  // The explicit dG/dp of G = sum_m w_m g(Z_m, p), from dgFndp at the nodes the
  // objective is evaluated on, nNodes per cell, with one weight per node.
  //
  // A spatial parameter takes its own value p_m at each of those nodes, so
  // dG/dp_m = w_m dg/dp_m: one row per node, (nPoints, np), the layout of G_p's
  // block for this objective. That is the discrete derivative the adjoint term
  // computes too -- d(residual)/dp_m through the cell projection -- which is
  // what lets the two halves of G_p add. A parameter that is one number gets
  // the sum over nodes, (1, np).
  Matrix explicitParameterDerivative(Index gIndex, GlobalState const &states,
                                     std::vector<Position> const &points,
                                     Vector const &weights, Index nNodes) const {
    Matrix dgdp = dgFndp(gIndex, states, points);
    checkDgFndpShape(dgdp, points.size());

    if (areParametersSpatial())
      return (dgdp * weights.asDiagonal()).transpose();

    Matrix out(1, np);
    out.setZero();
    const Index nCells = static_cast<Index>(points.size()) / nNodes;
    for (Index i = 0; i < nCells; i++)
      for (Index p = 0; p < np; p++)
        out(p) += dgdp.row(p).segment(i * nNodes, nNodes).dot(
            weights.segment(i * nNodes, nNodes));
    return out;
  }

  Matrix dGFndp(Index gIndex, DGSoln &y) const override {
    const Index nCells = y.getGrid().getNCells();
    const Index nNodes = y.getBasis().Order() + 1;
    Vector weights(nCells * nNodes);
    for (Index i = 0; i < nCells; i++)
      weights.segment(i * nNodes, nNodes) =
          y.getBasis().getIntegrationWeights(y.getGrid()[i]);

    return explicitParameterDerivative(gIndex, y.evalOnNodes(), y.getPoints(),
                                       weights, nNodes);
  }

  // At the star nodes, with u* and the star basis's weights b1: the derivative
  // of AdjointProblem::GFn's superconvergent form. Reads the u* from the last
  // pp.computeUStar(y).
  Matrix dGFndp(Index gIndex, DGSoln &y,
                Postprocessor const &pp) const override {
    const Index nCells = y.getGrid().getNCells();
    const Index nNodes = pp.starDoF();
    Vector weights(nCells * nNodes);
    for (Index i = 0; i < nCells; i++)
      weights.segment(i * nNodes, nNodes) = pp.starWeights(i);

    return explicitParameterDerivative(gIndex, pp.evalOnStarNodes(y),
                                       pp.starPoints(), weights, nNodes);
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
    throw std::runtime_error("Individual derivative function \"dgFn_dphi\" "
                             "deprecated; use vectorized version dg instead.");
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

  // A length-np vector arrives as (np, 1), which is the non-spatial shape; the
  // solver checks the shape against spatialParameters.
  Matrix dScalarG_dp(Index s, GlobalState const &y, GlobalState const &ydot,
                     std::vector<Position> const &abscissae, Values const &weights,
                     Matrix const &phiBoundary, Time t) override {
    PYBIND11_OVERRIDE(Matrix, AdjointProblem, dScalarG_dp, s, y, ydot, abscissae,
                      weights, phiBoundary, t);
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

    throw std::runtime_error("Individual derivative functions deprecated; use "
                             "vectorized version ComputePhysicsDerivatives instead.");
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

};

#endif // PYADJOINTPROBLEM_HPP
