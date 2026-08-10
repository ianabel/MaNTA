#ifndef STATE_HPP
#define STATE_HPP

#include "Logging.hpp"
#include "Types.hpp"

#include <stdexcept>
#include <string>

#ifdef DEBUG
// Eigen error messages are very unhelpful so we make our own
// Mainly for debugging, but also to make sure we don't accidentally mess up
// shapes when copying from python
template <typename A, typename B>
inline void checkShapeAndSet(A &&lhs, const B &rhs,
                             std::optional<std::string> varname) {
  static_assert(std::is_base_of<Eigen::MatrixBase<typename std::decay<A>::type>,
                                typename std::decay<A>::type>::value,
                "Input lhs must be an Eigen Matrix or Matrix Expression");
  static_assert(std::is_base_of<Eigen::MatrixBase<typename std::decay<B>::type>,
                                typename std::decay<B>::type>::value,
                "Input rhs must be an Eigen Matrix or Matrix Expression");
  if ((lhs.rows() == rhs.cols()) && (lhs.cols() == rhs.rows())) {
    logmsg<LOG_LEVEL::WARNING>("Transposing when copying {}; this will lead to "
                               "an error if compiled with DEBUG=off",
                               varname.value_or("variable"));
    lhs = rhs.transpose();
  } else {
    const auto lhs_rows = lhs.rows();
    const auto rhs_rows = rhs.rows();

    const auto lhs_cols = lhs.cols();
    const auto rhs_cols = rhs.cols();

    if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
      const std::string msg =
          "Shape mismatch when attempting to set " +
          varname.value_or("variable") + " during assignment. Input shape: (" +
          std::to_string(rhs_rows) + ", " + std::to_string(rhs_cols) +
          "). Required shape: (" + std::to_string(lhs_rows) + ", " +
          std::to_string(lhs_cols) + ")";
      throw std::runtime_error(msg);
    }
    lhs = rhs;
  }
}
#else
template <typename A, typename B>
inline void checkShapeAndSet(A &&lhs, const B &rhs,
                             std::optional<std::string> varname) {
  lhs = rhs;
}
#endif

class State {
public:
  State() = default;

  /// Born zeroed, not merely sized.
  ///
  /// These are handed to the derivative hooks as out-parameters, and Eigen's
  /// resize() leaves the memory indeterminate. That put the burden of an
  /// opening setZero() on every physics case, and a case that assigned only its
  /// nonzero entries -- which is the natural way to write one -- got whatever
  /// was in the buffer for the rest. That is not hypothetical: it is defect (2)
  /// in the ScalarTestLD3 post-mortem in Tests/README.md, where a missing v[2]
  /// put a garbage column into the scalar coupling matrix. The hooks may still
  /// call zero() and several do; it is redundant now rather than load-bearing.
  explicit State(Index nv, Index ns = 0, Index naux = 0) {
    Variable.setZero(nv);
    Derivative.setZero(nv);
    Flux.setZero(nv);
    Scalars.setZero(ns);
    Aux.setZero(naux);
  }

  void clone(const State &other) {
    Variable.setZero(other.Variable.size());
    Derivative.setZero(other.Derivative.size());
    Flux.setZero(other.Flux.size());
    Scalars.setZero(other.Scalars.size());
    Aux.setZero(other.Aux.size());
  }

  void zero() {
    Variable.setZero();
    Derivative.setZero();
    Flux.setZero();
    Scalars.setZero();
    Aux.setZero();
  }

private:
  // Defined before the accessors that call it: a deduced return type has to be
  // known at the point of use.
  template <typename V>
  static auto &checked(V &v, Index i, const char *what) {
#ifdef DEBUG
    if (i < 0 || i >= v.size())
      throw std::out_of_range(std::string("State: no ") + what + " " +
                              std::to_string(i) + " (there are " +
                              std::to_string(v.size()) + ")");
#else
    (void)what;
#endif
    return v[i];
  }

public:
  /*
      Named access.

      `s.Variable[0]` says where a number lives; `s.u(0)` says what it is. The
      raw vectors below are still public because the type casters and the
      autodiff layer construct whole RealVectors from them, but a physics case
      should reach for these.

      Two things they buy beyond readability:

      * bounds checking under DEBUG. Anything indexed per auxiliary variable is
        sized nAux, not nVars, and those coincide in nearly every case here --
        which is how dGdaux_Vec carried two confusions between them unnoticed.

      * an honest name for the flux. `Flux` holds the solver's *stored* sigma,
        which is -sigma_hat: the negative of what SigmaFn returned. sigma()
        gives that stored value, sigmaHat() gives the physical flux. See the
        header comment in TransportSystem.hpp.
  */
  double &u(Index i) { return checked(Variable, i, "variable"); }
  double u(Index i) const { return checked(Variable, i, "variable"); }

  double &q(Index i) { return checked(Derivative, i, "variable"); }
  double q(Index i) const { return checked(Derivative, i, "variable"); }

  /// The stored flux, sigma = -sigma_hat. This is what Flux has always held.
  double &sigma(Index i) { return checked(Flux, i, "variable"); }
  double sigma(Index i) const { return checked(Flux, i, "variable"); }

  /// The physical flux -- the quantity SigmaFn returns. Read-only: it is a
  /// negation of the stored value, so there is nothing to take a reference to.
  double sigmaHat(Index i) const { return -checked(Flux, i, "variable"); }

  double &phi(Index i) { return checked(Aux, i, "auxiliary variable"); }
  double phi(Index i) const { return checked(Aux, i, "auxiliary variable"); }

  double &scalar(Index i) { return checked(Scalars, i, "scalar"); }
  double scalar(Index i) const { return checked(Scalars, i, "scalar"); }

  Vector Variable, Derivative, Flux, Aux;
  Vector Scalars;
};

class GlobalState {
public:
  GlobalState() = default;

  /// Zeroed for the same reason State is: SystemSolver builds a fresh
  /// GlobalStateMatrix for every Jacobian evaluation and hands its columns
  /// straight to dSigmaFn_du and friends as out-parameters.
  explicit GlobalState(Index nCells, Index k, Index nv, Index ns = 0,
                       Index naux = 0) noexcept
      : nCells(nCells), k(k), nVars(nv), nScalars(ns), nAux(naux) {
    m_Variable.setZero(nVars, nCells * (k + 1));
    m_Derivative.setZero(nVars, nCells * (k + 1));
    m_Flux.setZero(nVars, nCells * (k + 1));
    m_Aux.setZero(nAux, nCells * (k + 1));
    m_Scalars.setZero(nScalars);
  }

  void setWithState(Index i, const State &s) {
    m_Variable.col(i) = s.Variable;
    m_Derivative.col(i) = s.Derivative;
    m_Flux.col(i) = s.Flux;
    m_Aux.col(i) = s.Aux;
    m_Scalars = s.Scalars;
  }

  // Return state at point i
  State operator[](Index i) const {
    State out(nVars, nScalars, nAux);

    out.Variable = m_Variable.col(i);
    out.Derivative = m_Derivative.col(i);
    out.Flux = m_Flux.col(i);
    out.Aux = m_Aux.col(i);
    out.Scalars = m_Scalars;

    return out;
  }

  // This is mainly for copying from python
  GlobalState &operator=(const GlobalState &other) {
    checkShapeAndSet(m_Variable, other.Variable(), "Variable");
    checkShapeAndSet(m_Derivative, other.Derivative(), "Derivative");
    checkShapeAndSet(m_Flux, other.Flux(), "Flux");
    if (nAux > 0) // Don't bother with Aux if nAux = 0
      checkShapeAndSet(m_Aux, other.Aux(), "Aux");
    // Guard-clause form, and braced. The `else` used to hang off the inner `if`
    // -- which is what was meant, so the behaviour here is unchanged -- but with
    // two unbraced nested ifs that is only true by the standard's
    // nearest-enclosing rule, not by anything the reader can see. gcc's
    // -Wdangling-else lives inside -Wparentheses, which Makefile.config disables
    // globally, so only clang reports it. It was the sole thing standing between
    // this codebase and a clean clang build.
    if (nScalars > 0)
    {
      if (m_Scalars.size() != other.Scalars().size())
        throw std::runtime_error("Shape of input scalar array must match "
                                 "nScalars (length of input = " +
                                 std::to_string(other.Scalars().size()) + ")");
      m_Scalars = other.Scalars();
    }
    return *this;
  }

  /*
      Variable
  */
  // Accessor methods for translating between python and C++
  Matrix &Variable() { return m_Variable; }
  const Matrix &Variable() const { return m_Variable; }
  // Accessor methods for getting elements at a point or in a cell
  VectorRef Variable(Index i) { return m_Variable.col(i); }
  // Grabs data on a whole cell for Jacobian computation, **implicitly assumes
  // we're doing interpolation
  Eigen::Ref<Matrix> cellwiseVariable(Index cell) {
    return m_Variable(Eigen::all,
                      Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
  }

  /*
      Derivative
  */
  Matrix &Derivative() { return m_Derivative; }
  const Matrix &Derivative() const { return m_Derivative; }
  VectorRef Derivative(Index i) { return m_Derivative.col(i); }
  Eigen::Ref<Matrix> cellwiseDerivative(Index cell) {
    return m_Derivative(Eigen::all,
                        Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
  }

  /*
      Flux
  */
  Matrix &Flux() { return m_Flux; }
  const Matrix &Flux() const { return m_Flux; }
  VectorRef Flux(Index i) { return m_Flux.col(i); }
  Eigen::Ref<Matrix> cellwiseFlux(Index cell) {
    return m_Flux(Eigen::all,
                  Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
  }

  /*
      Aux
  */
  Matrix &Aux() { return m_Aux; }
  const Matrix &Aux() const { return m_Aux; }
  VectorRef Aux(Index i) { return m_Aux.col(i); }
  Eigen::Ref<Matrix> cellwiseAux(Index cell) {
    return m_Aux(Eigen::all,
                 Eigen::seq(cell * (k + 1), (cell + 1) * (k + 1) - 1));
  }

  /*
      Scalars
  */
  Vector &Scalars() { return m_Scalars; }
  const Vector &Scalars() const { return m_Scalars; }

  size_t size() const { return static_cast<size_t>(nCells * (k + 1)); }

  Index cellDOF() const { return k; };
  Index getNCells() const { return nCells; };

  /// Recover the shape from the arrays themselves.
  ///
  /// Only for the pybind11 type_caster<GlobalState>: PYBIND11_TYPE_CASTER
  /// default-constructs the value and `load` then assigns the matrices, so
  /// without this the size members stay indeterminate and size()/operator[]
  /// read uninitialised memory. The solver never noticed because it only ever
  /// *assigns* a Python-loaded GlobalState into one it built itself, and
  /// operator= copies the arrays and not the sizes.
  ///
  /// A dict of (nPoints, nVars) arrays does not say how the points are shared
  /// out between cells, so this records nCells = nPoints and k = 0: size() is
  /// then correct and operator[] works, but cellwise* is meaningless on such an
  /// object.
  void setShapeFromData()
  {
    nVars = m_Variable.rows();
    nAux = m_Aux.rows();
    nScalars = m_Scalars.size();
    nCells = m_Variable.cols();
    k = 0;
  }

  friend class GlobalStateMatrix;

private:
  // We hold global state data in matrices that are (nVars x nPoints)
  Matrix m_Variable, m_Derivative, m_Flux, m_Aux;

  // Scalars are global so this is just a vector
  Vector m_Scalars;

  // Hold sizes internally for checking & preallocating memory.
  // Initialised here because the default constructor is used by the pybind11
  // type caster; leaving them indeterminate made size() unpredictable.
  Index nCells = 0, k = 0, nVars = 0, nScalars = 0, nAux = 0;
};

/*
    Helpers for the scalar constraint hooks.

    A scalar constraint is a functional of the whole solution, so writing one
    means contracting nodal values with either the quadrature weights or the
    boundary basis values. These put that arithmetic in one place: getting the
    cell/node flattening wrong is silent, because a wrong scalar Jacobian only
    slows Newton down.

    Each takes a single field's nodal values -- a row of GlobalState's
    (nVars x nPoints) matrices, e.g. `y.Variable().row(var)`.
*/
namespace ScalarHooks {

/// Int over the whole domain of one field. weights has one entry per node.
template <typename Row>
inline double integrate(Row const &nodalValues, Vector const &weights) {
  return (nodalValues.transpose().array() * weights.array()).sum();
}

/// The field's value at an end of the domain: 0 for the lower, 1 for the upper.
///
/// The nodes are interior, so this is a contraction of the first (or last)
/// cell's degrees of freedom with the basis functions evaluated there.
template <typename Row>
inline double boundaryValue(Row const &nodalValues, Matrix const &phiBoundary,
                            Index end) {
  const Index cellDoF = phiBoundary.rows(); // k + 1
  const Index offset = (end == 0) ? 0 : nodalValues.size() - cellDoF;
  double out = 0.0;
  for (Index j = 0; j < cellDoF; ++j)
    out += phiBoundary(j, end) * nodalValues[offset + j];
  return out;
}

/// d(boundaryValue)/d(nodal DOF), accumulated into a derivative row with the
/// given scale. The counterpart of boundaryValue for the ScalarGPrime side.
template <typename Row>
inline void addBoundaryDerivative(Row &&dRow, Matrix const &phiBoundary, Index end,
                                  double scale) {
  const Index cellDoF = phiBoundary.rows();
  const Index offset = (end == 0) ? 0 : dRow.size() - cellDoF;
  for (Index j = 0; j < cellDoF; ++j)
    dRow[offset + j] += scale * phiBoundary(j, end);
}

constexpr Index Lower = 0;
constexpr Index Upper = 1;

} // namespace ScalarHooks

// Wrapper class to make Jacobian computation cleaner
// In general, this class will be holding derivative data and not state data
class GlobalStateMatrix {
public:
  GlobalStateMatrix(Index nVars) noexcept : nVars(nVars) {
    m_data.reserve(nVars);
  };

  GlobalStateMatrix(const GlobalStateMatrix &other) {
    nVars = other.nVars;
    m_data = other.m_data;
  }

  GlobalStateMatrix(GlobalStateMatrix &&other) noexcept {
    nVars = other.nVars;
    m_data = std::move(other.m_data);
  }

  void add(Index nCells, Index k, Index nVars, Index nScalars, Index nAux) {
    m_data.emplace_back(nCells, k, nVars, nScalars, nAux);
  }

  GlobalStateMatrix &operator=(const std::vector<GlobalState> &other) {
    nVars = other.size();
    m_data = other;
    return *this;
  }

  void add(const GlobalState &g_in) { m_data.push_back(g_in); }
  /*
      Returns vector of Matrix for per-cell operations, index like
     Variable[Var1](Var2, i), where i is within-cell index
  */
  std::vector<Eigen::Ref<Matrix>> Variable(Index cell) {
    std::vector<Eigen::Ref<Matrix>> out;

    for (Index var = 0; var < nVars; var++) {
      out.emplace_back(m_data[var].cellwiseVariable(cell));
    }
    return out;
  }

  std::vector<Eigen::Ref<Matrix>> Derivative(Index cell) {
    std::vector<Eigen::Ref<Matrix>> out;

    for (Index var = 0; var < nVars; var++) {
      out.emplace_back(m_data[var].cellwiseDerivative(cell));
    }
    return out;
  }
  std::vector<Eigen::Ref<Matrix>> Flux(Index cell) {
    std::vector<Eigen::Ref<Matrix>> out;

    for (Index var = 0; var < nVars; var++) {
      out.emplace_back(m_data[var].cellwiseFlux(cell));
    }
    return out;
  }

  std::vector<Eigen::Ref<Matrix>> Aux(Index cell) {
    std::vector<Eigen::Ref<Matrix>> out;

    for (Index var = 0; var < nVars; var++) {
      out.emplace_back(m_data[var].cellwiseAux(cell));
    }
    return out;
  }
  GlobalState &operator[](Index var) { return m_data[var]; }

  const Index size() const { return nVars; }

private:
  std::vector<GlobalState> m_data;

  Index nVars;
};

#endif
