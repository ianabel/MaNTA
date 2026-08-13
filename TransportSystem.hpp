#ifndef TRANSPORTSYSTEM_HPP
#define TRANSPORTSYSTEM_HPP

#include "State.hpp"
#include "Types.hpp"
#include "SystemSpec.hpp"
#include "DGSoln.hpp"
#include "NetCDFIO.hpp"
#include "AdjointProblem.hpp"

/*
    Pure interface class
    defines a problem in the form
        a_i d_t u_i + d_x ( sigma_i ) = S_i( u(x), q(x), x, t ) ; S_i can depend on the entire u & q vector, but only locally.
        sigma_i = sigma_hat_i( u( x ), q( x ), x, t ) ; so can sigma_hat_i

    The second line is a sign convention, not an identity. SigmaFn returns
    sigma_hat, but the solver stores sigma_i = -sigma_hat_i: residual() forms the
    flux row as ( sigma_h + I_h sigma_hat, phi ) = 0. So the equation actually
    integrated is

        a_i d_t u_i - d_x[ sigma_hat_i( u, q, x, t ) ] = S_i

    Two things follow for anyone writing a case here. A manufactured source must
    carry that minus sign -- get it backwards and the case converges at the right
    rate to the wrong function, which no order study can detect. And State::Flux[i],
    which the hooks below receive, holds the negated sigma_h rather than the
    sigma_hat that SigmaFn returned.

    See CLAUDE.md, "The equation being solved".
 */


class TransportSystem : public std::enable_shared_from_this<TransportSystem>
{

protected:
  /// A case describes itself once, as data, and cannot be built without one.
  ///
  /// There is deliberately no default constructor. A case whose shape depends
  /// on its configuration builds the spec in a static helper and passes it up:
  ///
  ///     MyCase::MyCase(toml::value const &config, Grid const &grid)
  ///         : TransportSystem(buildSpec(config)) { ... }
  ///
  /// which is what forces the description to be complete before any hook can
  /// be called, rather than assembled by assignment part-way through a
  /// constructor body.
  explicit TransportSystem(SystemSpec spec)
      : m_spec(validated(std::move(spec))),
        nVars(m_spec.numVars()), nScalars(m_spec.numScalars()), nAux(m_spec.numAux())
  {
  }

  static SystemSpec validated(SystemSpec spec)
  {
    spec.validate();
    return spec;
  }

  const SystemSpec m_spec;

  // Derived from the spec, never assigned. They are const so that the old
  // `nVars = 2;` in a constructor body is a compile error rather than a second
  // source of truth.
  const Index nVars;
  const Index nScalars;
  const Index nAux;

public:
  virtual ~TransportSystem() = default;

  SystemSpec const &spec() const { return m_spec; }

  Index getNumVars() const { return nVars; };
  Index getNumScalars() const { return nScalars; };
  Index getNumAux() const { return nAux; };

  virtual void setRestartValues(const std::vector<double> &y, const std::vector<double> &dydt, const Grid &grid, Index k)
  {
    // Copy into vectors owned by TransportSystem. (These were written as
    // std::move(y), but y is a const lvalue reference -- std::move on it yields
    // a const rvalue, which binds to copy-assignment, not move-assignment. So
    // it was always a copy; the std::move only misled the reader. A copy is
    // what we want here, since the caller keeps ownership of its vectors.)
    restart_Y_data = y;
    restart_dYdt_data = dydt;

    // Create DGSolns to wrap restart data
    restart_Y = std::make_shared<DGSoln>(nVars, grid, k, restart_Y_data.data(), nScalars, nAux);
    restart_dYdt = std::make_shared<DGSoln>(nVars, grid, k, restart_dYdt_data.data(), nScalars, nAux);
    restarting = true;

    // Pull boundary conditions directly from restart values
    Position xL = grid.lowerBoundary();
    Position xR = grid.upperBoundary();

    uL.resize(nVars);
    uR.resize(nVars);

    for (Index i = 0; i < nVars; ++i)
    {
      // q, not sigma. A Neumann boundary value is applied to q -- see
      // SystemSolver.cpp's L_global assembly and docs/physics_interface.rst --
      // so seeding it from sigma handed the resumed run the wrong quantity, and
      // with the wrong sign into the bargain, since the stored sigma is
      // -sigma_hat. For sigma_hat = kappa q that made the resumed boundary
      // -kappa q rather than q: a sign flip even at kappa = 1. It stayed hidden
      // because it needs three things at once -- a Neumann boundary, a *nonzero*
      // value on it, and a case that does not override LowerBoundary, since an
      // overriding case never reads these at all.
      uL[i] = isLowerBoundaryDirichlet(i) ? restart_Y->u(i)(xL) : restart_Y->q(i)(xL);
      uR[i] = isUpperBoundaryDirichlet(i) ? restart_Y->u(i)(xR) : restart_Y->q(i)(xR);
    }
  }

  bool isRestarting() const { return restarting; };
  DGSoln &getRestartY() { return *restart_Y; };
  DGSoln &getRestartdYdt() { return *restart_dYdt; };

  // Function for passing boundary conditions to the solver
  virtual Value LowerBoundary(Index i, Time t) const { return uL[i]; };
  virtual Value UpperBoundary(Index i, Time t) const { return uR[i]; };

  // Not virtual: the boundary *kind* is part of what a case is, and lives in the
  // spec. Only the boundary *values* above depend on t and stay overridable.
  // This is also what retired the pair of uninitialised bools these used to
  // read, which a case that overrode neither function would silently consult.
  //
  // .at(), here and in the lookups below, deliberately. The old versions
  // answered for any index -- getScalarName(1) on a case with no scalars
  // returned "Scalar1", and isScalarDifferential(0) returned false -- so a
  // caller that had confused nAux with nVars, or looped to the wrong bound, got
  // a plausible answer instead of a complaint. That confusion is a documented
  // source of bugs here (dGdaux_Vec carried two of them). These are integer
  // lookups next to matrix solves; the bounds check is not measurable.
  bool isLowerBoundaryDirichlet(Index i) const { return m_spec.variables.at(i).lower == BoundaryKind::Dirichlet; };
  bool isUpperBoundaryDirichlet(Index i) const { return m_spec.variables.at(i).upper == BoundaryKind::Dirichlet; };

  // The same for the flux and source functions -- the vectors have length nVars
  virtual Value SigmaFn(Index i, const State &s, Position x, Time t) = 0;
  virtual Value Sources(Index i, const State &s, Position x, Time t) = 0;

  // This determines the a_i functions. Only one with a default option, but can be overridden
  virtual Value aFn(Index i, Position x) { return 1.0; };

  // We need derivatives of the flux functions
  virtual void dSigmaFn_du(Index i, VectorRef, const State &s, Position x, Time t) = 0;
  virtual void dSigmaFn_dq(Index i, VectorRef, const State &s, Position x, Time t) = 0;

  // and for the sources
  virtual void dSources_du(Index i, VectorRef, const State &, Position x, Time t) = 0;
  virtual void dSources_dq(Index i, VectorRef, const State &, Position x, Time t) = 0;
  virtual void dSources_dsigma(Index i, VectorRef, const State &, Position x, Time t) = 0;


 /*
 * Compute all fluxes and sources 
 */
  virtual PhysicsOutput ComputePhysics(GlobalState const &states, std::vector<Position> const & abscissae, Time time)
{
    m_sourceCache.resize(nVars); // make sure we have enough elements in cache
    PhysicsOutput out;
    out[0].resize(nVars);
    out[1].resize(nVars);
    out[2].resize(nAux);
    for (auto& p: out)
    {
      for (auto &v : p)
           v.resize(states.size());
    }
    for (Index i = 0; i < nVars; ++i) 
    {
        out[0][i] = SigmaFn(i, states, abscissae, time);
        out[1][i] = Sources(i, states, abscissae, time);
        m_sourceCache[i] = out[1][i];
    }
    for (Index i = 0; i < nAux; ++i)
      out[2][i] = AuxG(i, states, abscissae, time);
    return out;
  }
  // Wrapper functions which serialise batched evaluations
  //
  virtual Values SigmaFn(Index i, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
    Values out(states.size());
#pragma omp parallel for
    for (size_t j = 0; j < states.size(); ++j)
    {
      out(j) = SigmaFn(i, states[j], abscissae[j], time);
    }
    return out;
  };

  virtual Values Sources(Index i, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
    Values out(states.size());
#pragma omp parallel for
    for (size_t j = 0; j < states.size(); ++j)
    {
      out(j) = Sources(i, states[j], abscissae[j], time);
    }
    return out;
  };

  virtual void ComputePhysicsDerivatives(std::array<std::reference_wrapper<GlobalStateMatrix>, NPHYSICS_FUNCTIONS>&&  out, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
    GlobalStateMatrix& dSigma_vals = out[0];
    GlobalStateMatrix& dSource_vals = out[1];
    GlobalStateMatrix& dAux_vals = out[2];
    for (Index i = 0; i < nVars; i++)
    {
        dSigma(i, dSigma_vals[i], states, abscissae, time);
        dSources(i, dSource_vals[i], states, abscissae, time);
    }
    for (Index i = 0; i < nAux; i++)
    {
        AuxGPrime(i, dAux_vals[i], states, abscissae,  time);
    }
  }

  virtual void dSigma(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
#pragma omp parallel for
    for (size_t j = 0; j < states.size(); ++j)
    {
      dSigmaFn_du(i, out.Variable(j), states[j], abscissae[j], time);
      dSigmaFn_dq(i, out.Derivative(j), states[j], abscissae[j], time);
      if (nAux > 0)
        dSigma_dPhi(i, out.Aux(j), states[j], abscissae[j], time);
    }
  }

  virtual void dSources(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
#pragma omp parallel for
    for (size_t j = 0; j < states.size(); ++j)
    {
      dSources_du(i, out.Variable(j), states[j], abscissae[j], time);
      dSources_dq(i, out.Derivative(j), states[j], abscissae[j], time);
      dSources_dsigma(i, out.Flux(j), states[j], abscissae[j], time);
      if (nAux > 0)
        dSources_dPhi(i, out.Aux(j), states[j], abscissae[j], time);
    }
  }

  // and initial conditions for u & q
  virtual Value InitialValue(Index i, Position x) const = 0;
  virtual Value InitialDerivative(Index i, Position x) const = 0;

  virtual Value InitialScalarValue(Index s) const
  {
    if (nScalars != 0)
      throw std::logic_error("nScalars > 0 but no initial value provided");
    return 0.0;
  }

  // Only called if you set a scalar to be differential (rather than algebraic)
  virtual Value InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const
  {
    return 0.0;
  }

  /*
      The global (non-spatial) scalar constraints, G_s( mu, y, dy/dt, t ) = 0.

      There were four names and five signatures here: ScalarG, ScalarGExtended,
      ScalarGPrime and two ScalarGPrimeExtended overloads, one of which handed
      the case a std::function test function and an Interval and asked it to
      integrate against them with a quadrature rule of its own choosing. Two
      pieces:

        * `y` and `ydot` are the solution and its time derivative sampled on the
          element nodes -- the same GlobalState the flux and source hooks see.
          The basis is interpolatory, so a nodal value *is* the coefficient it
          multiplies.

        * `abscissae` gives the position of each node, for a constraint whose
          integrand depends on x itself -- a mirror plasma's needs the magnetic
          geometry at each point. Same vector the flux and source hooks get.

        * `weights` holds one quadrature weight per node, so an integral over
          the whole domain is `weights.dot(y.Variable().row(i))`. Using the
          framework's weights rather than a rule of your own is not a
          convenience: ScalarTestLD3 integrated its own mass with a global
          adaptive Kronrod rule over a piecewise polynomial, which is not a
          smooth function of the coefficients, and its finite-difference
          reference disagreed with the exact answer by 8% at k = 4 on 16 cells.

      This is the shape the Python trampoline already flattened these into, and
      it is now the one contract rather than a translation of another one.

        * `phiBoundary` is (k+1) x 2: the basis functions of the first and last
          cells evaluated at the two ends of the domain. It is what makes a
          constraint on a boundary *point* expressible at all -- MaNTA's nodes
          are Chebyshev points of the first kind, which are strictly interior,
          so sigma(x_R) cannot be read off the nodal values. boundaryValue() in
          State.hpp does that contraction. (The Python trampoline passed
          phiBoundary only to the derivative, so a Python case could not write
          such a constraint at all; both hooks take it now.)
  */
  virtual Value ScalarG(Index s, GlobalState const &y, GlobalState const &ydot,
                        std::vector<Position> const &abscissae, Values const &weights,
                        Matrix const &phiBoundary, Time t)
  {
    if (nScalars != 0)
      throw std::logic_error("nScalars > 0 but no scalar G provided");
    return 0.0;
  }

  /*
      dG_s/d(state) and dG_s/d(state_dot), for every s at once.

      Each is written as the derivative with respect to the *degrees of
      freedom*, not as a function of x: for G = mu - Int u dx the u entry at
      node j is -weights(j), because that is d/du_j of the integral. That is
      why `weights` is passed to the derivative as well as to G itself.

      `phiBoundary` is (k+1) x 2: the basis functions of the first and last
      cells evaluated at the two ends of the domain. It is what makes point
      functionals expressible -- a constraint involving sigma(x_R) has
      d/d(sigma DOF j of the last cell) = phiBoundary(j, 1) -- since the nodes
      need not include the endpoints.
  */
  virtual void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot,
                            GlobalState const &y, GlobalState const &ydot,
                            std::vector<Position> const &abscissae, Values const &weights,
                            Matrix const &phiBoundary, Time t)
  {
    if (nScalars != 0)
      throw std::logic_error("nScalars > 0 but no scalar G derivative provided");
  }

  // Also spec data. Consulted to decide whether InitialScalarDerivative is
  // asked for this scalar, and whether G_s carries an alpha * dG/dmu' term.
  bool isScalarDifferential(Index i) const { return m_spec.scalars.at(i).differential; }

  virtual void dSources_dScalars(Index, VectorRef, const State &, Position, Time)
  {
    if (nScalars != 0)
      throw std::logic_error("nScalars > 0 but no coupling function provided");
  }

  // Auxiliary variable functions

  virtual Value InitialAuxValue(Index i, Position x) const
  {
    if (nAux != 0)
      throw std::logic_error("nAux > 0 but no initial auxiliary value provided");
    return 0.0;
  }

  // G_i( a(x), {u_j(x), q_j(x), sigma_j(x)} , x ) = 0 is the equation
  // that defines the auxiliary variable a
  virtual Value AuxG(Index i, const State &, Position, Time)
  {
    if (nAux != 0)
      throw std::logic_error("nAux > 0 but no auxiliary G provided");
    return 0.0;
  }

  virtual Values AuxG(Index i, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
    Values out(states.size());
#pragma omp parallel for
    for (size_t j = 0; j < states.size(); ++j)
    {
      out(j) = AuxG(i, states[j], abscissae[j], time);
    }
    return out;
  }

  // AuxGPrime returns dG_i in out
  virtual void AuxGPrime(Index i, State &out, const State &, Position, Time)
  {
    throw std::logic_error("nAux > 0 but no G derivative provided");
  }

  virtual void AuxGPrime(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae, Time time)
  {
#pragma omp parallel for
    for (size_t j = 0; j < states.size(); ++j)
    {
      // Declared inside the loop, not outside it. One State shared across a
      // `#pragma omp parallel for` is a data race under OMP=on -- every thread
      // writing its own point's derivatives into the same vectors -- and it
      // also carried one point's values into the next when a hook wrote only
      // its nonzero entries. Per-iteration, it is private and starts zeroed.
      State temp(nVars, nScalars, nAux);
      AuxGPrime(i, temp, states[j], abscissae[j], time);
      out.setWithState(j, temp);
    }
  }

  virtual void dSources_dPhi(Index, VectorRef, const State &, Position, Time)
  {
    if (nAux != 0)
      throw std::logic_error("nAux > 0 but no coupling to the main sources provided");
  }

  virtual void dSigma_dPhi(Index, VectorRef, const State &, Position, Time)
  {
    if (nAux != 0)
      throw std::logic_error("nAux > 0 but no coupling to fluxes provided");
  }

  virtual std::unique_ptr<AdjointProblem> createAdjointProblem()
  {
    throw std::logic_error("Adjoint problem not implemented for this physics case");
  }

  // Nine virtuals' worth of naming, now one lookup each. These are what key the
  // netCDF groups and time series, so they are the case's own names for things
  // rather than Var0/Scalar1/AuxVariable2.
  std::string const &getVariableName(Index i) const { return m_spec.variables.at(i).name; }
  std::string const &getScalarName(Index i) const { return m_spec.scalars.at(i).name; }
  std::string const &getAuxVarName(Index i) const { return m_spec.aux.at(i).name; }

  std::string const &getVariableDescription(Index i) const { return m_spec.variables.at(i).description; }
  std::string const &getScalarDescription(Index i) const { return m_spec.scalars.at(i).description; }
  std::string const &getAuxDescription(Index i) const { return m_spec.aux.at(i).description; }

  std::string const &getVariableUnits(Index i) const { return m_spec.variables.at(i).units; }
  std::string const &getScalarUnits(Index i) const { return m_spec.scalars.at(i).units; }
  std::string const &getAuxUnits(Index i) const { return m_spec.aux.at(i).units; }

  // Hooks for adding extra NetCDF outputs
  virtual void initialiseDiagnostics(NetCDFIO &)
  {
    return;
  }

  virtual void writeDiagnostics(DGSoln const &y, DGSoln const &dydt, double t, NetCDFIO &nc, size_t tIndex)
  {
    writeDiagnostics(y, t, nc, tIndex);
  }

  // Parameters are ( solution, time, netcdf output object, time index )
  virtual void writeDiagnostics(DGSoln const &, double, NetCDFIO &, size_t)
  {
    return;
  }

  virtual void finaliseDiagnostics(NetCDFIO &)
  {
    return;
  }

  std::map<int, std::string> subVars = {{0, "u"}, {1, "q"}, {2, "sigma"}, {3, "S"}};

  virtual std::string getAdjointNames(Index pIndex) const { return "p" + std::to_string(pIndex); }

  Values& getSourceCache(Index var) { return m_sourceCache[var]; }

protected:
  bool restarting = false;
  std::vector<double> restart_Y_data;
  std::vector<double> restart_dYdt_data;
  std::shared_ptr<DGSoln> restart_Y = nullptr;
  std::shared_ptr<DGSoln> restart_dYdt = nullptr;
  
  std::vector<Values> m_sourceCache; // since sources might be expensive to calculate, cache them for use in outputs

  std::vector<Value> uL, uR;
};

#endif // TRANSPORTSYSTEM_HPP
