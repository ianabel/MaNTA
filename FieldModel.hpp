#ifndef FIELDMODEL_HPP
#define FIELDMODEL_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include "FieldModelSpec.hpp"
#include "State.hpp"
#include "Types.hpp"

#include <memory>

/*
    A magnetic-field model: a set of unknowns `psi` carried in the IDA vector,
    an algebraic (or differential) residual constraining them, and a geometry
    map from (psi, x) to the metric fields the transport physics reads.

    The residual is evaluated on every residual call. That is affordable
    because it is the *constraint*, not a solve: the model's own Newton is
    subsumed into IDA's, so its Jacobian is applied once per Newton iteration
    rather than being iterated to convergence per call. See the spec's reading
    of refs/NewtonGSMFEM.pdf.
 */
class FieldModel
{
public:
    explicit FieldModel(FieldModelSpec spec_) : spec(std::move(spec_))
    {
        spec.validate();
        B.setZero(spec.nFieldDOF(), spec.nFieldDOF());
    }
    virtual ~FieldModel() = default;

    FieldModelSpec const &getSpec() const { return spec; }
    Index nFieldDOF() const { return spec.nFieldDOF(); }
    Index nGeometry() const { return spec.nGeometry(); }
    bool isFieldDOFDifferential(Index i) const { return spec.dofs[i].differential; }

    // ---- Residual -------------------------------------------------------

    /// The constraint rows. `out` is length nFieldDOF and arrives zeroed.
    /// `states` and `points` are the transport solution sampled on the physics
    /// nodes -- the same GlobalState ScalarG receives -- and `weights` is one
    /// quadrature weight per node, so Int f dx is weights.dot(f_at_nodes).
    ///
    /// A model that cannot evaluate at this state (no x-point, a boundary that
    /// has left the domain) must throw. static_residual catches and returns 1,
    /// which IDA treats as recoverable and retries with a smaller step.
    virtual void FieldResidual(VectorRef out, Vector const &psi, Vector const &dpsidt,
                               GlobalState const &states, std::vector<Position> const &points,
                               Vector const &weights, Time t) = 0;

    // ---- Geometry -------------------------------------------------------

    /// The metric at one point. `out` is length nGeometry and arrives zeroed.
    virtual void Geometry(VectorRef out, Vector const &psi, Position x, Time t) = 0;

    /// d(geometry slot g)/d(psi_m), shape (nGeometry, nFieldDOF), arrives zeroed.
    virtual void dGeometry_dpsi(MatrixRef out, Vector const &psi, Position x, Time t) = 0;

    // ---- Derivatives of the residual ------------------------------------

    /// Every field row's derivative at once, in the shape ScalarGPrime uses.
    ///
    ///   dR    -- indexed by field row, d(row)/d(transport DOF at each node)
    ///   dRdot -- the same against d/dt of the transport DOFs
    ///   dRdpsi, dRddpsidt -- (nFieldDOF, nFieldDOF), the model's own block
    ///
    /// All four arrive zeroed. Reporting every row at once is deliberate: it is
    /// what lets a model that solves a coupled system internally do so once.
    ///
    /// **dRdot cannot be filled today, and leaving it zero is correct.**
    /// FieldResidual above receives `states` and no `states_dot` -- unlike
    /// ScalarG, which takes both y and ydot -- so a field row has no way to
    /// depend on the transport time derivatives in the first place. The slot is
    /// here because the coupling assembly already weights it by alpha, so the
    /// day the value hook gains ydot the derivative is right rather than
    /// silently one term short.
    ///
    /// What a model author must *not* do on finding it unfillable is put
    /// d(row)/d(psi') there instead. That belongs in dRddpsidt, which is the
    /// block IDA's alpha multiplies and which initialize() checks a
    /// differential DOF against; written into dRdot it would land in the A2
    /// coupling row at the wrong DOFs entirely, and nothing would say so --
    /// A2 is only ever applied, never assembled or printed. See the TODO entry.
    virtual void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &dRdot,
                                    MatrixRef dRdpsi, MatrixRef dRddpsidt,
                                    Vector const &psi, Vector const &dpsidt,
                                    GlobalState const &states, std::vector<Position> const &points,
                                    Vector const &weights, Time t) = 0;

    // ---- Initial condition ----------------------------------------------

    /// The starting guess for psi. Arrives zeroed.
    virtual void InitialFieldValue(VectorRef out) = 0;

    // ---- The model's own Jacobian solve ---------------------------------

    /// Assemble B = dRdpsi + alpha * dRddpsidt and prepare a solve. The default
    /// stores B densely and factorises it with a partial-pivot LU, which is
    /// right for a small block and is what the manufactured clients use. A
    /// model with a large or structured block overrides all four of the
    /// following; this is the seam a real Grad-Shafranov solver plugs into.
    virtual void updateFieldJacobian(MatrixRef dRdpsi, MatrixRef dRddpsidt, double alpha)
    {
        B = dRdpsi + alpha * dRddpsidt;
        Blu.compute(B);
    }

    virtual void applyB(VectorRef out, Vector const &v) const { out = B * v; }
    virtual void applyBTranspose(VectorRef out, Vector const &v) const { out = B.transpose() * v; }

    virtual void solveB(VectorRef out, Vector const &rhs) const
    {
        // Never slice an Eigen solve() result: assign first.
        Vector x = Blu.solve(rhs);
        out = x;
    }

    virtual void solveBTranspose(VectorRef out, Vector const &rhs) const
    {
        Vector x = Blu.transpose().solve(rhs);
        out = x;
    }

    /// Discard anything cached for one run. Called from
    /// SystemSolver::initialize on every run, because initialize() skips
    /// initialiseMatrices() when already initialised -- the RF_cellwise trap,
    /// which made a reused solver take its initial condition from the previous
    /// run's final state.
    virtual void resetForRun() {}

protected:
    FieldModelSpec spec;
    Matrix B;
    Eigen::PartialPivLU<Matrix> Blu;
};

#include "gridStructures.hpp"
#include <toml.hpp>

#include <functional>
#include <map>

template <typename T>
std::unique_ptr<FieldModel> createFieldModel(toml::value const &config, Grid const &grid)
{
    return std::make_unique<T>(config, grid);
}

struct FieldModels
{
public:
    typedef std::function<std::unique_ptr<FieldModel>(toml::value const &, Grid const &)> function_type;
    typedef std::map<std::string, function_type> map_type;

    static std::unique_ptr<FieldModel> InstantiateFieldModel(std::string const &s,
                                                             toml::value const &config,
                                                             Grid const &grid);

    // Throws on a duplicate name rather than quietly keeping the first, which
    // is what a bare map::insert would do -- a model whose name collided would
    // simply never be instantiated, with nothing said at build or run time.
    static void RegisterFieldModel(std::string const &s, function_type creator);

protected:
    static map_type *getMap();

public:
    static map_type *map;
};

template <typename T>
struct FieldModelRegister
{
    explicit FieldModelRegister(std::string const &name)
    {
        FieldModels::RegisterFieldModel(name, createFieldModel<T>);
    }
};

// A model only appears if its object file is linked in -- nothing references it
// directly, so a missing entry is a link-line problem with no compile error.
#define REGISTER_FIELD_MODEL_HEADER(T) static FieldModelRegister<T> registerFieldModel_##T;
#define REGISTER_FIELD_MODEL_IMPL(T) FieldModelRegister<T> T::registerFieldModel_##T(#T);

#endif // FIELDMODEL_HPP
