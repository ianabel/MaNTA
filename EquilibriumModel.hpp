#ifndef EQUILIBRIUMMODEL_HPP
#define EQUILIBRIUMMODEL_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include "FieldModelSpec.hpp"
#include "State.hpp"
#include "Types.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/*
    A magnetic equilibrium as a black box: tier 0 of
    docs/superpowers/specs/2026-10-23-field-coupling-tiers-design.md.

    What the transport needs from the field is one map, from the transport state
    at the physics nodes to the geometry at them, g = G(p). A tier-0 model
    supplies exactly that and nothing else -- no unknowns in MaNTA's state
    vector, no residual, no derivatives -- which is what a code driven as a
    black box (VMEC, DESC) can actually answer.

    The solver holds the answer as a cache, not as unknowns:

      * Solve is called whenever the transport input changes, and only then: a
        repeated residual at one state, or a Jacobian build at the last
        residual's state, reuses the last equilibrium. The input is the
        transport state at the basis nodes, every variable, as a FieldModel's
        residual is handed it.
      * Geometry is then asked for the slots at whichever points the solver is
        about to evaluate the physics at.
      * The Jacobian carries no coupling: the transport derivatives are taken at
        the geometry of the build's state, held fixed. So the solver's Newton
        converges linearly in the coupling, which is the price of asking for
        nothing but G.

    `Equilibrium` is opaque to the solver. It holds whatever the model caches --
    a spline, Fourier coefficients, a file handle -- and the solver only stores
    the most recent one and hands it back as the warm start for the next Solve.
 */
class Equilibrium
{
public:
    virtual ~Equilibrium() = default;
};

/// What a tier-0 model is, as data: FieldModelSpec without the unknowns.
///
/// `name` is the netCDF group the geometry slots are written into and `label`
/// names the coordinate they are a function of, exactly as for a FieldModel, and
/// they are checked the same way.
struct EquilibriumSpec
{
    std::vector<FieldSlot> geometry;
    std::string label;
    std::string name = "Field";

    Index nGeometry() const { return static_cast<Index>(geometry.size()); }

    void validate() const
    {
        if (geometry.empty())
            throw std::invalid_argument(
                "EquilibriumSpec: an equilibrium model must declare at least one geometry slot; "
                "geometry is its only channel into the transport physics");
        if (label.empty())
            throw std::invalid_argument("EquilibriumSpec: the spatial label must be named");
        if (name.empty())
            throw std::invalid_argument("EquilibriumSpec: the output group name cannot be empty");
        FieldModelSpec::checkNetCDFName(name, "output group name");
        for (size_t i = 0; i < geometry.size(); ++i)
        {
            if (geometry[i].name.empty())
                throw std::invalid_argument("EquilibriumSpec: an unnamed geometry slot");
            FieldModelSpec::checkNetCDFName(geometry[i].name, "geometry slot");
            for (size_t j = i + 1; j < geometry.size(); ++j)
                if (geometry[i].name == geometry[j].name)
                    throw std::invalid_argument("EquilibriumSpec: duplicate geometry slot name '" +
                                                geometry[i].name + "'");
        }
    }
};

class EquilibriumModel
{
public:
    explicit EquilibriumModel(EquilibriumSpec spec_) : spec(std::move(spec_)) { spec.validate(); }
    virtual ~EquilibriumModel() = default;

    EquilibriumSpec const &getSpec() const { return spec; }
    Index nGeometry() const { return spec.nGeometry(); }

    /// The equilibrium for this transport state.
    ///
    /// `states` are the transport variables at the basis nodes, `points` those
    /// nodes and `weights` the quadrature weights that integrate a nodal
    /// function over the domain. `warm` is this model's own last answer in the
    /// current run, or null on the run's first Solve.
    ///
    /// A failure to converge should throw: inside a residual it is a recoverable
    /// failure, and the integrator cuts its step.
    virtual std::unique_ptr<Equilibrium> Solve(GlobalState const &states,
                                               std::vector<Position> const &points,
                                               Vector const &weights, Time t,
                                               Equilibrium const *warm) = 0;

    /// The geometry slots of `e` at each of `points`: g is (nGeometry, points.size()).
    virtual void Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points) = 0;

    /// How accurately the geometry has to be resolved, relative and absolute,
    /// in the units of the slots. Handed over once per run, from initialize().
    ///
    /// An equilibrium converged no tighter than its own looser tolerance puts
    /// that error into every residual, where the integrator reads it as local
    /// error and cuts its step for it. So the solver asks for well below the
    /// run's own tolerances. Return false if that cannot be met: the run goes
    /// ahead, with a warning naming the model.
    virtual bool setTolerances(double /*relative*/, double /*absolute*/) { return true; }

    /// Called at the start of every run, from initialize()'s unconditional
    /// part. The solver drops its own cached equilibrium there too, so a reused
    /// solver starts from the same cold state as a fresh one.
    virtual void resetForRun() {}

protected:
    EquilibriumSpec spec;
};

#endif // EQUILIBRIUMMODEL_HPP
