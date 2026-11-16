#ifndef GEOMETRYSOURCE_HPP
#define GEOMETRYSOURCE_HPP

#include "EquilibriumModel.hpp"
#include "FieldModel.hpp"
#include "SolverConfig.hpp"
#include "SystemSolver.hpp"
#include "SystemSpec.hpp"

#include <memory>
#include <toml.hpp>

/*
    Where a run's geometry comes from, and how it is coupled: the one place a
    FieldModel or an EquilibriumModel becomes something a SystemSolver can be
    handed. runManta and PyRunner::configure both go through it, so the two
    surfaces resolve FieldCoupling the same way -- the reason applySolverConfig
    exists, for the same kind of key.

    At most one of the two models is the source. A FieldModel run at tier 2
    keeps `field` alone, and its unknowns join the DAE; run at tier 0 or 1 it is
    wrapped, and `equilibrium` is the FieldModelEquilibrium around it. `field`
    is then still set, so nField() can tell the two apart.
 */
struct GeometrySource
{
    std::shared_ptr<FieldModel> field;
    std::shared_ptr<EquilibriumModel> equilibrium;
    SystemSolver::EquilibriumCoupling coupling = SystemSolver::EquilibriumCoupling::Consistent;

    explicit operator bool() const { return field || equilibrium; }

    /// How many unknowns the source adds to the solution vector -- and so how
    /// many trailing entries of a restart file's Y belong to it.
    Index nField() const { return field && !equilibrium ? field->nFieldDOF() : 0; }

    /// Hand the source to a solver. Before applySolverConfig and before the
    /// solver is initialised: setFieldModel reshapes the solution vector.
    void attachTo(SystemSolver &system) const;
};

/// The source a configuration names, from the two registries: FieldModel or
/// EquilibriumModel, each handed `table` -- the whole parsed configuration, so
/// a model reads its own section -- and the grid. Empty when neither is named.
/// Throws std::invalid_argument for an unknown name, a model rejecting its own
/// table, a FieldCoupling with nothing to couple, and every refusal
/// chooseFieldCoupling or chooseEquilibriumCoupling makes.
GeometrySource geometrySourceFromConfig(SolverConfig const &config, toml::value const &table, Grid const &grid,
                                        SystemSpec const &spec);

/// The same resolution for models already built -- a Python EquilibriumModel,
/// say. At most one may be non-null; FieldCoupling is resolved exactly as for a
/// named model.
GeometrySource geometrySourceFor(SolverConfig const &config, SystemSpec const &spec,
                                 std::shared_ptr<FieldModel> field, std::shared_ptr<EquilibriumModel> equilibrium);

/// Refuse a configuration that would build a solver per level -- degree or
/// mesh adaptation, a ladder -- beside a geometry source: the adaptive drivers
/// cannot carry the model, which would silently not be attached.
void refuseAdaptationWithGeometry(SolverConfig const &config, GeometrySource const &source);

#endif // GEOMETRYSOURCE_HPP
