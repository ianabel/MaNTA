#include "GeometrySource.hpp"
#include "FieldModelEquilibrium.hpp"

void GeometrySource::attachTo(SystemSolver &system) const
{
    if (equilibrium)
        system.setEquilibriumModel(equilibrium, coupling);
    else if (field)
        system.setFieldModel(field);
}

GeometrySource geometrySourceFor(SolverConfig const &config, SystemSpec const &spec,
                                 std::shared_ptr<FieldModel> field, std::shared_ptr<EquilibriumModel> equilibrium)
{
    if (field && equilibrium)
        throw std::invalid_argument("A run takes its geometry from one source: a field model or an "
                                    "equilibrium model, not both.");

    GeometrySource source;
    if (field)
    {
        // A differential field unknown, or a row reading du/dt: either way the
        // field is not a function of the transport state.
        bool dynamics = field->getSpec().readsTransportTimeDerivatives;
        for (Index i = 0; i < field->nFieldDOF(); ++i)
            dynamics = dynamics || field->isFieldDOFDifferential(i);
        // The consistent and reduced modes run the same model as an
        // equilibrium: no unknowns in the state vector, and the wrapper refuses
        // a model whose field has dynamics, before anything else is built.
        const FieldCouplingMode mode = chooseFieldCoupling(config, spec, dynamics);
        source.field = std::move(field);
        if (mode != FieldCouplingMode::Coupled)
            source.equilibrium = std::make_shared<FieldModelEquilibrium>(source.field);
        if (mode == FieldCouplingMode::Reduced)
            source.coupling = SystemSolver::EquilibriumCoupling::Reduced;
    }
    else if (equilibrium)
    {
        const bool differenced = equilibrium->getSpec().response == EquilibriumResponse::Differenced;
        if (chooseEquilibriumCoupling(config, spec, differenced) == FieldCouplingMode::Reduced)
            source.coupling = SystemSolver::EquilibriumCoupling::Reduced;
        source.equilibrium = std::move(equilibrium);
    }
    else if (config.FieldCoupling != "auto")
        throw std::invalid_argument("FieldCoupling = \"" + config.FieldCoupling +
                                    "\" was given without a FieldModel or EquilibriumModel to couple.");
    return source;
}

GeometrySource geometrySourceFromConfig(SolverConfig const &config, toml::value const &table, Grid const &grid,
                                        SystemSpec const &spec)
{
    // loadSolverConfig has already refused both keys together.
    std::shared_ptr<FieldModel> field;
    std::shared_ptr<EquilibriumModel> equilibrium;
    try
    {
        if (!config.FieldModel.empty())
            field = FieldModels::InstantiateFieldModel(config.FieldModel, table, grid);
    }
    catch (std::invalid_argument const &e)
    {
        throw std::invalid_argument("Could not instantiate a field model for FieldModel = " + config.FieldModel +
                                    ": " + e.what());
    }
    try
    {
        if (!config.EquilibriumModel.empty())
            equilibrium = EquilibriumModels::InstantiateEquilibriumModel(config.EquilibriumModel, table, grid);
    }
    catch (std::invalid_argument const &e)
    {
        throw std::invalid_argument("Could not instantiate an equilibrium model for EquilibriumModel = " +
                                    config.EquilibriumModel + ": " + e.what());
    }
    return geometrySourceFor(config, spec, std::move(field), std::move(equilibrium));
}

void refuseAdaptationWithGeometry(SolverConfig const &config, GeometrySource const &source)
{
    if (source && (config.DegreeAdaptation || config.MeshAdaptation || !config.DegreeLadder.empty() ||
                   !config.GridLadder.empty()))
        throw std::invalid_argument(
            "DegreeAdaptation and MeshAdaptation cannot be combined with a FieldModel or an "
            "EquilibriumModel: the adaptive drivers build a solver per level and cannot carry the model.");
}
