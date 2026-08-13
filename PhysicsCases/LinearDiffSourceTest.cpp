#include "LinearDiffSourceTest.hpp"
#include <string>

REGISTER_PHYSICS_IMPL(LinearDiffSourceTest);

SystemSpec LinearDiffSourceTest::buildSpec(toml::value const &config)
{
    if (config.count("LinearDiffSourceTest") != 1)
        return {.variables = numberedFields(1)};

    auto const &InternalConfig = config.at("LinearDiffSourceTest");
    const Index nVars = toml::find_or(InternalConfig, "nVars", 1);

    // Per-variable, unlike the single-flag [AutodiffTransportSystem] keys.
    const auto lower = toml::find_or(InternalConfig, "LowerBoundaryConditions", std::vector<bool>(nVars, true));
    const auto upper = toml::find_or(InternalConfig, "UpperBoundaryConditions", std::vector<bool>(nVars, true));

    auto spec = SystemSpec{.variables = numberedFields(nVars)};
    for (Index i = 0; i < nVars; ++i)
    {
        spec.variables[i].lower = lower[i] ? BoundaryKind::Dirichlet : BoundaryKind::Neumann;
        spec.variables[i].upper = upper[i] ? BoundaryKind::Dirichlet : BoundaryKind::Neumann;
    }
    return spec;
}

LinearDiffSourceTest::LinearDiffSourceTest(toml::value const &config, Grid const &grid)
    : AutodiffTransportSystem(config, grid, buildSpec(config))
{
    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    if (config.count("LinearDiffSourceTest") == 1)
    {
        auto const &InternalConfig = config.at("LinearDiffSourceTest");
        if (InternalConfig.count("Kappa") != 1)
            Kappa = Matrix::Identity(nVars, nVars);
        else
        {
            std::vector<double> kappa_vec = toml::find<std::vector<double>>(InternalConfig, "Kappa");
            Kappa = MatrixWrapper(kappa_vec.data(), nVars, nVars);
        }
        uL = toml::find_or(InternalConfig, "uL", std::vector<double>(nVars, 0.0));
        uR = toml::find_or(InternalConfig, "uR", std::vector<double>(nVars, 0.0));

        nSources = toml::find_or(InternalConfig, "nSources", nVars);

        SourceWidth.resize(nSources);
        SourceStrength.resize(nSources);
        SourceCenter.resize(nSources);
        SourceTypes.resize(nSources);
        if (InternalConfig.count("SourceWidth") > 0)
        {
            auto SourceWidth_toml = toml::find(InternalConfig, "SourceWidth");
            SourceWidth = toml::get<std::vector<double>>(SourceWidth_toml);
        }
        else
        {
            SourceWidth = std::vector<double>(nSources, 1.0);
        }
        if (InternalConfig.count("SourceCenter") > 0)
        {
            auto SourceCenter_toml = toml::find(InternalConfig, "SourceCenter");
            SourceCenter = toml::get<std::vector<double>>(SourceCenter_toml);
        }
        else
        {
            SourceCenter = std::vector<double>(nSources, 0.0);
        }
        if (InternalConfig.count("SourceStrength") > 0)
        {
            auto SourceStrength_toml = toml::find(InternalConfig, "SourceStrength");
            SourceStrength = toml::get<std::vector<double>>(SourceStrength_toml);
        }
        else
        {
            SourceStrength = std::vector<double>(nSources, 1.0);
        }
        if (InternalConfig.count("SourceTypes") > 0)
        {
            auto SourceTypes_toml = toml::find<std::vector<int>>(InternalConfig, "SourceTypes");
            for (Index i = 0; i < nSources; i++)
                SourceTypes[i] = static_cast<Sources>(SourceTypes_toml[i]);
        }
        else
        {
            SourceTypes = std::vector<Sources>(nSources, Sources::PeakedEdge);
        }

        loadInitialConditionsFromFile = toml::find_or(InternalConfig, "useNcFile", false);
        if (loadInitialConditionsFromFile)
        {
            if (InternalConfig.count("InitialConditionFilename") > 0)
            {
                filename = toml::find<std::string>(InternalConfig, "InitialConditionFilename");
                LoadDataToSpline(filename);
            }
            else
            {
                throw std::invalid_argument("Please specify a filename to load initial conditions from file.");
            }
        }

        InitialWidth = toml::find_or(InternalConfig, "InitialWidth", std::vector<double>(nVars, 0.1));
        InitialHeight = toml::find_or(InternalConfig, "InitialHeight", std::vector<double>(nVars, 1.0));
    }
    else
    {
        throw std::invalid_argument("To use the Linear Diffusion Source Test physics model, a [LinearDiffSourceTest] configuration section is required.");
    }
}
Real2nd LinearDiffSourceTest::InitialFunction(Index i, Real2nd x, Real2nd t) const
{
    double center = 0.5 * (xL + xR);
    double shape = 1.0 / InitialWidth[i];

    return InitialHeight[i] * exp(-shape * (x - center) * (x - center));
};

Real LinearDiffSourceTest::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{
    RealVector sigma = Kappa * q;
    return sigma(i);
}

Real LinearDiffSourceTest::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t)
{

    return (x > -SourceWidth[i] / 2.0) && (x < SourceWidth[i] / 2.0) ? SourceStrength[i] : 0.0;
    // for (auto j = 0; j < nSources; ++j)
    // {
    //     auto s = SourceTypes[j];
    //     double shape = 1.0 / SourceWidth[j];
    //     switch (s)
    //     {
    //     case Sources::PeakedEdge:
    //         S += SourceStrength[j] * (exp(-shape * (x - xL) * (x - xL)) + exp(-shape * (x - xR) * (x - xR)));
    //         break;
    //     case Sources::Gaussian:
    //         S += SourceStrength[j] * exp(-shape * (x - SourceCenter[j]) * (x - SourceCenter[j]));
    //         break;
    //     case Sources::Uniform:
    //         S += SourceStrength[j];
    //         break;
    //     case Sources::Step:
    //         S += x < SourceCenter[j] ? 0.0 : SourceStrength[j];
    //         break;
    //     default:
    //         break;
    //     }
    // }
    // return S;
}

Value LinearDiffSourceTest::LowerBoundary(Index i, Time t) const
{
    return uL[i];
}
Value LinearDiffSourceTest::UpperBoundary(Index i, Time t) const
{
    return uR[i];
}

void LinearDiffSourceTest::initialiseDiagnostics(NetCDFIO &nc)
{
    nc.AddGroup("InitialProfile", "Initial profile, for reference");
    for (int j = 0; j < nVars; ++j)
        nc.AddVariable("InitialProfile", "Var" + std::to_string(j), "Initial profile", "-", [this, j](double V)
                       { return this->InitialFunction(j, V, 0.0).val.val; });
}

void LinearDiffSourceTest::writeDiagnostics(DGSoln const &y, DGSoln const&, Time t, NetCDFIO &nc, size_t tIndex)
{
    for (Index j = 0; j < nVars; ++j)
        nc.AppendToGroup("InitialProfile", tIndex, "Var" + std::to_string(j), [this, j, t](double x)
                         { return this->InitialFunction(j, x, t).val.val; });
}
