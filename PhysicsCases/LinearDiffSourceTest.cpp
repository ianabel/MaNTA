#include "LinearDiffSourceTest.hpp"

REGISTER_PHYSICS_IMPL(LinearDiffSourceTest);

LinearDiffSourceTest::LinearDiffSourceTest(toml::value const &config, Grid const &grid)
{
    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    nVars = toml::find<int>(config.at("configuration"), "nVars");
    nScalars = 0;

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

        SourceWidth = toml::find_or(InternalConfig, "SourceWidth", std::vector<double>(nVars, 0.1));
        SourceStrength = toml::find_or(InternalConfig, "SourceStrength", std::vector<double>(nVars, 1.0));
        SourceCenter = toml::find_or(InternalConfig, "SourceCenter", std::vector<double>(nVars, 0.0));

        std::vector<std::string> sourceType = toml::find_or(InternalConfig, "SourceTypes", std::vector<std::string>(nVars, "PeakedEdge"));
        for (auto &s : sourceType)
            SourceTypes.push_back(SourceMap[s]);

        InitialWidth = toml::find_or(InternalConfig, "InitialWidth", std::vector<double>(nVars, 0.1));
        InitialHeight = toml::find_or(InternalConfig, "InitialHeight", std::vector<double>(nVars, 1.0));
        lowerBoundaryConditions = toml::find_or(InternalConfig, "LowerBoundaryConditions", std::vector<bool>(nVars, true));
        upperBoundaryConditions = toml::find_or(InternalConfig, "UpperBoundaryConditions", std::vector<bool>(nVars, true));
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

Real LinearDiffSourceTest::Flux(Index i, RealVector u, RealVector q, Position x, Time t)
{
    RealVector sigma = Kappa * q;
    return sigma(i);
}

Real LinearDiffSourceTest::Source(Index i, RealVector u, RealVector q, RealVector sigma, Position x, Time t)
{
    auto s = SourceTypes[i];
    double shape = 1.0 / SourceWidth[i];
    switch (s)
    {
    case Sources::PeakedEdge:
        return SourceStrength[i] * (exp(-shape * (x - xL) * (x - xL)) + exp(-shape * (x - xR) * (x - xR)));
    case Sources::Gaussian:
        return SourceStrength[i] * exp(-shape * (x - SourceCenter[i]) * (x - SourceCenter[i]));
    case Sources::Uniform:
        return SourceStrength[i];
    default:
        return 0.0;
    }
}

Value LinearDiffSourceTest::LowerBoundary(Index i, Time t) const
{
    return uL[i];
}
Value LinearDiffSourceTest::UpperBoundary(Index i, Time t) const
{
    return uR[i];
}

bool LinearDiffSourceTest::isLowerBoundaryDirichlet(Index i) const
{
    return lowerBoundaryConditions[i];
}

bool LinearDiffSourceTest::isUpperBoundaryDirichlet(Index i) const
{
    return upperBoundaryConditions[i];
}