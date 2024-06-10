#include "NonlinAutodiff.hpp"

REGISTER_PHYSICS_IMPL(NonlinAutodiff)

NonlinAutodiff::NonlinAutodiff(toml::value const &config, Grid const &grid) : AutodiffTransportSystem(config, grid, 1, 0)
{

    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    uL.resize(nVars);
    uR.resize(nVars);

    isUpperDirichlet = true;
    isLowerDirichlet = true;

    // Construst your problem from user-specified config
    // throw an exception if you can't. NEVER leave a part-constructed object around
    // here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

    if (config.count("DiffusionProblem") != 1)
        throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the NonlinAutodiff physics model.");

    auto const &DiffConfig = config.at("DiffusionProblem");

    Kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
    Beta = toml::find_or(DiffConfig, "Beta", 0.0);
    InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
    InitialHeight = toml::find_or(DiffConfig, "InitialHeight", 1.0);
    SourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);

    // MMS
    useMMS = toml::find_or(DiffConfig, "UseMMS", false);
    growth = toml::find_or(DiffConfig, "growth", 1.0);
    growth_rate = toml::find_or(DiffConfig, "growth_rate", 0.5);

    uL[0] = InitialFunction(0, xL, 0.0).val.val;
    uR[0] = InitialFunction(0, xR, 0.0).val.val;
}

Real2nd NonlinAutodiff::InitialFunction(Index i, Real2nd x, Real2nd t) const
{
    double center = 0.5 * (xL + xR);
    double shape = 1.0 / InitialWidth;

    return InitialHeight * exp(-shape * (x - center) * (x - center));
}

Real NonlinAutodiff::Flux(Index, RealVector u, RealVector q, Real x, Time t)
{
    double x_mid = 0.5 * (xL + xR);
    Real s = sin(2 * pi * (x - x_mid) / (xR - xL));
    return Kappa * u[0] * (q[0] - Beta * s);
}

Real NonlinAutodiff::Source(Index, RealVector, RealVector, RealVector, Real, Time)
{
    return 0.0;
}

void NonlinAutodiff::initialiseDiagnostics(NetCDFIO &nc)
{
    nc.AddGroup("MMS", "Manufactured solutions");
    nc.AddVariable("MMS", "Var0", "Manufactured solution", "-", [this](double x)
                   { return this->InitialValue(0, x); });
}

void NonlinAutodiff::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{
    Fn s1 = [this, t](double x)
    { return this->MMS_Solution(0, x, t).val.val; };

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", s1}});
}
