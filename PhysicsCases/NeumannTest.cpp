
#include "NeumannTest.hpp"

/*
    Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(NeumannTest);

SystemSpec NeumannTest::buildSpec(toml::value const &config)
{
    bool lowerNeumann = false, upperNeumann = false;
    if (config.count("DiffusionProblem") == 1)
    {
        auto const &DiffConfig = config.at("DiffusionProblem");
        lowerNeumann = toml::find_or(DiffConfig, "LowerNeumann", false);
        upperNeumann = toml::find_or(DiffConfig, "UpperNeumann", false);
    }
    return {.variables = numberedFields(1,
                                        lowerNeumann ? BoundaryKind::Neumann : BoundaryKind::Dirichlet,
                                        upperNeumann ? BoundaryKind::Neumann : BoundaryKind::Dirichlet)};
}

NeumannTest::NeumannTest(toml::value const &config, Grid const &grid)
    : TransportSystem(buildSpec(config))
{
    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    // Construct your problem from user-specified config
    // throw an exception if you can't. NEVER leave a part-constructed object around
    // here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

    if (config.count("DiffusionProblem") != 1)
        throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the NeumannTest physics model.");

    auto const &DiffConfig = config.at("DiffusionProblem");

    kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
    InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
    InitialHeight = toml::find_or(DiffConfig, "InitialHeight", 1.0);
    Centre = toml::find_or(DiffConfig, "Centre", 0.5);
    SourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);

    growth = toml::find_or(DiffConfig, "growth", 1.0);
    growth_rate = toml::find_or(DiffConfig, "growth_rate", 0.5);

}

// A Neumann boundary fixes the flux, a Dirichlet one fixes u, so which of the
// two this returns follows from the boundary kind. That now comes from the spec
// rather than from a second pair of bools parsed out of the same config keys.
Value NeumannTest::LowerBoundary(Index, Time) const
{
    if (!isLowerBoundaryDirichlet(0))
        return InitialDerivative(0, xL);
    else
        return InitialValue(0, xL);
}

Value NeumannTest::UpperBoundary(Index, Time) const
{
    if (!isUpperBoundaryDirichlet(0))
        return InitialDerivative(0, xR);
    else
        return InitialValue(0, xR);
}

Value NeumannTest::SigmaFn(Index, const State &s, Position x, Time)
{
    return kappa * s.q(0);
}

Value NeumannTest::Sources(Index, const State &s, Position x, Time t)
{
    // double u = s.u(0);
    double S = SourceStrength;

    return S;
}

void NeumannTest::dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time)
{
    v[0] = kappa;
};

void NeumannTest::dSigmaFn_du(Index, VectorRef v, const State &, Position, Time)
{
    v[0] = 0.0;
};

void NeumannTest::dSources_du(Index, VectorRef v, const State &, Position, Time)
{
    v[0] = 0.0;
};

void NeumannTest::dSources_dq(Index, VectorRef v, const State &, Position, Time)
{
    v[0] = 0.0;
};

void NeumannTest::dSources_dsigma(Index, VectorRef v, const State &, Position, Time)
{
    v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value NeumannTest::InitialValue(Index, Position x) const
{
    double alpha = 1 / InitialWidth;
    double y = (x - Centre);
    return InitialHeight * ::exp(-alpha * y * y);
}

Value NeumannTest::InitialDerivative(Index, Position x) const
{
    double y = (x - Centre);
    double alpha = 1 / InitialWidth;
    return -InitialHeight * (2.0 * y) * ::exp(-alpha * y * y) * alpha;
}
