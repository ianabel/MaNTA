#include "ThetaPinchDiffusion.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

REGISTER_PHYSICS_IMPL(ThetaPinchDiffusion);

namespace
{
SystemSpec thetaPinchSpec()
{
    return {.variables = {{"N", "particles per unit flux per unit length, V' n", "n0 a^2",
                           BoundaryKind::Neumann, BoundaryKind::Dirichlet}},
            .aux = {{"p", "pressure, in units where 2 mu0 = 1", "B0^2"}},
            .regrid = RegridPolicy::InPlace};
}
} // namespace

// RegridPolicy::InPlace: nothing here depends on the grid.
ThetaPinchDiffusion::ThetaPinchDiffusion(toml::value const &config, Grid const &grid)
    : AutodiffTransportSystem(config, grid, thetaPinchSpec())
{
    toml::value table = toml::table{};
    if (config.is_table() && config.contains("ThetaPinchDiffusion"))
        table = config.at("ThetaPinchDiffusion");

    T = toml::find_or(table, "Temperature", 0.1);
    D0 = toml::find_or(table, "D0", 1.0);
    Bref = toml::find_or(table, "Bref", 1.0);
    gamma = toml::find_or(table, "Gamma", 1.0);
    sourceStrength = toml::find_or(table, "SourceStrength", 1.0);
    sourceWidth = toml::find_or(table, "SourceWidth", 0.3);
    edgeDensity = toml::find_or(table, "EdgeDensity", 0.1);
    centralDensity = toml::find_or(table, "CentralDensity", 1.0);
    wallRadius = toml::find_or(table, "WallRadius", 1.0);
    vacuumField = toml::find_or(table, "VacuumField", 1.0);
    vacuumGeometry = toml::find_or(table, "VacuumGeometry", false);

    if (!(T >= 0.0) || !(D0 > 0.0) || !(Bref > 0.0) || !(sourceWidth > 0.0) || !(wallRadius > 0.0) ||
        !(vacuumField > 0.0))
        throw std::invalid_argument("ThetaPinchDiffusion: Temperature must be non-negative, and D0, Bref, "
                                    "SourceWidth, WallRadius and VacuumField positive");
    if (grid.lowerBoundary() != 0.0 || grid.upperBoundary() != 1.0)
        throw std::invalid_argument(
            "ThetaPinchDiffusion: x is the normalised flux Phi / Phi_wall, so the grid must be [0, 1]");

    addP(D0);
    addP(T);
}

RealVector ThetaPinchDiffusion::metric(RealVector const &geom, Real x) const
{
    if (geom.size() >= 4)
        return geom.head(4);
    if (geom.size() > 0)
        throw std::invalid_argument("ThetaPinchDiffusion: the field model provides " +
                                    std::to_string(geom.size()) +
                                    " geometry slots; this case reads four, V', dV'/dx, <|grad x|^2> and B");
    if (!vacuumGeometry)
        throw std::invalid_argument("ThetaPinchDiffusion: no field model is attached. Set FieldModel = "
                                    "\"ThetaPinchEquilibrium\", or VacuumGeometry = true in "
                                    "[ThetaPinchDiffusion] to run on the vacuum metric");
    // A uniform B0 in a wall of radius a: x = r^2 / a^2.
    const double a2 = wallRadius * wallRadius;
    RealVector g(4);
    g(Vprime) = M_PI * a2;
    g(dVprime_dx) = 0.0;
    g(gradx2) = 4.0 * x / a2;
    g(Bfield) = vacuumField;
    return g;
}

Value ThetaPinchDiffusion::LowerBoundary(Index, Time) const { return 0.0; }

Value ThetaPinchDiffusion::UpperBoundary(Index, Time) const
{
    return M_PI * wallRadius * wallRadius * edgeDensity;
}

Real ThetaPinchDiffusion::Flux(Index, RealVector, RealVector, Real, Time)
{
    throw std::logic_error("ThetaPinchDiffusion reads geometry; its geometry-blind Flux is never meaningful");
}

Real ThetaPinchDiffusion::Flux(Index, RealVector u, RealVector q, RealVector geom, Real x, Time)
{
    const RealVector g = metric(geom, x);
    const Real Vp = g(Vprime);
    const Real n = u(0) / Vp;
    const Real dn_dx = (q(0) - n * g(dVprime_dx)) / Vp;
    const Real D = D0 * pow(Bref / g(Bfield), gamma);
    return Vp * D * g(gradx2) * dn_dx;
}

Real ThetaPinchDiffusion::Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time)
{
    throw std::logic_error("ThetaPinchDiffusion reads geometry; its geometry-blind Source is never meaningful");
}

Real ThetaPinchDiffusion::Source(Index, RealVector, RealVector, RealVector, RealVector, RealVector,
                                 RealVector geom, Real x, Time)
{
    const RealVector g = metric(geom, x);
    return g(Vprime) * sourceStrength * exp(-x * x / (sourceWidth * sourceWidth));
}

Real ThetaPinchDiffusion::GFunc(Index, RealVector, RealVector, RealVector, RealVector, Position, Time)
{
    throw std::logic_error("ThetaPinchDiffusion reads geometry; its geometry-blind GFunc is never meaningful");
}

Real ThetaPinchDiffusion::GFunc(Index, RealVector u, RealVector, RealVector, RealVector phi, RealVector geom,
                                Position x, Time)
{
    const RealVector g = metric(geom, x);
    return phi(0) - T * u(0) / g(Vprime);
}

// N on the vacuum metric, V' = pi a^2, from a parabolic density.
autodiff::dual2nd ThetaPinchDiffusion::InitialFunction(Index, autodiff::dual2nd x, autodiff::dual2nd) const
{
    return M_PI * wallRadius * wallRadius * (edgeDensity + (centralDensity - edgeDensity) * (1.0 - x * x));
}

// A guess, on the vacuum metric: p = T n. The pressure is algebraic, so
// IDACalcIC (or the steady solve) corrects it together with the field.
Value ThetaPinchDiffusion::InitialAuxValue(Index, Position x, Time) const
{
    return T.val * (edgeDensity + (centralDensity - edgeDensity) * (1.0 - x * x));
}
