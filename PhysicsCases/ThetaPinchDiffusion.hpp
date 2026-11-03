#ifndef THETAPINCHDIFFUSION_HPP
#define THETAPINCHDIFFUSION_HPP

#include "AutodiffTransportSystem.hpp"

/*
    Isothermal density transport in a theta-pinch, on the normalised flux label
    x = Phi / Phi_wall, reading its geometry from a field model
    (ThetaPinchEquilibrium). docs/superpowers/specs/2026-10-19-theta-pinch-field-model-design.md
    is the design.

    The evolved variable is N = V' n, particles per unit flux per unit length.
    With labels that move with the field, continuity is

        d_t N + d_x ( V' <Gamma . grad x> ) = V' S,      <Gamma . grad x> = -D <|grad x|^2> d_x n,

    which stays exactly conservative while V' changes in time, because V' n is
    what sits under d_t. In MaNTA's sign convention the flux is therefore

        sigma_hat = V' D <|grad x|^2> d_x n,   n = N / V',   d_x n = (q - n V'_x) / V',
        D = D0 (Bref / B)^gamma.

    The auxiliary variable is the pressure the field model reads,
    G = p - T N / V' = 0, with T constant.

    Geometry is read by slot index, in ThetaPinchEquilibrium's order: V', dV'/dx,
    <|grad x|^2>, B. A run without a field model is refused unless
    VacuumGeometry = true, which substitutes the vacuum metric (uniform B0 in a
    wall of radius a), so a frozen-geometry run is a choice rather than an
    accident.

    The boundaries are zero flux at the axis -- V' <|grad x|^2> vanishes there,
    so that is the only consistent choice -- and Dirichlet on N at the wall.
    The datum is N, not n, because a boundary value cannot read geometry; as
    V'(1) moves, the edge density moves with it.
 */
class ThetaPinchDiffusion : public AutodiffTransportSystem
{
public:
    ThetaPinchDiffusion(toml::value const &config, Grid const &grid);

    // Slots, in the field model's order.
    static constexpr Index Vprime = 0, dVprime_dx = 1, gradx2 = 2, Bfield = 3;

    // The adjoint parameters, as AutodiffTransportSystem numbers them.
    static constexpr Index pD0 = 0, pTemperature = 1;

protected:
    // Protected rather than private for the manufactured-solution study
    // (ThetaPinchMMSTests.cpp), which replaces the source, the wall datum and the
    // initial state and keeps the flux, the aux row and the parameters.
    Value LowerBoundary(Index, Time) const override;
    Value UpperBoundary(Index, Time) const override;

    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Flux(Index, RealVector u, RealVector q, RealVector geom, Real x, Time t) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;
    Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars,
                RealVector geom, Real x, Time t) override;
    Real GFunc(Index, RealVector, RealVector, RealVector, RealVector, Position, Time) override;
    Real GFunc(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector geom,
               Position x, Time t) override;

    autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t) const override;
    Value InitialAuxValue(Index, Position, Time) const override;

    /// The four slots, from the field model, or the vacuum metric if allowed.
    RealVector metric(RealVector const &geom, Real x) const;

    // Adjoint parameters, in this order: D0 enters the flux, T only the
    // pressure the field rows read, so dG/dT reaches the transport solely
    // through the field.
    Real D0, T;
    double Bref, gamma;
    double sourceStrength, sourceWidth;
    double edgeDensity, centralDensity;
    double wallRadius, vacuumField;
    bool vacuumGeometry;

    REGISTER_PHYSICS_HEADER(ThetaPinchDiffusion)
};

#endif // THETAPINCHDIFFUSION_HPP
