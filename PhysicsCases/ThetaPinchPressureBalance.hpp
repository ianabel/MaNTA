#ifndef THETAPINCHPRESSUREBALANCE_HPP
#define THETAPINCHPRESSUREBALANCE_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include "EquilibriumModel.hpp"
#include "gridStructures.hpp"

#include <toml.hpp>

#include <vector>

/*
    The theta-pinch equilibrium of ThetaPinchEquilibrium, as a native
    EquilibriumModel: radial pressure balance solved in closed form rather than
    projected onto a spline.

    The physics is the same. On every flux surface p + B^2 = C (2 mu0 = 1), and
    with B = Phi_a / V' that is V' = Phi_a / sqrt(C - p), pointwise; the wall
    fixes the enclosed area, Int_0^1 V' dx = pi a^2, which is one equation for
    the one constant C. So the equilibrium of a pressure profile is a scalar
    root-find, and the geometry slots follow from p and C directly:

        V'      = Phi_a / sqrt(C - p)
        dV'/dx  = (Phi_a / 2) p_x (C - p)^(-3/2)
        <|grad x|^2> = 4 pi A / V'^2,    A(x) = Int_0^x V'
        B       = Phi_a / V'

    What differs is the discretisation. ThetaPinchEquilibrium holds V' as a C^2
    cubic spline on the transport faces and imposes the balance by Galerkin
    projection, so its V' is smooth across faces and its dV'/dx is the spline's
    derivative. Here p is the transport's own piecewise polynomial p_h -- each
    cell's interpolant of the pressure at its basis nodes -- and V' is the
    balance evaluated on it exactly, so V' is discontinuous at the faces
    wherever p_h is. The integrals (the wall constraint and A) are per cell, by a
    16-point Gauss rule on a smooth integrand.

    p_x is not p_h's derivative, which is good only to O(h^k) and would cost the
    transport an order: dV'/dx goes straight into the flux. It is the derivative
    of a degree-(k + 1) polynomial fitted, by least squares, to the cell means of
    p_h over a centred patch of 2 ceil((k + 1) / 2) + 1 cells, moved inward at
    the ends. Cell means, not nodal values: p_h's interpolation error repeats its
    shape in every cell, a fit to the values differentiates that shape (at k = 1
    it decays to order 1), and a mean carries the error only as a smooth
    function. On an exact pressure dV'/dx then converges at 2.00, 2.99 and 5.00
    for k = 1, 2, 3, and the manufactured theta pinch at k + 1 throughout
    (ThetaPinchMMSTests.cpp, EquilibriumRegistryTests.cpp).

    The response G' is exact (EquilibriumResponse::Model): the same evaluation
    run in forward-mode duals, with dC from the derivative of the wall
    constraint, dC = Int (C-p)^(-3/2) dp / Int (C-p)^(-3/2). Its transpose, for
    the tier-1 adjoint, is exact too: the pressure columns of G' by one dual
    sweep each, assembled once per equilibrium and transposed, with no Solve.
    Parameters::differencedResponse falls back to the base class's forward
    difference of Solve and its assembled transpose, as FieldModelEquilibrium's
    flag does.

    Its table is [ThetaPinchPressureBalance]: PressureAux, WallRadius,
    VacuumField and DifferencedResponse, an unknown key refused.
 */
class ThetaPinchPressureBalance : public EquilibriumModel
{
public:
    struct Parameters
    {
        Index pressureAux = 0;
        double wallRadius = 1.0;
        double vacuumField = 1.0;
        bool differencedResponse = false;
    };

    ThetaPinchPressureBalance(toml::value const &config, Grid const &grid);
    ThetaPinchPressureBalance(Parameters const &parameters, Grid const &grid);

    static Parameters readParameters(toml::value const &config);
    static EquilibriumSpec buildSpec(bool differencedResponse);

    // Slots, in ThetaPinchEquilibrium's order, which is what ThetaPinchDiffusion reads.
    enum Slot : Index
    {
        Vprime = 0,
        dVprime_dx = 1,
        gradx2 = 2,
        Bfield = 3,
    };

    std::unique_ptr<Equilibrium> Solve(GlobalState const &states, std::vector<Position> const &points,
                                       Vector const &weights, Time t, Equilibrium const *warm) override;

    void Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points) override;

    void applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states, GlobalState const &dp,
                       std::vector<Position> const &inputPoints, Vector const &weights, Time t,
                       std::vector<Position> const &geometryPoints) override;

    void applyResponseTranspose(GlobalState &dp, Equilibrium const &e, GlobalState const &states,
                                Matrix const &dg, std::vector<Position> const &inputPoints,
                                Vector const &weights, Time t,
                                std::vector<Position> const &geometryPoints) override;

    /// The total pressure C of an equilibrium this model produced.
    static double totalPressure(Equilibrium const &e);

    /// Solve calls over the model's lifetime, including the ones the
    /// differenced response makes: the field-side cost.
    long solves() const { return nSolves; }

    // The root-find stops at a step this small relative to C.
    static constexpr double relativeStep = 1e-15;
    static constexpr int maxIterations = 200;

private:
    /// dg = G' dp for a direction dp in the nodal pressures alone.
    void pressureResponse(MatrixRef dg, Equilibrium const &e, Vector const &dp,
                          std::vector<Position> const &geometryPoints) const;

    Parameters parameters;
    Grid grid;
    double Phi;   // Phi_a = pi a^2 B0
    double wallA; // pi a^2
    long nSolves = 0;

    REGISTER_EQUILIBRIUM_MODEL_HEADER(ThetaPinchPressureBalance)
};

#endif // THETAPINCHPRESSUREBALANCE_HPP
