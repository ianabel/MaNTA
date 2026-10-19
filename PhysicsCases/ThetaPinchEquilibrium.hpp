#ifndef THETAPINCHEQUILIBRIUM_HPP
#define THETAPINCHEQUILIBRIUM_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/Splines>

#include "FieldModel.hpp"
#include "gridStructures.hpp"

#include <toml.hpp>

#include <vector>

/*
    A diamagnetic theta-pinch in a flux-conserving wall: the first field model
    with physics in it. docs/superpowers/specs/2026-10-19-theta-pinch-field-model-design.md
    is the design.

    The transport label is the normalised axial flux x = Phi / Phi_a. Radial force
    balance with no axial current is p + B^2 / 2 mu0 = C on every surface, and
    the wall fixes the area it encloses, Int_0^1 V' dx = pi a^2, where
    V' = dA/dx = Phi_a / B is the area per unit flux. That integral is what
    determines C. Units are normalised so that 2 mu0 = 1: p is then the local
    beta times the vacuum field squared.

    The unknowns are V' as a cubic B-spline on this model's own uniform grid of
    `Intervals` intervals (Intervals + 3 coefficients), then C. The rows are
    Galerkin against the same B-splines,

        R_m = Int phi_m ( Phi_a^2 / V'^2 + p - C ) dx,     R_C = Int V' dx - pi a^2.

    The pressure is an auxiliary variable of the transport case, whose index is
    `PressureAux`. The rows read it at the transport basis nodes and integrate
    each cell's interpolant of it exactly against the splines (see
    pressureProjection). So this model knows nothing about the transport
    variables beyond that one index.

    The basis comes from Eigen's Splines module, and the field block is solved by
    Eigen's SparseLU. B is banded, three either side, with one border row and
    column, so its LU has no fill to speak of.
 */
class ThetaPinchEquilibrium : public FieldModel
{
public:
    struct Parameters
    {
        Index intervals = 16;
        Index pressureAux = 0;
        double wallRadius = 1.0;
        double vacuumField = 1.0;
    };

    ThetaPinchEquilibrium(toml::value const &config, Grid const &grid);
    ThetaPinchEquilibrium(Parameters const &parameters, Grid const &grid);

    static Parameters readParameters(toml::value const &config);
    static FieldModelSpec buildSpec(Index intervals);

    enum Slot : Index
    {
        Vprime = 0,
        dVprime_dx = 1,
        gradx2 = 2,
        Bfield = 3,
    };

    Index nCoefficients() const { return parameters.intervals + 3; }
    double axialFlux() const { return Phi; }
    double wallArea() const { return wallA; }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &dpsidt,
                       GlobalState const &states, std::vector<Position> const &points,
                       Vector const &weights, Time t) override;

    void Geometry(VectorRef out, Vector const &psi, Position x, Time t) override;
    void dGeometry_dpsi(MatrixRef out, Vector const &psi, Position x, Time t) override;

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &dRdot,
                            MatrixRef dRdpsi, MatrixRef dRddpsidt,
                            Vector const &psi, Vector const &dpsidt,
                            GlobalState const &states, std::vector<Position> const &points,
                            Vector const &weights, Time t) override;

    /// The vacuum state, V' = pi a^2 everywhere and C = B0^2.
    void InitialFieldValue(VectorRef out) override;

    void updateFieldJacobian(MatrixRef dRdpsi, MatrixRef dRddpsidt, double alpha) override;
    void applyB(VectorRef out, Vector const &v) const override;
    void applyBTranspose(VectorRef out, Vector const &v) const override;
    void solveB(VectorRef out, Vector const &rhs) const override;
    void solveBTranspose(VectorRef out, Vector const &rhs) const override;

    // ---- the spline, public for the tests ---------------------------------

    /// The four nonzero cubic B-splines at x, numbers first..first+3: their
    /// values (row 0) and first and second x derivatives (rows 1 and 2).
    struct Local
    {
        Index first = 0;
        Eigen::Array<double, 3, 4> d;
    };
    Local localBasis(Position x) const;

    /// Int_0^x N_n for every coefficient n, exactly.
    Vector antiderivativeWeights(Position x) const;

    /// P(m, j) = Int phi_m l_j dx, l_j the Lagrange basis of node j's transport
    /// cell: Int phi_m p dx = P p for a nodal p. Recomputed when `points` moves.
    Matrix const &pressureProjection(std::vector<Position> const &points);

private:
    using Spline3 = Eigen::Spline<double, 1, 3>;
    using Spline4 = Eigen::Spline<double, 1, 4>;

    double VprimeAt(Vector const &v, Local const &l) const;
    /// Throws if V' is not positive, which no equilibrium has: the residual is
    /// then a recoverable failure and IDA cuts the step.
    static void requirePositive(double Vp, Position x);

    Parameters parameters;
    Grid grid;
    double h;      // field interval width
    double Phi;    // Phi_a = pi a^2 B0
    double wallA;  // pi a^2

    // Knots for the cubic basis, extended by three intervals beyond each end,
    // and for the quartic basis its antiderivative is built from, by four.
    Spline3::KnotVectorType knots3;
    Spline4::KnotVectorType knots4;
    Vector tailAtLower;  // h * sum_{j > n} N4_j(0), subtracted to start the integral at 0
    Vector basisIntegrals; // Int_0^1 N_n

    // The model's own Gauss rule, per field interval.
    std::vector<Position> qx;
    std::vector<double> qw;
    std::vector<Local> qBasis;

    std::vector<Position> projectedPoints;
    Matrix P;

    Eigen::SparseMatrix<double> Bsparse;
    // mutable because SparseLU::transpose() is not const.
    mutable Eigen::SparseLU<Eigen::SparseMatrix<double>> Blu;

    REGISTER_FIELD_MODEL_HEADER(ThetaPinchEquilibrium)
};

#endif // THETAPINCHEQUILIBRIUM_HPP
