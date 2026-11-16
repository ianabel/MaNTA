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

    The unknowns are V' as a C^2 cubic B-spline whose knots are the transport
    mesh's faces, each cell split into `IntervalsPerCell` equal intervals
    (nCells * IntervalsPerCell + 3 coefficients), then C. The rows are Galerkin
    against the same B-splines,

        R_m = Int phi_m ( Phi_a^2 / V'^2 + p - C ) dx,     R_C = Int V' dx - pi a^2.

    The pressure is an auxiliary variable of the transport case, whose index is
    `PressureAux`. The rows read it at the transport basis nodes and integrate
    each cell's interpolant of it exactly against the splines (see
    pressureProjection). So this model knows nothing about the transport
    variables beyond that one index.

    Why the knots follow the mesh, and why the spline stays C^2 across it: p_h
    jumps by O(h^(k+1)) at every transport face. A spline much finer than the
    mesh resolves those jumps, which puts spikes of about jump / (knot spacing)
    into dV'/dx at the transport nodes beside each face, and dV'/dx goes
    straight into the flux. Measured on the manufactured theta pinch
    (ThetaPinchMMSTests.cpp): one or two intervals per cell converge at k + 1
    and four per cell at k = 1 drops to order k; and IDA stalls at order one,
    under the exact field solve as much as the iterative one, at two per cell
    on 64 cells at k = 3 and at sixteen per cell on 8 at k = 6, where at one per
    cell the exact solve never has. Letting V' break at the faces
    instead (repeated knots) is worse at every k: V' then inherits p_h's
    derivative error, O(h^k), cell by cell. Raise IntervalsPerCell only where
    the spline's own O(h^4) is what limits a high-degree run.

    The basis comes from Eigen's Splines module, and the field block is solved by
    Eigen's SparseLU. B is banded, three either side, with one border row and
    column, so its LU has no fill to speak of.
 */
class ThetaPinchEquilibrium : public FieldModel
{
public:
    struct Parameters
    {
        Index intervalsPerCell = 1;
        Index pressureAux = 0;
        double wallRadius = 1.0;
        double vacuumField = 1.0;
    };

    ThetaPinchEquilibrium(toml::value const &config, Grid const &grid);
    ThetaPinchEquilibrium(Parameters const &parameters, Grid const &grid);

    static Parameters readParameters(toml::value const &config);
    static FieldModelSpec buildSpec(Index nCoefficients);
    /// The cubic knot vector: the transport faces, each cell split into
    /// intervalsPerCell intervals, extended three intervals beyond each end.
    static Vector knotVector(Parameters const &parameters, Grid const &grid);

    enum Slot : Index
    {
        Vprime = 0,
        dVprime_dx = 1,
        gradx2 = 2,
        Bfield = 3,
    };

    Index nCoefficients() const { return nCoef; }
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

    double VprimeAt(Vector const &v, Local const &l) const;
    /// Throws if V' is not positive, which no equilibrium has: the residual is
    /// then a recoverable failure and IDA cuts the step.
    static void requirePositive(double Vp, Position x);

    Parameters parameters;
    Grid grid;
    double Phi;    // Phi_a = pi a^2 B0
    double wallA;  // pi a^2

    Spline3::KnotVectorType knots;
    Index nCoef;
    std::vector<double> breaks;  // the distinct knots in [0, 1]
    Matrix cumulative;           // row b: Int_0^{breaks[b]} N_n, for every n
    Vector basisIntegrals;       // Int_0^1 N_n

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
