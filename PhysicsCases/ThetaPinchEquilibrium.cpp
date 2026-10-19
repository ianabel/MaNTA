#include "ThetaPinchEquilibrium.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

REGISTER_FIELD_MODEL_IMPL(ThetaPinchEquilibrium)

namespace
{
// Eight points integrate a polynomial of degree 15 exactly, which covers the
// pressure projection's cubic-times-degree-k integrand up to k = 12.
using Gauss = boost::math::quadrature::gauss<double, 8>;
constexpr Index gaussExactDegree = 15;

/// The 8-point Gauss-Legendre rule mapped onto [a, b].
void gaussOn(double a, double b, std::vector<double> &x, std::vector<double> &w)
{
    x.clear();
    w.clear();
    const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
    for (size_t i = 0; i < Gauss::abscissa().size(); ++i)
    {
        const double xi = Gauss::abscissa()[i], wi = Gauss::weights()[i];
        x.push_back(mid - half * xi);
        w.push_back(half * wi);
        x.push_back(mid + half * xi);
        w.push_back(half * wi);
    }
}
} // namespace

ThetaPinchEquilibrium::Parameters ThetaPinchEquilibrium::readParameters(toml::value const &config)
{
    Parameters p;
    if (!config.is_table() || !config.contains("ThetaPinchEquilibrium"))
        return p;
    auto const &table = config.at("ThetaPinchEquilibrium");
    p.intervals = toml::find_or(table, "Intervals", p.intervals);
    p.pressureAux = toml::find_or(table, "PressureAux", p.pressureAux);
    p.wallRadius = toml::find_or(table, "WallRadius", p.wallRadius);
    p.vacuumField = toml::find_or(table, "VacuumField", p.vacuumField);
    return p;
}

FieldModelSpec ThetaPinchEquilibrium::buildSpec(Index intervals)
{
    if (intervals < 1)
        throw std::invalid_argument("ThetaPinchEquilibrium: Intervals must be at least 1, not " +
                                    std::to_string(intervals));
    FieldModelSpec s;
    for (Index n = 0; n < intervals + 3; ++n)
        s.dofs.push_back({"v" + std::to_string(n), "cubic B-spline coefficient of V'", "a^2", false});
    s.dofs.push_back({"C", "total pressure p + B^2, with 2 mu0 = 1", "B0^2", false});
    s.geometry = {{"Vprime", "area per unit flux label, dA/dx", "a^2"},
                  {"dVprime_dx", "d V' / dx", "a^2"},
                  {"gradx2", "<|grad x|^2>", "a^-2"},
                  {"B", "axial magnetic field", "B0"}};
    s.label = "Phi/Phi_wall";
    s.name = "ThetaPinch";
    return s;
}

ThetaPinchEquilibrium::ThetaPinchEquilibrium(toml::value const &config, Grid const &grid_)
    : ThetaPinchEquilibrium(readParameters(config), grid_)
{
}

ThetaPinchEquilibrium::ThetaPinchEquilibrium(Parameters const &parameters_, Grid const &grid_)
    : FieldModel(buildSpec(parameters_.intervals)), parameters(parameters_), grid(grid_)
{
    if (parameters.pressureAux < 0)
        throw std::invalid_argument("ThetaPinchEquilibrium: PressureAux must be an auxiliary variable index");
    if (!(parameters.wallRadius > 0.0) || !(parameters.vacuumField > 0.0))
        throw std::invalid_argument("ThetaPinchEquilibrium: WallRadius and VacuumField must be positive");
    if (grid.lowerBoundary() != 0.0 || grid.upperBoundary() != 1.0)
        throw std::invalid_argument(
            "ThetaPinchEquilibrium: x is the normalised flux Phi / Phi_wall, so the grid must be [0, 1]");

    const Index M = parameters.intervals;
    h = 1.0 / static_cast<double>(M);
    wallA = M_PI * parameters.wallRadius * parameters.wallRadius;
    Phi = wallA * parameters.vacuumField;

    knots3.resize(M + 7);
    for (Index i = 0; i < M + 7; ++i)
        knots3(i) = static_cast<double>(i - 3) * h;
    knots4.resize(M + 9);
    for (Index i = 0; i < M + 9; ++i)
        knots4(i) = static_cast<double>(i - 4) * h;

    tailAtLower = Vector::Zero(nCoefficients());
    tailAtLower = antiderivativeWeights(0.0); // with tailAtLower zero, this is the raw tail
    basisIntegrals = antiderivativeWeights(1.0);

    std::vector<double> x, w;
    for (Index c = 0; c < M; ++c)
    {
        gaussOn(c * h, (c + 1) * h, x, w);
        for (size_t q = 0; q < x.size(); ++q)
        {
            qx.push_back(x[q]);
            qw.push_back(w[q]);
            qBasis.push_back(localBasis(x[q]));
        }
    }
}

// ---- the spline -------------------------------------------------------------

ThetaPinchEquilibrium::Local ThetaPinchEquilibrium::localBasis(Position x) const
{
    x = std::clamp(x, 0.0, 1.0);
    Local l;
    l.first = Spline3::Span(x, 3, knots3) - 3;
    l.d = Spline3::BasisFunctionDerivatives(x, 2, 3, knots3);
    return l;
}

// The antiderivative of a cubic B-spline is a sum of quartic ones on the same
// knots (de Boor): Int_{-inf}^x N3_n = h * sum_{j >= n+1} N4_j(x), the shift of
// one because knots4 starts an interval earlier than knots3. The quartic basis
// is a partition of unity on [0, 1], so the tail sum is 1 for every j left of
// the four that are nonzero at x and 0 for every j right of them.
Vector ThetaPinchEquilibrium::antiderivativeWeights(Position x) const
{
    x = std::clamp(x, 0.0, 1.0);
    const Index span = Spline4::Span(x, 4, knots4);
    const auto N4 = Spline4::BasisFunctions(x, 4, knots4);
    const Index first4 = span - 4;

    Vector out(nCoefficients());
    for (Index n = 0; n < nCoefficients(); ++n)
    {
        const Index j = n + 1;
        double tail = 0.0;
        if (j <= first4)
            tail = 1.0;
        else if (j <= span)
            for (Index r = j - first4; r <= 4; ++r)
                tail += N4(r);
        out(n) = h * tail - tailAtLower(n);
    }
    return out;
}

double ThetaPinchEquilibrium::VprimeAt(Vector const &v, Local const &l) const
{
    double Vp = 0.0;
    for (Index r = 0; r < 4; ++r)
        Vp += v(l.first + r) * l.d(0, r);
    return Vp;
}

void ThetaPinchEquilibrium::requirePositive(double Vp, Position x)
{
    if (!(Vp > 0.0))
        throw std::runtime_error("ThetaPinchEquilibrium: V' = " + std::to_string(Vp) + " at x = " +
                                 std::to_string(x) + ", and no equilibrium has a non-positive V'");
}

// ---- the pressure -----------------------------------------------------------

Matrix const &ThetaPinchEquilibrium::pressureProjection(std::vector<Position> const &points)
{
    if (points == projectedPoints)
        return P;

    const Index nCells = static_cast<Index>(grid.getNCells());
    const Index nPoints = static_cast<Index>(points.size());
    if (nPoints % nCells != 0)
        throw std::invalid_argument("ThetaPinchEquilibrium: " + std::to_string(nPoints) +
                                    " transport nodes do not divide among " + std::to_string(nCells) +
                                    " cells");
    const Index nPer = nPoints / nCells;
    if (3 + (nPer - 1) > gaussExactDegree)
        throw std::invalid_argument("ThetaPinchEquilibrium: the pressure projection is exact only up to "
                                    "polynomial degree 12 in the transport");

    P = Matrix::Zero(nCoefficients(), nPoints);
    std::vector<double> x, w;
    for (Index i = 0; i < nCells; ++i)
    {
        const Interval I = grid[i];
        // Split the cell at every field knot inside it, so each piece is a
        // polynomial in x and the Gauss rule integrates it exactly.
        std::vector<double> cuts{I.x_l};
        for (Index c = 1; c < parameters.intervals; ++c)
            if (c * h > I.x_l && c * h < I.x_u)
                cuts.push_back(c * h);
        cuts.push_back(I.x_u);

        for (size_t s = 0; s + 1 < cuts.size(); ++s)
        {
            gaussOn(cuts[s], cuts[s + 1], x, w);
            for (size_t q = 0; q < x.size(); ++q)
            {
                const Local l = localBasis(x[q]);
                for (Index j = 0; j < nPer; ++j)
                {
                    double lj = 1.0;
                    const Position xj = points[i * nPer + j];
                    for (Index m = 0; m < nPer; ++m)
                        if (m != j)
                            lj *= (x[q] - points[i * nPer + m]) / (xj - points[i * nPer + m]);
                    for (Index r = 0; r < 4; ++r)
                        P(l.first + r, i * nPer + j) += w[q] * l.d(0, r) * lj;
                }
            }
        }
    }
    projectedPoints = points;
    return P;
}

// ---- residual and derivatives ----------------------------------------------

void ThetaPinchEquilibrium::FieldResidual(VectorRef out, Vector const &psi, Vector const &,
                                          GlobalState const &states, std::vector<Position> const &points,
                                          Vector const &, Time)
{
    const Index nB = nCoefficients();
    const Vector v = psi.head(nB);
    const double C = psi(nB);

    for (size_t q = 0; q < qx.size(); ++q)
    {
        const Local &l = qBasis[q];
        const double Vp = VprimeAt(v, l);
        requirePositive(Vp, qx[q]);
        const double magnetic = Phi * Phi / (Vp * Vp);
        for (Index r = 0; r < 4; ++r)
            out(l.first + r) += qw[q] * l.d(0, r) * magnetic;
    }

    if (parameters.pressureAux >= states.Aux().rows())
        throw std::invalid_argument("ThetaPinchEquilibrium: PressureAux = " +
                                    std::to_string(parameters.pressureAux) + ", but the transport case has " +
                                    std::to_string(states.Aux().rows()) + " auxiliary variables");
    const Vector p = states.Aux().row(parameters.pressureAux).transpose();
    out.head(nB) += pressureProjection(points) * p;
    out.head(nB) -= C * basisIntegrals;
    out(nB) = basisIntegrals.dot(v) - wallA;
}

void ThetaPinchEquilibrium::FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                                               MatrixRef, Vector const &psi, Vector const &,
                                               GlobalState const &states, std::vector<Position> const &points,
                                               Vector const &, Time)
{
    const Index nB = nCoefficients();
    const Vector v = psi.head(nB);

    for (size_t q = 0; q < qx.size(); ++q)
    {
        const Local &l = qBasis[q];
        const double Vp = VprimeAt(v, l);
        requirePositive(Vp, qx[q]);
        const double dmagnetic = -2.0 * Phi * Phi / (Vp * Vp * Vp);
        for (Index r = 0; r < 4; ++r)
            for (Index s = 0; s < 4; ++s)
                dRdpsi(l.first + r, l.first + s) += qw[q] * l.d(0, r) * l.d(0, s) * dmagnetic;
    }
    dRdpsi.col(nB).head(nB) = -basisIntegrals;
    dRdpsi.row(nB).head(nB) = basisIntegrals.transpose();

    // dR_m / dp_j = P(m, j), through the pressure's auxiliary column. Written
    // through GlobalState::Aux(), the whole matrix, which is a real reference:
    // dR[m][j] would build a State by value and discard the write.
    if (parameters.pressureAux >= states.Aux().rows())
        throw std::invalid_argument("ThetaPinchEquilibrium: PressureAux is not an auxiliary variable");
    Matrix const &proj = pressureProjection(points);
    for (Index m = 0; m < nB; ++m)
        dR[m].Aux().row(parameters.pressureAux) = proj.row(m);
}

void ThetaPinchEquilibrium::InitialFieldValue(VectorRef out)
{
    // The splines are a partition of unity, so equal coefficients are a constant V'.
    out.head(nCoefficients()).setConstant(wallA);
    out(nCoefficients()) = parameters.vacuumField * parameters.vacuumField;
}

// ---- geometry ---------------------------------------------------------------

void ThetaPinchEquilibrium::Geometry(VectorRef out, Vector const &psi, Position x, Time)
{
    const Vector v = psi.head(nCoefficients());
    const Local l = localBasis(x);
    const double Vp = VprimeAt(v, l);
    requirePositive(Vp, x);
    double dVp = 0.0;
    for (Index r = 0; r < 4; ++r)
        dVp += v(l.first + r) * l.d(1, r);
    const double A = antiderivativeWeights(x).dot(v);

    out(Vprime) = Vp;
    out(dVprime_dx) = dVp;
    out(gradx2) = 4.0 * M_PI * A / (Vp * Vp);
    out(Bfield) = Phi / Vp;
}

void ThetaPinchEquilibrium::dGeometry_dpsi(MatrixRef out, Vector const &psi, Position x, Time)
{
    const Vector v = psi.head(nCoefficients());
    const Local l = localBasis(x);
    const double Vp = VprimeAt(v, l);
    requirePositive(Vp, x);
    const Vector dA = antiderivativeWeights(x);
    const double A = dA.dot(v);

    // gradx2 = 4 pi A / V'^2 reads every coefficient whose support starts left
    // of x, through A; the other three slots read only the four local ones.
    out.row(gradx2).head(nCoefficients()) = (4.0 * M_PI / (Vp * Vp)) * dA.transpose();
    for (Index r = 0; r < 4; ++r)
    {
        const Index n = l.first + r;
        out(Vprime, n) = l.d(0, r);
        out(dVprime_dx, n) = l.d(1, r);
        out(gradx2, n) -= 8.0 * M_PI * A * l.d(0, r) / (Vp * Vp * Vp);
        out(Bfield, n) = -Phi * l.d(0, r) / (Vp * Vp);
    }
}

// ---- the field block's own solve ------------------------------------------

void ThetaPinchEquilibrium::updateFieldJacobian(MatrixRef dRdpsi, MatrixRef dRddpsidt, double alpha)
{
    const Matrix M = dRdpsi + alpha * dRddpsidt;
    Bsparse = M.sparseView();
    Bsparse.makeCompressed();
    Blu.compute(Bsparse);
    if (Blu.info() != Eigen::Success)
        throw std::runtime_error("ThetaPinchEquilibrium: SparseLU could not factorise the field block: " +
                                 Blu.lastErrorMessage());
}

void ThetaPinchEquilibrium::applyB(VectorRef out, Vector const &v) const { out = Bsparse * v; }

void ThetaPinchEquilibrium::applyBTranspose(VectorRef out, Vector const &v) const
{
    out = Bsparse.transpose() * v;
}

void ThetaPinchEquilibrium::solveB(VectorRef out, Vector const &rhs) const
{
    // Never slice an Eigen solve() result: assign first.
    Vector x = Blu.solve(rhs);
    out = x;
}

void ThetaPinchEquilibrium::solveBTranspose(VectorRef out, Vector const &rhs) const
{
    Vector x = Blu.transpose().solve(rhs);
    out = x;
}
