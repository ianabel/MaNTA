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

    // Refused rather than ignored: find_or would read a misspelt key as absent
    // and run on the default without a word.
    static const std::vector<std::string> known = {"IntervalsPerCell", "PressureAux", "WallRadius",
                                                   "VacuumField"};
    for (auto const &[key, value] : table.as_table())
        if (std::find(known.begin(), known.end(), key) == known.end())
        {
            std::string list;
            for (auto const &k : known)
                list += (list.empty() ? "" : ", ") + k;
            throw std::invalid_argument("ThetaPinchEquilibrium: unknown key \"" + key +
                                        "\" in [ThetaPinchEquilibrium]; the keys are " + list);
        }

    p.intervalsPerCell = toml::find_or(table, "IntervalsPerCell", p.intervalsPerCell);
    p.pressureAux = toml::find_or(table, "PressureAux", p.pressureAux);
    p.wallRadius = toml::find_or(table, "WallRadius", p.wallRadius);
    p.vacuumField = toml::find_or(table, "VacuumField", p.vacuumField);
    return p;
}

Vector ThetaPinchEquilibrium::knotVector(Parameters const &p, Grid const &grid)
{
    if (p.intervalsPerCell < 1)
        throw std::invalid_argument("ThetaPinchEquilibrium: IntervalsPerCell must be at least 1, not " +
                                    std::to_string(p.intervalsPerCell));

    // Every transport face is a knot, and each cell is split into
    // intervalsPerCell equal intervals; every knot is simple, so the spline is
    // C^2 throughout.
    const Index nCells = static_cast<Index>(grid.getNCells());
    const Index m = p.intervalsPerCell;
    std::vector<double> interior;
    for (Index i = 0; i < nCells; ++i)
    {
        const Interval I = grid[i];
        if (i > 0)
            interior.push_back(I.x_l);
        for (Index j = 1; j < m; ++j)
            interior.push_back(I.x_l + (I.x_u - I.x_l) * static_cast<double>(j) / static_cast<double>(m));
    }

    // Extended three intervals past each end at the end interval's width, so the
    // basis on [0, 1] is the uniform spline's own wherever the mesh is uniform.
    const double hLower = (grid[0].x_u - grid[0].x_l) / static_cast<double>(m);
    const double hUpper = (grid[nCells - 1].x_u - grid[nCells - 1].x_l) / static_cast<double>(m);
    Vector t(static_cast<Index>(interior.size()) + 8);
    for (Index r = 0; r < 4; ++r)
    {
        t(r) = static_cast<double>(r - 3) * hLower;
        t(t.size() - 1 - r) = 1.0 + static_cast<double>(3 - r) * hUpper;
    }
    for (size_t j = 0; j < interior.size(); ++j)
        t(4 + static_cast<Index>(j)) = interior[j];
    return t;
}

FieldModelSpec ThetaPinchEquilibrium::buildSpec(Index nCoefficients)
{
    FieldModelSpec s;
    for (Index n = 0; n < nCoefficients; ++n)
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
    : FieldModel(buildSpec(knotVector(parameters_, grid_).size() - 4)), parameters(parameters_), grid(grid_)
{
    if (parameters.pressureAux < 0)
        throw std::invalid_argument("ThetaPinchEquilibrium: PressureAux must be an auxiliary variable index");
    if (!(parameters.wallRadius > 0.0) || !(parameters.vacuumField > 0.0))
        throw std::invalid_argument("ThetaPinchEquilibrium: WallRadius and VacuumField must be positive");
    if (grid.lowerBoundary() != 0.0 || grid.upperBoundary() != 1.0)
        throw std::invalid_argument(
            "ThetaPinchEquilibrium: x is the normalised flux Phi / Phi_wall, so the grid must be [0, 1]");

    wallA = M_PI * parameters.wallRadius * parameters.wallRadius;
    Phi = wallA * parameters.vacuumField;

    knots = knotVector(parameters, grid);
    nCoef = knots.size() - 4;
    for (Index i = 3; i < knots.size() - 3; ++i)
        if (breaks.empty() || knots(i) > breaks.back())
            breaks.push_back(knots(i));

    // The model's Gauss rule, per interval between distinct knots, and the
    // running integral of every basis function at each knot. Eight points are
    // exact for the cubics many times over.
    cumulative = Matrix::Zero(static_cast<Index>(breaks.size()), nCoef);
    std::vector<double> x, w;
    for (size_t b = 0; b + 1 < breaks.size(); ++b)
    {
        cumulative.row(b + 1) = cumulative.row(b);
        gaussOn(breaks[b], breaks[b + 1], x, w);
        for (size_t q = 0; q < x.size(); ++q)
        {
            const Local l = localBasis(x[q]);
            qx.push_back(x[q]);
            qw.push_back(w[q]);
            qBasis.push_back(l);
            for (Index r = 0; r < 4; ++r)
                cumulative(b + 1, l.first + r) += w[q] * l.d(0, r);
        }
    }
    basisIntegrals = cumulative.row(cumulative.rows() - 1).transpose();
}

// ---- the spline -------------------------------------------------------------

ThetaPinchEquilibrium::Local ThetaPinchEquilibrium::localBasis(Position x) const
{
    x = std::clamp(x, 0.0, 1.0);
    Local l;
    l.first = Spline3::Span(x, 3, knots) - 3;
    l.d = Spline3::BasisFunctionDerivatives(x, 2, 3, knots);
    return l;
}

// The integral to the last distinct knot at or left of x, then the rest of the
// way by two-point Gauss, which is exact for the cubic pieces.
Vector ThetaPinchEquilibrium::antiderivativeWeights(Position x) const
{
    x = std::clamp(x, 0.0, 1.0);
    const auto it = std::upper_bound(breaks.begin(), breaks.end(), x);
    const Index b = std::min<Index>(static_cast<Index>(it - breaks.begin()) - 1,
                                    static_cast<Index>(breaks.size()) - 2);
    Vector out = cumulative.row(b).transpose();
    const double a = breaks[b], mid = 0.5 * (a + x), half = 0.5 * (x - a);
    if (half > 0.0)
        for (double s : {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)})
        {
            // Evaluated inside [a, x], so the span is the one starting at a
            // even where a carries repeated knots.
            const Local l = localBasis(mid + half * s);
            for (Index r = 0; r < 4; ++r)
                out(l.first + r) += half * l.d(0, r);
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
        for (double b : breaks)
            if (b > I.x_l && b < I.x_u)
                cuts.push_back(b);
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
