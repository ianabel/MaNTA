#include "ThetaPinchPressureBalance.hpp"

#include <autodiff/forward/dual.hpp>
#include <boost/math/quadrature/gauss.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>

REGISTER_EQUILIBRIUM_MODEL_IMPL(ThetaPinchPressureBalance)

namespace
{
using Gauss = boost::math::quadrature::gauss<double, 16>;
using autodiff::dual;

/// Where p_h lives: the transport's basis nodes, cell by cell, and the cells.
/// Fixed for a run, so shared by every equilibrium of it.
struct Layout
{
    Index nCells = 0, nPer = 0;
    std::vector<double> lo, hi;
    std::vector<Position> nodes; // cell-major, nPer per cell

    // The gradient recovery: per cell, the cells of its patch, and the
    // least-squares map from the patch's nodal values -- through their cell
    // means -- to the coefficients of a degree-(k + 1) polynomial in
    // s = (x - centre) / h. See buildRecovery.
    std::vector<std::vector<Index>> patch;
    std::vector<Matrix> fit;
    int recoveryDegree() const { return static_cast<int>(nPer); } // k + 1

    void buildRecovery()
    {
        patch.assign(static_cast<size_t>(nCells), {});
        fit.assign(static_cast<size_t>(nCells), Matrix());
        // A centred patch of 2m + 1 cells, m = ceil((nPer) / 2), moved inward
        // at the ends so it stays on the grid, and never wider than the grid.
        const Index m = (nPer + 1) / 2;
        const Index width = std::min<Index>(2 * m + 1, nCells);
        const int d = std::min<int>(recoveryDegree(), static_cast<int>(width) - 1);
        std::vector<double> l, dl;
        for (Index i = 0; i < nCells; ++i)
        {
            auto &P = patch[static_cast<size_t>(i)];
            const Index first = std::clamp<Index>(i - m, 0, nCells - width);
            for (Index c = first; c < first + width; ++c)
                P.push_back(c);
            const double c0 = 0.5 * (lo[static_cast<size_t>(i)] + hi[static_cast<size_t>(i)]);
            const double h = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];

            // Row c: the mean over cell c of each monomial in s = (x - c0) / h,
            // against the mean of p_h over cell c as a function of its nodal
            // values. Means of p_h carry its interpolation error only as a
            // smooth O(h^(k+1)) function, so a fit to them recovers p_x at
            // O(h^(k+1)); a fit to p_h itself also sees the error's repeating
            // shape inside each cell, and differentiates it.
            const Eigen::Index nCols = static_cast<Eigen::Index>(P.size()) * nPer;
            Matrix V = Matrix::Zero(static_cast<Eigen::Index>(P.size()), d + 1), E = Matrix::Zero(V.rows(), nCols);
            for (size_t ci = 0; ci < P.size(); ++ci)
            {
                const Index c = P[ci];
                const double a = lo[static_cast<size_t>(c)], b = hi[static_cast<size_t>(c)];
                const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
                for (size_t q = 0; q < Gauss::abscissa().size(); ++q)
                    for (double sgn : {-1.0, 1.0})
                    {
                        const double x = mid + sgn * half * Gauss::abscissa()[q];
                        const double w = 0.5 * Gauss::weights()[q]; // over the cell's length, a mean
                        const double sx = (x - c0) / h;
                        double v = 1.0;
                        for (int r = 0; r <= d; ++r, v *= sx)
                            V(static_cast<Eigen::Index>(ci), r) += w * v;
                        basis(c, x, l, dl);
                        for (Index j = 0; j < nPer; ++j)
                            E(static_cast<Eigen::Index>(ci), static_cast<Eigen::Index>(ci) * nPer + j) +=
                                w * l[static_cast<size_t>(j)];
                    }
            }
            // coefficients = (V^T V)^-1 V^T E p_patch; assigned before use, as
            // a solve() result must be.
            Matrix VtV = V.transpose() * V, VtE = V.transpose() * E;
            Matrix coef = VtV.ldlt().solve(VtE);
            fit[static_cast<size_t>(i)] = coef; // (d + 1, nCols)
        }
    }

    /// Weights w such that the recovered p_x at x in cell i is sum_r w_r p_patch_r.
    Vector recoveredDerivativeWeights(Index i, Position x) const
    {
        const double c0 = 0.5 * (lo[static_cast<size_t>(i)] + hi[static_cast<size_t>(i)]);
        const double h = hi[static_cast<size_t>(i)] - lo[static_cast<size_t>(i)];
        const double sx = (x - c0) / h;
        const int d = static_cast<int>(fit[static_cast<size_t>(i)].rows()) - 1;
        Vector dphi = Vector::Zero(d + 1);
        double v = 1.0;
        for (int m = 1; m <= d; ++m, v *= sx)
            dphi(m) = m * v / h;
        return fit[static_cast<size_t>(i)].transpose() * dphi;
    }

    /// The cell holding x, and whether x sits on the interior face at its left.
    Index cellOf(Position x, bool &onFace) const
    {
        onFace = false;
        const auto it = std::upper_bound(lo.begin(), lo.end(), x);
        Index i = std::max<Index>(static_cast<Index>(it - lo.begin()) - 1, 0);
        i = std::min(i, nCells - 1);
        onFace = i > 0 && x == lo[static_cast<size_t>(i)];
        return i;
    }

    /// Lagrange basis of cell i at x, and its x derivative.
    void basis(Index i, Position x, std::vector<double> &l, std::vector<double> &dl) const
    {
        l.assign(static_cast<size_t>(nPer), 0.0);
        dl.assign(static_cast<size_t>(nPer), 0.0);
        const Position *xn = &nodes[static_cast<size_t>(i * nPer)];
        for (Index j = 0; j < nPer; ++j)
        {
            double lj = 1.0;
            for (Index m = 0; m < nPer; ++m)
                if (m != j)
                    lj *= (x - xn[m]) / (xn[j] - xn[m]);
            l[static_cast<size_t>(j)] = lj;

            double dlj = 0.0;
            for (Index r = 0; r < nPer; ++r)
            {
                if (r == j)
                    continue;
                double term = 1.0 / (xn[j] - xn[r]);
                for (Index m = 0; m < nPer; ++m)
                    if (m != j && m != r)
                        term *= (x - xn[m]) / (xn[j] - xn[m]);
                dlj += term;
            }
            dl[static_cast<size_t>(j)] = dlj;
        }
    }
};

struct BalanceState : public Equilibrium
{
    std::shared_ptr<Layout const> layout;
    std::vector<double> p; // nodal pressure, cell-major
    double C = 0.0;

    // G' restricted to the pressure, assembled for the transpose on first use:
    // rows the geometry at `responsePoints`, flattened point by point with the
    // slot fastest; columns the nodal pressures. Mutable as a cache of what the
    // equilibrium already determines.
    mutable std::vector<Position> responsePoints;
    mutable Matrix response;
};

BalanceState const &asState(Equilibrium const &e)
{
    auto const *s = dynamic_cast<BalanceState const *>(&e);
    if (!s)
        throw std::logic_error("ThetaPinchPressureBalance was handed an equilibrium it did not produce");
    return *s;
}

// Everything below is written once over the value type, so the response is the
// same arithmetic run in duals. Intermediates are declared T rather than auto:
// autodiff's expression templates hold references to their operands.

template <class T>
T pressureAt(Layout const &L, std::vector<T> const &p, Index i, Position x, T *dpdx)
{
    std::vector<double> l, dl;
    L.basis(i, x, l, dl);
    T v = 0.0;
    for (Index j = 0; j < L.nPer; ++j)
        v += l[static_cast<size_t>(j)] * p[static_cast<size_t>(i * L.nPer + j)];
    if (dpdx)
    {
        // The recovered derivative, from the patch fit.
        const Vector w = L.recoveredDerivativeWeights(i, x);
        T d = 0.0;
        Eigen::Index r = 0;
        for (Index c : L.patch[static_cast<size_t>(i)])
            for (Index j = 0; j < L.nPer; ++j, ++r)
                d += w(r) * p[static_cast<size_t>(c * L.nPer + j)];
        *dpdx = d;
    }
    return v;
}

/// Int_a^b (C - p_h)^(-power) over a piece of cell i, by the 16-point rule.
template <class T>
T cellIntegral(Layout const &L, std::vector<T> const &p, T const &C, Index i, double a, double b, double power)
{
    using std::pow;
    T sum = 0.0;
    const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
    if (!(half > 0.0))
        return sum;
    for (size_t q = 0; q < Gauss::abscissa().size(); ++q)
        for (double sgn : {-1.0, 1.0})
        {
            const double x = mid + sgn * half * Gauss::abscissa()[q];
            const T pq = pressureAt(L, p, i, x, static_cast<T *>(nullptr));
            const T gap = C - pq;
            const T f = pow(gap, -power);
            sum += half * Gauss::weights()[q] * f;
        }
    return sum;
}

/// A at every face, A(0) = 0.
template <class T>
std::vector<T> faceAreas(Layout const &L, std::vector<T> const &p, T const &C, double Phi)
{
    std::vector<T> A(static_cast<size_t>(L.nCells + 1), T(0.0));
    for (Index i = 0; i < L.nCells; ++i)
    {
        const T Ii = cellIntegral(L, p, C, i, L.lo[static_cast<size_t>(i)], L.hi[static_cast<size_t>(i)], 0.5);
        A[static_cast<size_t>(i + 1)] = A[static_cast<size_t>(i)] + Phi * Ii;
    }
    return A;
}

/// The four slots at x, evaluated from cell i's side.
template <class T>
void slotsInCell(Layout const &L, std::vector<T> const &p, T const &C, std::vector<T> const &A, double Phi,
                 Index i, Position x, T out[4])
{
    using std::pow;
    using std::sqrt;
    T px = 0.0;
    const T pv = pressureAt(L, p, i, x, &px);
    const T gap = C - pv;
    if (!(autodiff::val(gap) > 0.0))
        throw std::runtime_error(std::format("ThetaPinchPressureBalance: p = {} reaches the total pressure "
                                             "C = {} at x = {}, where no equilibrium exists",
                                             autodiff::val(pv), autodiff::val(C), x));
    const T Vp = Phi / sqrt(gap);
    const T Ax = A[static_cast<size_t>(i)] +
                 Phi * cellIntegral(L, p, C, i, L.lo[static_cast<size_t>(i)], x, 0.5);
    out[ThetaPinchPressureBalance::Vprime] = Vp;
    out[ThetaPinchPressureBalance::dVprime_dx] = 0.5 * Phi * px * pow(gap, -1.5);
    out[ThetaPinchPressureBalance::gradx2] = 4.0 * M_PI * Ax / (Vp * Vp);
    out[ThetaPinchPressureBalance::Bfield] = Phi / Vp;
}

/// The slots at x; on an interior face, the mean of the two sides, since V'
/// is discontinuous there wherever p_h is.
template <class T>
void slotsAt(Layout const &L, std::vector<T> const &p, T const &C, std::vector<T> const &A, double Phi,
             Position x, T out[4])
{
    bool onFace = false;
    const Index i = L.cellOf(x, onFace);
    slotsInCell(L, p, C, A, Phi, i, x, out);
    if (onFace)
    {
        T left[4];
        slotsInCell(L, p, C, A, Phi, i - 1, x, left);
        for (int s = 0; s < 4; ++s)
            out[s] = 0.5 * (out[s] + left[s]);
    }
}

/// The largest p_h anywhere the model evaluates it inside a cell.
double maxPressure(Layout const &L, std::vector<double> const &p)
{
    double m = -std::numeric_limits<double>::infinity();
    for (double v : p)
        m = std::max(m, v);
    for (Index i = 0; i < L.nCells; ++i)
    {
        const double a = L.lo[static_cast<size_t>(i)], b = L.hi[static_cast<size_t>(i)];
        const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
        for (size_t q = 0; q < Gauss::abscissa().size(); ++q)
            for (double sgn : {-1.0, 1.0})
                m = std::max(m, pressureAt(L, p, i, mid + sgn * half * Gauss::abscissa()[q],
                                           static_cast<double *>(nullptr)));
    }
    return m;
}
} // namespace

ThetaPinchPressureBalance::Parameters ThetaPinchPressureBalance::readParameters(toml::value const &config)
{
    Parameters p;
    if (!config.is_table() || !config.contains("ThetaPinchPressureBalance"))
        return p;
    auto const &table = config.at("ThetaPinchPressureBalance");

    // Refused rather than ignored, as ThetaPinchEquilibrium's table is.
    static const std::vector<std::string> known = {"PressureAux", "WallRadius", "VacuumField",
                                                   "DifferencedResponse"};
    for (auto const &[key, value] : table.as_table())
        if (std::find(known.begin(), known.end(), key) == known.end())
        {
            std::string list;
            for (auto const &k : known)
                list += (list.empty() ? "" : ", ") + k;
            throw std::invalid_argument("ThetaPinchPressureBalance: unknown key \"" + key +
                                        "\" in [ThetaPinchPressureBalance]; the keys are " + list);
        }

    p.pressureAux = toml::find_or(table, "PressureAux", p.pressureAux);
    p.wallRadius = toml::find_or(table, "WallRadius", p.wallRadius);
    p.vacuumField = toml::find_or(table, "VacuumField", p.vacuumField);
    p.differencedResponse = toml::find_or(table, "DifferencedResponse", p.differencedResponse);
    return p;
}

EquilibriumSpec ThetaPinchPressureBalance::buildSpec(bool differencedResponse)
{
    return {.geometry = {{"Vprime", "area per unit flux label, dA/dx", "a^2"},
                         {"dVprime_dx", "d V' / dx", "a^2"},
                         {"gradx2", "<|grad x|^2>", "a^-2"},
                         {"B", "axial magnetic field", "B0"}},
            .label = "Phi/Phi_wall",
            .name = "ThetaPinch",
            .response = differencedResponse ? EquilibriumResponse::Differenced : EquilibriumResponse::Model};
}

ThetaPinchPressureBalance::ThetaPinchPressureBalance(toml::value const &config, Grid const &grid_)
    : ThetaPinchPressureBalance(readParameters(config), grid_)
{
}

ThetaPinchPressureBalance::ThetaPinchPressureBalance(Parameters const &parameters_, Grid const &grid_)
    : EquilibriumModel(buildSpec(parameters_.differencedResponse)), parameters(parameters_), grid(grid_)
{
    if (parameters.pressureAux < 0)
        throw std::invalid_argument("ThetaPinchPressureBalance: PressureAux must be an auxiliary variable index");
    if (!(parameters.wallRadius > 0.0) || !(parameters.vacuumField > 0.0))
        throw std::invalid_argument("ThetaPinchPressureBalance: WallRadius and VacuumField must be positive");
    if (grid.lowerBoundary() != 0.0 || grid.upperBoundary() != 1.0)
        throw std::invalid_argument(
            "ThetaPinchPressureBalance: x is the normalised flux Phi / Phi_wall, so the grid must be [0, 1]");
    wallA = M_PI * parameters.wallRadius * parameters.wallRadius;
    Phi = wallA * parameters.vacuumField;
}

double ThetaPinchPressureBalance::totalPressure(Equilibrium const &e) { return asState(e).C; }

// The wall constraint F(C) = Phi Int (C - p_h)^(-1/2) - pi a^2 falls from +inf
// at C = max p_h to -pi a^2, convexly, so it has one root. Newton from the warm
// start, kept inside a bracket and bisecting whenever a step would leave it.
std::unique_ptr<Equilibrium> ThetaPinchPressureBalance::Solve(GlobalState const &states,
                                                              std::vector<Position> const &points,
                                                              Vector const &, Time, Equilibrium const *warm)
{
    ++nSolves;
    if (parameters.pressureAux >= states.Aux().rows())
        throw std::invalid_argument(std::format("ThetaPinchPressureBalance: PressureAux = {}, but the "
                                                "transport case has {} auxiliary variables",
                                                parameters.pressureAux, states.Aux().rows()));

    auto out = std::make_unique<BalanceState>();
    if (warm && asState(*warm).layout->nodes == points)
        out->layout = asState(*warm).layout;
    else
    {
        auto L = std::make_shared<Layout>();
        L->nCells = static_cast<Index>(grid.getNCells());
        const Index nPoints = static_cast<Index>(points.size());
        if (nPoints % L->nCells != 0)
            throw std::invalid_argument(std::format("ThetaPinchPressureBalance: {} transport nodes do not "
                                                    "divide among {} cells",
                                                    nPoints, L->nCells));
        L->nPer = nPoints / L->nCells;
        L->nodes = points;
        for (Index i = 0; i < L->nCells; ++i)
        {
            L->lo.push_back(grid[i].x_l);
            L->hi.push_back(grid[i].x_u);
        }
        L->buildRecovery();
        out->layout = std::move(L);
    }
    Layout const &L = *out->layout;

    out->p.resize(points.size());
    for (size_t j = 0; j < points.size(); ++j)
        out->p[j] = states.Aux()(parameters.pressureAux, static_cast<Eigen::Index>(j));
    std::vector<double> const &p = out->p;

    auto F = [&](double C) -> double
    {
        double I = 0.0;
        for (Index i = 0; i < L.nCells; ++i)
            I += cellIntegral(L, p, C, i, L.lo[static_cast<size_t>(i)], L.hi[static_cast<size_t>(i)], 0.5);
        return Phi * I - wallA;
    };
    auto dF = [&](double C) -> double
    {
        double I = 0.0;
        for (Index i = 0; i < L.nCells; ++i)
            I += cellIntegral(L, p, C, i, L.lo[static_cast<size_t>(i)], L.hi[static_cast<size_t>(i)], 1.5);
        return -0.5 * Phi * I;
    };

    const double pmax = maxPressure(L, p);
    const double B02 = parameters.vacuumField * parameters.vacuumField;
    double lo = pmax + 1e-14 * std::max(std::abs(pmax), B02);
    double hi = std::max(warm ? asState(*warm).C : lo, pmax + B02);
    for (int grow = 0; F(hi) > 0.0; ++grow)
    {
        if (grow > 100)
            throw std::runtime_error("ThetaPinchPressureBalance: could not bracket the total pressure");
        lo = hi;
        hi = pmax + 2.0 * (hi - pmax);
    }

    double C = (warm && asState(*warm).C > lo && asState(*warm).C < hi) ? asState(*warm).C : hi;
    for (int it = 0; it < maxIterations; ++it)
    {
        const double f = F(C);
        if (f > 0.0)
            lo = C;
        else
            hi = C;
        double next = C - f / dF(C);
        if (!(next > lo && next < hi))
            next = 0.5 * (lo + hi);
        const double step = next - C;
        C = next;
        if (std::abs(step) <= relativeStep * std::abs(C) || hi - lo <= relativeStep * std::abs(C))
        {
            out->C = C;
            return out;
        }
    }
    throw std::runtime_error(std::format("ThetaPinchPressureBalance: the wall constraint did not converge in {} "
                                         "iterations",
                                         maxIterations));
}

void ThetaPinchPressureBalance::Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points)
{
    BalanceState const &s = asState(e);
    const std::vector<double> A = faceAreas(*s.layout, s.p, s.C, Phi);
    double slot[4];
    for (size_t j = 0; j < points.size(); ++j)
    {
        slotsAt(*s.layout, s.p, s.C, A, Phi, points[j], slot);
        for (int r = 0; r < 4; ++r)
            g(r, static_cast<Eigen::Index>(j)) = slot[r];
    }
}

// The geometry along a pressure direction dp, in duals: p carries dp as its
// derivative part, C its response to dp through the wall constraint, and every
// slot's derivative part is then dg. Nothing but the pressure reaches the
// balance, so nothing else of the input is read.
void ThetaPinchPressureBalance::pressureResponse(MatrixRef dg, Equilibrium const &e, Vector const &dp,
                                                 std::vector<Position> const &geometryPoints) const
{
    BalanceState const &s = asState(e);
    Layout const &L = *s.layout;
    std::vector<dual> p(s.p.size());
    for (size_t j = 0; j < p.size(); ++j)
    {
        p[j].val = s.p[j];
        p[j].grad = dp(static_cast<Eigen::Index>(j));
    }

    // F = Phi Int (C-p)^(-1/2) - pi a^2 = 0 along dp: dF/dp . dp + dF/dC dC = 0,
    // with dF/dC = -(Phi/2) Int (C-p)^(-3/2), and Fp.grad the first integral's
    // derivative along dp, still to be scaled by Phi.
    const dual C0 = s.C; // zero derivative part
    dual Fp = 0.0;
    double I32 = 0.0;
    for (Index i = 0; i < L.nCells; ++i)
    {
        Fp += cellIntegral(L, p, C0, i, L.lo[static_cast<size_t>(i)], L.hi[static_cast<size_t>(i)], 0.5);
        I32 += cellIntegral(L, s.p, s.C, i, L.lo[static_cast<size_t>(i)], L.hi[static_cast<size_t>(i)], 1.5);
    }
    dual C = s.C;
    C.grad = (Phi * Fp.grad) / (0.5 * Phi * I32);

    const std::vector<dual> A = faceAreas(L, p, C, Phi);
    dual slot[4];
    for (size_t j = 0; j < geometryPoints.size(); ++j)
    {
        slotsAt(L, p, C, A, Phi, geometryPoints[j], slot);
        for (int r = 0; r < 4; ++r)
            dg(r, static_cast<Eigen::Index>(j)) = slot[r].grad;
    }
}

void ThetaPinchPressureBalance::applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states,
                                              GlobalState const &dp, std::vector<Position> const &inputPoints,
                                              Vector const &weights, Time t,
                                              std::vector<Position> const &geometryPoints)
{
    if (spec.response == EquilibriumResponse::Differenced)
    {
        EquilibriumModel::applyResponse(dg, e, states, dp, inputPoints, weights, t, geometryPoints);
        return;
    }
    pressureResponse(dg, e, dp.Aux().row(parameters.pressureAux).transpose(), geometryPoints);
}

// Exact: the pressure columns of G', one dual sweep per nodal pressure,
// assembled once per equilibrium and kept on it, then transposed. Every other
// row of dp stays zero, since nothing else reaches the balance. No Solve is
// made, unlike the base class's assembly under a differenced response.
void ThetaPinchPressureBalance::applyResponseTranspose(GlobalState &dp, Equilibrium const &e,
                                                       GlobalState const &states, Matrix const &dg,
                                                       std::vector<Position> const &inputPoints,
                                                       Vector const &weights, Time t,
                                                       std::vector<Position> const &geometryPoints)
{
    if (spec.response == EquilibriumResponse::Differenced)
    {
        EquilibriumModel::applyResponseTranspose(dp, e, states, dg, inputPoints, weights, t, geometryPoints);
        return;
    }

    BalanceState const &s = asState(e);
    const Eigen::Index nP = static_cast<Eigen::Index>(s.p.size());
    const Eigen::Index nG = static_cast<Eigen::Index>(geometryPoints.size());
    if (s.responsePoints != geometryPoints || s.response.cols() != nP)
    {
        s.response.resize(nGeometry() * nG, nP);
        Matrix column(nGeometry(), nG);
        Vector unit = Vector::Zero(nP);
        for (Eigen::Index c = 0; c < nP; ++c)
        {
            unit(c) = 1.0;
            pressureResponse(column, e, unit, geometryPoints);
            unit(c) = 0.0;
            s.response.col(c) = Eigen::Map<Vector const>(column.data(), column.size());
        }
        s.responsePoints = geometryPoints;
    }
    const Vector flat = Eigen::Map<Vector const>(dg.data(), dg.size());
    dp.Aux().row(parameters.pressureAux) += (s.response.transpose() * flat).transpose();
}
