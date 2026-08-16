#ifndef MANUFACTUREDFIELDS_HPP
#define MANUFACTUREDFIELDS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../../FieldModel.hpp"
#include "../../gridStructures.hpp"

#include <cmath>
#include <tuple>

/*
    Manufactured field models: test fixtures, deliberately not registered for
    production.

    The exact transport solution is the one MMSHarness.hpp already shares,
    u = sin(pi x)(1 + t) on [0,1].

    A note on how the constraint is compensated, because the obvious way is
    wrong. A compensating term written against the *discrete* state can be an
    exact row operation: residual() evaluates the hooks on the same states at
    the same abscissae and pushes them through the same projection, so a
    compensation of that shape cancels identically and the study silently
    measures the uncoupled problem. Everything below is therefore compensated
    against u_exact(x, t), never against the state it is handed.
*/

inline Value manufacturedU(Position x, Time t) { return std::sin(M_PI * x) * (1.0 + t); }

/// Int_0^1 sin(pi x)(1+t) dx = (2/pi)(1+t)
inline Value manufacturedPsiExact(Time t) { return (2.0 / M_PI) * (1.0 + t); }

/// The shape function the single-DOF model's geometry slot carries. Chosen
/// nonconstant so that dGeometry/dpsi varies across a cell -- a constant would
/// be annihilated by exactly the operator that hid the mass-matrix confusion in
/// DerivativeSubVector, so it would fail to distinguish a right answer from a
/// wrong one.
inline Value manufacturedC(Position x) { return std::cos(M_PI * x); }

/// The SPD tridiagonal standing in for a 1-D elliptic operator: the usual
/// second-difference stencil with unit Dirichlet ends.
inline Matrix manufacturedL(Index n)
{
    Matrix L = Matrix::Zero(n, n);
    for (Index i = 0; i < n; ++i)
    {
        L(i, i) = 2.0;
        if (i > 0)
            L(i, i - 1) = -1.0;
        if (i + 1 < n)
            L(i, i + 1) = -1.0;
    }
    return L;
}

/// nFieldDOF = 1, algebraic:  R = psi - Int u dx,  g(x; psi) = 1 + psi c(x).
class ManufacturedField : public FieldModel
{
public:
    ManufacturedField(toml::value const &, Grid const &) : FieldModel(buildSpec()) {}

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.dofs = {{"psi", "the manufactured field unknown", "1", false}};
        s.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}};
        s.label = "x";
        return s;
    }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &,
                       GlobalState const &states, std::vector<Position> const &,
                       Vector const &weights, Time) override
    {
        double integral = 0.0;
        for (Index j = 0; j < weights.size(); ++j)
            integral += weights(j) * states[j].u(0);
        out(0) = psi(0) - integral;
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + psi(0) * manufacturedC(x);
    }

    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        out(0, 0) = manufacturedC(x);
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef, Vector const &, Vector const &, GlobalState const &,
                            std::vector<Position> const &, Vector const &weights, Time) override
    {
        dRdpsi(0, 0) = 1.0;
        // dR/du_j = -w_j, for every node j at once.
        //
        // Not "dR[0][j].u(0) = -weights(j)": GlobalState::operator[](Index)
        // const returns a State *by value* -- it builds one from the stored
        // column and hands back a copy, the same way evalOnNode() does. Writing
        // through a temporary like that compiles (State::u(Index) returns a
        // real double&) and modifies nothing, because the temporary is gone at
        // the end of the statement. ScalarTestLD3::ScalarGPrime writes the same
        // shape of derivative through GlobalState::Variable(), the whole
        // (nVars, nPoints) matrix, which is a real reference into the data --
        // that is the pattern to copy.
        dR[0].Variable().row(0) = -weights.transpose();
    }

    void InitialFieldValue(VectorRef out) override { out(0) = manufacturedPsiExact(0.0); }
};

/// nFieldDOF = n:  L psi = strength * f(state), f_m sampling the transport solution.
/// B = L is tridiagonal and dGeometry/dpsi is dense, because geometry at x
/// interpolates every entry of psi.
///
/// `strength` is the coupling dial: B = L is untouched by it and A2 is
/// proportional to it, so the block Gauss-Seidel iteration matrix
/// M = B^-1 A2 A^-1 A1 is linear in it. That is what lets a test *measure* the
/// spectral radius and assert the fixture is in the regime it claims, rather
/// than assuming a pairing is divergent and having the assertion go vacuous when
/// it stops being.
class ManufacturedFieldVector : public FieldModel
{
public:
    static constexpr Index N = 5;

    ManufacturedFieldVector(toml::value const &, Grid const &, double strength_ = 1.0)
        : FieldModel(buildSpec()), strength(strength_), L(manufacturedL(N))
    {
    }

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}};
        s.label = "x";
        for (Index m = 0; m < N; ++m)
            s.dofs.push_back({"psi" + std::to_string(m), "manufactured field unknown", "1", false});
        return s;
    }

    /// psi is sampled at N equispaced points in [0,1]; geometry at x is the
    /// piecewise-linear interpolant, which is what makes dGeometry/dpsi dense
    /// in the sense that matters (every x sees more than one psi entry).
    static Position node(Index m) { return static_cast<double>(m) / static_cast<double>(N - 1); }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &,
                       GlobalState const &states, std::vector<Position> const &points,
                       Vector const &weights, Time) override
    {
        out = L * psi - strength * f(states, points, weights);
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + interpolate(psi, x);
    }

    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        for (Index m = 0; m < N; ++m)
            out(0, m) = basis(m, x);
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef, Vector const &, Vector const &, GlobalState const &,
                            std::vector<Position> const &points, Vector const &weights,
                            Time) override
    {
        dRdpsi = L;
        // f_m = Int c_m(x) u(x) dx, so df_m/du_j = -w_j c_m(x_j) in the
        // residual. Written through GlobalState::Variable() -- see the
        // comment in ManufacturedField::FieldResidualPrime for why dR[m][j]
        // would silently discard the write instead.
        for (Index m = 0; m < N; ++m)
            for (Index j = 0; j < weights.size(); ++j)
                dR[m].Variable()(0, j) = -strength * weights(j) * basis(m, points[j]);
    }

    void InitialFieldValue(VectorRef out) override
    {
        // The exact psi at t = 0, from L psi = strength * f(u_exact).
        Vector rhs = strength * fExact(0.0);
        Vector x = L.partialPivLu().solve(rhs);
        out = x;
    }

    /// The exact psi at time t, for the order study to compare against.
    Vector psiExact(Time t) const
    {
        Vector rhs = strength * fExact(t);
        Vector x = L.partialPivLu().solve(rhs);
        return x;
    }

private:
    /// The hat function centred on node m, evaluated at x.
    static double basis(Index m, Position x)
    {
        const double h = 1.0 / static_cast<double>(N - 1);
        const double d = std::abs(x - node(m)) / h;
        return d >= 1.0 ? 0.0 : 1.0 - d;
    }

    static double interpolate(Vector const &psi, Position x)
    {
        double v = 0.0;
        for (Index m = 0; m < N; ++m)
            v += psi(m) * basis(m, x);
        return v;
    }

    /// The residual's own term: f_m = Int c_m(x) u(x) dx against the *discrete*
    /// state, which is what the constraint L psi = f(state) means.
    Vector f(GlobalState const &states, std::vector<Position> const &points,
             Vector const &weights) const
    {
        Vector out = Vector::Zero(N);
        for (Index m = 0; m < N; ++m)
            for (Index j = 0; j < weights.size(); ++j)
                out(m) += weights(j) * basis(m, points[j]) * states[j].u(0);
        return out;
    }

    /// The same integral against u_exact, for psiExact and the initial value.
    /// This -- not f() above -- is the "compensate against u_exact" the header
    /// comment describes: it is what the order study compares to, so it must
    /// not be a function of the discrete state.
    Vector fExact(Time t) const
    {
        // Int c_m(x) sin(pi x)(1+t) dx, by a fine Simpson rule; the constraint
        // only has to be *consistent*, not analytic, for the order study.
        const Index nq = 4001;
        Vector out = Vector::Zero(N);
        const double h = 1.0 / static_cast<double>(nq - 1);
        for (Index m = 0; m < N; ++m)
        {
            double s = 0.0;
            for (Index j = 0; j < nq; ++j)
            {
                const double x = j * h;
                const double w = (j == 0 || j == nq - 1) ? 1.0 : (j % 2 ? 4.0 : 2.0);
                s += w * basis(m, x) * manufacturedU(x, t);
            }
            out(m) = s * h / 3.0;
        }
        return out;
    }

    double strength;
    Matrix L;
};

/// Sample u_exact on a grid's nodes and return the GlobalState, the abscissae
/// and the integration weights, so a test can call a field hook without a
/// solver. Defined in ManufacturedFieldTests.cpp.
std::tuple<GlobalState, std::vector<Position>, Vector>
sampleExactOnNodes(Grid const &grid, Index k, Time t);

#endif // MANUFACTUREDFIELDS_HPP
