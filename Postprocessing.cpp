#include "Postprocessing.hpp"

#include <Eigen/Dense>
#include <Eigen/LU>

#include <stdexcept>

/*
 * Postprocessing algorithm derived from SuperconvergentHDG-I.pdf, variable names
 * correspond to matrices from that paper (referred to as paper I below)
 */

Postprocessor::Postprocessor(Grid const &Grid_, unsigned int Order, Index n_var,
                             Index Scalars, Index aux)
    : grid(Grid_), k(Order), nVars(n_var), nScalars(Scalars), nAux(aux),
      basis(NodalBasis::getBasis(Order)), starBasis(NodalBasis::getBasis(Order + 1))
{
    // The degree-0 NodalBasis is incomplete: its constructor returns early
    // (Basis.hpp:369-377) without building Vandermonde or BarycentricWeights, so
    // Evaluate() at anything other than the single node reads an empty vector.
    // Nothing here can work around that, and paper I requires k >= 1 for the
    // superconvergence anyway.
    if (k < 1)
        throw std::invalid_argument(
            "Superconvergent postprocessing requires Polynomial_degree >= 1");

    const Index nCells = grid.getNCells();
    const Index nStar = k + 2; // dofs per cell in P_{k+1}
    const Index nLocal = k + 1;

    B11_.reserve(nCells);
    B12_.reserve(nCells);
    V_.reserve(nCells);
    A9_.reserve(nCells);
    b1_.reserve(nCells);

    Vector const &starNodes = starBasis.getNodes();

    starPoints_.reserve(nCells * nStar);

    for (Index cell = 0; cell < nCells; ++cell)
    {
        Interval const &I(grid[cell]);

        for (Index m = 0; m < nStar; ++m)
            starPoints_.push_back(I.fromRef(starNodes(m)));

        // A1 = ( d_x chi_j, d_x chi_i )_K   -- the P_{k+1} stiffness matrix,
        // singular with the constants in its kernel; that is what b1 is for.
        Matrix A1(nStar, nStar);
        for (Index i = 0; i < nStar; ++i)
            for (Index j = 0; j < nStar; ++j)
                A1(i, j) = starBasis.CellProduct(
                    I, [&](double x) { return starBasis.Prime(I, i, x); },
                    [&](double x) { return starBasis.Prime(I, j, x); });

        // A2 = ( phi_j, d_x chi_i )_K
        Matrix A2(nStar, nLocal);
        for (Index i = 0; i < nStar; ++i)
            for (Index j = 0; j < nLocal; ++j)
                A2(i, j) = starBasis.CellProduct(
                    I, [&](double x) { return starBasis.Prime(I, i, x); },
                    [&](double x) { return basis.Evaluate(I, j, x); });

        // b1 = ( chi_j, 1 )_K and b2 = ( phi_j, 1 )_K
        Vector b1 = starBasis.getIntegrationWeights(I);
        Vector b2 = basis.getIntegrationWeights(I);

        // A9 = ( chi_m, phi_i )_K
        Matrix A9(nLocal, nStar);
        for (Index i = 0; i < nLocal; ++i)
            for (Index m = 0; m < nStar; ++m)
                A9(i, m) = basis.CellProduct(
                    I, [&](double x) { return basis.Evaluate(I, i, x); },
                    [&](double x) { return starBasis.Evaluate(I, m, x); });

        // V = phi_j evaluated at the star nodes
        Matrix V(nStar, nLocal);
        for (Index m = 0; m < nStar; ++m)
            for (Index j = 0; j < nLocal; ++j)
                V(m, j) = basis.Evaluate(I, j, I.fromRef(starNodes(m)));

        // The bordered system [ A1 b1^T ; b1 0 ]. It is nonsingular because the
        // only kernel vector of A1 is the constant, and ( 1, 1 )_K = h != 0.
        Matrix S(nStar + 1, nStar + 1);
        S.setZero();
        S.topLeftCorner(nStar, nStar) = A1;
        S.block(0, nStar, nStar, 1) = b1;
        S.block(nStar, 0, 1, nStar) = b1.transpose();

        // Right-hand sides as operators on alpha_q and beta_u separately, so the
        // solve yields B11 and B12 directly rather than a vector.
        Matrix Rq(nStar + 1, nLocal);
        Rq.setZero();
        Rq.topRows(nStar) = A2;

        Matrix Ru(nStar + 1, nLocal);
        Ru.setZero();
        Ru.row(nStar) = b2.transpose();

        Eigen::FullPivLU<Matrix> Slu(S);

        // Materialise the solves before slicing. Eigen's solve() returns a lazy
        // Solve<> expression with no coefficient accessor, so Solve<>::topRows()
        // is not a valid block of an evaluated matrix -- it compiles, and then
        // corrupts the heap. Under -DEIGEN_USE_BLAS (which this project builds
        // with) the symptom was a SIGSEGV inside free() in this constructor.
        const Matrix Xq = Slu.solve(Rq);
        const Matrix Xu = Slu.solve(Ru);

        B11_.emplace_back(Xq.topRows(nStar));
        B12_.emplace_back(Xu.topRows(nStar));
        V_.emplace_back(std::move(V));
        A9_.emplace_back(std::move(A9));
        b1_.emplace_back(b1);
    }

    starMem_.assign(nCells * nVars * nStar, 0.0);
    uStar_.reserve(nVars);
    for (Index var = 0; var < nVars; ++var)
    {
        uStar_.emplace_back(grid, starBasis);
        uStar_.back().Map(starMem_.data() + var * nStar,
                          static_cast<size_t>(nVars * nStar));
    }
}

void Postprocessor::computeUStar(DGSoln const &Y)
{
    const Index nCells = grid.getNCells();

    for (Index cell = 0; cell < nCells; ++cell)
    {
        for (Index var = 0; var < nVars; ++var)
        {
            uStar_[var].getCoeff(cell).second =
                B11_[cell] * Y.q(var).getCoeff(cell).second +
                B12_[cell] * Y.u(var).getCoeff(cell).second;
        }
    }
}

GlobalState Postprocessor::evalOnStarNodes(DGSoln const &Y) const
{
    const Index nCells = grid.getNCells();
    const Index nStar = k + 2;

    // GlobalState's second argument is a per-cell dof count minus one, not a
    // polynomial degree: passing k+1 is what makes cellwiseVariable() and its
    // siblings return nStar columns.
    GlobalState out(nCells, k + 1, nVars, nScalars, nAux);

    for (Index cell = 0; cell < nCells; ++cell)
    {
        Matrix const &Vc = V_[cell];

        for (Index var = 0; var < nVars; ++var)
        {
            // u* is nodal on the star nodes, so its coefficients *are* its
            // values there. Everything else is a degree-k field sampled at
            // those nodes.
            out.cellwiseVariable(cell).row(var) =
                uStar_[var].getCoeff(cell).second.transpose();
            out.cellwiseDerivative(cell).row(var) =
                (Vc * Y.q(var).getCoeff(cell).second).transpose();
            out.cellwiseFlux(cell).row(var) =
                (Vc * Y.sigma(var).getCoeff(cell).second).transpose();
        }

        for (Index a = 0; a < nAux; ++a)
            out.cellwiseAux(cell).row(a) =
                (Vc * Y.Aux(a).getCoeff(cell).second).transpose();
    }

    for (Index s = 0; s < nScalars; ++s)
        out.Scalars()(s) = Y.Scalar(s);

    return out;
}
