// Tests for the cellwise Jacobian blocks in Matrices.cpp.
//
// These are the matrices that make up MX, the local DG block that
// updateMatricesForJacSolve factorises. Nothing downstream checks them: the
// Jacobian is never assembled, so a wrong block shows up only as slow Newton
// convergence -- or, if it is wrong enough, as IDA giving up.
//
// Two structural facts make most of this testable cheaply:
//
//  * Several builders exist in *pairs* -- one taking a pointer-to-member and
//    evaluating the physics per node, one reading precomputed batched values
//    out of a GlobalStateMatrix. The batched forms are what the solver actually
//    calls; the per-node forms are the older, simpler code. Where they compute
//    the same quantity they must produce the same matrix, and that equivalence
//    is a far stronger check than either one against a hand-built reference.
//
//  * With interpolation the blocks are all `MassMatrix * diag(f(nodes))`, so
//    when f is constant across a cell the result is just `f * M` -- exactly
//    what the quadrature-based builders give too. Making the mock's
//    derivatives constant therefore lets the interpolatory and quadrature
//    forms be compared *exactly*, isolating the column layout from the
//    (legitimate) difference between the two quadrature schemes.

#include <boost/test/unit_test.hpp>

#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <boost/math/quadrature/gauss.hpp>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <toml.hpp>

using namespace toml::literals::toml_literals;

namespace
{

// A transport system whose derivatives are all distinct and easy to predict.
//
// nVars = 2, nAux = 1, nScalars = 2 so that every block has a nontrivial shape
// and a misplaced index cannot hide inside a square 1x1 layout.
//
// The sigma/source derivatives vary with position and state (so the per-node
// diagonal really is a diagonal of *different* numbers, and an interpolatory
// builder that silently used one value everywhere would be caught). The aux
// derivatives are deliberately CONSTANT and pairwise coprime -- see the header
// comment: that is what makes the two dAux_Mat overloads exactly comparable,
// and it makes any mis-slotted entry obvious by inspection.
class MatrixMock : public TransportSystem
{
public:
    MatrixMock()
        : TransportSystem({.variables = numberedFields(2),
                           .scalars = numberedScalars(2),
                           .aux = numberedAux(1)})
    {
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index i, const State &s, Position x, Time) override
    {
        return (1.0 + 0.3 * i) * s.q(i) + 0.2 * s.u(0) * s.u(0) +
               0.5 * s.phi(0) * x;
    }
    Value Sources(Index i, const State &s, Position x, Time) override
    {
        return 0.7 * s.u(i) + 0.1 * x + 0.4 * s.scalar(0);
    }

    // d(sigma_i)/dq_j -- varies with x and with the state.
    void dSigmaFn_dq(Index i, VectorRef v, const State &s, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 0.31 + 0.11 * (i * nVars + j) + 0.7 * x + 0.13 * s.u(j);
    }
    void dSigmaFn_du(Index i, VectorRef v, const State &s, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 0.17 + 0.23 * (i * nVars + j) - 0.5 * x + 0.09 * s.q(j);
    }
    void dSources_du(Index i, VectorRef v, const State &s, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 1.3 + 0.19 * (i * nVars + j) + 0.4 * x * x + 0.07 * s.u(j);
    }
    void dSources_dq(Index i, VectorRef v, const State &s, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = -0.6 + 0.29 * (i * nVars + j) + 0.8 * x - 0.05 * s.q(j);
    }
    void dSources_dsigma(Index i, VectorRef v, const State &s, Position x, Time) override
    {
        for (Index j = 0; j < nVars; ++j)
            v[j] = 0.43 - 0.07 * (i * nVars + j) + 0.25 * x + 0.02 * s.sigma(j);
    }

    // Constant, so the interpolatory and quadrature builders must agree exactly.
    void dSources_dPhi(Index i, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
        v[0] = 1.9 + 0.5 * i;
    }
    void dSigma_dPhi(Index i, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
        v[0] = 0.6 + 0.25 * i;
    }
    void dSources_dScalars(Index i, VectorRef v, const State &, Position x, Time) override
    {
        for (Index j = 0; j < nScalars; ++j)
            v[j] = 2.1 + 0.3 * (i * nScalars + j) + 0.9 * x;
    }

    Value AuxG(Index, const State &s, Position, Time) override
    {
        return s.phi(0) - s.u(0) * s.u(0);
    }

    // Distinct primes in every slot: if a builder puts dG/du where dG/dq
    // belongs, the resulting matrix cannot accidentally still be right.
    void AuxGPrime(Index, State &out, const State &, Position, Time) override
    {
        out.zero();
        out.u(0) = 2.0;
        out.u(1) = 3.0;
        out.q(0) = 5.0;
        out.q(1) = 7.0;
        out.sigma(0) = 11.0;
        out.sigma(1) = 13.0;
        out.phi(0) = 17.0;
    }

    Value InitialValue(Index i, Position x) const override
    {
        return (1.0 + 0.5 * i) * x * (1.0 - x);
    }
    Value InitialDerivative(Index i, Position x) const override
    {
        return (1.0 + 0.5 * i) * (1.0 - 2.0 * x);
    }
    Value InitialAuxValue(Index, Position x) const override { return 0.25 + x * x; }
    Value InitialScalarValue(Index s) const override { return 0.5 + 0.25 * s; }

    Value ScalarG(Index s, GlobalState const &y, GlobalState const &,
                  std::vector<Position> const &, Values const &, Matrix const &,
                  Time) override
    {
        return y.Scalars()(s) - 1.0;
    }
};

// Everything a matrix test needs: a solver with matrices built, initial
// conditions applied, and yJac pointing at that state.
struct MatrixFixture
{
    static constexpr Index k = 3;
    static constexpr Index nCells = 4;

    Grid grid{0.0, 1.0, nCells};
    MatrixMock problem;
    SystemSolver sys{grid, k, &problem};
    SUNContext ctx = nullptr;
    N_Vector Y = nullptr, dYdt = nullptr;

    MatrixFixture()
    {
        sys.setTau(1.0);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext_Create(SUN_COMM_NULL, &ctx);

        DGSoln shape(problem.getNumVars(), grid, k, problem.getNumScalars(),
                     problem.getNumAux());
        Y = N_VNew_Serial(shape.getDoF(), ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);

        sys.setJacTime(0.0);
        sys.setAlpha(1.0);
        sys.setJacEvalY(Y, dYdt);
        sys.updateBoundaryConditions(0.0);
    }

    ~MatrixFixture()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }

    // Reproduce the batched derivative evaluation that updateMatricesForJacSolve
    // performs, so the batched builders can be driven the same way the solver
    // drives them.
    void computeBatched(GlobalStateMatrix &dSigma, GlobalStateMatrix &dSource,
                        GlobalStateMatrix &dAux)
    {
        const Index nVars = problem.getNumVars();
        const Index nScalars = problem.getNumScalars();
        const Index nAux = problem.getNumAux();

        for (Index var = 0; var < nVars; ++var)
        {
            dSigma.add(nCells, k, nVars, nScalars, nAux);
            dSource.add(nCells, k, nVars, nScalars, nAux);
        }
        for (Index a = 0; a < nAux; ++a)
            dAux.add(nCells, k, nVars, nScalars, nAux);

        problem.ComputePhysicsDerivatives({dSigma, dSource, dAux},
                                          sys.yJac.evalOnNodes(), sys.yJac.getPoints(),
                                          0.0);
    }
};

// Reference for every interpolatory block: M * diag(f evaluated at the nodes).
Matrix interpolatoryReference(DGSoln const &Y, Index cell,
                              void (TransportSystem::*dX_dZ)(Index, VectorRef,
                                                             const State &, Position,
                                                             Time),
                              TransportSystem &problem, Index nVars, Index k,
                              Grid const &grid)
{
    Matrix ref(nVars * (k + 1), nVars * (k + 1));
    ref.setZero();
    const Matrix M = Y.getBasis().MassMatrix(grid[cell]);

    for (Index XVar = 0; XVar < nVars; ++XVar)
    {
        for (Index j = 0; j < k + 1; ++j)
        {
            Vector vals(nVars);
            vals.setZero();
            const double xj = grid[cell].fromRef(Y.getBasis().Nodes(j));
            State s = Y.evalOnNode(cell, j);
            (problem.*dX_dZ)(XVar, vals, s, xj, 0.0);
            for (Index ZVar = 0; ZVar < nVars; ++ZVar)
                ref(XVar * (k + 1) + j, ZVar * (k + 1) + j) = vals[ZVar];
        }
        for (Index ZVar = 0; ZVar < nVars; ++ZVar)
            ref.block(XVar * (k + 1), ZVar * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
    }
    return ref;
}

} // namespace

BOOST_AUTO_TEST_SUITE(system_solver_matrix_tests)

// ------------------------------------------ the two DerivativeSubMatrix forms --

BOOST_AUTO_TEST_CASE(derivative_sub_matrix_overloads_agree)
{
    // The solver calls the batched (GlobalStateMatrix) overload; the per-node
    // (pointer-to-member) overload is what NLqMat/NLuMat/dSourced*_Mat use.
    // Both implement the same formula, so they must produce bit-comparable
    // matrices for every block and every cell. If the batched path ever picks
    // up the wrong slice of the GlobalState, this catches it immediately.
    MatrixFixture f;
    const Index nVars = f.problem.getNumVars();
    const Index dof = nVars * (MatrixFixture::k + 1);

    GlobalStateMatrix dSigma(nVars), dSource(nVars), dAux(f.problem.getNumAux());
    f.computeBatched(dSigma, dSource, dAux);

    struct Case
    {
        const char *name;
        std::vector<Eigen::Ref<Matrix>> (GlobalStateMatrix::*slice)(Index);
        GlobalStateMatrix *src;
        void (TransportSystem::*fn)(Index, VectorRef, const State &, Position, Time);
    };

    const Case cases[] = {
        {"dSigma/dq", &GlobalStateMatrix::Derivative, &dSigma, &TransportSystem::dSigmaFn_dq},
        {"dSigma/du", &GlobalStateMatrix::Variable, &dSigma, &TransportSystem::dSigmaFn_du},
        {"dS/dsigma", &GlobalStateMatrix::Flux, &dSource, &TransportSystem::dSources_dsigma},
        {"dS/dq", &GlobalStateMatrix::Derivative, &dSource, &TransportSystem::dSources_dq},
        {"dS/du", &GlobalStateMatrix::Variable, &dSource, &TransportSystem::dSources_du},
    };

    for (auto const &c : cases)
        for (Index i = 0; i < MatrixFixture::nCells; ++i)
        {
            Matrix batched(dof, dof), perNode(dof, dof);
            f.sys.DerivativeSubMatrix(batched, (c.src->*c.slice)(i), f.sys.yJac, i);
            f.sys.DerivativeSubMatrix(perNode, c.fn, f.sys.yJac, i);

            BOOST_TEST((batched - perNode).norm() < 1e-12,
                       c.name << " cell " << i << ": batched and per-node forms differ by "
                              << (batched - perNode).norm());
        }
}

BOOST_AUTO_TEST_CASE(derivative_sub_matrix_is_mass_times_nodal_diagonal)
{
    // Pin the formula itself, not just self-consistency: M * diag(f(nodes)).
    MatrixFixture f;
    const Index nVars = f.problem.getNumVars();
    const Index dof = nVars * (MatrixFixture::k + 1);

    for (Index i = 0; i < MatrixFixture::nCells; ++i)
    {
        Matrix NLq(dof, dof), NLu(dof, dof);
        f.sys.NLqMat(NLq, f.sys.yJac, i);
        f.sys.NLuMat(NLu, f.sys.yJac, i);

        const Matrix refQ =
            interpolatoryReference(f.sys.yJac, i, &TransportSystem::dSigmaFn_dq,
                                   f.problem, nVars, MatrixFixture::k, f.grid);
        const Matrix refU =
            interpolatoryReference(f.sys.yJac, i, &TransportSystem::dSigmaFn_du,
                                   f.problem, nVars, MatrixFixture::k, f.grid);

        BOOST_TEST((NLq - refQ).norm() < 1e-12);
        BOOST_TEST((NLu - refU).norm() < 1e-12);

        // The off-diagonal (XVar, ZVar) blocks must be present -- the mock has
        // genuine cross-variable coupling, so a builder that only filled the
        // diagonal blocks would still pass a norm test against zero.
        BOOST_TEST(NLq.block(0, MatrixFixture::k + 1, MatrixFixture::k + 1,
                             MatrixFixture::k + 1)
                       .norm() > 1e-3);
    }

    // And the source blocks, which route through the same helper.
    for (Index i = 0; i < MatrixFixture::nCells; ++i)
    {
        Matrix Su(dof, dof), Sq(dof, dof), Ss(dof, dof);
        f.sys.dSourcedu_Mat(Su, f.sys.yJac, i);
        f.sys.dSourcedq_Mat(Sq, f.sys.yJac, i);
        f.sys.dSourcedsigma_Mat(Ss, f.sys.yJac, i);

        BOOST_TEST((Su - interpolatoryReference(f.sys.yJac, i,
                                                &TransportSystem::dSources_du, f.problem,
                                                nVars, MatrixFixture::k, f.grid))
                       .norm() < 1e-12);
        BOOST_TEST((Sq - interpolatoryReference(f.sys.yJac, i,
                                                &TransportSystem::dSources_dq, f.problem,
                                                nVars, MatrixFixture::k, f.grid))
                       .norm() < 1e-12);
        BOOST_TEST((Ss - interpolatoryReference(f.sys.yJac, i,
                                                &TransportSystem::dSources_dsigma,
                                                f.problem, nVars, MatrixFixture::k,
                                                f.grid))
                       .norm() < 1e-12);
    }
}

// --------------------------------------------------------------- scalars --

BOOST_AUTO_TEST_CASE(d_sources_d_scalars_mat_is_the_projected_derivative)
{
    // v_(var,j),s = Int_I dS_var/dmu_s * phi_j dx, built by Gauss quadrature.
    // The mock's dSources_dScalars is affine in x, so a two-point rule is
    // already exact and an independent high-order quadrature must agree to
    // round-off.
    MatrixFixture f;
    const Index nVars = f.problem.getNumVars();
    const Index nScalars = f.problem.getNumScalars();
    const Index k = MatrixFixture::k;

    for (Index cell = 0; cell < MatrixFixture::nCells; ++cell)
    {
        Matrix mat(nVars * (k + 1), nScalars);
        f.sys.dSources_dScalars_Mat(mat, f.sys.yJac, cell, 0.0);

        Interval const &I = f.grid[cell];
        auto const &basis = f.sys.yJac.getBasis();

        // Independent 20-point Gauss-Legendre reference.
        boost::math::quadrature::gauss<double, 20> gauss;

        for (Index var = 0; var < nVars; ++var)
            for (Index s = 0; s < nScalars; ++s)
                for (Index j = 0; j < k + 1; ++j)
                {
                    auto integrand = [&](double x)
                    {
                        Vector vals(nScalars);
                        vals.setZero();
                        State st = f.sys.yJac.eval(x);
                        f.problem.dSources_dScalars(var, vals, st, x, 0.0);
                        return vals[s] * basis.Evaluate(I, j, x);
                    };
                    const double ref = gauss.integrate(integrand, I.x_l, I.x_u);
                    BOOST_TEST(mat(var * (k + 1) + j, s) == ref,
                               boost::test_tools::tolerance(1e-10));
                }
    }
}

// ------------------------------------------------- the auxiliary couplings --

BOOST_AUTO_TEST_CASE(d_phi_mat_and_d_sourced_phi_mat_agree_for_a_constant_derivative)
{
    // dPhi_Mat is interpolatory (M * diag) and dSourcedPhi_Mat integrates
    // dS/dphi * phi_j * phi_l by quadrature. Those differ in general, but the
    // mock's dSources_dPhi is constant on each cell, where both reduce to
    // c * M. Any disagreement here is a layout error, not a quadrature choice.
    MatrixFixture f;
    const Index nVars = f.problem.getNumVars();
    const Index nAux = f.problem.getNumAux();
    const Index k = MatrixFixture::k;

    GlobalStateMatrix dSigma(nVars), dSource(nVars), dAux(nAux);
    f.computeBatched(dSigma, dSource, dAux);

    for (Index i = 0; i < MatrixFixture::nCells; ++i)
    {
        Matrix interp(nVars * (k + 1), nAux * (k + 1));
        Matrix quad(nVars * (k + 1), nAux * (k + 1));

        f.sys.dPhi_Mat(interp, dSource.Aux(i), f.sys.yJac, i);
        f.sys.dSourcedPhi_Mat(quad, f.sys.yJac, i);

        BOOST_TEST((interp - quad).norm() < 1e-10,
                   "cell " << i << ": ||interp - quad|| = " << (interp - quad).norm());

        // Not vacuous: the block is genuinely nonzero.
        BOOST_TEST(interp.norm() > 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(d_aux_mat_overloads_agree)
{
    // Both overloads fill the aux row of MX: the nAux*(k+1) rows against the
    // full [ sigma | q | u | phi ] column layout. The solver calls the batched
    // one, from both updateMatricesForJacSolve and
    // initializeMatricesForAdjointSolve; the per-node one is unused but is
    // the reference implementation of the layout, and it is the layout MX's
    // other blocks are written against.
    //
    // The mock's aux derivatives are constant, so the interpolatory and
    // quadrature forms coincide and the two must agree exactly.
    MatrixFixture f;
    const Index nVars = f.problem.getNumVars();
    const Index nAux = f.problem.getNumAux();
    const Index k = MatrixFixture::k;
    const Index cols = (3 * nVars + nAux) * (k + 1);

    GlobalStateMatrix dSigma(nVars), dSource(nVars), dAux(nAux);
    f.computeBatched(dSigma, dSource, dAux);

    for (Index i = 0; i < MatrixFixture::nCells; ++i)
    {
        Matrix batched(nAux * (k + 1), cols), perNode(nAux * (k + 1), cols);
        f.sys.dAux_Mat(batched, dAux, f.sys.yJac, i);
        f.sys.dAux_Mat(perNode, f.sys.yJac, i);

        BOOST_TEST((batched - perNode).norm() < 1e-10,
                   "cell " << i << ": ||batched - perNode|| = "
                           << (batched - perNode).norm());
    }
}

BOOST_AUTO_TEST_CASE(d_aux_mat_puts_each_derivative_in_the_right_column_block)
{
    // The column layout of MX is [ sigma | q | u | phi ], each block
    // nVars*(k+1) wide (nAux*(k+1) for phi) -- see updateMatricesForJacSolve,
    // which writes NLq at column nVars*(k+1) and NLu at 2*nVars*(k+1).
    //
    // With constant aux derivatives every block must be exactly
    // (that derivative) * MassMatrix, so this reads off which value landed
    // where. Pinning it explicitly means a future refactor of the batched
    // builder cannot quietly reintroduce a scramble that the
    // overloads-agree test would only catch if *both* forms were changed.
    MatrixFixture f;
    const Index nVars = f.problem.getNumVars();
    const Index nAux = f.problem.getNumAux();
    const Index k = MatrixFixture::k;
    const Index cols = (3 * nVars + nAux) * (k + 1);
    const Index cell = 1;

    GlobalStateMatrix dSigma(nVars), dSource(nVars), dAux(nAux);
    f.computeBatched(dSigma, dSource, dAux);

    Matrix mat(nAux * (k + 1), cols);
    f.sys.dAux_Mat(mat, dAux, f.sys.yJac, cell);

    const Matrix M = f.sys.yJac.getBasis().MassMatrix(f.grid[cell]);

    // The values AuxGPrime writes, by slot.
    const double dG_dsigma[2] = {11.0, 13.0};
    const double dG_dq[2] = {5.0, 7.0};
    const double dG_du[2] = {2.0, 3.0};
    const double dG_dphi = 17.0;

    for (Index var = 0; var < nVars; ++var)
    {
        const Index sigmaCol = var * (k + 1);
        const Index qCol = nVars * (k + 1) + var * (k + 1);
        const Index uCol = 2 * nVars * (k + 1) + var * (k + 1);

        BOOST_TEST((mat.block(0, sigmaCol, k + 1, k + 1) - dG_dsigma[var] * M).norm() < 1e-10,
                   "sigma block, var " << var);
        BOOST_TEST((mat.block(0, qCol, k + 1, k + 1) - dG_dq[var] * M).norm() < 1e-10,
                   "q block, var " << var);
        BOOST_TEST((mat.block(0, uCol, k + 1, k + 1) - dG_du[var] * M).norm() < 1e-10,
                   "u block, var " << var);
    }

    BOOST_TEST((mat.block(0, 3 * nVars * (k + 1), k + 1, k + 1) - dG_dphi * M).norm() < 1e-10,
               "phi block");
}

BOOST_FIXTURE_TEST_CASE(initialiseMatrices_rebuilds_rather_than_grows, MatrixFixture)
{
    // Every cellwise container is filled by emplace_back, so initialiseMatrices()
    // has to clear them first or a second call appends a second set. It clears
    // through clearCellwiseVecs(), whose list used to omit D_cellwise, CEBlocks and
    // MXSolvers -- and appending to those is worse than a leak, because indices run
    // 0..nCells-1 and so keep reaching the *stale* front half.
    //
    // The fixture has already called initialiseMatrices() once, so this is the
    // second call. It is not a hypothetical route: PrintDebugInfo() calls it
    // unguarded on an already-initialised solver.
    sys.initialiseMatrices();

    auto sizes = {std::pair{"XMats", sys.XMats.size()},
                  std::pair{"MBlocks", sys.MBlocks.size()},
                  std::pair{"CG_cellwise", sys.CG_cellwise.size()},
                  std::pair{"RF_cellwise", sys.RF_cellwise.size()},
                  std::pair{"A_cellwise", sys.A_cellwise.size()},
                  std::pair{"B_cellwise", sys.B_cellwise.size()},
                  std::pair{"D_cellwise", sys.D_cellwise.size()},
                  std::pair{"E_cellwise", sys.E_cellwise.size()},
                  std::pair{"C_cellwise", sys.C_cellwise.size()},
                  std::pair{"G_cellwise", sys.G_cellwise.size()},
                  std::pair{"H_cellwise", sys.H_cellwise.size()},
                  std::pair{"Csigma_cellwise", sys.Csigma_cellwise.size()},
                  std::pair{"Cq_cellwise", sys.Cq_cellwise.size()},
                  std::pair{"CEBlocks", sys.CEBlocks.size()},
                  std::pair{"MXSolvers", sys.MXSolvers.size()}};

    for (auto const &[name, size] : sizes)
        BOOST_TEST(size == static_cast<size_t>(nCells),
                   name << " holds " << size << " entries after two "
                        << "initialiseMatrices() calls, expected " << nCells);

    // And every surviving entry is properly shaped. Weaker than the size checks
    // above -- dropping any of the three clear() calls is caught by those, not by
    // these -- but it is what would catch MXSolvers being sized somewhere other
    // than the emplace_back in initialiseMatrices, which is how it came to hold
    // 2 * nCells entries with default-constructed (rows() == 0) ones at the front.
    for (Index i = 0; i < nCells; ++i)
    {
        BOOST_TEST(sys.MXSolvers[i].rows() == sys.MBlocks[i].rows(),
                   "cell " << i << " MX solver is " << sys.MXSolvers[i].rows()
                           << " rows, expected " << sys.MBlocks[i].rows());
        BOOST_TEST(sys.CEBlocks[i].rows() == sys.MBlocks[i].rows(),
                   "cell " << i << " CEBlock has the wrong height");
        BOOST_TEST(sys.D_cellwise[i].rows() == problem.getNumVars() * (k + 1),
                   "cell " << i << " D block has the wrong height");
    }
}

BOOST_AUTO_TEST_SUITE_END()
