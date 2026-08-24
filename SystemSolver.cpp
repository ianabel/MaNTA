#include "SystemSolver.hpp"
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of sunrealtype, sunindextype  */
#include <nvector/nvector_serial.h>         /* access to serial N_Vector            */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <toml.hpp>
#include <ostream>
#include <print>

#include "State.hpp"
#include "Types.hpp"
#include "gridStructures.hpp"
// The scalar hooks are handed the quadrature weights and the boundary basis
// values, so the case does not have to pick a rule of its own. Both are cached
// here, keyed on basis order and grid.
#include "PyIntegrator.hpp"

SystemSolver::SystemSolver(Grid const &Grid, unsigned int polyNum, TransportSystem *transpSystem)
    : grid(Grid), k(polyNum), nCells(Grid.getNCells()), nVars(transpSystem->getNumVars()), nScalars(transpSystem->getNumScalars()), nAux(transpSystem->getNumAux()), y(nVars, grid, k, nScalars, nAux), dydt(nVars, grid, k, nScalars, nAux), yJac(nVars, grid, k, nScalars, nAux), dydtJac(nVars, grid, k, nScalars, nAux), problem(transpSystem)
{
    if (SUNContext_Create(SUN_COMM_NULL, &ctx) < 0)
        throw std::runtime_error("Unable to allocate SUNDIALS Context, aborting.");
    yJacMem = new double[yJac.getDoF()];
    yJac.Map(yJacMem);
    dydtJacMem = new double[yJac.getDoF()];
    dydtJac.Map(dydtJacMem);
    S_DOF = k + 1;
    U_DOF = k + 1;
    Q_DOF = k + 1;
    SQU_DOF = U_DOF + Q_DOF + S_DOF;

    AUX_DOF = k + 1;
    localDOF = nVars * SQU_DOF + nAux * AUX_DOF;

    logmsg<LOG_LEVEL::INFO>("Total HDG degrees of freedom {}", (localDOF)*nCells + (nCells + 1) * nVars + nScalars );
    if (nScalars > 0)
    {
        v = new N_Vector[nScalars];
        w = new N_Vector[nScalars];
        for (Index i = 0; i < nScalars; ++i)
        {
            v[i] = N_VNew_Serial(y.getDoF(), ctx);
            w[i] = N_VNew_Serial(y.getDoF(), ctx);
        }
    }
    else
    {
        v = nullptr;
        w = nullptr;
    }
    initialised = false; // Need to know tau to call this
}

SystemSolver::~SystemSolver()
{
    delete[] yJacMem;
    delete[] dydtJacMem;
    if (nScalars > 0)
    {
        for (Index i = 0; i < nScalars; ++i)
        {
            N_VDestroy(v[i]);
            N_VDestroy(w[i]);
        }
        delete[] v;
        delete[] w;
    }
    SUNContext_Free(&ctx);
}

void SystemSolver::setInitialConditions(N_Vector &Y, N_Vector &dYdt)
{
    logmsg<LOG_LEVEL::INFO>("Setting initial conditions");
    t = t0;
    y.Map(N_VGetArrayPointer(Y));
    dydt.Map(N_VGetArrayPointer(dYdt));

    resetCoeffs();
    if (!initialised)
        throw std::logic_error("setInitialConditions can only be called after initialising the matrices");

    // The initial dydt below is solved out of the residual equation, which reads
    // RF_cellwise and L_global -- the Dirichlet and Neumann boundary data. Both
    // are functions of time, and nothing else has evaluated them at t0 yet:
    // initialiseMatrices only sizes them, and on a *second* initialize() it is
    // skipped entirely, so they would otherwise still hold whatever the previous
    // run's last residual evaluation left behind -- its final-time boundary
    // values. That made the initial condition of every run after the first
    // subtly wrong, and was the difference that turned the once-broken
    // IDACalcIC (see Solver.cpp) into a hard failure on the second run only.
    updateBoundaryConditions(t0);

    if (problem->isRestarting())
    {
        DGSoln const &restart = problem->getRestartY();
        const bool sameGrid = restart.getGrid() == grid;
        const bool sameDiscretisation = restart.getBasis().Order() == k && sameGrid;

        // Recorded because initialize() decides whether to skip IDACalcIC from it.
        // A restart skips by default on the grounds that it resumes from a state
        // the previous run had already driven onto the constraint manifold -- and
        // that is only true on the copy path. The projection below transfers u, q,
        // aux and the scalars and then *rebuilds* sigma and the trace, so the state
        // it produces is a guess like any other. Measured on AuxVarTest resuming at
        // a lower degree: skipping there fails with IDA_ERR_FAIL, and running
        // IDACalcIC completes the run.
        restartWasProjected = !sameDiscretisation;

        if (sameDiscretisation)
        {
            // Copy restart values into y. Bit for bit, which is what the restart
            // round-trip regression cases compare, so this stays the path taken
            // whenever the discretisation matches.
            y.copy(problem->getRestartY());
        }
        else
        {
            // A restart whose polynomial degree differs from the run's. copy()
            // refuses it -- DGApproxImpl::copy throws on a different order and
            // DGSolnImpl::copy on a different grid -- so the state is projected
            // instead, by evaluating the stored element polynomials.
            //
            // AssignU's operator= is an L2 projection rather than an
            // interpolation, and that is the better of the two here: refining
            // puts the old polynomial inside the new space, where both are
            // exact, while coarsening makes the L2 projection the optimal
            // approximation and interpolation merely a choice of points.
            //
            // q is projected as its own field rather than differentiated from
            // the projected u. It is an unknown of the system in its own right,
            // and making it consistent with u is worth about 30% of the
            // subsequent solve; a fit-then-differentiate route was measured
            // 0.245 out pointwise on a field ranging to -1.0.
            y.AssignU([&restart](Index i, Position x) { return restart.u(i)(x); });
            y.AssignQ([&restart](Index i, Position x) { return restart.q(i)(x); });

            if (nAux > 0)
                y.AssignAux([&restart](Index i, Position x) { return restart.Aux(i)(x); });

            // Scalars carry no spatial discretisation, so they transfer as they
            // are whatever the degree.
            for (Index s = 0; s < nScalars; ++s)
                y.Scalar(s) = restart.Scalar(s);

            // Neither does the trace. lambda is one value per face per variable
            // -- DGSoln::Map gives it nCells + 1 entries and no basis at all --
            // so a change of *degree* leaves it transferable verbatim even though
            // copy() refuses the state as a whole. Only a change of mesh makes it
            // meaningless, and then there is nothing to do but rebuild it.
            if (sameGrid)
                for (Index v = 0; v < nVars; ++v)
                    y.lambda(v) = restart.lambda(v);
        }

        ApplyDirichletBCs(y); // If dirichlet, overwrite with those boundary conditions

        // sigma is *loaded* on the copy path, not recomputed.
        //
        // y.copy() above already brought it across -- DGSoln::copy transfers sigma
        // along with u, q, lambda, the scalars and aux -- and nothing between here
        // and there disturbs the inputs it was built from: ApplyDirichletBCs
        // touches lambda only, and u and q are bit for bit the file's. So
        // AssignSigma would rebuild sigma from exactly the state the file's sigma
        // was already built from, at the price of a full ComputePhysics over every
        // node.
        //
        // That price is the point. ComputePhysics evaluates SigmaFn *and* Sources
        // for every variable and AuxG for every auxiliary one, at every node, and
        // this block uses only the first of the three -- the sources and the aux
        // values are computed and dropped. On a case whose flux is expensive that
        // is a whole physics sweep bought for nothing, on the very path that
        // exists because the state is already right.
        //
        // The projection path still needs it: there sigma was never transferred,
        // since a degree change leaves the stored coefficients in the wrong space,
        // and rebuilding it from the projected u and q is what makes the sigma row
        // exact.
        if (!sameDiscretisation)
        {
            GlobalState initialState = y.evalOnNodes(); // only need u and q so this is ok
            const auto points = y.getPoints();
            auto physics_vals = problem->ComputePhysics(initialState, points, t);
            for (Index var = 0; var < nVars; var++)
            {
                // set flux for each variable, casting to a row vector and making sure to remember minus sign
                initialState.Flux().row(var) = -static_cast<Eigen::Matrix<double, 1, Eigen::Dynamic>>(physics_vals[0][var]);
            }
            y.AssignSigma(initialState);
        }

        // Keep the trace the file carries, on the path where it is meaningful.
        //
        // EvaluateLambda() sets lambda to {{u}}, the DG average of the two cell
        // traces (DGSoln.hpp). That is a reasonable *guess* and it is not the
        // equation lambda satisfies -- the HDG trace row is
        // Csigma sigma + Cq q + G_c u + H lambda = L(t) -- so applying it to a
        // restart throws away a converged trace and replaces it with something
        // that does not solve anything. Measured on a TestDiffusion round trip at
        // Absolute_tolerance = 1e-8, that single call takes the weighted residual
        // of the stored state from 2.6e-3 to 556, and it is why a restart used to
        // need about ten times as many residual evaluations inside IDACalcIC as a
        // cold start (Tests/README.md).
        //
        // The projection path keeps it too, whenever the *mesh* matches. That used
        // to read "there is no stored trace to keep", which was wrong: lambda has
        // no polynomial degree -- DGSoln::Map gives it nCells + 1 entries and no
        // basis -- so a change of degree leaves it transferable verbatim even
        // though copy() refuses the state as a whole. Only a change of mesh makes
        // it meaningless, and {{u}} is then the right guess to build.
        //
        // It is worth more than it looks. The q row carries a <lambda, v n> term,
        // so discarding the trace breaks q as well as lambda: on a LinearDiffusion
        // restart coarsened from k = 4 to k = 3 at atol 1e-10, keeping it takes the
        // q block from 7.3e7 to 3.2e-7 -- machine zero, the same as the copy path.
        // That is the L2 projection commuting with the weak q equation, which it
        // does exactly, and only the trace was standing in the way.
        //
        // It does not make a projected restart consistent, and nothing can. A
        // change of degree changes the *equations*, not just the representation,
        // and each direction breaks the block whose equation moved:
        //
        //   coarsening   q transfers exactly (above), lambda does not -- u and q
        //                lost information, so the numerical flux at each face
        //                moved and the stored trace no longer balances it
        //   refinement   lambda transfers exactly (measured 4e-9, nothing at the
        //                faces changed) and q does not -- the degree-(k+1) test
        //                functions impose a q row the coarse state was never asked
        //                to satisfy
        //
        // sigma is never the problem, linear flux or not, because it is not
        // transferred at all on this path: AssignSigma rebuilds it from the
        // projected u and q by evaluating the physics at the run's own nodes, so
        // it satisfies its own row by construction. Measured at 1e-18 in every
        // direction on both LinearDiffusion and NonlinDiffTest.
        //
        // The nonlinearity does show up, but in lambda rather than sigma. On
        // refinement a linear flux gives sigma = -kappa q, which is embedded
        // exactly along with q, so the face flux does not move and lambda stays at
        // 4e-9. A nonlinear sigma_hat interpolated on the *new* node set is a
        // different polynomial from the old interpolant, so the face flux does
        // move: NonlinDiffTest refining 3 -> 4 leaves lambda at 4.7e8.
        //
        // q on its own can be repaired without any Jacobian. Its row is
        // A q = -B^T u + C^T lambda - RF with A the cell mass matrix, so given u
        // and lambda it is one (k+1)x(k+1) dense solve per cell per variable and
        // touches no physics; tried as an experiment it took q to 5e-15 in every
        // direction on both cases. It is not done here because it changes no
        // outcome -- every degree-transfer run tested returns exactly what it did
        // without it -- and the inconsistency simply migrates into lambda, whose
        // row then reads off a different q and sigma. lambda is the coupled one:
        // its row needs sigma, sigma needs q, q needs lambda, so closing the loop
        // is a nonlinear solve on the algebraic subsystem rather than a
        // substitution. That is exactly what IDACalcIC does, which is why
        // initialize() runs it on this path rather than skipping as it does for a
        // copied restart.
        //
        // ApplyDirichletBCs stays *above* this rather than below it, which looks
        // wrong and is not. EvaluateLambda overwrites every entry including the
        // boundary ones, so in that order the boundary data it writes is
        // discarded again on the projection path -- but moving it below breaks
        // the AuxVarTest restart round trip outright (IDA_CONV_FAIL at the first
        // step of the resumed run). The two differ only in what a Dirichlet end's
        // trace holds -- the boundary datum, or u's own trace there -- and that
        // node has an identically zero row and column, so it is not the residual
        // that notices: it is IDA's error test, which weights lambda like
        // everything else.
        //
        // The reason given for that used to be that the case sat close to the
        // edge -- rtol 1e-6 with nAux > 0, where restarting was documented as
        // marginal. That premise is gone: AuxVarTest's missing dSigma_dPhi block
        // is declared and all three round trips now survive 1e-6 / 1e-8 with the
        // CalcIC skip armed. So the ordering may no longer be load-bearing at
        // all. It has not been re-measured since, which is the only reason it is
        // still written this way; reordering is a behaviour change wanting its
        // own regression run, and is a separate question from keeping the trace.
        if (!sameDiscretisation && !sameGrid)
            y.EvaluateLambda();
    }
    else
    {
        // Lambdas rather than std::bind_front. libstdc++ gives _Bind_front two
        // implementations of operator(), chosen by #if
        // __cpp_explicit_this_parameter -- four cv/ref-qualified overloads, or one
        // with an explicit object parameter. clang only defines that macro from
        // clang 20, and in libstdc++ before 14.4 the deducing-this overload fails
        // std::function's _Callable check when the _Bind_front is converted as an
        // lvalue. So (clang >= 20, libstdc++ < 14.4) rejected these three calls,
        // which is exactly what ubuntu-24.04 gives CI's clang legs. Nothing here
        // needed bind_front, so the portable spelling is also the clearer one.
        y.AssignU([this](Index i, Position x) { return problem->InitialValue(i, x); });
        y.AssignQ([this](Index i, Position x) { return problem->InitialDerivative(i, x); });

        for (Index s = 0; s < nScalars; ++s)
        {
            y.Scalar(s) = problem->InitialScalarValue(s);
        }

        if (nAux > 0)
        {
            y.AssignAux([this](Index i, Position x) { return problem->InitialAuxValue(i, x); });
        }

        ApplyDirichletBCs(y);

        // Zero most of dydt, we only have to set it to nonzero values for the differential parts of y

        // Vectorize initial flux calculation
        GlobalState initialState = y.evalOnNodes(); // only need u and q so this is ok
        const auto points = y.getPoints();
        auto physics_vals = problem->ComputePhysics(initialState, points, t);
        for (Index var = 0; var < nVars; var++)
        {
            // set flux for each variable, casting to a row vector and making sure to remember minus sign
            initialState.Flux().row(var) = -static_cast<Eigen::Matrix<double, 1, Eigen::Dynamic>>(physics_vals[0][var]);
        }
        y.AssignSigma(initialState);

        y.EvaluateLambda();
    }

    dydt.zeroCoeffs();

    auto Source_vals = problem->ComputePhysics(y.evalOnNodes(), y.getPoints(), t)[1];
    for (Index var = 0; var < nVars; var++)
    {
        // Solver For dudt with dudt = X^-1( -B*Sig - D*U - E*Lam + F )
        Eigen::Vector2d lamCell;
        for (Index i = 0; i < nCells; i++)
        {
            Interval I = grid[i];

            // Evaluate Source Function
            Eigen::VectorXd S_cellwise(k + 1);

            auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);

            S_cellwise = y.getBasis().InterpolateOntoBasis( I, Source_vals[var](ind) );

            lamCell[0] = y.lambda(var)[i];
            lamCell[1] = y.lambda(var)[i + 1];
            // dudt.coeffs[ var ][ i ].second.setZero();
            auto const &sigma_vec = y.sigma(var).getCoeff(i).second;
            auto const &u_vec = y.u(var).getCoeff(i).second;
            dydt.u(var).getCoeff(i).second =
                XMats[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1).inverse() *
                (-B_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * sigma_vec - D_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * u_vec - E_cellwise[i].block(var * (k + 1), var * 2, k + 1, 2) * lamCell + RF_cellwise[i].block(nVars * (k + 1) + var * (k + 1), 0, k + 1, 1) + S_cellwise);
            // <cellwise derivative matrix> * dydt.u( var ).getCoeff( i ).second;
        }
    }
    for (Index s = 0; s < nScalars; ++s)
    {
        if (problem->isScalarDifferential(s))
        {
            dydt.Scalar(s) = problem->InitialScalarDerivative(s, y, dydt);
        }
    }
}

void SystemSolver::ApplyDirichletBCs(DGSoln &Y)
{
    for (Index i = 0; i < nVars; ++i)
    {
        if (problem->isLowerBoundaryDirichlet(i))
        {
            Y.lambda(i)(0) = problem->LowerBoundary(i, t);
        }

        if (problem->isUpperBoundaryDirichlet(i))
        {
            Y.lambda(i)(grid.getNCells()) = problem->UpperBoundary(i, t);
        }
    }
}

void SystemSolver::initialiseMatrices()
{
    // The postprocessed u* is reported in every output, so the reconstruction is
    // built whether or not the superconvergent residual is switched on. k = 0 is
    // the exception: the degree-0 NodalBasis returns from its constructor before
    // building Vandermonde or BarycentricWeights (Basis.hpp:369-377), so it
    // cannot be evaluated off-node and there is nothing to reconstruct.
    if (k >= 1)
        postprocessor = std::make_unique<Postprocessor>(grid, k, nVars, nScalars, nAux);
    else if (superconvergent)
        throw std::invalid_argument(
            "Superconvergent postprocessing requires Polynomial_degree >= 1");

    // These are temporary working space
    // Matrices we need per cell
    Eigen::MatrixXd A(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd B(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd D(nVars * (k + 1), nVars * (k + 1));
    // Two endpoints per cell
    Eigen::MatrixXd C(2 * nVars, nVars * (k + 1));
    Eigen::MatrixXd E(nVars * (k + 1), 2 * nVars);

    // Temporary per-variable matrices that will be assembled into the larger cell matrices as blocks
    Eigen::MatrixXd Avar(k + 1, k + 1);
    Eigen::MatrixXd Bvar(k + 1, k + 1);
    Eigen::MatrixXd Dvar(k + 1, k + 1);
    Eigen::MatrixXd Cvar(2, k + 1);
    Eigen::MatrixXd Evar(k + 1, 2);

    Eigen::MatrixXd HGlobalMat(nVars * (nCells + 1), nVars * (nCells + 1));
    HGlobalMat.setZero();
    K_global.resize(nVars * (nCells + 1), nVars * (nCells + 1));
    K_global.setZero();
    L_global.resize(nVars * (nCells + 1));
    L_global.setZero();

    clearCellwiseVecs();
    for (unsigned int i = 0; i < nCells; i++)
    {
        A.setZero();
        B.setZero();
        C.setZero();
        D.setZero();
        E.setZero();
        Interval const &I(grid[i]);
        for (Index var = 0; var < nVars; var++)
        {
            Avar.setZero();
            Bvar.setZero();
            Dvar.setZero();
            // A_ij = ( phi_j, phi_i )
            y.getBasis().MassMatrix(I, Avar);
            // B_ij = ( phi_i, phi_j' )
            y.getBasis().DerivativeMatrix(I, Bvar);

            // Now do all the boundary terms
            for (Eigen::Index i = 0; i < k + 1; i++)
            {
                for (Eigen::Index j = 0; j < k + 1; j++)
                {
                    Dvar(i, j) +=
                        tau(I.x_l) * y.getBasis().Evaluate(I, j, I.x_l) * y.getBasis().Evaluate(I, i, I.x_l) +
                        tau(I.x_u) * y.getBasis().Evaluate(I, j, I.x_u) * y.getBasis().Evaluate(I, i, I.x_u);
                }
            }

            A.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Avar;
            D.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Dvar;
            B.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Bvar;
        }

        A_cellwise.emplace_back(A);
        B_cellwise.emplace_back(B);
        D_cellwise.emplace_back(D);

        // M is the local DG Matrix
        Eigen::MatrixXd M(localDOF, localDOF);
        M.setZero();

        // row1
        M.block(0, 0, nVars * (k + 1), nVars * (k + 1)) = A;
        M.block(0, nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero();     // NLq added at Jac step
        M.block(0, 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero(); // NLu added at Jac step

        // row2
        M.block(nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)).setZero();
        M.block(nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = -A;
        M.block(nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = -B.transpose();

        // row3
        M.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)) = B;
        M.block(2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero();
        M.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = D; // X added at Jac step

        // TODO:  Consider factorization here (is M sparse nough to warrant a sparse implementation?)
        MBlocks.emplace_back(M);

        Eigen::MatrixXd CE_vec(localDOF, 2 * nVars);
        CE_vec.setZero();
        for (Index var = 0; var < nVars; var++)
        {
            Cvar.setZero();
            Evar.setZero();
            for (Index i = 0; i < k + 1; i++)
            {
                // C_ij = < psi_i, phi_j * n_x > , where psi_i are edge degrees of
                // freedom and n_x is the unit normal in the x direction
                // for a line, edge degrees of freedom are just 1 at each end
                Cvar(0, i) = -y.getBasis().Evaluate(I, i, I.x_l);
                Cvar(1, i) = y.getBasis().Evaluate(I, i, I.x_u);

                // E_ij = < phi_i, (- tau ) lambda >
                Evar(i, 0) = y.getBasis().Evaluate(I, i, I.x_l) * (-tau(I.x_l));
                Evar(i, 1) = y.getBasis().Evaluate(I, i, I.x_u) * (-tau(I.x_u));

                if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
                {
                    Cvar(0, i) = 0;
                    Evar(i, 0) = 0;
                }
                // should this be is upper boundary dirichlet?
                if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
                {
                    Cvar(1, i) = 0;
                    Evar(i, 1) = 0;
                }
            }

            // Construct per-cell Matrix solutions
            // ( A   NLu   NLq )^-1 [  0  ]
            // ( 0    A    B^T )    [ C^T ]
            // ( B    0     D  )    [  E  ]
            // These are the homogeneous solution, that depend on lambda
            C.block(var * 2, var * (k + 1), 2, k + 1) = Cvar;
            E.block(var * (k + 1), var * 2, k + 1, 2) = Evar;
        }

        CE_vec.block(0, 0, nVars * (k + 1), nVars * 2).setZero();
        CE_vec.block(nVars * (k + 1), 0, nVars * (k + 1), nVars * 2) = C.transpose();
        CE_vec.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * 2) = E;
        CE_vec.block(3 * nVars * (k + 1), 0, nAux * (k + 1), nVars * 2).setZero();
        CEBlocks.emplace_back(CE_vec);
        C_cellwise.emplace_back(C);
        E_cellwise.emplace_back(E);

        // To store the RHS. Sized and zeroed here; the boundary data that goes in
        // it is *time dependent*, so filling it is updateBoundaryConditions's job
        // and setInitialConditions calls that for t0 before reading it. This used
        // to carry a third copy of that loop, evaluated at a hardcoded t = 0.0 --
        // wrong for any run with t0 != 0, and stale on a second initialize(),
        // where initialiseMatrices is skipped and RF_cellwise still held the
        // previous run's final-time boundary values.
        RF_cellwise.emplace_back(nVars * 2 * (k + 1));

        // R is composed of parts of the values of
        // u on the total domain boundary
        // don't need to do RHS terms here, those are now in 'Sources'
        RF_cellwise[i].setZero();

        // For Neumann, need a structure more like
        // If (boundary cell)
        // C is multiplied by Q and not sigma, so we need to change the structure of CG cellwise, also need to zero the contribution of sigma at the boundary so it can float
        // if the flux is a complicated nonlinear function of the other variables, zero sigma does not necessarily mean zero q
        // So we set up a "double" C matrix that includes all of this information and set the sigma and q portions at once

        // Per-cell contributions to the global matrices K and F.
        // First fill G
        Eigen::MatrixXd G(2 * nVars, nVars * (k + 1));
        G.setZero();
        Eigen::MatrixXd Cq(2 * nVars, nVars * (k + 1));
        Eigen::MatrixXd Csigma(2 * nVars, nVars * (k + 1));
        Cq.setZero();
        Csigma.setZero();
        for (Index var = 0; var < nVars; var++)
        {
            Eigen::MatrixXd Gvar(2, k + 1);
            Eigen::MatrixXd Cq_var(2, k + 1);
            Eigen::MatrixXd Csigma_var(2, k + 1);
            Cq_var.setZero();
            Csigma_var.setZero();
            Gvar.setZero();
            for (Index i = 0; i < k + 1; i++)
            {
                // C_ij = < psi_i, phi_j * n_x > , where psi_i are edge degrees of
                // freedom and n_x is the unit normal in the x direction
                // for a line, edge degrees of freedom are just 1 at each end

                // Always set this for sigma
                Csigma_var(1, i) = y.getBasis().Evaluate(I, i, I.x_u);
                Csigma_var(0, i) = -y.getBasis().Evaluate(I, i, I.x_l);

                // Every non-Dirichlet end, Neumann and Mixed alike, is assembled
                // from the same two lines. The row this face contributes is
                //     (b q + d sigma).n + tau (u - lambda) + n a lambda = n c
                // so dividing by the outward normal leaves the case author's
                // `a u + b q + d sigma = c` with no normals in it -- the
                // convention today's Neumann already follows, the +-phi here and
                // the +-c in L_global cancelling to give `q = g` at both ends.
                //
                // Neumann arrives here as b = 1, or d = 1 under zeroFlux, through
                // effectiveLowerBoundary; that is the whole of what the flag
                // means, and it is now the only place the flag is read. The two
                // hand-written branches this replaces said the same thing in four
                // copies, one per (end, flag) pair, with the zeroFlux arms
                // restating an assignment made three lines above.
                if (I.x_l == grid.lowerBoundary() && !problem->isLowerBoundaryDirichlet(var))
                {
                    auto const bc = effectiveLowerBoundary(var);
                    Csigma_var(0, i) = -bc.d * y.getBasis().Evaluate(I, i, I.x_l);
                    Cq_var(0, i) = -bc.b * y.getBasis().Evaluate(I, i, I.x_l);
                }
                if (I.x_u == grid.upperBoundary() && !problem->isUpperBoundaryDirichlet(var))
                {
                    auto const bc = effectiveUpperBoundary(var);
                    Csigma_var(1, i) = bc.d * y.getBasis().Evaluate(I, i, I.x_u);
                    Cq_var(1, i) = bc.b * y.getBasis().Evaluate(I, i, I.x_u);
                }

                Gvar(0, i) = tau(I.x_l) * y.getBasis().Evaluate(I, i, I.x_l);

                // If Dirichlet, proceed as normal
                if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
                {
                    Csigma_var(0, i) = 0.0;
                    Gvar(0, i) = 0.0;
                }

                Gvar(1, i) = tau(I.x_u) * y.getBasis().Evaluate(I, i, I.x_u);
                if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
                {
                    Csigma_var(1, i) = 0.0;
                    Gvar(1, i) = 0.0;
                }
            }

            Csigma.block(2 * var, (k + 1) * var, 2,(k + 1)) = Csigma_var;
            Cq.block(2 * var, (k + 1) * var, 2, (k + 1)) = Cq_var;

            G.block(2 * var, (k + 1) * var, 2, (k + 1)) = Gvar;
        }

        //[ C 0 G 0 ] (4th index is aux vars)
        CG_cellwise.emplace_back(2 * nVars, localDOF);
        CG_cellwise[i].setZero();
      
        CG_cellwise[i].block(0, 2 * nVars * (k + 1), 2 * nVars, nVars * (k + 1)) = G; // this is the U block
        CG_cellwise[i].block(0, 0, 2 * nVars, nVars * (k + 1)) = Csigma; // This is the sigma block
        CG_cellwise[i].block(0, nVars * (k + 1), 2 * nVars, nVars * (k + 1)) = Cq; // This is the q block

        G_cellwise.emplace_back(G);
        Csigma_cellwise.emplace_back(Csigma);
        Cq_cellwise.emplace_back(Cq);

        // Now fill H
        Eigen::MatrixXd H(2 * nVars, 2 * nVars);
        H.setZero();
        for (Index var = 0; var < nVars; var++)
        {
            Eigen::MatrixXd Hvar(2, 2);
            Hvar.setZero();
            Hvar(0, 0) = -tau(I.x_l);
            Hvar(1, 0) = 0.0;
            Hvar(0, 1) = 0.0;
            Hvar(1, 1) = -tau(I.x_u);

            // A Mixed end's `a` coefficient lives here, on the lambda column,
            // rather than in G on the interior u. That is the form the HDG
            // literature uses -- the condition relates the *numerical flux* to
            // the *trace unknown*, not to the interior trace (Cui & Zhang,
            // refs/HDG-Helmholtz-Robin.pdf eq. 2.3 and its impedance condition)
            // -- and it is what keeps the row solvable for lambda when b = d = 0
            // is the only thing left. It carries the outward normal, so that
            // dividing the row through by n leaves a plain `a u` for the case
            // author: -a below, +a above.
            if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
                Hvar(0, 0) = 0.0;
            else if (I.x_l == grid.lowerBoundary())
                Hvar(0, 0) = -tau(I.x_l) - effectiveLowerBoundary(var).a;

            if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
                Hvar(1, 1) = 0.0;
            else if (I.x_u == grid.upperBoundary())
                Hvar(1, 1) = -tau(I.x_u) + effectiveUpperBoundary(var).a;

            H.block(2 * var, 2 * var, 2, 2) = Hvar;
            HGlobalMat.block(var * (nCells + 1) + i, var * (nCells + 1) + i, 2, 2) += Hvar;
        }

        H_cellwise.emplace_back(H);

        // L is the Neumann counterpart of RF_cellwise above, and equally time
        // dependent: updateBoundaryConditions fills it. It was zeroed before the
        // loop and stays zero until then.

        Eigen::MatrixXd X(nVars * (k + 1), nVars * (k + 1));
        X.setZero();
        for (Index var = 0; var < nVars; var++)
        {
            Eigen::MatrixXd Xvar(k + 1, k + 1);
            y.getBasis().MassMatrix(I, Xvar, [this, var](double x)
                    { return problem->aFn(var, x); });
            X.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Xvar;
        }
        XMats.emplace_back(X);
    
        // The solver for this cell's MX block, pre-sized so that the compute() in
        // updateMatricesForJacSolve does not have to reallocate. This is the only
        // place MXSolvers is sized: the constructor used to size it to nCells as
        // well, so it ended up 2 * nCells long with the *default-constructed*
        // entries at the front -- which are the ones every index reaches, making
        // the pre-sizing here dead and the compute() reallocate after all.
        Eigen::Index nDof = nVars * SQU_DOF + nAux * AUX_DOF;
        MXSolvers.emplace_back( nDof, nDof );
    }
    // Factorise the global H matrix
    H_global.compute(HGlobalMat);
    H_global_mat = HGlobalMat;

    // Just zero v & w
    for (Index i = 0; i < nScalars; ++i)
    {
        N_VConst(0.0, v[i]);
        N_VConst(0.0, w[i]);
    }

    // and zeros for N_global
    N_global = Matrix::Zero(nScalars, nScalars);

    initialised = true;
}

// Every per-cell container initialiseMatrices() appends to, so that calling it a
// second time rebuilds rather than grows. That has to be *all* of them: the list
// used to omit D_cellwise, CEBlocks and MXSolvers, so a second initialiseMatrices
// left those three holding 2 * nCells entries with the stale ones at the front --
// which is where every index into them lands. Only PrintDebugInfo() calls it
// unguarded today, so nothing had reached it, but the coupling is invisible from
// the append site. If you add a cellwise vector, add it here.
void SystemSolver::clearCellwiseVecs()
{
    XMats.clear();
    MBlocks.clear();
    CG_cellwise.clear();
    RF_cellwise.clear();
    A_cellwise.clear();
    B_cellwise.clear();
    D_cellwise.clear();
    E_cellwise.clear();
    C_cellwise.clear();
    G_cellwise.clear();
    H_cellwise.clear();
    Csigma_cellwise.clear();
    Cq_cellwise.clear();
    CEBlocks.clear();
    MXSolvers.clear();
}

// Memory Layout for a sundials Y is, if i indexes the components of u / q / sigma
// Y = [ sigma[ cell0, i=0 ], ..., sigma[ cell0, i= nVars - 1], q[ cell0, i = 0 ], ..., q[ cell0, i = nVars-1 ], u[ cell0, i = 0 ], .. u[ cell0, i = nVars - 1], sigma[ cell1, i=0 ], .... , u[ cellN-1, i = nVars - 1 ], Lambda[ cell0, i=0 ],.. ]
//
// This API is now in DGSoln

void SystemSolver::updateBoundaryConditions(double t)
{
    L_global.setZero();
    for (unsigned int i = 0; i < nCells; i++)
    {
        Interval const &I(grid[i]);
        RF_cellwise[i].setZero();

        for (Index var = 0; var < nVars; var++)
        {
            if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
            {
                for (Eigen::Index j = 0; j < k + 1; j++)
                {
                    // < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 )
                    RF_cellwise[i](j + var * (k + 1)) += -y.getBasis().Evaluate(I, j, I.x_l) * (-1) * problem->LowerBoundary(var, t);
                    // < ( tau ) g_D, w >
                    RF_cellwise[i](nVars * (k + 1) + j + var * (k + 1)) += y.getBasis().Evaluate(I, j, I.x_l) * tau(I.x_l) * problem->LowerBoundary(var, t);
                }
            }

            if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
            {
                for (Eigen::Index j = 0; j < k + 1; j++)
                {
                    // < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 )
                    RF_cellwise[i](j + var * (k + 1)) += -y.getBasis().Evaluate(I, j, I.x_u) * (+1) * problem->UpperBoundary(var, t);
                    RF_cellwise[i](nVars * (k + 1) + j + var * (k + 1)) += y.getBasis().Evaluate(I, j, I.x_u) * tau(I.x_u) * problem->UpperBoundary(var, t);
                }
            }

            if (I.x_l == grid.lowerBoundary() && /* is b.d. Neumann at lower boundary */ !problem->isLowerBoundaryDirichlet(var))
                L_global(var * (nCells + 1) + i) += -problem->LowerBoundary(var, t);
            if (I.x_u == grid.upperBoundary() && /* is b.d. Neumann at upper boundary */ !problem->isUpperBoundaryDirichlet(var))
                L_global(var * (nCells + 1) + i + 1) += problem->UpperBoundary(var, t);
        }
    }
}

void SystemSolver::resetCoeffs()
{
    y.zeroCoeffs();
    dydt.zeroCoeffs();
}

// Where and at what state the physics derivatives are evaluated. Shared with the
// algebraic-derivative solve, which has to make exactly the same choice: a
// Jacobian consistent with a different residual is the one failure mode this
// solver cannot detect from its answers, only from its iteration counts.
SystemSolver::PhysicsNodes
SystemSolver::evaluatePhysicsDerivatives(DGSoln const &Y, Time tEval,
                                         GlobalStateMatrix &dSigma_vals,
                                         GlobalStateMatrix &dSource_vals,
                                         GlobalStateMatrix &dAux_vals)
{
    // With the superconvergent scheme the derivatives are wanted at the star
    // nodes and evaluated with u* in place of u_h, exactly as the residual does.
    if (superconvergent)
        postprocessor->computeUStar(Y);

    PhysicsNodes nodes{superconvergent ? postprocessor->starPoints() : Y.getPoints(),
                       superconvergent ? postprocessor->evalOnStarNodes(Y) : Y.evalOnNodes()};

    // GlobalState's second argument is a per-cell dof count minus one; passing
    // k+1 is what makes cellwise*() hand back the k+2 star values.
    const Index derivK = superconvergent ? k + 1 : k;

    for (Index var = 0; var < nVars; var++)
    {
      dSigma_vals.add(nCells, derivK, nVars, nScalars, nAux);
      dSource_vals.add(nCells, derivK, nVars, nScalars, nAux);
    }
    for (Index aux = 0; aux < nAux; aux++)
    {
      dAux_vals.add(nCells, derivK, nVars, nScalars, nAux);
    }

    problem->ComputePhysicsDerivatives({dSigma_vals, dSource_vals, dAux_vals}, nodes.states,
                                       nodes.points, tEval);

    return nodes;
}

// One cell's Jacobian block. See the declaration for what alphaValue is; there is
// no other difference between the forward solve's block and the one the
// algebraic-derivative solve wants.
Matrix SystemSolver::assembleCellMatrix(Index i, DGSoln const &Y,
                                        GlobalStateMatrix &dSigma_vals,
                                        GlobalStateMatrix &dSource_vals,
                                        GlobalStateMatrix &dAux_vals, double alphaValue)
{
    Eigen::MatrixXd X(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd NLq(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd NLu(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd Ssig(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd Sq(nVars * (k + 1), nVars * (k + 1));
    Eigen::MatrixXd Su(nVars * (k + 1), nVars * (k + 1));


    Eigen::MatrixXd Sigma_phi(nVars * (k + 1), nAux * (k + 1));
    Eigen::MatrixXd Sphi(nVars * (k + 1), nAux * (k + 1));

    Interval const &I(grid[i]);
    Eigen::MatrixXd MX(nVars * SQU_DOF + nAux * AUX_DOF, nVars * SQU_DOF + nAux * AUX_DOF);
    MX = MBlocks[i];

    // X matrix
    X.setZero();
    for (Index var = 0; var < nVars; var++)
    {
        std::function<double(double)> alphaF = [=, this](double x)
        { return alphaValue * problem->aFn(var, x); };
        Eigen::MatrixXd Xsubmat((k + 1), (k + 1));
        Y.getBasis().MassMatrix(I, Xsubmat, alphaF);
        X.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Xsubmat;
    }
    MX.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) += X;

    if (superconvergent)
    {
        Postprocessor const &pp = *postprocessor;

        // sigma_hat's dependence on q is twofold: directly, and through u*,
        // which the reconstruction builds from q as well as u. B11 carries
        // the second path -- it is the only genuinely new coupling the
        // superconvergent scheme introduces.
        NLq.setZero();
        accumulateStarBlocks(NLq, dSigma_vals.Derivative(i), pp.V(i), nVars, nVars, i);
        accumulateStarBlocks(NLq, dSigma_vals.Variable(i), pp.B11(i), nVars, nVars, i);
        MX.block(0, nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLq;

        NLu.setZero();
        accumulateStarBlocks(NLu, dSigma_vals.Variable(i), pp.B12(i), nVars, nVars, i);
        MX.block(0, 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLu;

        if (nAux > 0)
        {
            Sigma_phi.setZero();
            accumulateStarBlocks(Sigma_phi, dSigma_vals.Aux(i), pp.V(i), nVars, nAux, i);
            MX.block(0, 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) = Sigma_phi;
        }

        Ssig.setZero();
        accumulateStarBlocks(Ssig, dSource_vals.Flux(i), pp.V(i), nVars, nVars, i);
        MX.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)) -= Ssig;

        Sq.setZero();
        accumulateStarBlocks(Sq, dSource_vals.Derivative(i), pp.V(i), nVars, nVars, i);
        accumulateStarBlocks(Sq, dSource_vals.Variable(i), pp.B11(i), nVars, nVars, i);
        MX.block(2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) -= Sq;

        Su.setZero();
        accumulateStarBlocks(Su, dSource_vals.Variable(i), pp.B12(i), nVars, nVars, i);
        MX.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) -= Su;

        if (nAux > 0)
        {
            Sphi.setZero();
            accumulateStarBlocks(Sphi, dSource_vals.Aux(i), pp.V(i), nVars, nAux, i);
            MX.block(2 * nVars * (k + 1), 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) -= Sphi;

            // The aux constraint rows, in MX's [ sigma | q | u | phi ] column
            // order. Same four chains as above.
            auto auxRows = MX.block(3 * nVars * (k + 1), 0, nAux * (k + 1),
                                    (3 * nVars + nAux) * (k + 1));
            auxRows.setZero();
            const Index sigmaBlock = 0;
            const Index qBlock = nVars * (k + 1);
            const Index uBlock = 2 * nVars * (k + 1);
            const Index phiBlock = 3 * nVars * (k + 1);

            accumulateStarBlocks(auxRows.middleCols(sigmaBlock, nVars * (k + 1)),
                                 dAux_vals.Flux(i), pp.V(i), nAux, nVars, i);
            accumulateStarBlocks(auxRows.middleCols(qBlock, nVars * (k + 1)),
                                 dAux_vals.Derivative(i), pp.V(i), nAux, nVars, i);
            accumulateStarBlocks(auxRows.middleCols(qBlock, nVars * (k + 1)),
                                 dAux_vals.Variable(i), pp.B11(i), nAux, nVars, i);
            accumulateStarBlocks(auxRows.middleCols(uBlock, nVars * (k + 1)),
                                 dAux_vals.Variable(i), pp.B12(i), nAux, nVars, i);
            accumulateStarBlocks(auxRows.middleCols(phiBlock, nAux * (k + 1)),
                                 dAux_vals.Aux(i), pp.V(i), nAux, nAux, i);
        }
    }
    else
    {
        // NLq Matrix
        DerivativeSubMatrix(NLq, dSigma_vals.Derivative(i), Y, i);
        MX.block(0, nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLq;

        // NLu Matrix
        DerivativeSubMatrix(NLu, dSigma_vals.Variable(i), Y, i);
        MX.block(0, 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLu;

        dPhi_Mat(Sigma_phi, dSigma_vals.Aux(i), Y, i);
        MX.block(0, 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) = Sigma_phi;

        // S_sig Matrix
        DerivativeSubMatrix(Ssig, dSource_vals.Flux(i), Y, i);
        MX.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)) -= Ssig;

        // S_q Matrix
        DerivativeSubMatrix(Sq, dSource_vals.Derivative(i), Y, i);
        MX.block(2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) -= Sq;

        // S_u Matrix
        DerivativeSubMatrix(Su, dSource_vals.Variable(i), Y, i);
        MX.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) -= Su;

        dPhi_Mat(Sphi, dSource_vals.Aux(i), Y, i);
        MX.block(2 * nVars * (k + 1), 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) -= Sphi;

        // Set Parts of Matrix due to aux variables
        dAux_Mat(MX.block(3 * nVars * (k + 1), 0, nAux * (k + 1), (3 * nVars + nAux) * (k + 1)), dAux_vals, Y, i);
    }

    return MX;
}

// The scalar coupling blocks. See the declaration for why they are written
// through the caller's storage rather than straight into v, w and N_global.
void SystemSolver::assembleScalarCoupling(DGSoln const &Y, DGSoln const &Ydot,
                                          PhysicsNodes const &nodes, Time tEval,
                                          double alphaValue, std::vector<DGSoln> &v_map,
                                          std::vector<DGSoln> &w_map, Matrix &N_out)
{
    // The N_HDG_DOF x N_Scalar matrix v, which contains the effect of the scalars
    // on the main variables (through the sources. nothing else is allowed to
    // depend on scalars)
    for (Index i = 0; i < nCells; ++i)
    {
        Matrix v_tmp(nVars * U_DOF, nScalars);
        if (superconvergent)
            dSources_dScalars_StarMat(v_tmp, nodes.states, nodes.points, i, tEval);
        else
            dSources_dScalars_Mat(v_tmp, Y, i, tEval);
        for (Index j = 0; j < nScalars; ++j)
            for (Index v = 0; v < nVars; ++v)
                v_map[j].u(v).getCoeff(i).second = v_tmp.block(v * U_DOF, j, U_DOF, 1);
    }

    // The N_Scalar x N_HDG_DOF matrix w, which contains the Jacobian of the
    // scalars with respect to the other variables; also the scalar-scalar
    // coupling matrix N.
    for (Index j = 0; j < nScalars; ++j)
        w_map[j].zeroCoeffs();

    GlobalStateMatrix ScalarG_vals(nScalars);
    GlobalStateMatrix ScalarG_dt_vals(nScalars);
    for (Index s = 0; s < nScalars; s++)
    {
      ScalarG_vals.add(nCells, k, nVars, nScalars, nAux);
      ScalarG_dt_vals.add(nCells, k, nVars, nScalars, nAux);
    }

    problem->ScalarGPrime(ScalarG_vals, ScalarG_dt_vals, Y.evalOnNodes(), Ydot.evalOnNodes(),
                          Y.getPoints(),
                          Integrator::getIntegrationWeights(Y.getBasis(), grid),
                          Integrator::getPhiBoundary(Y.getBasis(), grid), tEval);

    for ( Index j = 0; j < nScalars; ++j ) {
        const auto& s = ScalarG_vals[j];
        const auto& s_dt = ScalarG_dt_vals[j];
        for (Index i = 0; i < nCells; ++i)
        {
            for ( Index l = 0; l < k + 1; ++l ) {
                for ( Index v = 0; v < nVars; ++v ) {
                    w_map[ j ].sigma( v ).getCoeff( i ).second( l ) = s[i * (k + 1) + l].sigma(v)       + alphaValue * s_dt[i * (k + 1) + l].sigma(v);
                    w_map[ j ].q( v ).getCoeff( i ).second( l )     = s[i * (k + 1) + l].q(v) + alphaValue * s_dt[i * (k + 1) + l].q(v);
                    w_map[ j ].u( v ).getCoeff( i ).second( l )     = s[i * (k + 1) + l].u(v)   + alphaValue * s_dt[i * (k + 1) + l].u(v);
                }
                for (Index a = 0; a < nAux; ++a)
                    w_map[j].Aux(a).getCoeff(i).second(l) = s[i * (k + 1) + l].phi(a) + alphaValue * s_dt[i * (k + 1) + l].phi(a);
            }
            for (Index m = 0; m < nScalars; ++m)
                N_out(j, m) = s.Scalars()[m] + alphaValue * s_dt.Scalars()[m];
        }
    }
}

void SystemSolver::updateMatricesForJacSolve()
{
    ++nJacBuilds;
    updateBoundaryConditions(jt);
    // We know where the jacobian is to be evaluated -- yJac
    // std::cerr << "Updating Jacobian at t=" << jt << std::endl;
    GlobalStateMatrix dSigma_vals(nVars);
    GlobalStateMatrix dSource_vals(nVars);
    GlobalStateMatrix dAux_vals(nAux);

    const PhysicsNodes nodes =
        evaluatePhysicsDerivatives(yJac, jt, dSigma_vals, dSource_vals, dAux_vals);

    // Cell-independent: iteration i reads MBlocks[i] and grid[i] and writes only
    // MXSolvers[i]. The quadrature `assembleCellMatrix` reaches through
    // Basis::MassMatrix is a boost::math::quadrature::gauss<double, 30>, whose
    // integrate() is const over static tables, so the shared static integrator is
    // not a race either.
#pragma omp parallel for
    for (unsigned int i = 0; i < nCells; i++)
    {
        MXSolvers[i].compute(
            assembleCellMatrix(i, yJac, dSigma_vals, dSource_vals, dAux_vals, alpha));
    }

    if (nScalars > 0)
    {
      std::vector<DGSoln> v_map, w_map;
      for (Index i = 0; i < nScalars; ++i)
      {
          v_map.emplace_back(nVars, grid, k, N_VGetArrayPointer(v[i]), nScalars, nAux);
          w_map.emplace_back(nVars, grid, k, N_VGetArrayPointer(w[i]), nScalars, nAux);
      }

      assembleScalarCoupling(yJac, dydtJac, nodes, jt, alpha, v_map, w_map, N_global);
    }
}

void SystemSolver::mapDGtoSundials(std::vector<VectorWrapper> &SQU_cell, VectorWrapper &lam, sunrealtype *const &Y) const
{
    SQU_cell.clear();
    for (Index i = 0; i < nCells; i++)
    {
        SQU_cell.emplace_back(VectorWrapper(Y + i * localDOF, localDOF));
    }

    new (&lam) VectorWrapper(Y + nCells * localDOF, nVars * (nCells + 1));
}

void SystemSolver::setJacEvalY(N_Vector yy, N_Vector yp)
{
    DGSoln yyMap(nVars, grid, k, nScalars, nAux);
    assert(static_cast<size_t>(N_VGetLength(yy)) == yyMap.getDoF());
    yyMap.Map(N_VGetArrayPointer(yy));
    yJac.copy(yyMap); // Deep copy -- yyMap only aliases the N_Vector, this copies the data

    DGSoln ypMap(nVars, grid, k, nScalars, nAux);
    assert(static_cast<size_t>(N_VGetLength(yp)) == ypMap.getDoF());
    ypMap.Map(N_VGetArrayPointer(yp));
    dydtJac.copy(ypMap); // Deep copy
}

// Over-arching Jacobian function. If there's no coupled B-field solve, or auxiliary variables, then just do the
// HDG Jacobian solve
void SystemSolver::solveJacEq(N_Vector res_g, N_Vector delY)
{
    ++nJacSolves;
    if (nScalars > 0)
    {
        // TODO: move temporaries into private variables of the class and allocate/destroy once
        // allocate temporary working space for gauss elimination of scalars.

        N_Vector d = N_VClone(delY);

        N_Vector *e = new N_Vector[nScalars];
        for (Index i = 0; i < nScalars; ++i)
            e[i] = N_VClone(delY);

        N_Vector g = N_VClone(delY);

        DGSoln res_g_map(nVars, grid, k, N_VGetArrayPointer(res_g), nScalars, nAux);

        DGSoln del_y(nVars, grid, k, N_VGetArrayPointer(delY), nScalars, nAux);

        // Let A be the HDG linear operator solved in solveHDGJac

        // First solve A d = res_g ;
        solveHDGJac(res_g, d);

        // Now A e = v ; Do as a loop over nScalars
        for (Index i = 0; i < nScalars; ++i)
        {
            solveHDGJac(v[i], e[i]);
        }

        Vector tmp_N = (N_global.inverse() * res_g_map.Scalars());
        N_VLinearCombination(nScalars, tmp_N.data(), e, g); // g = Sum_i tmp_N[i]*e[i]
        N_VLinearSum(1.0, g, 1.0, d, g);                    // g += d;

        Vector wDotg(nScalars);
        for (Index i = 0; i < nScalars; ++i)
        {
            wDotg[i] = N_VDotProd(w[i], g);
        }
        Matrix wTe(nScalars, nScalars);
        for (Index i = 0; i < nScalars; ++i)
            for (Index j = 0; j < nScalars; ++j)
                wTe(i, j) = N_VDotProd(w[i], e[j]);

        Matrix Nwe = N_global + wTe;
        Vector NweInv_w_g = -1 * Nwe.inverse() * wDotg;             // Uses PartialPivLU internally, never really does inverse (except for small matrices)
        N_VLinearCombination(nScalars, NweInv_w_g.data(), e, delY); // Set delY = - [ e  ( N + w^T e )^-1  w ]  g
        N_VLinearSum(1.0, delY, 1.0, g, delY);                      // delY += g; so delY = g - (....), which is the final answer

        // Finally, set the components of delY related to the change of the scalars

        del_y.Scalars() = Values::Zero(nScalars);

        Vector del_y_scalars(nScalars);
        for (Index i = 0; i < nScalars; ++i)
            del_y_scalars(i) = res_g_map.Scalar(i) - N_VDotProd(w[i], delY);

        del_y.Scalars() = N_global.inverse() * del_y_scalars;

        for (Index i = 0; i < nScalars; ++i)
            N_VDestroy(e[i]);
        N_VDestroy(d);
        N_VDestroy(g);
        delete[] e;
    }
    else
    {
        solveHDGJac(res_g, delY);
    }
}

// Solve the HDG part of the Jacobian
// NB: This is called repeatedly, *possibly with the same jacobian*
// don't do any matrix re-assembly here
void SystemSolver::solveHDGJac(N_Vector g, N_Vector delY)
{
    // DGsoln object that will map the data from delY
    DGSoln del_y(nVars, grid, k, nScalars, nAux);
#ifdef DEBUG
    // Provide view on g for debugging
    DGSoln gMap(nVars, grid, k, nScalars, nAux);
    assert(static_cast<size_t>(N_VGetLength(g)) == gMap.getDoF());
    gMap.Map(N_VGetArrayPointer(g));
#endif

    assert(static_cast<size_t>(N_VGetLength(delY)) == del_y.getDoF());
    del_y.Map(N_VGetArrayPointer(delY));

    std::vector<VectorWrapper> g1g2g3_cellwise;
    VectorWrapper g4(nullptr, 0);

    // Eigen::Vector wrapper
    VectorWrapper delYVec(N_VGetArrayPointer(delY), N_VGetLength(delY));
    delYVec.setZero();

    K_global.setZero();

    // Assemble RHS g into cellwise form and solve for SQU blocks
    mapDGtoSundials(g1g2g3_cellwise, g4, N_VGetArrayPointer(g));

    std::vector<Eigen::VectorXd> SQU_f(nCells);
    std::vector<Eigen::MatrixXd> SQU_0(nCells);
    std::vector<Eigen::MatrixXd> K_cell(nCells);

    // The two back-substitutions are the expensive part and are cell-independent,
    // so they parallelise; the assembly into K_global below does not, and the two
    // are separate loops for that reason. See the comment there.
#pragma omp parallel for
    for (Index i = 0; i < nCells; i++)
    {
        // Interval const& I( grid[ i ] );

        // SQU_f
        Eigen::VectorXd const &g1g2g3 = g1g2g3_cellwise[i];

        SQU_f[i] = MXSolvers[i].solve(g1g2g3);

        // SQU_0
        Eigen::MatrixXd const &CE = CEBlocks[i];
        SQU_0[i] = MXSolvers[i].solve(CE);
        // std::cerr << SQU_0[i] << std::endl << std::endl;
        // std::cerr << CE << std::endl << std::endl;

        K_cell[i] = H_cellwise[i] - CG_cellwise[i] * SQU_0[i];
    }

    // Serial, and it has to be: lambda lives on cell *faces*, so cell i's 2x2
    // block starts at trace index i and cell i+1's at i+1 -- they overlap in the
    // face they share, and two threads would `+=` into it at once. The work here
    // is 4 * nVars^2 adds per cell against two dense solves above, so nothing is
    // lost by leaving it out of the parallel region.
    // K
    for (Index i = 0; i < nCells; i++)
        for (Index varI = 0; varI < nVars; varI++)
            for (Index varJ = 0; varJ < nVars; varJ++)
                K_global.block<2, 2>(varI * (nCells + 1) + i, varJ * (nCells + 1) + i) += K_cell[i].block<2, 2>(varI * 2, varJ * 2);

    // Construct the RHS of K Lambda = F
    Eigen::VectorXd F(nVars * (nCells + 1));
    F = g4;
    for (Index i = 0; i < nCells; i++)
    {
        for (Index var = 0; var < nVars; var++)
        {
            F.block<2, 1>(var * (nCells + 1) + i, 0) -= (CG_cellwise[i] * SQU_f[i]).block(var * 2, 0, 2, 1);
        }
    }

    // Factorise the global matrix ( size n_cells * n_variables )
    EigenGlobalSolver globalKSolver(K_global);
    // This solves for the lambdas of all variables at once (drop it in the memory sundials reserved for it)
    Index LambdaOffset = nCells * localDOF;

    delYVec.segment(LambdaOffset, nVars * (nCells + 1)) = globalKSolver.solve(F);

    /*
     * We really should do something here.
    // If the BCs are Dirichlet, enforce that (Y + delY).lambda( v )[0,N] are the right values
    for ( Index i=0; i < nVars; i++ ) {
    if ( problem->isLowerBoundaryDirichlet( i ) )
    del_y.lambda( i )[ 0 ] = problem->LowerBoundary( i, t ) - y.lambda( i )[ 0 ];
    if ( problem->isUpperBoundaryDirichlet( i ) )
    del_y.lambda( i )[ nCells ] = problem->UpperBoundary( i, t ) - y.lambda( i )[ nCells ];
    }
    */

    // Now find del sigma, del q and del u to eventually find del Y
    // this can be done in parallel over each cell
    for (Index i = 0; i < nCells; i++)
    {
        Vector delSQU(nVars * SQU_DOF);

        // Reorganise the data from variable-major to cell-major
        Vector delLambdaCell(2 * nVars);

        for (Index var = 0; var < nVars; var++)
        {
            delLambdaCell.block<2, 1>(2 * var, 0) = delYVec.segment(LambdaOffset + var * (nCells + 1) + i, 2);
        }

        /*
        // Try mapping the memory by using the magic runes (future update)
        Eigen::Map< Vector, 0, Eigen::InnerStride<nCells + 1> >
        delLambdaCell( delYVec.data() + LambdaOffset + i, 2 * nVars, Eigen::InnerStride<nCells + 1> );
        */

        delSQU = SQU_f[i] - SQU_0[i] * delLambdaCell;
        for (Index var = 0; var < nVars; var++)
        {
            del_y.sigma(var).getCoeff(i).second = delSQU.segment(var * S_DOF, S_DOF);
            del_y.q(var).getCoeff(i).second = delSQU.segment(nVars * S_DOF + var * Q_DOF, Q_DOF);
            del_y.u(var).getCoeff(i).second = delSQU.segment(nVars * (S_DOF + Q_DOF) + var * U_DOF, U_DOF);
        }
        for (Index aux = 0; aux < nAux; aux++)
            del_y.Aux(aux).getCoeff(i).second = delSQU.segment(nVars * SQU_DOF + aux * AUX_DOF, AUX_DOF);
    }
}

int static_residual(sunrealtype tres, N_Vector Y, N_Vector dYdt, N_Vector resval, void *user_data)
{
    auto system = static_cast<SystemSolver *>(user_data);
    try
    {
        return system->residual(tres, Y, dYdt, resval);
    }
    catch (std::exception &e)
    {
        std::println("Caught exception : {} ; Retrying. ", e.what());
        return 1;
    }
}

int SystemSolver::residual(sunrealtype tres, N_Vector Y, N_Vector dYdt, N_Vector resval)
{
    ++nResidualEvals;
    updateBoundaryConditions(tres);

    DGSoln Y_h(nVars, grid, k, N_VGetArrayPointer(Y), nScalars, nAux);
    DGSoln dYdt_h(nVars, grid, k, N_VGetArrayPointer(dYdt), nScalars, nAux);
    DGSoln res(nVars, grid, k, N_VGetArrayPointer(resval), nScalars, nAux);

    VectorWrapper resVec(N_VGetArrayPointer(resval), N_VGetLength(resval));

    resVec.setZero();

    // With the superconvergent scheme the physics is evaluated at the k+2 nodes
    // of the degree-(k+1) basis, with the postprocessed u* standing in for u_h,
    // and the resulting P_{k+1} interpolant is projected onto the P_k test space
    // by A9 instead of by the mass matrix InterpolateOntoBasis applies. Both
    // halves matter: interpolating a non-polynomial into P_k contributes an
    // O(h^(k+1)) consistency error with no orthogonality against the test space,
    // and that alone caps the rate at k+1. See Postprocessing.hpp.
    if (superconvergent)
        postprocessor->computeUStar(Y_h);

    const Index physicsDoF = superconvergent ? k + 2 : k + 1;

    const std::vector<Position> points =
        superconvergent ? postprocessor->starPoints() : Y_h.getPoints();

    const GlobalState states = superconvergent ? postprocessor->evalOnStarNodes(Y_h)
                                               : Y_h.evalOnNodes();

    auto values = problem->ComputePhysics(states, points, tres);

    // ( X, phi_i )_K for a physics value X sampled on the cell's nodes: A9 times
    // the star-node values with the superconvergent scheme, the interpolatory
    // mass-matrix form of arXiv:1811.09667 otherwise.
    auto projectOntoTestSpace = [&](Index cell, Interval const &I,
                                    auto const &vals) -> Eigen::VectorXd
    {
        if (superconvergent)
            return postprocessor->A9(cell) * vals;
        return y.getBasis().InterpolateOntoBasis(I, vals);
    };

    std::vector<Values> Sigma_vals = std::move(values[0]);
    std::vector<Values> Source_vals = std::move(values[1]);
    std::vector<Values> Aux_vals = std::move(values[2]);

    // residual.lambda = C*sigma + G*u + H*lambda - L
    for (Index i = 0; i < nCells; i++)
    {
        // C_cellwise * sigma_cellwise
        for (Index var = 0; var < nVars; var++)
        {
            res.lambda(var).segment<2>(i) += Csigma_cellwise[i].block(var * 2, var * (k + 1), 2, k + 1) * Y_h.sigma(var).getCoeff(i).second + Cq_cellwise[i].block(var * 2, var * (k + 1), 2, k + 1) * Y_h.q(var).getCoeff(i).second + G_cellwise[i].block(var * 2, var * (k + 1), 2, k + 1) * Y_h.u(var).getCoeff(i).second + H_cellwise[i].block(2 * var, 2 * var, 2, 2) * Y_h.lambda(var).segment<2>(i) - L_global.segment<2>(var * (nCells + 1) + i);
        }
    }

    for (Index i = 0; i < nCells; i++)
    {
        Eigen::MatrixXd Mass( k + 1, k + 1 );
        Interval I = grid[i];
        y.getBasis().MassMatrix( I, Mass );
        Eigen::VectorXd lamCell(2 * nVars);

        for (Index var = 0; var < nVars; var++)
        {
            auto const &lCell = Y_h.lambda(var);
            lamCell[2 * var] = lCell[i];
            lamCell[2 * var + 1] = lCell[i + 1];
        }

        // length = nVars*(k+1)
        for (Index var = 0; var < nVars; var++)
        {
            // std::function<double(double)> kappaFunc = [=, this, &Y_h](double x)
            // {
            //     State s = Y_h.eval(x);
            //     return problem->SigmaFn(var, s, x, tres);
            // };

            // std::function<double(double)> sourceFunc = [=, this, &Y_h](double x)
            // {
            //     State s = Y_h.eval(x);
            //     return problem->Sources(var, s, x, tres);
            // };
            auto ind = Eigen::seq(i * physicsDoF, (i + 1) * physicsDoF - 1);
            // Evaluate Diffusion Function
            Eigen::VectorXd kappa_cellwise = projectOntoTestSpace(i, I, Sigma_vals[var](ind));

            // Evaluate Source Function
            Eigen::VectorXd S_cellwise = projectOntoTestSpace(i, I, Source_vals[var](ind));

            auto const &lambda = lamCell.segment<2>(2 * var);

            // We should normalise the components of the residual such that the `sigma' component of res
            // has tolerances that are the same as `sigma' itself.
            //
            // For sigma and q, just make the 'sigma-determinitive' equation the sigma component and the same for q.
            // as these equations are proportional to the variables themselves, we are done

            res.sigma(var).getCoeff(i).second = A_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.sigma(var).getCoeff(i).second + kappa_cellwise;

            res.q(var).getCoeff(i).second =
                -A_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.q(var).getCoeff(i).second - B_cellwise[i].transpose().block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.u(var).getCoeff(i).second + C_cellwise[i].transpose().block(var * (k + 1), var * 2, k + 1, 2) * lambda - RF_cellwise[i].block(var * (k + 1), 0, k + 1, 1);

            // For the 'u' component of the residual, we also have a factor of d/dt. Thus we should multiply this equation by some frequency estimate.
            // For the moment we leave it as it is.
            res.u(var).getCoeff(i).second =
                B_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.sigma(var).getCoeff(i).second + D_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.u(var).getCoeff(i).second + E_cellwise[i].block(var * (k + 1), var * 2, k + 1, 2) * lambda - RF_cellwise[i].block(nVars * (k + 1) + var * (k + 1), 0, k + 1, 1) - S_cellwise + XMats[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * dYdt_h.u(var).getCoeff(i).second;
        }
        for (Index aux = 0; aux < nAux; aux++)
        {
            // For the auxiliary variable bits
            // Set (res_aux_i)_j = < G_i, phi_j >
            // so we enforce G = 0 by projection
            auto ind = Eigen::seq(i * physicsDoF, (i + 1) * physicsDoF - 1);
            res.Aux(aux).getCoeff(i).second = projectOntoTestSpace(i, I, Aux_vals[aux](ind));
        }
    }


    if (nScalars > 0)
    {
        // Sampled once for all the scalars, not once each: evalOnNodes walks
        // every cell and node, and the scalars all see the same state.
        const GlobalState scalarStates = Y_h.evalOnNodes();
        const GlobalState scalarStates_dt = dYdt_h.evalOnNodes();
        const Vector &weights = Integrator::getIntegrationWeights(Y_h.getBasis(), grid);
        const Matrix &phiBoundary = Integrator::getPhiBoundary(Y_h.getBasis(), grid);

        for (Index j = 0; j < nScalars; j++)
            res.Scalar(j) = problem->ScalarG(j, scalarStates, scalarStates_dt, Y_h.getPoints(),
                                             weights, phiBoundary, tres);
    }

    return 0;
}

void SystemSolver::initializeMatricesForAdjointSolve()
{
    // With the superconvergent scheme the objective is a functional of u*, so
    // dG/dy runs through the reconstruction just as the residual's Jacobian does.
    // See the G_y assembly below for why the u and q rows contract with the star
    // mass matrix rather than with A9.
    if (superconvergent)
        postprocessor->computeUStar(y);

    const Index derivK = superconvergent ? k + 1 : k;

    GlobalState dGdvars(grid.getNCells(), derivK, nVars, nScalars, nAux);
    const std::vector<Position> points =
        superconvergent ? postprocessor->starPoints() : y.getPoints();
    const GlobalState states =
        superconvergent ? postprocessor->evalOnStarNodes(y) : y.evalOnNodes();
    adjointProblem->dg(0, dGdvars, states, points);
    Vector dGdu(nVars * (k + 1));
    Vector dGdq(nVars * (k + 1));
    Vector dGdsigma(nVars * (k + 1));

    Vector dGdaux(nAux * (k + 1));

    for (Index i = 0; i < nCells; ++i)
    {
        G_y.emplace_back(3 * nVars * (k + 1) + nAux * (k + 1));

        if (superconvergent)
        {
            // The exact derivative of the objective AdjointProblem::GFn's
            // superconvergent form computes, G = b1 . g over the star nodes:
            //
            //     dG/dZ = chain^T diag( dg/dW ) b1
            //
            // with the same chain matrices the residual's Jacobian uses -- V where
            // the field is simply sampled at the star nodes, B12 for the u
            // coefficients, and B11 for the q coefficients' path through u*.
            // Because this is the exact derivative of the reported objective and
            // not merely a consistent discretisation of it, the finite-difference
            // gradient check has nothing to absorb.
            Postprocessor const &pp = *postprocessor;
            const Vector &b1 = pp.starWeights(i);

            G_y[i].setZero();

            auto dFlux = dGdvars.cellwiseFlux(i);
            auto dDeriv = dGdvars.cellwiseDerivative(i);
            auto dVar = dGdvars.cellwiseVariable(i);

            for (Index var = 0; var < nVars; ++var)
            {
                const Vector w_sigma = dFlux.row(var).transpose().cwiseProduct(b1);
                const Vector w_q = dDeriv.row(var).transpose().cwiseProduct(b1);
                const Vector w_u = dVar.row(var).transpose().cwiseProduct(b1);

                G_y[i].segment(var * (k + 1), k + 1) = pp.V(i).transpose() * w_sigma;
                G_y[i].segment(nVars * (k + 1) + var * (k + 1), k + 1) =
                    pp.V(i).transpose() * w_q + pp.B11(i).transpose() * w_u;
                G_y[i].segment(2 * nVars * (k + 1) + var * (k + 1), k + 1) =
                    pp.B12(i).transpose() * w_u;
            }

            if (nAux > 0)
            {
                auto dAux = dGdvars.cellwiseAux(i);
                for (Index a = 0; a < nAux; ++a)
                    G_y[i].segment(3 * nVars * (k + 1) + a * (k + 1), k + 1) =
                        pp.V(i).transpose() *
                        Vector(dAux.row(a).transpose().cwiseProduct(b1));
            }
        }
        else
        {
        DerivativeSubVector(0, dGdsigma, dGdvars.cellwiseFlux(i), y, i);
        G_y[i].block(0, 0, nVars * (k + 1), 1) = dGdsigma;

        DerivativeSubVector(0, dGdq, dGdvars.cellwiseDerivative(i), y, i);
        G_y[i].block(nVars * (k + 1), 0, nVars * (k + 1), 1) = dGdq;

        DerivativeSubVector(0, dGdu, dGdvars.cellwiseVariable(i), y, i);
        G_y[i].block(2 * nVars * (k + 1), 0, nVars * (k + 1), 1) = dGdu;


         if (nAux > 0)
         {
             dGdaux_Vec(0, dGdaux, dGdvars.cellwiseAux(i), y, i);
             G_y[i].block(3 * nVars * (k + 1), 0, nAux * (k + 1), 1) = dGdaux;
         }
        }
    }


    GlobalStateMatrix dSigma_vals(nVars);
    GlobalStateMatrix dSource_vals(nVars);
    GlobalStateMatrix dAux_vals(nAux);

    for (Index var = 0; var < nVars; var++)
    {
      dSigma_vals.add(nCells, derivK, nVars, nScalars, nAux);
      dSource_vals.add(nCells, derivK, nVars, nScalars, nAux);
    }
    for (Index aux = 0; aux < nAux; aux++)
    {
      dAux_vals.add(nCells, derivK, nVars, nScalars, nAux);
    }

    problem->ComputePhysicsDerivatives({dSigma_vals, dSource_vals, dAux_vals}, states, points, jt);

   // We have to remake the M matrices because they're in the wrong order
    // We also need to calculate the dSigmadX and dSourcedX matrices at the same time

    for (unsigned int i = 0; i < nCells; i++)
    {
        Eigen::MatrixXd NLq(nVars * (k + 1), nVars * (k + 1));
        Eigen::MatrixXd NLu(nVars * (k + 1), nVars * (k + 1));
        Eigen::MatrixXd Ssig(nVars * (k + 1), nVars * (k + 1));
        Eigen::MatrixXd Sq(nVars * (k + 1), nVars * (k + 1));
        Eigen::MatrixXd Su(nVars * (k + 1), nVars * (k + 1));

        Eigen::MatrixXd Sigma_phi(nVars * (k + 1), nAux * (k + 1));
        Eigen::MatrixXd Sphi(nVars * (k + 1), nAux * (k + 1));

        if (superconvergent)
        {
            // The same blocks updateMatricesForJacSolve builds; M below is their
            // transpose, so keeping the two in step is what keeps the adjoint the
            // adjoint of the residual actually being solved.
            Postprocessor const &pp = *postprocessor;

            NLq.setZero();
            accumulateStarBlocks(NLq, dSigma_vals.Derivative(i), pp.V(i), nVars, nVars, i);
            accumulateStarBlocks(NLq, dSigma_vals.Variable(i), pp.B11(i), nVars, nVars, i);

            NLu.setZero();
            accumulateStarBlocks(NLu, dSigma_vals.Variable(i), pp.B12(i), nVars, nVars, i);

            Ssig.setZero();
            accumulateStarBlocks(Ssig, dSource_vals.Flux(i), pp.V(i), nVars, nVars, i);

            Sq.setZero();
            accumulateStarBlocks(Sq, dSource_vals.Derivative(i), pp.V(i), nVars, nVars, i);
            accumulateStarBlocks(Sq, dSource_vals.Variable(i), pp.B11(i), nVars, nVars, i);

            Su.setZero();
            accumulateStarBlocks(Su, dSource_vals.Variable(i), pp.B12(i), nVars, nVars, i);

            // phi is not reconstructed -- u* stands in for u only -- so both
            // auxiliary columns take the plain evaluation chain A9 diag(dX/dphi) V.
            Sigma_phi.setZero();
            Sphi.setZero();
            if (nAux > 0)
            {
                accumulateStarBlocks(Sigma_phi, dSigma_vals.Aux(i), pp.V(i), nVars, nAux, i);
                accumulateStarBlocks(Sphi, dSource_vals.Aux(i), pp.V(i), nVars, nAux, i);
            }
        }
        else
        {
        // NLq Matrix
        DerivativeSubMatrix(NLq, dSigma_vals.Derivative(i), yJac, i);

        // NLu Matrix
        DerivativeSubMatrix(NLu, dSigma_vals.Variable(i), yJac, i);

        // S_sig Matrix
        DerivativeSubMatrix(Ssig, dSource_vals.Flux(i), yJac, i);

        // S_q Matrix
        DerivativeSubMatrix(Sq, dSource_vals.Derivative(i), yJac, i);

        // S_u Matrix
        DerivativeSubMatrix(Su, dSource_vals.Variable(i), yJac, i);
        }

        // M is the local DG Matrix
        Eigen::MatrixXd M(localDOF, localDOF);
        M.setZero();
        auto A = A_cellwise[i];
        auto B = B_cellwise[i];
        auto D = D_cellwise[i];
        // row1
        M.block(0, 0, nVars * (k + 1), nVars * (k + 1)) = A;
        M.block(0, nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero() = NLq;
        M.block(0, 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLu; // NLu added at Jac step

        // row2
        M.block(nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)).setZero();
        M.block(nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = -A;
        M.block(nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = -B.transpose();

        // row3
        M.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)) = B - Ssig;
        M.block(2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = Sq;
        M.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = (D - Su);

        if (nAux > 0)
        {
            // Both auxiliary column blocks, in either discretisation. The
            // dSigma/dPhi one at row 1 was missing entirely: M is zeroed and
            // nothing ever wrote column block 3 of row 1, so whenever the flux
            // depended on an auxiliary variable the matrix stored here was not
            // the transpose of the forward Jacobian that
            // updateMatricesForJacSolve builds (which does write it, above).
            //
            // On the forward side an inconsistent Jacobian only costs Newton
            // iterations. Here it costs correctness: M.transpose() *is* the
            // adjoint operator, so a missing block gives a silently wrong
            // gradient. Nothing caught it because no adjoint test had nAux > 0;
            // python/Tests/test_adjoint_aux.py now does.

            if (superconvergent)
            {
                // Sigma_phi and Sphi already hold their star forms from above;
                // recomputing them here would discard the chain rule.
                Postprocessor const &pp = *postprocessor;
                M.block(0, 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) = Sigma_phi;
                M.block(2 * nVars * (k + 1), 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) -= Sphi;

                auto auxRows = M.block(3 * nVars * (k + 1), 0, nAux * (k + 1),
                                       (3 * nVars + nAux) * (k + 1));
                auxRows.setZero();
                const Index qBlock = nVars * (k + 1);
                const Index uBlock = 2 * nVars * (k + 1);
                const Index phiBlock = 3 * nVars * (k + 1);

                accumulateStarBlocks(auxRows.middleCols(0, nVars * (k + 1)),
                                     dAux_vals.Flux(i), pp.V(i), nAux, nVars, i);
                accumulateStarBlocks(auxRows.middleCols(qBlock, nVars * (k + 1)),
                                     dAux_vals.Derivative(i), pp.V(i), nAux, nVars, i);
                accumulateStarBlocks(auxRows.middleCols(qBlock, nVars * (k + 1)),
                                     dAux_vals.Variable(i), pp.B11(i), nAux, nVars, i);
                accumulateStarBlocks(auxRows.middleCols(uBlock, nVars * (k + 1)),
                                     dAux_vals.Variable(i), pp.B12(i), nAux, nVars, i);
                accumulateStarBlocks(auxRows.middleCols(phiBlock, nAux * (k + 1)),
                                     dAux_vals.Aux(i), pp.V(i), nAux, nAux, i);
            }
            else
            {
            // dPhi_Mat, not dSourcedPhi_Mat: the residual interpolates the
            // sources onto the basis, so the consistent block is the
            // interpolatory Mass * diag(dX/dphi at the nodes), which is what the
            // forward Jacobian uses. dSourcedPhi_Mat integrates by quadrature
            // and also re-evaluates the physics hooks that
            // ComputePhysicsDerivatives has already batched into dSource_vals.
            dPhi_Mat(Sigma_phi, dSigma_vals.Aux(i), yJac, i);
            M.block(0, 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) = Sigma_phi;

            dPhi_Mat(Sphi, dSource_vals.Aux(i), yJac, i);
            M.block(2 * nVars * (k + 1), 3 * nVars * (k + 1), nVars * (k + 1), nAux * (k + 1)) -= Sphi;

            // Set Parts of Matrix due to aux variables

            dAux_Mat(M.block(3 * nVars * (k + 1), 0, nAux * (k + 1), (3 * nVars + nAux) * (k + 1)), dAux_vals, yJac, i);

            // TODO: Consider factorization here (is M sparse enough to warrant a sparse implementation?)
            }
        }

        // Note we save the transpose for adjoints

        Eigen::MatrixXd CE_vec(localDOF, 2 * nVars);
        CE_vec.setZero();
        auto C = C_cellwise[i];
        auto E = E_cellwise[i];
        CE_vec.block(0, 0, nVars * (k + 1), nVars * 2).setZero();
        CE_vec.block(nVars * (k + 1), 0, nVars * (k + 1), nVars * 2) = C.transpose();
        CE_vec.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * 2) = E;
        CE_vec.block(3 * nVars * (k + 1), 0, nAux * (k + 1), nVars * 2).setZero();
        adjoint_CEBlocks.emplace_back(CE_vec.transpose());

        // //[ C 0 G 0 ] (4th index is aux vars)
        auto G = G_cellwise[i];
        Eigen::MatrixXd CG_vec(2 * nVars, localDOF);

        CG_vec.setZero();
        CG_vec.block(0, 0, 2 * nVars, nVars * (k + 1)) = Csigma_cellwise[i];
        CG_vec.block(0, nVars * (k + 1), 2 * nVars, nVars * (k + 1)) = Cq_cellwise[i];
        CG_vec.block(0, 2 * nVars * (k + 1), 2 * nVars, nVars * (k + 1)) = G;

        adjoint_CGBlocks.emplace_back(CG_vec.transpose());

        MXSolvers[i].compute(M.transpose());
    }

    // no computation of scalars

    initialised = true;
}

void SystemSolver::solveAdjointState(Index gIndex)
{

    K_global.setZero();

    std::vector<Eigen::VectorXd> SQU_f(nCells);
    std::vector<Eigen::MatrixXd> SQU_0(nCells);
    for (Index i = 0; i < nCells; i++)
    {
        // Interval const& I( grid[ i ] );

        // SQU_f
        Vector g1g2g3 = G_y[i];

        SQU_f[i] = MXSolvers[i].solve(g1g2g3);

        // SQU_0
        Eigen::MatrixXd const &CG = adjoint_CGBlocks[i];
        SQU_0[i] = MXSolvers[i].solve(CG);
        // std::cerr << SQU_0[i] << std::endl << std::endl;
        // std::cerr << CE << std::endl << std::endl;

        Eigen::MatrixXd K_cell(nVars * 2, nVars * 2);
        K_cell = H_cellwise[i].transpose() - adjoint_CEBlocks[i] * SQU_0[i];

        // K
        for (Index varI = 0; varI < nVars; varI++)
            for (Index varJ = 0; varJ < nVars; varJ++)
                K_global.block<2, 2>(varI * (nCells + 1) + i, varJ * (nCells + 1) + i) += K_cell.block<2, 2>(varI * 2, varJ * 2);
    }

    // Construct the RHS of K Lambda = F
    Eigen::VectorXd F(nVars * (nCells + 1));
    F.setZero();
    for (Index i = 0; i < nCells; i++)
    {
        for (Index var = 0; var < nVars; var++)
        {
            F.block<2, 1>(var * (nCells + 1) + i, 0) -= (adjoint_CEBlocks[i] * SQU_f[i]).block(var * 2, 0, 2, 1);
        }
    }

    // Factorise the global matrix ( size n_cells * n_variables )
    EigenGlobalSolver globalKSolver(K_global);

    adjoint_lambdas = globalKSolver.solve(F);

    /*
     * We really should do something here.
    // If the BCs are Dirichlet, enforce that (Y + delY).lambda( v )[0,N] are the right values
    for ( Index i=0; i < nVars; i++ ) {
    if ( problem->isLowerBoundaryDirichlet( i ) )
    del_y.lambda( i )[ 0 ] = problem->LowerBoundary( i, t ) - y.lambda( i )[ 0 ];
    if ( problem->isUpperBoundaryDirichlet( i ) )
    del_y.lambda( i )[ nCells ] = problem->UpperBoundary( i, t ) - y.lambda( i )[ nCells ];
    }
    */

    // Now find del sigma, del q and del u to eventually find del Y
    // this can be done in parallel over each cell
    for (Index i = 0; i < nCells; i++)
    {

        // Reorganise the data from variable-major to cell-major
        Vector LambdaCell(2 * nVars);

        for (Index var = 0; var < nVars; var++)
        {
            LambdaCell.block<2, 1>(2 * var, 0) = adjoint_lambdas.segment(var * (nCells + 1) + i, 2);
        }

        /*
        // Try mapping the memory by using the magic runes (future update)
        Eigen::Map< Vector, 0, Eigen::InnerStride<nCells + 1> >
        delLambdaCell( delYVec.data() + LambdaOffset + i, 2 * nVars, Eigen::InnerStride<nCells + 1> );
        */

        adjoint_squ.emplace_back(SQU_f[i] - SQU_0[i] * LambdaCell);
    }
}

void SystemSolver::computeAdjointGradients()
{

    GlobalStateMatrix dSigmadp(nVars);
    GlobalStateMatrix dSourcedp(nVars);
    GlobalStateMatrix dAuxdp(nAux);

    // Spatial adjoint parameters index the parameter vector by node, so the star
    // node set would silently redefine how many parameters there are. Combined
    // with the fact that spatial adjoint output has never worked (WriteAdjoints
    // is disabled for exactly that reason, Solver.cpp:350), refusing is better
    // than producing a gradient whose meaning is unclear.
    if (superconvergent && adjointProblem->areParametersSpatial())
        throw std::invalid_argument(
            "Superconvergent postprocessing is not supported with spatial adjoint "
            "parameters");

    if (superconvergent)
        postprocessor->computeUStar(y);

    const std::vector<Position> points =
        superconvergent ? postprocessor->starPoints() : y.getPoints();
    const GlobalState states =
        superconvergent ? postprocessor->evalOnStarNodes(y) : y.evalOnNodes();

    const Index derivK = superconvergent ? k + 1 : k;

    const Index np_internal = adjointProblem->getNpInternal();
    logmsg<LOG_LEVEL::INFO>("Computing adjoints for {} parameters.", adjointProblem->getNp());
    for (Index var = 0; var < nVars; var++)
    {
      dSigmadp.add(nCells, derivK, np_internal, nScalars, np_internal);
      dSourcedp.add(nCells, derivK, np_internal, nScalars, np_internal);
    }
    for (Index aux = 0; aux < nAux; aux++)
    {
      dAuxdp.add(nCells, derivK, np_internal, nScalars, np_internal);
    }
    adjointProblem->ComputePhysicsDerivatives({dSigmadp, dSourcedp, dAuxdp}, states, points);
    
    // Spatial parameters effectively mean we have nCells * np parameters, but we store as a matrix to make output easier to interpret
    if (adjointProblem->areParametersSpatial())
        G_p.resize(adjointProblem->getNg() * nCells * (k + 1), adjointProblem->getNp());
    else 
        G_p.resize(adjointProblem->getNg(), adjointProblem->getNp());

    G_p.setZero();

    for (Index i = 0; i < adjointProblem->getNg(); i++)
    {
        if (adjointProblem->areParametersSpatial())
        {
            checkShapeAndSet(G_p.block(i * nCells * (k + 1), 0, nCells * (k + 1), adjointProblem->getNp()), adjointProblem->dGFndp(i, y), "dGdp in SystemSolver");
        }
        else
            G_p.row(i) = adjointProblem->dGFndp(i, y);
    }
    
    //Index np = adjointProblem->areParametersSpatial() ? np_internal * nCells * (k + 1) + adjointProblem->getNpBoundary() : adjointProblem->getNp();
    for (Index pIndex = 0; pIndex < adjointProblem->getNp(); ++pIndex)
    {
        for (Index i = 0; i < nCells; ++i)
        {
            Matrix F_p;
            if (adjointProblem->areParametersSpatial())
                F_p.resize(3 * nVars * (k + 1) + nAux * (k + 1), (k + 1));
            else
            {
                F_p.resize(3 * nVars * (k + 1) + nAux * (k + 1), 1);
            }
            F_p.setZero();

            Interval I = grid[i];

            for (Index var = 0; var < nVars; ++var)
            {
                
                Eigen::VectorXd dkappa_dp_phi(k + 1);
                Eigen::VectorXd dSdp_cellwise(k + 1);
                if (adjointProblem->areParametersSpatial())
                {
                    for (Index j = 0; j < k + 1; j++)
                    {
                        dkappa_dp_phi.setZero();
                        if (adjointProblem->isAdjointIndexInternal(pIndex))
                        {
                            const auto dSigmadp_cell = dSigmadp.Variable(i)[var];
                            Vector temp(k + 1);
                            temp.setZero();
                            temp(j) = dSigmadp_cell(pIndex, j);
                            dkappa_dp_phi = y.getBasis().InterpolateOntoBasis(I, temp);
                        }

                        // Evaluate Source Function

                        dSdp_cellwise.setZero();
                        if (adjointProblem->isAdjointIndexInternal(pIndex))
                        {
                            const auto dSourcedp_cell = dSourcedp.Variable(i)[var];
                            Vector temp(k + 1);
                            temp.setZero();
                            temp(j) = dSourcedp_cell(pIndex, j);
                            dSdp_cellwise = y.getBasis().InterpolateOntoBasis(I, temp);
                        }

                 
                        F_p.block(var * (k + 1), j, k + 1, 1) = dkappa_dp_phi;

                        F_p.block(var * (k + 1) + 2 * nVars * (k + 1), j, k + 1, 1) = -dSdp_cellwise;
                    }
                }
                else
                {
                    // Same projection the residual uses for these terms: A9 over
                    // the star nodes with the superconvergent scheme, the
                    // interpolatory mass-matrix form otherwise. The parameters do
                    // not enter u*, so there is no chain matrix here.
                    dkappa_dp_phi.setZero();
                    if( adjointProblem->isAdjointIndexInternal( pIndex ) )
                    {
                        const auto dSigmadp_cell = dSigmadp.Variable(i)[var];
                        dkappa_dp_phi = superconvergent
                            ? Vector(postprocessor->A9(i) * Vector(dSigmadp_cell.row(pIndex)))
                            : y.getBasis().InterpolateOntoBasis( I, dSigmadp_cell.row(pIndex) );
                    }

                    // Evaluate Source Function

                    dSdp_cellwise.setZero();
                    if( adjointProblem->isAdjointIndexInternal( pIndex ) )
                    {
                        const auto dSourcedp_cell = dSourcedp.Variable(i)[var];
                        dSdp_cellwise = superconvergent
                            ? Vector(postprocessor->A9(i) * Vector(dSourcedp_cell.row(pIndex)))
                            : y.getBasis().InterpolateOntoBasis( I, dSourcedp_cell.row(pIndex) );
                    }

                
                    F_p.block(var * (k + 1), 0, k + 1, 1) = dkappa_dp_phi;

                    //auto C_cell = C_cellwise[i];
                    F_p.block(var * (k + 1) + 2 * nVars * (k + 1), 0, k + 1, 1) = -dSdp_cellwise;
                }

                

                // Boundary conditions
                // p = g_D in this case, so the derivatives are just the basis functions
                //
                // "in this case" is load bearing, and is now enforced. This term is
                // the derivative of a *Dirichlet* datum, which reaches the residual
                // through RF_cellwise in the cell rows. A Neumann or Mixed datum
                // reaches it through L_global in the lambda row instead, and F_p has
                // no lambda rows at all -- it is 3*nVars*(k+1) + nAux*(k+1) tall, and
                // the lambda contribution exists only as the commented-out block
                // further down. So a boundary parameter on such an end would be
                // handed a Dirichlet-shaped derivative and return a plausible wrong
                // gradient with a perfectly good G, which is the failure mode this
                // file's dSigma/dPhi comment records. Refuse it instead.
                if (I.x_l == grid.lowerBoundary() && adjointProblem->computeLowerBoundarySensitivity(var, pIndex))
                {
                    if (!problem->isLowerBoundaryDirichlet(var))
                        throw std::logic_error(
                            "Adjoint parameter " + std::to_string(pIndex) + " is declared a lower "
                            "boundary sensitivity for variable '" + problem->getVariableName(var) +
                            "', whose lower boundary is not Dirichlet. Only a Dirichlet datum has a "
                            "derivative here: a Neumann or Mixed one enters through L_global in the "
                            "trace row, which F_p does not represent.");
                    for (Eigen::Index j = 0; j < k + 1; j++)
                    {
                        F_p(nVars * (k + 1) + j + var * (k + 1)) += y.getBasis().Evaluate(I, j, I.x_l);
                    }
                }

                if (I.x_u == grid.upperBoundary() && adjointProblem->computeUpperBoundarySensitivity(var, pIndex))
                {
                    if (!problem->isUpperBoundaryDirichlet(var))
                        throw std::logic_error(
                            "Adjoint parameter " + std::to_string(pIndex) + " is declared an upper "
                            "boundary sensitivity for variable '" + problem->getVariableName(var) +
                            "', whose upper boundary is not Dirichlet. Only a Dirichlet datum has a "
                            "derivative here: a Neumann or Mixed one enters through L_global in the "
                            "trace row, which F_p does not represent.");
                    for (Eigen::Index j = 0; j < k + 1; j++)
                    {
                        // < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 )
                        F_p(nVars * (k + 1) + j + var * (k + 1)) += y.getBasis().Evaluate(I, j, I.x_u);
                    }
                }
                // TODO: implement this 
               
            }
            for (Index aux = 0; aux < nAux; ++aux)
            {

                Eigen::VectorXd daux_dp_phi(k + 1);
                if (adjointProblem->areParametersSpatial())
                {
                  for (Index j = 0; j < k + 1; j++)
                  {
                    daux_dp_phi.setZero();
                    if( adjointProblem->isAdjointIndexInternal( pIndex ) )
                    {
                        const auto dAuxdp_cell = dAuxdp.Variable(i)[aux];
                        Vector temp(k + 1);
                        temp.setZero();
                        temp(j) = dAuxdp_cell(pIndex, j);
                        daux_dp_phi = y.getBasis().InterpolateOntoBasis(I, temp);
                    }
                    F_p.block(3 * nVars * (k + 1) + aux * (k + 1), j, k + 1, 1) = daux_dp_phi;
                  }
                }
                else
                {
                  daux_dp_phi.setZero();
                  if( adjointProblem->isAdjointIndexInternal( pIndex ) )
                  {
                      const auto dAuxdp_cell = dAuxdp.Variable(i)[aux];
                      daux_dp_phi = superconvergent
                          ? Vector(postprocessor->A9(i) * Vector(dAuxdp_cell.row(pIndex)))
                          : y.getBasis().InterpolateOntoBasis( I, dAuxdp_cell.row(pIndex) );
                  }
                  F_p.block(3 * nVars * (k + 1) + aux * (k + 1), 0, k + 1, 1) = daux_dp_phi;
                }
            }

            // SQU portion
        
            for (Index gIndex = 0; gIndex < adjointProblem->getNg(); gIndex++)
            {
                if (adjointProblem->areParametersSpatial())
                {
                    for (Index j = 0; j < k + 1; j ++)
                    {
                        G_p(gIndex * nCells * (k + 1) + i * (k + 1) + j, pIndex) -= adjoint_squ[i].transpose() * F_p.col(j);
                    }
                }
                else
                    G_p(gIndex, pIndex) -= adjoint_squ[i].transpose() * static_cast<Vector>(F_p);
            }


            // Eigen::VectorXd dkappa_lambda = C_cell * dkappa_dp_phi;
            // // // // // Lambda portion
            // G_p(pIndex) -= adjoint_lambdas.segment(i, 2).transpose() * dkappa_lambda;
        }
    }
}

void SystemSolver::print(std::ostream &out, double t, int nOut, N_Vector const &tempY, bool printSources)
{
    DGSoln tmp_y(nVars, grid, k, N_VGetArrayPointer(tempY), nScalars, nAux);

    std::println(out, "# t = {:g}", t);
    for (Index v = 0; v < nVars; ++v)
    {
        std::print(out, "# Lambda ({}) = ", v);
        for (Index i = 0; i < nCells; ++i)
            std::print(out, "{:g}, ", tmp_y.lambda(v)[i]);
        std::println(out, "{:g}", tmp_y.lambda(v)[nCells]);
    }

    if (nScalars > 0)
    {
        std::print(out, "# Scalars : ");
        for (Index i = 0; i < nScalars - 1; ++i)
            std::print(out, "{:g}, ", tmp_y.Scalar(i));
        std::println(out, "{:g}", tmp_y.Scalar(nScalars - 1));
    }

    double delta_x = (grid.upperBoundary() - grid.lowerBoundary()) * (1.0 / (nOut - 1.0));

    // Sources are vectorized so we interpolate to the output grid.
    //
    // The cache holds one value per node of whichever basis the residual last
    // evaluated the physics on: k+1 per cell normally, k+2 with the
    // superconvergent scheme. Wrapping it in a view of the wrong order reads
    // across cell boundaries, so the order and stride follow the scheme.
    const NodalBasis &sourceBasis =
        superconvergent ? postprocessor->getStarBasis() : y.getBasis();
    const size_t sourceStride = superconvergent ? k + 2 : k + 1;

    std::vector<DGApproxImpl<NodalBasis>> source_interp;
    if (printSources)
    {
        for (Index v = 0; v < nVars; ++v)
       {
          auto& Source_vals = problem->getSourceCache(v);
          source_interp.emplace_back(grid, sourceBasis, Source_vals.data(), sourceStride);
       }
    }

    if (postprocessor)
        postprocessor->computeUStar(tmp_y);

    for (int i = 0; i < nOut; ++i)
    {
        double x = static_cast<double>(i) * delta_x + grid.lowerBoundary();
        std::print(out, "{:g}", x);
        State s = tmp_y.eval(x);
        for (Index v = 0; v < nVars; ++v)
        {
            std::print(out, "\t{:g}\t{:g}\t{:g}", s.u(v), s.q(v), s.sigma(v));
            std::print(out, "\t{:g}", postprocessor ? postprocessor->uStar(v)(x) : s.u(v));
            if (printSources)
                std::print(out, "\t{:g}", source_interp[v](x));
        }

        for (Index a = 0; a < nAux; ++a)
            std::print(out, "\t{:g}", s.phi(a));

        std::println(out, "");
    }
    std::println(out, "");
    std::println(out, "");
}

void SystemSolver::print(std::ostream &out, double t, int nOut, bool printSources)
{

    std::println(out, "# t = {:g}", t);
    for (Index v = 0; v < nVars; ++v)
    {
        std::print(out, "# Lambda ({}) = ", v);
        for (Index i = 0; i < nCells; ++i)
            std::print(out, "{:g}, ", y.lambda(v)[i]);
        std::println(out, "{:g}", y.lambda(v)[nCells]);
    }

    if (nScalars > 0)
    {
        std::print(out, "# Scalars : ");
        for (Index i = 0; i < nScalars - 1; ++i)
            std::print(out, "{:g}, ", y.Scalar(i));
        std::println(out, "{:g}", y.Scalar(nScalars - 1));
    }

    double delta_x = (grid.upperBoundary() - grid.lowerBoundary()) * (1.0 / (nOut - 1.0));


    // See the sibling overload above for why the basis and stride follow the
    // scheme rather than being fixed at k+1.
    const NodalBasis &sourceBasis =
        superconvergent ? postprocessor->getStarBasis() : y.getBasis();
    const size_t sourceStride = superconvergent ? k + 2 : k + 1;

    std::vector<DGApproxImpl<NodalBasis>> source_interp;
    if (printSources)
    {
        for (Index v = 0; v < nVars; ++v)
       {
          auto& Source_vals = problem->getSourceCache(v);
          source_interp.emplace_back(grid, sourceBasis, Source_vals.data(), sourceStride);
       }
    }

    if (postprocessor)
        postprocessor->computeUStar(y);

    for (int i = 0; i < nOut; ++i)
    {
        double x = static_cast<double>(i) * delta_x + grid.lowerBoundary();
        std::print(out, "{:g}", x);
        State s = y.eval(x);
        for (Index v = 0; v < nVars; ++v)
        {
            std::print(out, "\t{:g}\t{:g}\t{:g}", s.u(v), s.q(v), s.sigma(v));
            std::print(out, "\t{:g}", postprocessor ? postprocessor->uStar(v)(x) : s.u(v));
            if (printSources)
                std::print(out, "\t{:g}", source_interp[v](x));
        }

        for (Index a = 0; a < nAux; ++a)
            std::print(out, "\t{:g}", s.phi(a));

        std::println(out, "");
    }
    std::println(out, "");
    std::println(out, ""); // Two blank lines needed to make gnuplot happy
}

void SystemSolver::printOnNodes(std::ostream &out, double t, N_Vector const& tempY, bool printSources)
{

    DGSoln tmp_y(nVars, grid, k, N_VGetArrayPointer(tempY), nScalars, nAux);
    std::println(out, "# t = {:g}", t);
    for (Index v = 0; v < nVars; ++v)
    {
        std::print(out, "# Lambda ({}) = ", v);
        for (Index i = 0; i < nCells; ++i)
            std::print(out, "{:g}, ", tmp_y.lambda(v)[i]);
        std::println(out, "{:g}", tmp_y.lambda(v)[nCells]);
    }

    if (nScalars > 0)
    {
        std::print(out, "# Scalars : ");
        for (Index i = 0; i < nScalars - 1; ++i)
            std::print(out, "{:g}, ", tmp_y.Scalar(i));
        std::println(out, "{:g}", tmp_y.Scalar(nScalars - 1));
    }

    std::vector<Values> sources(nVars);
    if (printSources) 
    {
        for (Index v = 0; v < nVars; ++v)
        {
            sources[v] = problem->Sources(v, tmp_y.evalOnNodes(), tmp_y.getPoints(), t);
        }
    }
    const auto states = tmp_y.evalOnNodes();
    const auto points = tmp_y.getPoints();

    if (postprocessor)
        postprocessor->computeUStar(tmp_y);

    for (size_t i = 0; i < points.size(); ++i)
    {
        const auto& x = points[i];
        const State s = states[i];

        std::print(out, "{:g}", x);
        for (Index v = 0; v < nVars; ++v)
        {
            std::print(out, "\t{:g}\t{:g}\t{:g}", s.u(v), s.q(v), s.sigma(v));
            std::print(out, "\t{:g}", postprocessor ? postprocessor->uStar(v)(x) : s.u(v));
            if (printSources)
                std::print(out, "\t{:g}", sources[v](i));
        }
        for (Index a = 0; a < nAux; ++a)
            std::print(out, "\t{:g}", s.phi(a));

        std::println(out, "");
       
    }
    std::println(out, "");
    std::println(out, ""); // Two blank lines needed to make gnuplot happy
}

int SystemSolver::getErrorWeights(N_Vector y_sundials, N_Vector ewt_sundials)
{
    DGSoln y(nVars, grid, k, N_VGetArrayPointer(y_sundials), nScalars, nAux);
    DGSoln ewt(nVars, grid, k, N_VGetArrayPointer(ewt_sundials), nScalars, nAux);
    for (Index i = 0; i < nCells; ++i)
    {
        double absTol = 1e-8;
        for (Index v = 0; v < nVars; ++v)
        {
            if (atol.size() == 1)
            {
                absTol = atol[0];
            }
            else if (atol.size() == nVars)
            {
                absTol = atol[v];
            }

            ewt.u(v).getCoeff(i).second = 1.0 / (rtol * abs(y.u(v).getCoeff(i).second.array()) + absTol);
            ewt.q(v).getCoeff(i).second = 1.0 / (rtol * abs(y.q(v).getCoeff(i).second.array()) + absTol);
            ewt.sigma(v).getCoeff(i).second = 1.0 / (rtol * abs(y.sigma(v).getCoeff(i).second.array()) + absTol);
        }

        for (Index a = 0; a < nAux; ++a)
        {

            ewt.Aux(a).getCoeff(i).second = 1.0 / (rtol * abs(y.Aux(a).getCoeff(i).second.array()) + absTol);
        }
    }

    for (Index v = 0; v < nVars; ++v)
    {

        double absTol = 1e-8;

        if (atol.size() == 1)
        {
            absTol = atol[0];
        }
        else if (atol.size() == nVars)
        {
            absTol = atol[v];
        }
        ewt.lambda(v) = 1.0 / (rtol * abs(y.lambda(v).array()) + absTol);
    }

    for (Index i = 0; i < nScalars; ++i)
    {
        double absTol = atol[0];
        ewt.Scalar(i) = ::sqrt(localDOF * nCells) / (rtol * abs(y.Scalar(i)) + absTol);
    }

    return 0;
}

int SystemSolver::getErrorWeights_static(N_Vector y, N_Vector ewt, void *sys)
{
    return static_cast<SystemSolver *>(sys)->getErrorWeights(y, ewt);
}

void SystemSolver::PrintDebugInfo()
{
    initialiseMatrices();
    for (Index i = 0; i < nCells; i++)
    {
        std::println("Cell {} M Matrix: ", i);
        // Eigen has no std::formatter, so write the block out row by row.
        for (Index r = 0; r < MBlocks[i].rows(); ++r)
        {
            for (Index c = 0; c < MBlocks[i].cols(); ++c)
                std::print("{:g} ", MBlocks[i](r, c));
            std::println("");
        }
        std::println("");
    }
}

AccuracyEstimate SystemSolver::accuracyEstimate(Index var)
{
    if (postprocessor == nullptr)
        throw std::runtime_error(
            "No postprocessed solution to estimate an error from: u* is not "
            "built at k = 0.");

    if (yJacMem == nullptr)
        throw std::runtime_error(
            "accuracyEstimate called before initialize(), so there is no "
            "solution to estimate the error of.");

    postprocessor->computeUStar(yJac);
    return postprocessor->accuracyIndicator(yJac, var);
}

std::vector<double> SystemSolver::stateVector() const
{
    if (yJacMem == nullptr)
        throw std::runtime_error("stateVector called before initialize().");
    return std::vector<double>(yJacMem, yJacMem + yJac.getDoF());
}

std::vector<double> SystemSolver::derivativeVector() const
{
    if (dydtJacMem == nullptr)
        throw std::runtime_error("derivativeVector called before initialize().");
    return std::vector<double>(dydtJacMem, dydtJacMem + dydtJac.getDoF());
}
