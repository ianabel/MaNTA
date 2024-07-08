#include "SystemSolver.hpp"
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of sunrealtype, sunindextype  */
#include <nvector/nvector_serial.h>			/* access to serial N_Vector            */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <toml.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(Grid const &Grid, unsigned int polyNum, TransportSystem *transpSystem)
	: grid(Grid), k(polyNum), nCells(Grid.getNCells()), nVars(transpSystem->getNumVars()), nScalars(transpSystem->getNumScalars()), MXSolvers(Grid.getNCells()), y(nVars, grid, k, nScalars), dydt(nVars, grid, k, nScalars), yJac(nVars, grid, k, nScalars), problem(transpSystem)
{
	if (SUNContext_Create(SUN_COMM_NULL, &ctx) < 0)
		throw std::runtime_error("Unable to allocate SUNDIALS Context, aborting.");
	yJacMem = new double[yJac.getDoF()];
	yJac.Map(yJacMem);
	S_DOF = k + 1;
	U_DOF = k + 1;
	Q_DOF = k + 1;
	SQU_DOF = U_DOF + Q_DOF + S_DOF;
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
	t = t0;
	y.Map(N_VGetArrayPointer(Y));
	dydt.Map(N_VGetArrayPointer(dYdt));

	resetCoeffs();
	if (!initialised)
		throw std::logic_error("setInitialConditions can only be called after initialising the matrices");

	// slightly minging syntax. blame C++
	auto initial_u = std::bind_front(&TransportSystem::InitialValue, problem);
	auto initial_q = std::bind_front(&TransportSystem::InitialDerivative, problem);
	y.AssignU(initial_u);
	y.AssignQ(initial_q);

	y.EvaluateLambda();
	// y.EvaluateLambda( tauc );

	/*
	for ( Index i = 0; i < nCells; i++ )
	{
		for ( Index var = 0; var < nVars; var++ )
		{
			y.q( var ).getCoeff( i ).second = A_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1).inverse() * (
					 - B_cellwise[i].transpose().block(var * (k + 1), var * (k + 1), k + 1, k + 1) * y.u(var).getCoeff(i).second
					 + C_cellwise[i].transpose().block(var * (k + 1), var * 2, k + 1, 2) * y.lambda(var).segment<2>( i )
					 - RF_cellwise[i].block(var * (k + 1), 0, k + 1, 1)
					);
		}
	}
	*/

	for (Index s = 0; s < nScalars; ++s)
		y.Scalar(s) = problem->InitialScalarValue(s);

	ApplyDirichletBCs(y); // If dirichlet, overwrite with those boundary conditions

	auto sigma_wrapper = [this](Index i, const State &s, Position x, Time t)
	{ return -problem->SigmaFn(i, s, x, t); };
	y.AssignSigma(sigma_wrapper);

	for (Index var = 0; var < nVars; var++)
	{
		// Solver For dudt with dudt = X^-1( -B*Sig - D*U - E*Lam + F )
		Eigen::Vector2d lamCell;
		for (Index i = 0; i < nCells; i++)
		{
			Interval I = grid[i];

			// Evaluate Source Function
			Eigen::VectorXd S_cellwise(k + 1);
			S_cellwise.setZero();

			auto const &x_vals = DGApprox::Integrator().abscissa();
			auto const &x_wgts = DGApprox::Integrator().weights();
			const size_t n_abscissa = x_vals.size();

			S_cellwise.setZero();
			for (size_t i = 0; i < n_abscissa; ++i)
			{
				double x_val = I.x_l + (1 + x_vals[i]) * I.h() / 2.0;
				double wgt = x_wgts[i] * (I.h() / 2.0);

				State s = y.eval(x_val);
				double sourceVal = problem->Sources(var, s, x_val, t);
				for (Eigen::Index j = 0; j < k + 1; j++)
					S_cellwise(j) += wgt * sourceVal * LegendreBasis::Evaluate(I, j, x_val);
			}
			// And for the other half of the abscissas
			for (size_t i = 0; i < n_abscissa; ++i)
			{
				double x_val = I.x_l + (1 - x_vals[i]) * I.h() / 2.0;
				double wgt = x_wgts[i] * (I.h() / 2.0);

				State s = y.eval(x_val);
				double sourceVal = problem->Sources(var, s, x_val, t);
				for (Eigen::Index j = 0; j < k + 1; j++)
					S_cellwise(j) += wgt * sourceVal * LegendreBasis::Evaluate(I, j, x_val);
			}

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
		dydt.q(var).zeroCoeffs();
		dydt.sigma(var).zeroCoeffs();
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
			DGApprox::MassMatrix(I, Avar);
			// B_ij = ( phi_i, phi_j' )
			DGApprox::DerivativeMatrix(I, Bvar);

			// Now do all the boundary terms
			for (Eigen::Index i = 0; i < k + 1; i++)
			{
				for (Eigen::Index j = 0; j < k + 1; j++)
				{
					Dvar(i, j) +=
						tau(I.x_l) * LegendreBasis::Evaluate(I, j, I.x_l) * LegendreBasis::Evaluate(I, i, I.x_l) +
						tau(I.x_u) * LegendreBasis::Evaluate(I, j, I.x_u) * LegendreBasis::Evaluate(I, i, I.x_u);
				}
			}

			A.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Avar;
			D.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Dvar;
			B.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Bvar;
		}

		A_cellwise.emplace_back(A);
		B_cellwise.emplace_back(B);
		D_cellwise.emplace_back(D);

		Eigen::MatrixXd M(3 * nVars * (k + 1), 3 * nVars * (k + 1));
		M.setZero();

		// row1
		M.block(0, 0, nVars * (k + 1), nVars * (k + 1)) = A;
		M.block(0, nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero();	 // NLq added at Jac step
		M.block(0, 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero(); // NLu added at Jac step

		// why negative?
		// row2
		M.block(nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)).setZero();
		M.block(nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = -A;
		M.block(nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = -B.transpose();

		// row3
		M.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)) = B;
		M.block(2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)).setZero();
		M.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = D; // X added at Jac step

		// TODO: Consider factorization here (is M sparse enough to warrant a sparse implementation?)
		MBlocks.emplace_back(M);

		Eigen::MatrixXd CE_vec(3 * nVars * (k + 1), 2 * nVars);
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

				Cvar(0, i) = -LegendreBasis::Evaluate(I, i, I.x_l);
				Cvar(1, i) = LegendreBasis::Evaluate(I, i, I.x_u);

				// E_ij = < phi_i, (- tau ) lambda >
				Evar(i, 0) = LegendreBasis::Evaluate(I, i, I.x_l) * (-tau(I.x_l));
				Evar(i, 1) = LegendreBasis::Evaluate(I, i, I.x_u) * (-tau(I.x_u));

				if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
				{
					Cvar(0, i) = 0;
					Evar(i, 0) = 0;
				}

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
		CEBlocks.emplace_back(CE_vec);

		C_cellwise.emplace_back(C);
		E_cellwise.emplace_back(E);

		// To store the RHS
		RF_cellwise.emplace_back(nVars * 2 * (k + 1));

		// R is composed of parts of the values of
		// u on the total domain boundary
		// don't need to do RHS terms here, those are now in 'Sources'
		// (should we double check that RF_cellwise[ i ] == RF_cellwise.back()?
		RF_cellwise[i].setZero();

		for (Index var = 0; var < nVars; var++)
		{
			if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
			{
				for (Eigen::Index j = 0; j < k + 1; j++)
				{
					// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 )
					RF_cellwise[i](j + var * (k + 1)) += -LegendreBasis::Evaluate(I, j, I.x_l) * (-1) * problem->LowerBoundary(var, 0.0);
					// < ( tau ) g_D, w >
					RF_cellwise[i](nVars * (k + 1) + j + var * (k + 1)) += LegendreBasis::Evaluate(I, j, I.x_l) * tau(I.x_l) * problem->LowerBoundary(var, 0.0);
				}
			}

			if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
			{
				for (Eigen::Index j = 0; j < k + 1; j++)
				{
					// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 )
					RF_cellwise[i](j + var * (k + 1)) += -LegendreBasis::Evaluate(I, j, I.x_u) * (+1) * problem->UpperBoundary(var, 0.0);
					RF_cellwise[i](nVars * (k + 1) + j + var * (k + 1)) += LegendreBasis::Evaluate(I, j, I.x_u) * tau(I.x_u) * problem->UpperBoundary(var, 0.0);
				}
			}
		}

		// Per-cell contributions to the global matrices K and F.
		// First fill G
		Eigen::MatrixXd G(2 * nVars, nVars * (k + 1));
		G.setZero();
		for (Index var = 0; var < nVars; var++)
		{
			Eigen::MatrixXd Gvar(2, k + 1);
			for (Index i = 0; i < k + 1; i++)
			{
				Gvar(0, i) = tau(I.x_l) * LegendreBasis::Evaluate(I, i, I.x_l);
				if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
					Gvar(0, i) = 0.0;
				Gvar(1, i) = tau(I.x_u) * LegendreBasis::Evaluate(I, i, I.x_u);
				if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
					Gvar(1, i) = 0.0;
			}
			G.block(2 * var, (k + 1) * var, 2, (k + 1)) = Gvar;
		}

		//[ C 0 G ]
		CG_cellwise.emplace_back(2 * nVars, 3 * nVars * (k + 1));
		CG_cellwise[i].setZero();
		CG_cellwise[i].block(0, 0, 2 * nVars, nVars * (k + 1)) = C;
		CG_cellwise[i].block(0, nVars * (k + 1), 2 * nVars, nVars * (k + 1)).setZero();
		CG_cellwise[i].block(0, 2 * nVars * (k + 1), 2 * nVars, nVars * (k + 1)) = G;
		G_cellwise.emplace_back(G);

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

			if (I.x_l == grid.lowerBoundary() && problem->isLowerBoundaryDirichlet(var))
				Hvar(0, 0) = 0.0;

			if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
				Hvar(1, 1) = 0.0;

			H.block(2 * var, 2 * var, 2, 2) = Hvar;
			HGlobalMat.block(var * (nCells + 1) + i, var * (nCells + 1) + i, 2, 2) += Hvar;
		}

		H_cellwise.emplace_back(H);

		// Finally fill L
		for (Index var = 0; var < nVars; var++)
		{
			if (I.x_l == grid.lowerBoundary() && /* is b.d. Neumann at lower boundary */ !problem->isLowerBoundaryDirichlet(var))
				L_global(var * (nCells + 1) + i) += problem->LowerBoundary(var, 0.0);
			if (I.x_u == grid.upperBoundary() && /* is b.d. Neumann at upper boundary */ !problem->isUpperBoundaryDirichlet(var))
				L_global(var * (nCells + 1) + i + 1) += problem->UpperBoundary(var, 0.0);
		}

		Eigen::MatrixXd X(nVars * (k + 1), nVars * (k + 1));
		X.setZero();
		for (Index var = 0; var < nVars; var++)
		{
			Eigen::MatrixXd Xvar(k + 1, k + 1);
			DGApprox::MassMatrix(I, Xvar, [this, var](double x)
								 { return problem->aFn(var, x); });
			X.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Xvar;
		}
		XMats.emplace_back(X);

		MXSolvers.emplace_back(nVars * SQU_DOF);
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

void SystemSolver::clearCellwiseVecs()
{
	XMats.clear();
	MBlocks.clear();
	CG_cellwise.clear();
	RF_cellwise.clear();
	A_cellwise.clear();
	B_cellwise.clear();
	E_cellwise.clear();
	C_cellwise.clear();
	G_cellwise.clear();
	H_cellwise.clear();
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
					RF_cellwise[i](j + var * (k + 1)) += -LegendreBasis::Evaluate(I, j, I.x_l) * (-1) * problem->LowerBoundary(var, t);
					// < ( tau ) g_D, w >
					RF_cellwise[i](nVars * (k + 1) + j + var * (k + 1)) += LegendreBasis::Evaluate(I, j, I.x_l) * tau(I.x_l) * problem->LowerBoundary(var, t);
				}
			}

			if (I.x_u == grid.upperBoundary() && problem->isUpperBoundaryDirichlet(var))
			{
				for (Eigen::Index j = 0; j < k + 1; j++)
				{
					// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 )
					RF_cellwise[i](j + var * (k + 1)) += -LegendreBasis::Evaluate(I, j, I.x_u) * (+1) * problem->UpperBoundary(var, t);
					RF_cellwise[i](nVars * (k + 1) + j + var * (k + 1)) += LegendreBasis::Evaluate(I, j, I.x_u) * tau(I.x_u) * problem->UpperBoundary(var, t);
				}
			}

			if (I.x_l == grid.lowerBoundary() && /* is b.d. Neumann at lower boundary */ !problem->isLowerBoundaryDirichlet(var))
				L_global(var * (nCells + 1) + i) += problem->LowerBoundary(var, t);
			if (I.x_u == grid.upperBoundary() && /* is b.d. Neumann at upper boundary */ !problem->isUpperBoundaryDirichlet(var))
				L_global(var * (nCells + 1) + i + 1) += problem->UpperBoundary(var, t);
		}
	}
}

Vector SystemSolver::resEval(std::vector<Vector> resTerms)
{
	Vector maxTerms, res;
	maxTerms.resize(resTerms[0].size());
	res.resize(resTerms[0].size());
	maxTerms.setZero();
	res.setZero();
	for (auto term : resTerms)
	{
		res += term;
		for (int i = 0; i < term.size(); i++)
		{
			if (std::abs(term[i]) > maxTerms[i])
				maxTerms[i] = std::abs(term[i]);
		}
	}
	for (int i = 0; i < resTerms[0].size(); i++)
	{
		if (res[i] < maxTerms[i] * 10e-10)
			res[i] = 0.0;
	}
	return res;
}

void SystemSolver::resetCoeffs()
{
	y.zeroCoeffs();
	dydt.zeroCoeffs();
}

void SystemSolver::updateMatricesForJacSolve()
{
	updateBoundaryConditions(jt);
	// We know where the jacobian is to be evaluated -- yJac
	for (unsigned int i = 0; i < nCells; i++)
	{

		Eigen::MatrixXd X(nVars * (k + 1), nVars * (k + 1));
		Eigen::MatrixXd NLq(nVars * (k + 1), nVars * (k + 1));
		Eigen::MatrixXd NLu(nVars * (k + 1), nVars * (k + 1));
		Eigen::MatrixXd Ssig(nVars * (k + 1), nVars * (k + 1));
		Eigen::MatrixXd Sq(nVars * (k + 1), nVars * (k + 1));
		Eigen::MatrixXd Su(nVars * (k + 1), nVars * (k + 1));

		Interval const &I(grid[i]);
		Eigen::MatrixXd MX(3 * nVars * (k + 1), 3 * nVars * (k + 1));
		MX = MBlocks[i];

		// X matrix
		for (Index var = 0; var < nVars; var++)
		{
			std::function<double(double)> alphaF = [=, this](double x)
			{ return alpha * problem->aFn(var, x); };
			Eigen::MatrixXd Xsubmat((k + 1), (k + 1));
			DGApprox::MassMatrix(I, Xsubmat, alphaF);
			X.block(var * (k + 1), var * (k + 1), k + 1, k + 1) = Xsubmat;
		}
		MX.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) += X;

		// NLq Matrix
		NLqMat(NLq, yJac, I);
		MX.block(0, nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLq;

		// NLu Matrix
		NLuMat(NLu, yJac, I);
		MX.block(0, 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) = NLu;

		// S_sig Matrix
		dSourcedsigma_Mat(Ssig, yJac, I);
		MX.block(2 * nVars * (k + 1), 0, nVars * (k + 1), nVars * (k + 1)) -= Ssig;

		// S_q Matrix
		dSourcedq_Mat(Sq, yJac, I);
		MX.block(2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) -= Sq;

		// S_u Matrix
		dSourcedu_Mat(Su, yJac, I);
		MX.block(2 * nVars * (k + 1), 2 * nVars * (k + 1), nVars * (k + 1), nVars * (k + 1)) -= Su;

		MXSolvers[i].compute(MX);
	}

	// Construct the N_HDG_DOF x N_Scalar matrix v which
	// contains the effect of the scalars on the main variables (through the sources. nothing else is allowed to depend on scalars)

	std::vector<DGSoln> v_map;
	for (Index i = 0; i < nScalars; ++i)
		v_map.emplace_back(nVars, grid, k, N_VGetArrayPointer(v[i]), nScalars);

	for (Index i = 0; i < nCells; ++i)
	{
		Matrix v_tmp(nVars * U_DOF, nScalars);
		dSources_dScalars_Mat(v_tmp, yJac, grid[i]);
		for (Index j = 0; j < nScalars; ++j)
			for (Index v = 0; v < nVars; ++v)
				v_map[j].u(v).getCoeff(i).second = v_tmp.block(v * U_DOF, j, U_DOF, 1);
	}
	v_map.clear();

	// Construct N_Scalar x N_HDG_DOF matrix w which contains the Jacobian
	// of the scalars with respect to the other variables
	// also construct the scalar-scalar coupling matrix N

	std::vector<DGSoln> w_map;
	for (Index i = 0; i < nScalars; ++i)
		w_map.emplace_back(nVars, grid, k, N_VGetArrayPointer(w[i]), nScalars);

	for (Index i = 0; i < nCells; ++i)
	{
		Interval const &I(grid[i]);
		for (Index j = 0; j < nScalars; ++j)
		{
			State s(nVars, nScalars);
			State s_dt(nVars, nScalars);
			for (Index l = 0; l < k + 1; ++l)
			{
				problem->ScalarGPrimeExtended(j, s, s_dt, yJac, [=](double x)
											  { return LegendreBasis::Evaluate(I, l, x); }, I, jt);
				for (Index v = 0; v < nVars; ++v)
				{
					w_map[j].sigma(v).getCoeff(i).second(l) = s.Flux[v];
					w_map[j].q(v).getCoeff(i).second(l) = s.Derivative[v];
					w_map[j].u(v).getCoeff(i).second(l) = s.Variable[v];
				}
				if (l == 0 && i == 0)
				{
					/* dG_i/d(Scalar_j) should vanish for  i!=j and doesn't depend on test function, so we can just do this once  */
					N_global(j, j) = s.Scalars[j];
				}
			}
		}
	}
	w_map.clear();
}

void SystemSolver::mapDGtoSundials(std::vector<VectorWrapper> &SQU_cell, VectorWrapper &lam, sunrealtype *const &Y) const
{
	SQU_cell.clear();
	for (Index i = 0; i < nCells; i++)
	{
		SQU_cell.emplace_back(VectorWrapper(Y + i * nVars * SQU_DOF, nVars * SQU_DOF));
	}

	new (&lam) VectorWrapper(Y + nVars * (nCells) * (3 * k + 3), nVars * (nCells + 1));
}

void SystemSolver::setJacEvalY(N_Vector &yy)
{
	DGSoln yyMap(nVars, grid, k, nScalars);
	assert(static_cast<size_t>(N_VGetLength(yy)) == yyMap.getDoF());
	yyMap.Map(N_VGetArrayPointer(yy));

	yJac.copy(yyMap); // Deep copy -- yyMap only aliases the N_Vector, this copies the data
}

// Over-arching Jacobian function. If there's no coupled B-field solve, or auxiliar variables, then just do the
// HDG Jacobian solve
void SystemSolver::solveJacEq(N_Vector res_g, N_Vector delY)
{
	if (nScalars > 0)
	{
		// TODO: move temporaries into private variables of the class and allocate/destroy once
		// allocate temporary working space for gauss elimination of scalars.
		N_Vector d = N_VClone(delY);
		N_Vector *e = new N_Vector[nScalars];
		for (Index i = 0; i < nScalars; ++i)
			e[i] = N_VClone(delY);
		N_Vector g = N_VClone(delY);

		// Let A be the HDG linear operator solved in solveHDGJac

		// First solve A d = res_g ;
		solveHDGJac(res_g, d);

		// Now A e = v ; Do as a loop over nScalars
		for (Index i = 0; i < nScalars; ++i)
		{
			solveHDGJac(e[i], v[i]);
		}

		Vector tmp_N = (N_global.inverse() * res_g_map.Scalars());
		N_VLinearCombination(nScalars, tmp_N.data(), e, g); // g = Sum_i tmp_N[i]*e[i]
		N_VLinearSum(1.0, g, 1.0, d, g);					// g += d;

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
		Vector NweInv_w_g = -1 * Nwe.inverse() * wDotg;				// Uses PartialPivLU internally, never really does inverse (except for small matrices)
		N_VLinearCombination(nScalars, NweInv_w_g.data(), e, delY); // Set delY = - [ e  ( N + w^T e )^-1  w ]  g
		N_VLinearSum(1.0, delY, 1.0, g, delY);						// delY += g; so delY = g - (....), which is the final answer

		// Finally, set the components of delY related to the change of the scalars
		DGSoln del_y(nVars, grid, k, N_VGetArrayPointer(delY), nScalars);

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
	DGSoln del_y(nVars, grid, k, nScalars);
#ifdef DEBUG
	// Provide view on g for debugging
	DGSoln gMap(nVars, grid, k, nScalars);
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

		Eigen::MatrixXd K_cell(nVars * 2, nVars * 2);
		K_cell = H_cellwise[i] - CG_cellwise[i] * SQU_0[i];

		// K
		for (Index varI = 0; varI < nVars; varI++)
			for (Index varJ = 0; varJ < nVars; varJ++)
				K_global.block<2, 2>(varI * (nCells + 1) + i, varJ * (nCells + 1) + i) += K_cell.block<2, 2>(varI * 2, varJ * 2);
	}

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
	Index LambdaOffset = nVars * nCells * SQU_DOF;

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
#pragma omp parallel for
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
	}
}

int static_residual(sunrealtype tres, N_Vector Y, N_Vector dYdt, N_Vector resval, void *user_data)
{
	auto system = reinterpret_cast<SystemSolver *>(user_data);
	try
	{
		return system->residual(tres, Y, dYdt, resval);
	}
	catch (std::exception &e)
	{
		std::cout << "Caught exception : " << e.what() << " ; Retrying. " << std::endl;
		return 1;
	}
}

int SystemSolver::residual(sunrealtype tres, N_Vector Y, N_Vector dYdt, N_Vector resval)
{
	updateBoundaryConditions(tres);

	DGSoln Y_h(nVars, grid, k, N_VGetArrayPointer(Y), nScalars);
	DGSoln dYdt_h(nVars, grid, k, N_VGetArrayPointer(dYdt), nScalars);
	DGSoln res(nVars, grid, k, N_VGetArrayPointer(resval), nScalars);

	VectorWrapper resVec(N_VGetArrayPointer(resval), N_VGetLength(resval));

	resVec.setZero();

	// residual.lambda = C*sigma + G*u + H*lambda - L

	for (Index i = 0; i < nCells; i++)
	{
		// C_cellwise * sigma_cellwise
		for (Index var = 0; var < nVars; var++)
		{
			res.lambda(var).segment<2>(i) +=
				C_cellwise[i].block(var * 2, var * (k + 1), 2, k + 1) * Y_h.sigma(var).getCoeff(i).second + G_cellwise[i].block(var * 2, var * (k + 1), 2, k + 1) * Y_h.u(var).getCoeff(i).second + H_cellwise[i].block(2 * var, 2 * var, 2, 2) * Y_h.lambda(var).segment<2>(i) - L_global.segment<2>(var * (nCells + 1));
		}
	}

#pragma omp parallel for
	for (Index i = 0; i < nCells; i++)
	{
		Interval I = grid[i];
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
			std::function<double(double)> kappaFunc = [=, this, &Y_h](double x)
			{
				State s = Y_h.eval(x);
				return problem->SigmaFn(var, s, x, tres);
			};

			std::function<double(double)> sourceFunc = [=, this, &Y_h](double x)
			{
				State s = Y_h.eval(x);
				return problem->Sources(var, s, x, tres);
			};

			// Evaluate Diffusion Function
			Eigen::VectorXd kappa_cellwise(k + 1);
			kappa_cellwise.setZero();
			for (Eigen::Index j = 0; j < k + 1; j++)
				kappa_cellwise(j) = DGApprox::CellProduct(I, kappaFunc, LegendreBasis::phi(I, j));

			// Evaluate Source Function
			Eigen::VectorXd S_cellwise(k + 1);
			S_cellwise.setZero();
			for (Eigen::Index j = 0; j < k + 1; j++)
				S_cellwise(j) = DGApprox::CellProduct(I, sourceFunc, LegendreBasis::phi(I, j));

			auto const &lambda = lamCell.segment<2>(2 * var);

			// We should normalise the components of the residual such that the `sigma' component of res
			// has tolerances that are the same as `sigma' itself.
			//
			// For sigma and q, just make the 'sigma-determinitive' equation the sigma component and the same for q.
			// as these equations are proportional to the variables themselves, we are done

			res.sigma(var).getCoeff(i).second = Y_h.sigma(var).getCoeff(i).second + kappa_cellwise;

			res.q(var).getCoeff(i).second =
				-A_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.q(var).getCoeff(i).second - B_cellwise[i].transpose().block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.u(var).getCoeff(i).second + C_cellwise[i].transpose().block(var * (k + 1), var * 2, k + 1, 2) * lambda - RF_cellwise[i].block(var * (k + 1), 0, k + 1, 1);

			// For the 'u' component of the residual, we also have a factor of d/dt. Thus we should multiply this equation by some frequency estimate.
			// For the moment we leave it as it is.
			res.u(var).getCoeff(i).second =
				B_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.sigma(var).getCoeff(i).second + D_cellwise[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * Y_h.u(var).getCoeff(i).second + E_cellwise[i].block(var * (k + 1), var * 2, k + 1, 2) * lambda - RF_cellwise[i].block(nVars * (k + 1) + var * (k + 1), 0, k + 1, 1) - S_cellwise + XMats[i].block(var * (k + 1), var * (k + 1), k + 1, k + 1) * dYdt_h.u(var).getCoeff(i).second;
		}
	}

	for (Index j = 0; j < nScalars; j++)
	{
		res.Scalar(j) = problem->ScalarG(j, Y_h, tres);
	}

	return 0;
}

void SystemSolver::print(std::ostream &out, double t, int nOut, N_Vector const &tempY, bool printSources)
{
	DGSoln tmp_y(nVars, grid, k, N_VGetArrayPointer(tempY), nScalars);

	out << "# t = " << t << std::endl;
	for (Index v = 0; v < nVars; ++v)
	{
		out << "# Lambda (" << v << ") = ";
		for (Index i = 0; i < nCells; ++i)
			out << tmp_y.lambda(v)[i] << ", ";
		out << tmp_y.lambda(v)[nCells] << std::endl;
	}

	if (nScalars > 0)
	{
		out << "# Scalars : ";
		for (Index i = 0; i < nScalars - 1; ++i)
			out << tmp_y.Scalar(i) << ", ";
		out << tmp_y.Scalar(nScalars - 1) << std::endl;
	}

	double delta_x = (grid.upperBoundary() - grid.lowerBoundary()) * (1.0 / (nOut - 1.0));
	for (int i = 0; i < nOut; ++i)
	{
		double x = static_cast<double>(i) * delta_x + grid.lowerBoundary();
		out << x;
		State s = tmp_y.eval(x);
		for (Index v = 0; v < nVars; ++v)
		{
			out << "\t" << s.Variable[v] << "\t" << s.Derivative[v] << "\t" << s.Flux[v];
			if (printSources)
				out << "\t" << problem->Sources(v, s, x, t);
		}
		out << std::endl;
	}
	out << std::endl;
	out << std::endl;
}

void SystemSolver::print(std::ostream &out, double t, int nOut, bool printSources)
{

	out << "# t = " << t << std::endl;
	for (Index v = 0; v < nVars; ++v)
	{
		out << "# Lambda (" << v << ") = ";
		for (Index i = 0; i < nCells; ++i)
			out << y.lambda(v)[i] << ", ";
		out << y.lambda(v)[nCells] << std::endl;
	}

	if (nScalars > 0)
	{
		out << "# Scalars : ";
		for (Index i = 0; i < nScalars - 1; ++i)
			out << y.Scalar(i) << ", ";
		out << y.Scalar(nScalars - 1) << std::endl;
	}

	double delta_x = (grid.upperBoundary() - grid.lowerBoundary()) * (1.0 / (nOut - 1.0));
	for (int i = 0; i < nOut; ++i)
	{
		double x = static_cast<double>(i) * delta_x + grid.lowerBoundary();
		out << x;
		State s = y.eval(x);
		for (Index v = 0; v < nVars; ++v)
		{
			out << "\t" << s.Variable[v] << "\t" << s.Derivative[v] << "\t" << s.Flux[v];
			if (printSources)
				out << "\t" << problem->Sources(v, s, x, t);
		}
		out << std::endl;
	}
	out << std::endl;
	out << std::endl; // Two blank lines needed to make gnuplot happy
}
