#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Variables, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c)
	: grid(Grid), k(polyNum), nCells(N_cells), nVar(N_Variables), dt(Dt), RHS(rhs), tau(Tau), c_fn(c), sig(grid,k), q(grid,k), u(grid,k), dudt(grid,k), dqdt(grid,k), dsigdt(grid,k), BCs (nullptr)
{
}

void SystemSolver::setInitialConditions( std::function< double ( double )> u_0, std::function< double ( double )> gradu_0, std::function< double ( double )> sigma_0, N_Vector& Y , N_Vector& dYdt ) 
{
	if ( !initialised )
		initialiseMatrices();

	if(!lambda.has_value())
	{
		double memBlock[nVar*(nCells+1)];
		lambda = VectorWrapper(memBlock, nVar*(nCells+1));
		dlamdt = VectorWrapper(memBlock, nVar*(nCells+1));
	}
	mapDGtoSundials( sig, q, u, lambda.value(), N_VGetArrayPointer( Y ));
	mapDGtoSundials( dsigdt, dqdt, dudt, dlamdt.value(), N_VGetArrayPointer( dYdt ));

	/*
	std::function<double( double )> f = [ = ]( double x ){ 
	//return 0.0;
	// return 2.0 * ( 2.0 * x*x*x*x - 7.0 * x*x + 2.0 ) * ::exp( -x*x ); 
	return (1.0/4)*(::exp(-0.75*(x-5)*(x-5)) * (73.0 + 3.0*x*(x-10)) + ::exp(-0.25 * (x-5)*(x-5)) * (23 + x*(x-10)));
	};
	*/

	resetCoeffs();

	u = u_0;
	q = gradu_0;
	sig = sigma_0;
	double dx = ::abs(BCs->UpperBound - BCs->LowerBound)/nCells;
	for(int var = 0; var < nVar; var++)
	{
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];
			lambda.value()[var*(nCells+1) + i] += u.Basis.Evaluate(I, u.coeffs[var][i].second, i*(dx))/2;
			lambda.value()[var*(nCells+1) + i+1] += u.Basis.Evaluate(I, u.coeffs[var][i].second, (i+1)*(dx))/2;

			dlamdt.value()[var*(nCells+1) + i] += dudt.Basis.Evaluate(I, dudt.coeffs[var][i].second, i*(dx))/2;
			dlamdt.value()[var*(nCells+1) + i+1] += dudt.Basis.Evaluate(I, dudt .coeffs[var][i].second, (i+1)*(dx))/2;
		}
		lambda.value()[var*(nCells+1)] = 0.0;
		lambda.value()[var*(nCells+1) + nCells] = 0.0;
		dlamdt.value()[var*(nCells+1)] = 0.0;
		dlamdt.value()[var*(nCells+1) + nCells] = 0.0;
	}

	//u and q have to be built for all variables before building sigma due to coupling
	/*
	for(int var = 0; var < nVar; var++)
	{
		//??to do implement funtion to take analytical q and u
		std::function< double (double) > minus_kappa = [ = ]( double x ) { return -1.0*diffObj->getKappaFunc(var)( x, q, u); };
		sig.setToFunc(minus_kappa, var);
	}
	*/

	for(int var = 0; var < nVar; var++)
	{
		//Solver For dudt with dudt = X^-1( -B*Sig - D*U - E*Lam + F )
		Eigen::Vector2d lamCell;
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];
			auto cTInv = Eigen::FullPivLU< Eigen::MatrixXd >(C_cellwise[i].transpose());
			lamCell[0] = lambda.value()[var*(nCells+1) + i]; lamCell[1] = lambda.value()[var*(nCells+1) + i+1];
			dudt.coeffs[ var ][ i ].second.setZero();		
			//dudt.coeffs[ var ][ i ].second = Eigen::FullPivLU< Eigen::MatrixXd >(XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)).solve( -B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*sig.coeffs[ var ][ i ].second - D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*u.coeffs[ var ][ i ].second - E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell + RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ));		
		}
	}

	// Differentiate dudt_0 to get dqdt_0 because dx & dt commute
	for(int var = 0; var < nVar; var++)
	{
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];
			Eigen::MatrixXd M_hr( k + 1, k + 1 );
			dudt.DerivativeMatrix( I, M_hr );
			dqdt.coeffs[ var ][ i ].second.setZero();
			//dqdt.coeffs[ var ][ i ].second = ( M_hr * dudt.coeffs[ var ][ i ].second ).block( 0, 0, k + 1, 1 );
		}
	}

	//this assues dkappadt = 0
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::MatrixXd NLq(nVar*(k+1), nVar*(k+1));
		Eigen::MatrixXd NLu(nVar*(k+1), nVar*(k+1));
		diffObj->NLqMat( NLq, q, u, I);
		diffObj->NLuMat( NLu, q, u, I);

		for(int vari = 0; vari < nVar; vari++)
		{
			for(int varj = 0; varj < nVar; varj++)
			{
				//dsigdt.coeffs[ vari ][ i ].second -= NLu.block(vari*(k+1), varj*(k+1), k+1, k+1)*dudt.coeffs[varj][i].second + NLq.block(vari*(k+1), varj*(k+1), k+1, k+1)*dqdt.coeffs[varj][i].second;
				dsigdt.coeffs[ vari ][ i ].second.setZero();
			}
		}
	}

	//print(std::cout, 1.0, 20, 0);
}

void SystemSolver::initialiseMatrices()
{
	// These are temporary working space
	// Matrices we need per cell
	Eigen::MatrixXd A( nVar*(k + 1), nVar*(k + 1) );
	Eigen::MatrixXd B( nVar*(k + 1), nVar*(k + 1) );
	Eigen::MatrixXd D( nVar*(k + 1), nVar*(k + 1) );
	// Two endpoints per cell
	Eigen::MatrixXd C( 2*nVar, nVar*(k + 1) );
	Eigen::MatrixXd E( nVar*(k + 1), 2*nVar );

	// Tomporary variable matrices that will be printed into the larger cell matrices as blocks
	Eigen::MatrixXd Avar( k + 1, k + 1 );
	Eigen::MatrixXd Bvar( k + 1, k + 1 );
	Eigen::MatrixXd Dvar( k + 1, k + 1 );
	Eigen::MatrixXd Cvar( 2, k + 1 );
	Eigen::MatrixXd Evar( k + 1, 2 );

	Eigen::MatrixXd HGlobalMat( nVar*(nCells+1), nVar*(nCells+1) );
	HGlobalMat.setZero();
	K_global.resize( nVar*(nCells + 1), nVar*(nCells + 1) );
	K_global.setZero();
	L_global.resize( nVar*(nCells + 1) );
	L_global.setZero();

	clearCellwiseVecs();
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		A.setZero();
		B.setZero();
		D.setZero();
		Interval const& I( grid.gridCells[ i ] );
		for(int var=0; var<nVar; var++)
		{
			Avar.setZero();
			Bvar.setZero();
			Dvar.setZero();
			// A_ij = ( phi_j, phi_i )
			u.MassMatrix( I, Avar );
			// B_ij = ( phi_i, phi_j' )
			u.DerivativeMatrix( I, Bvar );
			// D_ij = -(c phi_j, phi_i') + < w, tau u > 
			u.DerivativeMatrix( I, Dvar, c_fn );
			// As DerivativeMatrix gives the weighted product (c phi_i, phi_j')
			// we flip the sign and the indices on D.
			Dvar *= -1.0;
			Dvar.transposeInPlace();

			// Now do all the boundary terms
			for ( Eigen::Index i=0; i<k+1;i++ )
			{
				for ( Eigen::Index j=0; j<k+1;j++ )
				{
					Dvar( i, j ) += 
						tau( I.x_l )*u.Basis.phi( I, j )( I.x_l )*u.Basis.phi( I, i )( I.x_l ) +
						tau( I.x_u )*u.Basis.phi( I, j )( I.x_u )*u.Basis.phi( I, i )( I.x_u )	;
				}
			}

			A.block(var*(k+1),var*(k+1),k+1,k+1) = Avar;
			D.block(var*(k+1),var*(k+1),k+1,k+1) = Dvar;
			B.block(var*(k+1),var*(k+1),k+1,k+1) = Bvar;
		}

		A_cellwise.emplace_back(A);
		B_cellwise.emplace_back(B);
		D_cellwise.emplace_back(D);

		Eigen::MatrixXd M( 3*nVar*(k + 1), 3*nVar*(k + 1) );
		M.setZero();
		//row1
		M.block( 0, 0, nVar*(k+1), nVar*(k+1) ).setZero();
		M.block( 0, nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = -A;
		M.block( 0, 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = -B.transpose();

		//row2
		M.block( nVar*(k+1), 0, nVar*(k+1), nVar*(k+1) ) = B;
		M.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ).setZero();
		M.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = D;			//X added at Jac step

		//row3
		M.block( 2*nVar*(k+1), 0, nVar*(k+1), nVar*(k+1) ) = A;
		M.block( 2*nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ).setZero();		//NLq added at Jac step
		M.block( 2*nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ).setZero();	//NLu added at Jac step

		// 2*( k + 1) is normally only medium-sized (10 - 20)
		// so just do a full LU factorization to solve 
		// ?Now this is nVar*3*(k+1) so maybe this should be changed?
		MBlocks.emplace_back( M );

		Eigen::MatrixXd CE_vec( 3*nVar*(k + 1), 2*nVar );
		CE_vec.setZero();
		for(int var=0; var<nVar; var++)
		{
			Cvar.setZero();
			Evar.setZero();
			for ( Eigen::Index i=0; i<k+1;i++ )
			{
				// C_ij = < psi_i, phi_j * n_x > , where psi_i are edge degrees of
				// freedom and n_x is the unit normal in the x direction
				// for a line, edge degrees of freedom are just 1 at each end

				Cvar( 0, i ) = -u.Basis.phi( I, i )( I.x_l );
				Cvar( 1, i ) = u.Basis.phi( I, i )( I.x_u ); 

				// E_ij = < phi_i, (c n_x - tau ) lambda >
				Evar( i, 0 ) = u.Basis.phi( I, i )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) );
				Evar( i, 1 ) = u.Basis.phi( I, i )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) );

				if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
				{
					Cvar( 0, i ) = 0;
					Evar( i, 0 ) = 0;
				}

				if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
				{
					Cvar( 1, i ) = 0;
					Evar( i, 1 ) = 0;
				}
			}

			// Construct per-cell Matrix solutions
			// ( A   NLu   NLq )^-1 [  0  ]
			// ( 0    A    B^T )    [ C^T ]
			// ( B    0     D )     [  E  ]
			// These are the homogeneous solution, that depend on lambda
			C.block(var*2,var*(k+1),2,k+1) = Cvar;
			E.block(var*(k+1),var*2,k+1,2) = Evar;
		}
		CE_vec.block( 0, 0, nVar*(k + 1), nVar*2 ) = C.transpose();
		CE_vec.block( nVar*(k+1), 0, nVar*(k + 1), nVar*2 ) = E;
		CE_vec.block( 2*nVar*(k+1), 0, nVar*(k + 1), nVar*2 ).setZero();
		CEBlocks.emplace_back(CE_vec);

		C_cellwise.emplace_back(C);
		E_cellwise.emplace_back(E);

		// To store the RHS
		RF_cellwise.emplace_back(Eigen::VectorXd(nVar*2*(k+1)));

		// R is composed of parts of the values of 
		// u on the total domain boundary
		RF_cellwise[ i ].setZero();
		// Take components of f
		for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			RF_cellwise[ i ]( nVar*(k + 1) + j ) = u.CellProduct( I, RHS, u.Basis.phi( I, j%(k+1) ) );

		if ( I.x_l == BCs->LowerBound  && BCs->isLBoundDirichlet )
		{

			for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			{
				// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
				RF_cellwise[ i ]( j ) = -u.Basis.phi( I, j%(k+1) )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l, 0.0 );
				// - < ( c.n - tau ) g_D, w >
				RF_cellwise[ i ]( nVar*(k + 1) + j ) -= u.Basis.phi( I, j%(k+1) )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l, 0.0 );
			}
		}

		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			{
				// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
				RF_cellwise[ i ]( j ) += -u.Basis.phi( I, j%k+1 )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u, 0.0 );
				RF_cellwise[ i ]( nVar*(k + 1) + j ) -= u.Basis.phi( I, j%(k+1) )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u, 0.0 );
			}
		}

		// Per-cell contributions to the global matrices K and F.
		// First fill G
		Eigen::MatrixXd G( 2*nVar, nVar*(k + 1) );
		G.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Gvar( 2, k + 1 );
			for ( Eigen::Index i = 0; i < k+1; i++ )
			{
				Gvar( 0, i ) = tau( I.x_l )*u.Basis.phi( I, i )( I.x_l );
				if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
					Gvar( 0, i ) = 0.0;
				Gvar( 1, i ) = tau( I.x_u )*u.Basis.phi( I, i )( I.x_u );
				if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
					Gvar( 1, i ) = 0.0;
			}
			G.block(2*var,(k+1)*var,2, (k + 1)) = Gvar;
		}

		//[ C 0 G ]
		CG_cellwise.emplace_back(Eigen::MatrixXd(2*nVar, 3*nVar*(k+1) ));
		CG_cellwise[i].setZero();
		CG_cellwise[ i ].block( 0, 0, 2*nVar, nVar*(k + 1) ) = C;
		CG_cellwise[ i ].block( 0, nVar*(k + 1), 2*nVar, nVar*(k + 1) ).setZero();
		CG_cellwise[ i ].block( 0, 2*nVar*(k + 1), 2*nVar, nVar*(k + 1) ) = G;
		G_cellwise.emplace_back(G);

		// Now fill H
		Eigen::MatrixXd H( 2*nVar, 2*nVar );
		H.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Hvar( 2, 2 );
			Hvar.setZero();
			Hvar( 0, 0 ) = -c_fn( I.x_l ) - tau( I.x_l );
			Hvar( 1, 0 ) = 0.0;
			Hvar( 0, 1 ) = 0.0;
			Hvar( 1, 1 ) = c_fn( I.x_u ) - tau( I.x_u );

			if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
					Hvar( 0, 0 ) = Hvar( 1, 0 ) = Hvar( 0, 1 ) = 0.0;

			if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
					Hvar( 1, 1 ) = Hvar( 1, 0 ) = Hvar( 0, 1 ) = 0.0;

			H.block(2*var,2*var,2,2) = Hvar;
			HGlobalMat.block(var*(nCells+1) + i,var*(nCells+1) + i, 2, 2) += Hvar;
		}

		H_cellwise.emplace_back(H);

		// Finally fill L
		for(int var = 0; var < nVar; var++)
		{
			if ( I.x_l == BCs->LowerBound && /* is b.d. Neumann at lower boundary */ !BCs->isLBoundDirichlet )
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound, 0.0 );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound, 0.0 );
		}

		Eigen::MatrixXd X(nVar*(k+1), nVar*(k+1));
		X.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Xvar( k + 1, k + 1 );
			u.MassMatrix( I, Xvar);
			X.block(var*(k+1), var*(k+1), k+1, k+1) = Xvar;
		}
		XMats.emplace_back(X);
	}
	H_global = static_cast<Eigen::FullPivLU< Eigen::MatrixXd >>(HGlobalMat);
	H_global_mat = HGlobalMat;
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

void SystemSolver::mapDGtoSundials(DGApprox& sigma, DGApprox& q, DGApprox& u, Eigen::Map<Eigen::VectorXd>& lam, realtype* Y)
{
	std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> cellSigCoeffs, cellQCoeffs, cellUCoeffs;

	for(int var = 0; var < nVar; var++)
	{
		for(int i=0; i<nCells; i++)
		{
			cellSigCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y +                var*(k+1) + i*3*nVar*(k+1), k+1 ));
			cellQCoeffs.emplace_back( grid.gridCells[ i ],   VectorWrapper( Y + nVar*(k+1)   + var*(k+1) + i*3*nVar*(k+1), k+1 ));
			cellUCoeffs.emplace_back( grid.gridCells[ i ],   VectorWrapper( Y + 2*nVar*(k+1) + var*(k+1) + i*3*nVar*(k+1), k+1 ));
		}
		sigma.coeffs.push_back(cellSigCoeffs);
		q.coeffs.push_back(cellQCoeffs);
		u.coeffs.push_back(cellUCoeffs);
		cellSigCoeffs.clear();
		cellQCoeffs.clear();
		cellUCoeffs.clear();

		new (&lam) VectorWrapper( Y + nVar*(nCells)*(3*k+3), nVar*(nCells+1) );
	}
}

void SystemSolver::mapDGtoSundials(DGApprox& u, realtype* Y)
{
	std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> cellUCoeffs;

	for(int var = 0; var < nVar; var++)
	{
		for(int i=0; i<nCells; i++)
		{
			cellUCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y + 2*nVar*(k+1) + var*(k+1) + i*3*nVar*(k+1), k+1 ));  
		}
		u.coeffs.push_back(cellUCoeffs);
		cellUCoeffs.clear();
	}
}

void SystemSolver::mapDGtoSundials(std::vector< Eigen::Map<Eigen::VectorXd > >& SQU_cell, Eigen::Map<Eigen::VectorXd>& lam, realtype* const& Y)
{
	SQU_cell.clear();
	for(int i=0; i<nCells; i++)
	{
		SQU_cell.emplace_back( VectorWrapper( Y + i*3*nVar*(k+1), nVar*3*(k+1) ) );
	}

		new (&lam) VectorWrapper( Y + nVar*(nCells)*(3*k+3), nVar*(nCells+1) );
}

void SystemSolver::updateBoundaryConditions(double t)
{
	L_global.setZero();
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		RF_cellwise[ i ].setZero();

		//??To Do: this should be a time dependent RHS function. Fairly easy implementation
		for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			RF_cellwise[ i ]( nVar*(k + 1) + j ) = u.CellProduct( I, RHS, u.Basis.phi( I, j%(k+1) ) );

		if ( I.x_l == BCs->LowerBound  && BCs->isLBoundDirichlet )
		{

			for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			{
				// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
				RF_cellwise[ i ]( j ) = -u.Basis.phi( I, j%(k+1) )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l, t );
				// - < ( c.n - tau ) g_D, w >
				RF_cellwise[ i ]( nVar*(k + 1) + j ) -= u.Basis.phi( I, j%(k+1) )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l, t );
			}
		}


		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			{
				// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
				RF_cellwise[ i ]( j ) += -u.Basis.phi( I, j%k+1 )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u, t );
				RF_cellwise[ i ]( nVar*(k + 1) + j ) -= u.Basis.phi( I, j%(k+1) )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u, t );
			}
		}


		for(int var = 0; var < nVar; var++)
		{
			if ( I.x_l == BCs->LowerBound && /* is b.d. Neumann at lower boundary */ !BCs->isLBoundDirichlet )
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound, t );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound, t );
		}
	}
}

void SystemSolver::resetCoeffs()
{
	sig.zeroCoeffs();
	q.zeroCoeffs();
	u.zeroCoeffs();
	dsigdt.zeroCoeffs();
	dqdt.zeroCoeffs();
	dudt.zeroCoeffs();
	lambda.value().setZero();
	dlamdt.value().setZero();
}

void SystemSolver::updateMForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& MXsolvers, double const alpha, DGApprox& delQ, DGApprox& delU)
{
	std::function<double( double )> alphaF = [ = ]( double x ){ return alpha;};

	//DGApprox newU(grid, k), newQ(grid, k);
	//double qMemBlock[nVar*nCells*(k+1)], uMemBlock[nVar*nCells*(k+1)]; //??need to assign memory block as DGAs don't own memory
	//newQ.setCoeffsToArrayMem(qMemBlock, nVar, nCells, grid);
	//newU.setCoeffsToArrayMem(uMemBlock, nVar, nCells, grid);
	//newQ.sum(q, delQ);
	//newU.sum(u, delU);


	Eigen::MatrixXd X( nVar*(k + 1), nVar*(k + 1) );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		u.MassMatrix( I, X, alphaF);
		auto MX = MBlocks[i];
		MX.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) += X;
		Eigen::MatrixXd NLq(nVar*(k+1), nVar*(k+1));
		Eigen::MatrixXd NLu(nVar*(k+1), nVar*(k+1));
		//diffObj->NLqMat( NLq, newQ, newU, I);
		diffObj->NLqMat( NLq, q, u, I);
		MX.block( 2*nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1)) = NLq;
		//diffObj->NLuMat( NLu, newQ, newU, I);
		diffObj->NLuMat( NLu, q, u, I);
		MX.block( 2*nVar*(k+1),2* nVar*(k+1), nVar*(k+1), nVar*(k+1)) = NLu;
		MXsolvers.emplace_back(MX);
	}
}

void SystemSolver::solveJacEq(N_Vector& g, N_Vector& delY)
{
	DGApprox delSig(grid,k), delQ(grid,k), delU(grid,k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> delLambda(memBlock, nVar*(nCells+1)), g4(memBlock, nVar*(nCells+1));
	std::vector< Eigen::Map<Eigen::VectorXd > > g1g2g3_cellwise;
	K_global.setZero(); 

	//print(std::cout, 1.0, 20, 0, g );

	VectorWrapper delYVec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) );
	//std::cerr << delYVec << std::endl << std::endl;

	//u.printCoeffs(0);
	//q.printCoeffs(0);
	//sig.printCoeffs(0);

	//std::cerr << g4 << std::endl << std::endl;

	mapDGtoSundials(delSig, delQ, delU, delLambda, N_VGetArrayPointer( delY ));

	//assemble temp cellwise M blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > factorisedM{};
	updateMForJacSolve(factorisedM, alpha, delQ, delU);//??To Do: find a way to acces the current guess of Y

	// Assemble RHS g into cellwise form and solve for SQU blocks
	mapDGtoSundials(g1g2g3_cellwise, g4, N_VGetArrayPointer( g ));

	std::vector< Eigen::VectorXd > SQU_f( nCells );
	std::vector< Eigen::MatrixXd > SQU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{  
		Interval const& I( grid.gridCells[ i ] );

		//SQU_f
		Eigen::VectorXd g1g2g3 = g1g2g3_cellwise[ i ];
		//std::cerr << g1g2g3 << std::endl << std::endl;

		SQU_f[ i ] =  factorisedM[ i ].solve( g1g2g3 );

		//SQU_0
		Eigen::MatrixXd CE = CEBlocks[ i ];
		SQU_0[ i ] = factorisedM[ i ].solve( CE );

		Eigen::MatrixXd K_cell(nVar*2,nVar*2);
		K_cell = H_cellwise[i] - CG_cellwise[ i ] * SQU_0[i];

		//K
		for(int var = 0; var < nVar; var++)
		{
			K_global.block( var*(nCells + 1) + i, var*(nCells + 1) + i, 2, 2 ) += K_cell.block(var*2,var*2,2,2);
		}
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nVar*(nCells + 1) );
	F.setZero();
	F = g4;
	for ( unsigned int i=0; i < nCells; i++ )
	{
		for(int var = 0; var < nVar; var++)
		{
			F.block<2,1>( var*(nCells + 1) + i, 0 ) -= (CG_cellwise[ i ] * SQU_f[ i ]).block(var*2,0,2,1);
		}
	}

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	delLambda = lu.solve( F );

	// Now find del sigma, del q and del u to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd delSQU( 3*nVar*( k + 1 ) );
		Eigen::VectorXd delLambdaCell(2*nVar);
		for(int var = 0; var < nVar; var++)
		{
			delLambdaCell.block<2,1>(2*var,0) = delLambda.block<2,1>(var*(nCells + 1) + i,0);
		}
		delSQU = SQU_f[ i ] - SQU_0[ i ] * delLambdaCell;
		for(int var = 0; var < nVar; var++)
		{
			delSig.coeffs[ var ][ i ].second = delSQU.block( var*(k+1), 0, k + 1, 1 );
			delQ.coeffs[ var ][ i ].second =   delSQU.block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 );
			delU.coeffs[ var ][ i ].second =   delSQU.block( 2*nVar*(k + 1) + var*(k+1), 0, k + 1, 1 );
		}
	}
	//delSig.printCoeffs(0);
	//delQ.printCoeffs(0);
	//delU.printCoeffs(0);
	//std::cerr << delLambda << std::endl << std::endl;

	//print(std::cout, 1.0, 20, 0, delY);
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	auto k = system->k;
	auto grid(system->grid);
	auto nCells = system->nCells;
	auto c_fn = system->getcfn();
	auto nVar = system->nVar;

	//system->updateBoundaryConditions(tres);

	DGApprox tempSig(grid, k), tempU(grid, k), tempQ(grid, k), tempdudt(grid, k);
	DGApprox res1(grid, k), res2(grid, k), res3(grid, k) ;
	DGApprox tempKappa(grid, k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> tempLambda(memBlock, nVar*(nCells+1)), res4(memBlock, nVar*(nCells+1));

	system->mapDGtoSundials(tempSig, tempQ, tempU, tempLambda, N_VGetArrayPointer( Y ));
	system->mapDGtoSundials(tempdudt, N_VGetArrayPointer( dydt )); 
	system->mapDGtoSundials(res1, res2, res3, res4, N_VGetArrayPointer( resval )); 

	//Solve for Lambda with Lam = (H^T)^-1*[ -C*Sig - G*U + L ] 
	Eigen::VectorXd Lambda( nVar*(nCells + 1) ), CsGuL_global(nVar*(nCells+1));
	CsGuL_global.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd LVarCell(2);
		Eigen::VectorXd CsGuLVarCell(2);
		CsGuLVarCell.setZero();
		LVarCell.setZero();
		for(int var = 0; var < nVar; var++)
		{
			LVarCell = system->L_global.block<2,1>(var*(nCells+1) + i,0);
			CsGuLVarCell = LVarCell - system->C_cellwise[i].block(var*2,var*(k+1),2,k+1)*tempSig.coeffs[ var ][ i ].second - system->G_cellwise[i].block(var*2,var*(k+1),2,k+1)*tempU.coeffs[ var ][ i ].second;
			CsGuL_global.block(var*(nCells + 1) + i, 0, 2, 1) += CsGuLVarCell;
		}
	}
	Lambda = system->H_global.solve(CsGuL_global);
	res4 = -tempLambda + Lambda;//??To Do: No need to invert H

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd lamCell(2*nVar);
		Eigen::VectorXd multiVarR(nVar*(k+1));

		for(int var = 0; var < nVar; var++)
		{
			lamCell[2*var] = tempLambda[var*(nCells+1) + i]; lamCell[2*var + 1] = tempLambda[var*(nCells+1) + i+1];
			multiVarR.block(var*(k+1), 0, k+1, 1) = system->RF_cellwise[i].block(var*(k+1),0,k+1,1);
		}

		//length = nVar*(k+1)
		for(int var = 0; var < nVar; var++)
		{
			std::function< double (double) > kappaFunc = [ = ]( double x ) { return system->getDiffObj()->getKappaFunc(var)( x, tempQ, tempU); };
			Eigen::VectorXd kappa_cellwise(k+1);
			for ( Eigen::Index j = 0; j < k+1; j++ )
				kappa_cellwise( j ) = tempU.CellProduct( I, kappaFunc, tempU.Basis.phi( I, j%(k+1) ) );
	
			res1.coeffs[ var ][ i ].second = -system->A_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempQ.coeffs[var][i].second - system->B_cellwise[i].transpose().block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->C_cellwise[i].transpose().block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( var*(k+1), 0, k + 1, 1 );
			res2.coeffs[ var ][ i ].second = system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempSig.coeffs[ var ][ i ].second + system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) + tempdudt.coeffs[ var ][ i ].second;
 			res3.coeffs[ var ][ i ].second = tempSig.coeffs[ var ][ i ].second + kappa_cellwise;
		}
	}

	//res1.printCoeffs(0);
	//res2.printCoeffs(0);
	//res3.printCoeffs(0);
	//std::cerr << res4 << std::endl << std::endl;

	VectorWrapper Vec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
	std::cerr << Vec.maxCoeff() << "	" << Vec.norm() << "	" << tres << std::endl << std::endl;

	//system->print(std::cout, 1.0, 20, 0, resval);

	return 0;
}


void SystemSolver::print( std::ostream& out, double t, int nOut, int var )
{
	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut ) );
		out << x << "\t" << EvalCoeffs( u.Basis, u.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, q.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, sig.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dudt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dqdt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dudt.coeffs, x, var ) << std::endl;
	}
	out << std::endl;
}

void SystemSolver::print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY )
{
	DGApprox Sig(grid,k), Q(grid,k), U(grid,k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> Lambda(memBlock, nVar*(nCells+1));

	mapDGtoSundials(Sig, Q, U, Lambda, N_VGetArrayPointer( tempY ));

	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut ) );
		out << x << "\t" << EvalCoeffs( U.Basis, U.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Q.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Sig.coeffs, x, var ) << std::endl;
	}
	out << std::endl;
}

double SystemSolver::EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var )
{
	for ( auto const & pair : cs[var] )
	{
		if ( pair.first.contains( x ) )
			return B.Evaluate( pair.first, pair.second, x );
	}
	return std::nan( "" );
}

std::shared_ptr<DiffusionObj> SystemSolver::getDiffObj()
{
	if(diffObj) return diffObj;
	else return nullptr;
}

/*
void SystemSolver::solveNonIDA(N_Vector& Y, N_Vector& dYdt, double dt
{
	std::vector< Eigen::VectorXd > QU_cellwise;
	Eigen::VectorXd tempLam(nCells+1);
	sundialsToDGVecConversion(Y, u, q);
	K_global.setZero();

	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > ABBDXSolvers{};
	updateABBDForJacSolve(ABBDXSolvers, 1/dt);

	//update F
	std::vector< Eigen::VectorXd > QU_f( nCells );
	std::vector< Eigen::MatrixXd > QU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );

		//QU_f
		Eigen::VectorXd RF = RF_cellwise[ i ];
		RF.block( k + 1, 0, k + 1, 1 ) += (1/dt)*u.coeffs[ i ].second;
		QU_f[ i ] = ABBDXSolvers[i].solve( RF );

		//QU_0
		Eigen::MatrixXd CE = CEBlocks[ i ];
		QU_0[ i ] = ABBDXSolvers[i].solve( CE );

		Eigen::Matrix2d K_cell = H_cellwise[i] - CG_cellwise[ i ] * QU_0[i];

		//K
		K_global.block( i, i, 2, 2 ) += K_cell;
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nCells + 1 );
	F = L_global;
	for ( unsigned int i=0; i < nCells; i++ )
		F.block<2,1>( i, 0 ) -= CG_cellwise[ i ] * QU_f[ i ];

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	tempLam = lu.solve( F );

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd QU( 2*( k + 1 ) );
		QU = QU_f[ i ] - QU_0[ i ] * tempLam.block<2,1>( i, 0 );
		q.coeffs[ i ].second = QU.block( 0, 0, k + 1, 1 );
		dudt.coeffs[ i ].second = (QU.block( k + 1, 0, k + 1, 1 ) - u.coeffs[ i ].second)/dt;
		u.coeffs[ i ].second = QU.block( k + 1, 0, k + 1, 1 );
	}

	DGtoSundialsVecConversion(q, u, Y);
}
*/