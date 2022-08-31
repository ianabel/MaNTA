#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Variables, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c, Eigen::MatrixXd kappaMat)
	: grid(Grid), k(polyNum), nCells(N_cells), nVar(N_Variables), dt(Dt), RHS(rhs), tau(Tau), c_fn(c), u(grid,k), q(grid,k), dudt(grid,k), BCs (nullptr)
{
	setKappaInv(kappaMat);
}

void SystemSolver::setInitialConditions( std::function< double ( double )> u_0 , N_Vector& Y , N_Vector& dYdt) 
{
	if ( !initialised )
		initialiseMatrices();

	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 

	mapDGtoSundials( q, u, N_VGetArrayPointer( Y ));
	mapDGtoSundials( dudt, N_VGetArrayPointer( dYdt ));

	for(int var = 0; var < nVar; var++)
	{
		u = u_0;

		// Differentiate u_0 to get q_0
		//DGApprox u_0_hr( grid, 2*k, u_0 ); ?To Do: Interpolate q from a denser u grid?
		for ( unsigned int i=0; i < nCells; i++ ) {

			Interval I = grid.gridCells[ i ];
			Eigen::MatrixXd M_hr( k + 1, k + 1 );
			u.DerivativeMatrix( I, M_hr );
			q.coeffs[ var ][ i ].second = -( M_hr * u.coeffs[ var ][ i ].second ).block( 0, 0, k + 1, 1 ); //?To Do:kappa will have to be added here
		}

		//Solve for Lambda with Lam = (H^T)^-1*[ -C*Q - G^T*U + L ] 
		Eigen::VectorXd Lambda( nVar*(nCells + 1) ), CqGuL_global( nVar*(nCells + 1) );
		CqGuL_global.setZero();
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];
			Eigen::Vector2d Lcell;
			Lcell[0] = L_global[i]; Lcell[1] = L_global[i+1];

			Eigen::Vector2d CqGuL;
			CqGuL = Lcell - C_cellwise[i].block(var*2, var*(k+1), 2, k+1)*q.coeffs[ var ][ i ].second - G_cellwise[i].block(var*2, var*(k+1), 2, k+1)*u.coeffs[ var ][ i ].second;
			CqGuL_global.block(i,0,2,1) += CqGuL;
		} 
		Lambda = H_global.solve(CqGuL_global);

		//Solver For dudt with dudt = X^-1( -B*Q - D*U - E*Lam + F )
		Eigen::Vector2d lamCell;
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid.gridCells[ i ];
			auto cTInv = Eigen::FullPivLU< Eigen::MatrixXd >(C_cellwise[i].transpose());
			lamCell[0] = Lambda[i]; lamCell[1] = Lambda[i+1];
			dudt.coeffs[ var ][ i ].second = Eigen::FullPivLU< Eigen::MatrixXd >(XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)).solve( -B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*q.coeffs[ var ][ i ].second - D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*u.coeffs[ var ][ i ].second - E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell + RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ));
		}
	}
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
	Eigen::MatrixXd A_mn( k + 1, k + 1 );
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
		Interval const& I( grid.gridCells[ i ] );

		for(int m=0; m<nVar; m++)
		{
			for(int n=0; n<nVar; n++)
			{
				A_mn.setZero();
				// A_ij_mn = ( phi_j*(kappa^-1)_mn, phi_i )
				auto kappa_inv = [ & ]( double x ) { return kappaInv(m,n);};
				u.MassMatrix( I, A_mn, kappa_inv );
				A.block(m*(k+1), n*(k+1), k+1, k+1) = A_mn;
			}
		}
		for(int var=0; var<nVar; var++)
		{

			Bvar.setZero();
			Dvar.setZero();
			// B_ij = ( phi_i, phi_j' )
			u.DerivativeMatrix( I, Bvar );
			// D_ij = -(c phi_j, phi_i') + < w, tau u > 
			u.DerivativeMatrix( I, Dvar, c_fn );
			// As DerivativeMatrix gives the weighted product (c phi_i, phi_j')
			// we flip the sign and the indices on D.
			Dvar *= -1.0;
			Dvar.transposeInPlace();
			//Note: D matix is unaffected in initialisation. During ech timestep it will see an extra mass matix*alpha to reflect the addition of the del udot term
			//When checking the residual eq the udot term will be added seperately.

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

			D.block(var*(k+1),var*(k+1),k+1,k+1) = Dvar;
			B.block(var*(k+1),var*(k+1),k+1,k+1) = Bvar;
		}

		A_cellwise.emplace_back(A);
		B_cellwise.emplace_back(B);
		D_cellwise.emplace_back(D);

		Eigen::MatrixXd ABBD( 2*nVar*(k + 1), 2*nVar*(k + 1) );
		ABBD.block( 0, 0, nVar*(k+1), nVar*(k+1) ) = A;
		ABBD.block( nVar*(k+1), 0, nVar*(k+1), nVar*(k+1) ) = B;
		ABBD.block( 0, nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = -B.transpose();
		ABBD.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = D;

		// 2*( k + 1) is normally only medium-sized (10 - 20)
		// so just do a full LU factorization to solve 
		// ?Now this is nVar*2*(k+1) so maybe this should be changed?
		ABBDBlocks.emplace_back( ABBD );

		Eigen::MatrixXd CE_vec( 2*nVar*(k + 1), 2*nVar );
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
			// ( A  -B^T )^-1 [ C^T ]
			// ( B    D  )    [ E   ]
			// These are the homogeneous solution, that depend on lambda
			C.block(var*2,var*(k+1),2,k+1) = Cvar;
			E.block(var*(k+1),var*2,k+1,2) = Evar;
		}
		CE_vec.block( 0    , 0, nVar*(k + 1), nVar*2 ) = C.transpose();
		CE_vec.block( nVar*(k + 1), 0, nVar*(k + 1), nVar*2 ) = E;
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
				RF_cellwise[ i ]( j ) = -u.Basis.phi( I, j%(k+1) )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l );
				// - < ( c.n - tau ) g_D, w >
				RF_cellwise[ i ]( nVar*(k + 1) + j ) -= u.Basis.phi( I, j%(k+1) )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l );
			}
		}

		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for ( Eigen::Index j = 0; j < nVar*(k+1); j++ )
			{
				// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
				RF_cellwise[ i ]( j ) += -u.Basis.phi( I, j%k+1 )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u );
				RF_cellwise[ i ]( nVar*(k + 1) + j ) -= u.Basis.phi( I, j%(k+1) )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u );
			}
		}


		// Now RF_cellwise[ i ] holds the steady-state RHS. 
		// We have to add the previous timestep to it when doing the time-dependent solve

		// Per-cell contributions to the global matrices K and F.
		// First fill G
		Eigen::MatrixXd G( 2*nVar, nVar*(k + 1) );
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

		CG_cellwise.emplace_back(Eigen::MatrixXd(2*nVar, nVar*2*(k+1) ));
		CG_cellwise[ i ].block( 0, 0, 2*nVar, nVar*(k + 1) ) = C;
		CG_cellwise[ i ].block( 0, nVar*(k + 1), 2*nVar, nVar*(k + 1) ) = G;
		G_cellwise.emplace_back(G);

		// Now fill H
		Eigen::MatrixXd H( 2*nVar, 2*nVar );
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Hvar( 2, 2 );
			Hvar( 0, 0 ) = -c_fn( I.x_l ) - tau( I.x_l );
			Hvar( 1, 0 ) = 0.0;
			Hvar( 0, 1 ) = 0.0;
			Hvar( 1, 1 ) = c_fn( I.x_u ) - tau( I.x_u );

			if ( I.x_l == BCs->LowerBound && BCs->isLBoundDirichlet )
					H( 0, 0 ) = H( 1, 0 ) = H( 0, 1 ) = 0.0;

			if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
					H( 1, 1 ) = H( 1, 0 ) = H( 0, 1 ) = 0.0;

			H.block(2*var,2*var,2,2) = Hvar;
			HGlobalMat.block(var*(nCells+1) + i,var*(nCells+1) + i, 2, 2) += Hvar;
		}

		H_cellwise.emplace_back(H);

		// Finally fill L
		for(int var = 0; var < nVar; var++)
		{
			if ( I.x_l == BCs->LowerBound && /* is b.d. Neumann at lower boundary */ !BCs->isLBoundDirichlet )
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound );
		}

		Eigen::MatrixXd X(nVar*(k+1), nVar*(k+1));
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Xvar( k + 1, k + 1 );
			u.MassMatrix( I, Xvar);
			X.block(var*(k+1), var*(k+1), k+1, k+1) = Xvar;
		}
		XMats.emplace_back(X);
	}
	H_global = static_cast<Eigen::FullPivLU< Eigen::MatrixXd >>(HGlobalMat);
	initialised = true;
}

void SystemSolver::clearCellwiseVecs()
{
	XMats.clear();
	ABBDBlocks.clear();
	CG_cellwise.clear();
	RF_cellwise.clear();
	A_cellwise.clear();
	B_cellwise.clear();
	E_cellwise.clear();
	C_cellwise.clear();
	G_cellwise.clear();
	H_cellwise.clear();
}

void SystemSolver::mapDGtoSundials(DGApprox& q, DGApprox& u, realtype* Y)
{
	std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> varQCoeffs, varUCoeffs;

	for(int var = 0; var < nVar; var++)
	{
		for(int i=0; i<nCells; i++)
		{
			varQCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y + var*(nCells)*(2*k+2) + i*(2*k+2), k+1 ));
			varUCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y + var*(nCells)*(2*k+2) + i*(2*k+2) + k+1, k+1 ));
		}
		q.coeffs.push_back(varQCoeffs);
		u.coeffs.push_back(varUCoeffs);
		varQCoeffs.clear();
		varUCoeffs.clear();
	}
}

void SystemSolver::mapDGtoSundials(DGApprox& u, realtype* Y)
{
	std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>> varUCoeffs;

	for(int var = 0; var < nVar; var++)
	{
		for(int i=0; i<nCells; i++)
		{
			VectorWrapper emptyVec( Y + var*(nCells)*(2*k+2) + i*(2*k+2), k+1 ); 
			emptyVec.setZero();
			varUCoeffs.emplace_back( grid.gridCells[ i ], VectorWrapper( Y + var*(nCells)*(2*k+2) + i*(2*k+2) + k+1, k+1 ));
		}
		u.coeffs.push_back(varUCoeffs);
		varUCoeffs.clear();
	}
}

void SystemSolver::mapDGtoSundials(std::vector< Eigen::VectorXd >& QU_cell, realtype* const& Y)
{
	//?To Do: This uses lots of copying, which is kind of slow. A better version would require a reconstruction of the shape of the Y vector to to avoid this
	//This would require the structure of the Y vector to have the QU blocks stuctured to be the appropriate design for the ABBDX multiplication step?
	QU_cell.resize(nCells);
	for(int i=0; i<nCells; i++)
	{
		QU_cell[i].resize(nVar*2*(k+1));
		for(int var=0; var<nVar; var++)
		{
			QU_cell[i].block(var*(k+1), 0, k+1, 1) = VectorWrapper( Y+var*nCells*(2*k+2)+i*(2*k+2), k+1 );
			QU_cell[i].block(nVar*(k+1) + var*(k+1), 0, k+1, 1) = VectorWrapper( Y+var*nCells*(2*k+2)+i*(2*k+2) + k+1, k+1 );
		}
	}

}

void SystemSolver::updateABBDForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& ABBDXsolvers, double const alpha)
{
	std::function<double( double )> alphaF = [ = ]( double x ){ return alpha;};

	Eigen::MatrixXd X( nVar*(k + 1), nVar*(k + 1) );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		u.MassMatrix( I, X, alphaF);
		auto ABBDX = ABBDBlocks[i];
		ABBDX.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ) += X;
		ABBDXsolvers.emplace_back(ABBDX);
	}
}

void SystemSolver::solveJacEq(N_Vector const& g, N_Vector& delY)
{
	Eigen::VectorXd delLambda( nVar*(nCells + 1) );
	DGApprox delU(grid,k), delQ(grid,k);
	std::vector< Eigen::VectorXd > g1g2_cellwise;
	K_global.setZero();

	mapDGtoSundials(delQ, delU, N_VGetArrayPointer( delY ));

	//assemble temp cellwise ABBD blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > ABBDXSolvers{};
	updateABBDForJacSolve(ABBDXSolvers, alpha);

	// Assemble RHS g into cellwise form and solve for QU blocks
	mapDGtoSundials(g1g2_cellwise, N_VGetArrayPointer( g ));
	std::vector< Eigen::VectorXd > QU_f( nCells );
	std::vector< Eigen::MatrixXd > QU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{  
		Interval const& I( grid.gridCells[ i ] );

		//QU_f
		Eigen::VectorXd g1g2 = g1g2_cellwise[ i ];

		QU_f[ i ] =  ABBDXSolvers[ i ].solve( g1g2 );

		//QU_0
		Eigen::MatrixXd CE = CEBlocks[ i ];
		QU_0[ i ] = ABBDXSolvers[ i ].solve( CE );

		Eigen::MatrixXd K_cell(nVar*2,nVar*2);
		K_cell = H_cellwise[i] - CG_cellwise[ i ] * QU_0[i];

		//K
		for(int var = 0; var < nVar; var++)
		{
			K_global.block( var*(nCells + 1) + i, var*(nCells + 1) + i, 2, 2 ) += K_cell.block(var*2,var*2,2,2);
		}
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nVar*(nCells + 1) );
	F.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		for(int var = 0; var < nVar; var++)
		{
			F.block<2,1>( var*(nCells + 1) + i, 0 ) -= (CG_cellwise[ i ] * QU_f[ i ]).block(var*2,0,2,1);
		}
	}

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	delLambda = lu.solve( F );

	// Now find del u and del q to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd delQU( 2*nVar*( k + 1 ) );
		Eigen::VectorXd delLambdaCell(2*nVar);
		for(int var = 0; var < nVar; var++)
		{
			delLambdaCell.block<2,1>(2*var,0) = delLambda.block<2,1>(var*(nCells + 1) + i,0);
		}
		delQU = QU_f[ i ] - QU_0[ i ] * delLambdaCell;
		for(int var = 0; var < nVar; var++)
		{
			delQ.coeffs[ var ][ i ].second = delQU.block( var*(k+1), 0, k + 1, 1 );
			delU.coeffs[ var ][ i ].second = delQU.block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 );
		}
	}
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	auto k = system->k;
	auto grid(system->grid);
	auto nCells = system->nCells;
	auto c_fn = system->getcfn();
	auto nVar = system->nVar;

	DGApprox tempU(grid, k), tempQ(grid, k), tempdudt(grid, k), temp0(grid, k);
	DGApprox res1(grid, k), res2(grid, k);

	system->mapDGtoSundials(tempQ, tempU, N_VGetArrayPointer( Y ));
	system->mapDGtoSundials(tempdudt, N_VGetArrayPointer( dydt )); 
	system->mapDGtoSundials(res1, res2, N_VGetArrayPointer( resval )); 


	//Solve for Lambda with Lam = (H^T)^-1*[ -C*Q - G^T*U + L ] 
	Eigen::VectorXd Lambda( nVar*(nCells + 1) ), CqGuL_global(nVar*(nCells+1));
	CqGuL_global.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd LVarCell(2);
		Eigen::VectorXd CqGuLVarCell(2);
		for(int var = 0; var < nVar; var++)
		{
			LVarCell = system->L_global.block<2,1>(var*(nCells+1) + i,0);
			CqGuLVarCell = LVarCell - system->C_cellwise[i].block(var*2,var*(k+1),2,k+1)*tempQ.coeffs[ var ][ i ].second - system->G_cellwise[i].block(var*2,var*(k+1),2,k+1)*tempU.coeffs[ var ][ i ].second;
			CqGuL_global.block(var*(nCells + 1) + i, 0, 2, 1) += CqGuLVarCell;
		}
	} 
	Lambda = system->H_global.solve(CqGuL_global);

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd lamCell(2*nVar);
		Eigen::VectorXd multiVarR(nVar*(k+1));

		for(int var = 0; var < nVar; var++)
		{
			lamCell[2*var] = Lambda[var*(nCells+1) + i]; lamCell[2*var + 1] = Lambda[var*(nCells+1) + i+1];
			multiVarR.block(var*(k+1), 0, k+1, 1) = system->RF_cellwise[i].block(var*(k+1),0,k+1,1);
		}

		//Res1 = A*Q - B^T*U + C^T*Lam - R
		// length = nVar*(k+1)
		Eigen::VectorXd res1CellVec(nVar*(k+1));
		res1CellVec = system->A_cellwise[i]*tempQ.multiVarCoeffs(i) - system->B_cellwise[i].transpose()*tempU.multiVarCoeffs(i) + system->C_cellwise[i].transpose()*lamCell - multiVarR;

		//Res2 = B*Q + D*U + X*Udot + E*Lam - F		
		// length = nVar*(k+1)
		for(int var = 0; var < nVar; var++)
		{
			res1.coeffs[ var ][ i ].second = res1CellVec.block(var*(k+1) , 0, k+1, 1);
			res2.coeffs[ var ][ i ].second = system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempQ.coeffs[ var ][ i ].second + system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) + tempdudt.coeffs[ var ][ i ].second;
		}
	} 

	return 0;
}


void SystemSolver::print( std::ostream& out, double t, int nOut, int var )
{
	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut ) );
		out << x << "\t" << EvalCoeffs( u.Basis, u.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dudt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, q.coeffs, x, var ) << std::endl;
	}
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

void SystemSolver::setKappaInv(Eigen::MatrixXd kappa)
{
	kappaInv = kappa.inverse();
}

/*
void SystemSolver::solveNonIDA(N_Vector& Y, N_Vector& dYdt, double dt)
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