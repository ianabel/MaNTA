#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c, Fn const& kappa, BoundaryConditions const& boundary )
	: grid(Grid), k(polyNum), nCells(N_cells), dt(Dt), RHS(rhs), tau(Tau), c_fn(c), kappa_fn(kappa), u(grid,k), q(grid,k), dudt(grid,k), BCs(boundary)
{
}

void SystemSolver::setInitialConditions( std::function< double ( double )> u_0 , N_Vector Y , N_Vector dYdt) 
{
	// Someone should have initialised this, but still...
	if ( !initialised )
		initialiseMatrices();

	u = u_0;

	// Differentiate u_0 to get q_0
	DGApprox u_0_hr( grid, 2*k, u_0 );
	for ( unsigned int i=0; i < nCells; i++ ) {
		Interval I = grid.gridCells[ i ];
		Eigen::MatrixXd M_hr( 2*k + 1, 2*k + 1 );
		u_0_hr.DerivativeMatrix( I, M_hr );
		q.coeffs[ I ] = -( M_hr * u_0_hr.coeffs[ I ] ).block( 0, 0, k + 1, 1 ); //To Do:kappa will have to be added here
	}

	//Solve for Lambda with Lam = (H^T)^-1*[ -C*Q - G^T*U + L ] 
	Eigen::VectorXd Lambda2( nCells + 1 ), Lambda( nCells + 1 ), CqGuL_global(nCells+1);
	CqGuL_global.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::Vector2d Lcell;
		Lcell[0] = L_global[i]; Lcell[1] = L_global[i+1];

		Eigen::Vector2d CqGuL;
		CqGuL = Lcell - C_cellwise[i]*q.coeffs[I] - G_cellwise[i]*u.coeffs[I];
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

		dudt.coeffs[I] = Eigen::FullPivLU< Eigen::MatrixXd >(XMats[i]).solve( -B_cellwise[i]*q.coeffs[I] - D_cellwise[i]*u.coeffs[I] - E_cellwise[i]*lamCell + RF_cellwise[ i ].block( k + 1, 0, k + 1, 1 ) );
	}
	
	DGtoSundialsVecConversion(q, u, Y);
	DGApprox zero{grid,k};
	DGtoSundialsVecConversion(zero, dudt, dYdt);
}

void SystemSolver::initialiseMatrices()
{
	// These are temporary working space
	// Matrices we need per cell
	Eigen::MatrixXd A( k + 1, k + 1 );
	Eigen::MatrixXd B( k + 1, k + 1 );
	Eigen::MatrixXd D( k + 1, k + 1 );
	// Two endpoints per cell
	Eigen::MatrixXd C( 2, k + 1 );
	Eigen::MatrixXd E( k + 1, 2 );
	Eigen::MatrixXd HGlobalMat( nCells+1, nCells+1 );
	HGlobalMat.setZero();
	K_global.resize( nCells + 1, nCells + 1 );
	K_global.setZero();
	L_global.resize( nCells + 1 );
	L_global.setZero();

	clearCellwiseVecs();
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		// A_ij = ( phi_j/kappa, phi_i )
		auto kappa_inv = [ & ]( double x ) { return 1.0/kappa_fn( x );};
		u.MassMatrix( I, A, kappa_inv );
		// B_ij = ( phi_i, phi_j' )
		u.DerivativeMatrix( I, B );
		// D_ij = -(c phi_j, phi_i') + < w, tau u > 
		u.DerivativeMatrix( I, D, c_fn );
		// As DerivativeMatrix gives the weighted product (c phi_i, phi_j')
		// we flip the sign and the indices on D.
		D *= -1.0;
		D.transposeInPlace();
		//Note: D matix is unaffected in initialisation. During ech timestep it will see an extra mass matix*alpha to reflect the addition of the del udot term
		//When checking the residual eq the udot term will be added seperately.

		// Now do all the boundary terms
		for ( Eigen::Index i=0; i<k+1;i++ )
		{
			for ( Eigen::Index j=0; j<k+1;j++ )
			{
				D( i, j ) += 
					tau( I.x_l )*u.Basis.phi( I, j )( I.x_l )*u.Basis.phi( I, i )( I.x_l ) +
					tau( I.x_u )*u.Basis.phi( I, j )( I.x_u )*u.Basis.phi( I, i )( I.x_u )	;
			}
		}

		A_cellwise.emplace_back(A);
		B_cellwise.emplace_back(B);
		D_cellwise.emplace_back(D);

		Eigen::MatrixXd ABBD( 2*k + 2, 2*k + 2 );
		ABBD.block( 0, 0, k+1, k+1 ) = A;
		ABBD.block( k+1, 0, k+1, k+1 ) = B;
		ABBD.block( 0, k+1, k+1, k+1 ) = -B.transpose();
		ABBD.block( k+1, k+1, k+1, k+1 ) = D;

		// 2*( k + 1) is normally only medium-sized (10 - 20)
		// so just do a full LU factorization to solve 
		ABBDBlocks.emplace_back( ABBD );

		for ( Eigen::Index i=0; i<k+1;i++ )
		{
			// C_ij = < psi_i, phi_j * n_x > , where psi_i are edge degrees of
			// freedom and n_x is the unit normal in the x direction
			// for a line, edge degrees of freedom are just 1 at each end

			C( 0, i ) = -u.Basis.phi( I, i )( I.x_l );
			C( 1, i ) = u.Basis.phi( I, i )( I.x_u ); 

			// E_ij = < phi_i, (c n_x - tau ) lambda >
			E( i, 0 ) = u.Basis.phi( I, i )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) );
			E( i, 1 ) = u.Basis.phi( I, i )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) );

			if ( I.x_l == BCs.LowerBound && BCs.isLBoundDirichlet )
			{
				C( 0, i ) = 0;
				E( i, 0 ) = 0;
			}

			if ( I.x_u == BCs.UpperBound && BCs.isUBoundDirichlet )
			{
				C( 1, i ) = 0;
				E( i, 1 ) = 0;
			}
		}

		// Construct per-cell Matrix solutions
		// ( A  -B^T )^-1 [ C^T ]
		// ( B    D  )    [ E   ]
		// These are the homogeneous solution, that depend on lambda
		Eigen::MatrixXd CE_vec( 2*k + 2, 2 );
		CE_vec.block( 0    , 0, k + 1, 2 ) = C.transpose();
		CE_vec.block( k + 1, 0, k + 1, 2 ) = E;
		CEBlocks.emplace_back(CE_vec);

		C_cellwise.emplace_back(C);
		E_cellwise.emplace_back(E);

		// To store the RHS
		RF_cellwise.emplace_back(Eigen::VectorXd(2*k+2));

		// R is composed of parts of the values of 
		// u on the total domain boundary
		RF_cellwise[ i ].setZero();
		// Take components of f
		for ( Eigen::Index j = 0; j <= k; j++ )
			RF_cellwise[ i ]( k + 1 + j ) = u.CellProduct( I, RHS, u.Basis.phi( I, j ) );

		if ( I.x_l == BCs.LowerBound  && BCs.isLBoundDirichlet )
		{

			for ( Eigen::Index j = 0; j <= k; j++ )
			{
				// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
				RF_cellwise[ i ]( j ) = -u.Basis.phi( I, j )( I.x_l ) * ( -1 ) * BCs.g_D( I.x_l );
				// - < ( c.n - tau ) g_D, w > 
				RF_cellwise[ i ]( k + 1 + j ) -= u.Basis.phi( I, j )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs.g_D( I.x_l );
			}
		}

		if ( I.x_u == BCs.UpperBound && BCs.isUBoundDirichlet )
		{
			for ( Eigen::Index j = 0; j <= k; j++ )
			{
				// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
				RF_cellwise[ i ]( j ) += -u.Basis.phi( I, j )( I.x_u ) * ( +1 ) * BCs.g_D( I.x_u );
				RF_cellwise[ i ]( k + 1 + j ) -= u.Basis.phi( I, j )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs.g_D( I.x_u );
			}
		}


		// Now RF_cellwise[ i ] holds the steady-state RHS. 
		// We have to add the previous timestep to it when doing the time-dependent solve

		// Per-cell contributions to the global matrices K and F.
		// First fill G
		Eigen::MatrixXd G( 2, k + 1 );
		for ( Eigen::Index i = 0; i < k+1; i++ )
		{
			G( 0, i ) = tau( I.x_l )*u.Basis.phi( I, i )( I.x_l );
			if ( I.x_l == BCs.LowerBound && BCs.isLBoundDirichlet )
				G( 0, i ) = 0.0;
			G( 1, i ) = tau( I.x_u )*u.Basis.phi( I, i )( I.x_u );
			if ( I.x_u == BCs.UpperBound && BCs.isUBoundDirichlet )
				G( 1, i ) = 0.0;
		}

		CG_cellwise.emplace_back(Eigen::MatrixXd(2, 2*(k+1) ));
		CG_cellwise[ i ].block( 0, 0, 2, k + 1 ) = C;
		CG_cellwise[ i ].block( 0, k + 1, 2, k + 1 ) = G;
		G_cellwise.emplace_back(G);

		// Now fill H
		Eigen::MatrixXd H( 2, 2 );
		H( 0, 0 ) = -c_fn( I.x_l ) - tau( I.x_l );
		H( 1, 0 ) = 0.0;
		H( 0, 1 ) = 0.0;
		H( 1, 1 ) = c_fn( I.x_u ) - tau( I.x_u );

		if ( I.x_l == BCs.LowerBound && BCs.isLBoundDirichlet )
				H( 0, 0 ) = H( 1, 0 ) = H( 0, 1 ) = 0.0;

		if ( I.x_u == BCs.UpperBound && BCs.isUBoundDirichlet )
				H( 1, 1 ) = H( 1, 0 ) = H( 0, 1 ) = 0.0;

		H_cellwise.emplace_back(H);
		HGlobalMat.block(i,i,2,2) += H;

		// Finally fill L
		if ( I.x_l == BCs.LowerBound && /* is b.d. Neumann at lower boundary */ !BCs.isLBoundDirichlet )
			L_global( i )     += BCs.g_N( BCs.LowerBound );
		if ( I.x_u == BCs.UpperBound && /* is b.d. Neumann at upper boundary */ !BCs.isUBoundDirichlet )
			L_global( i + 1 ) += BCs.g_N( BCs.UpperBound );

		//Now build the matrices used in the residual equation. These are largely similar as those for the Jac solve but no inversion is required
		//First we solve for the Q and U resuduals
		// [R_1]       ( A -B^T C^T ) [q]       [R]   ( 0  0  0 ) [  0  ]
		// [R_2]   =   ( B   D   E  ) [u]   -   [F] + ( 0  X  0 ) [du/dt]
		// [R_3]     ( C   G   H  ) [Lam]     [L]   ( 0  0  0 ) [  0  ]

		Eigen::MatrixXd ABCBCECGH(2*(k+1) + 2, 2*(k+1) + 2);
		ABCBCECGH.block( 0, 0, 2*k+2, 2*k+2 ) = ABBD;
		ABCBCECGH.block( 0, 2*k+2, 2*k+2, 2) = CE_vec;
		ABCBCECGH.block( 2*k+2, 0, 2, k+1) = C;
		ABCBCECGH.block( 2*k+2, k+1, 2, k+1) = G;
		ABCBCECGH.block( 2*k+2, 2*k+2, 2, 2) = H;

		Eigen::MatrixXd X( k + 1, k + 1 );
		u.MassMatrix( I, X);
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

void SystemSolver::sundialsToDGVecConversion(N_Vector const& g, std::vector< Eigen::VectorXd >& g1g2_cell)
{
	g1g2_cell.clear();

	if( N_VGetLength(g) != 2*nCells*(k+1))
		throw std::invalid_argument( "Sundials Vecotor does not match size \n" );

	VectorWrapper gVec( N_VGetArrayPointer( g ), N_VGetLength( g ) ); 

	Eigen::VectorXd g1g2Coeffs(2*k+2);
	for(int i=0; i<nCells; i++)
	{
		for(int j=0; j<k+1; j++)
		{
			g1g2Coeffs[j] = gVec[i*(k+1) + j];
			g1g2Coeffs[k+1+j] = gVec[nCells*(k+1) + i*(k+1) + j];
		}
		g1g2_cell.emplace_back(g1g2Coeffs);
	}
}

void SystemSolver::sundialsToDGVecConversion(N_Vector const& Y, DGApprox& U, DGApprox& Q)
{
	if( N_VGetLength(Y) != 2*nCells*(k+1))
		throw std::invalid_argument( "Sundials Vecotor does not match size \n" );

	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 

	for(int i=0; i<nCells; i++)
	{
		Interval const& I = grid.gridCells[ i ];
		for(int j=0; j<k+1; j++)
		{
			Q.coeffs[I][j] = yVec[i*(k+1) + j];
			U.coeffs[I][j] = yVec[nCells*(k+1) + i*(k+1) + j];
		}
	}
}

void SystemSolver::DGtoSundialsVecConversion(DGApprox delQ, DGApprox delU, N_Vector& delY)
{
	if (N_VGetLength(delY) != 2*nCells*(k+1))
		throw std::invalid_argument( "Sundials Vecotor does not match size \n" );

	VectorWrapper delYVec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) );

	Eigen::VectorXd uCellCoeffs(k+1), qCellCoeffs(k+1);
	for(int i=0; i<nCells; i++)
	{
		Interval const& I = grid.gridCells[ i ];
		qCellCoeffs = delQ.coeffs[I];
		uCellCoeffs = delU.coeffs[I];
		for (int j=0; j<k+1; j++)
		{
			delYVec[i*(k+1)+j] = qCellCoeffs[j];
			delYVec[nCells*(k+1) + i*(k+1)+j] = uCellCoeffs[j];
		}
	}
}

void SystemSolver::updateCoeffs(N_Vector const& Y, N_Vector const& dYdt)
{
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) );
	VectorWrapper dydtVec( N_VGetArrayPointer( dYdt ), N_VGetLength( dYdt ) );


	for(int i=0; i<nCells; i++)
	{
		Interval const& I( grid.gridCells[ i ] );
		for(int j=0; j<k+1; j++)
		{
			q.coeffs[I][j] = yVec[i*(k+1) + j];
			u.coeffs[I][j] = yVec[nCells*(k+1) + i*(k+1) + j];
			dudt.coeffs[I][j] = dydtVec[nCells*(k+1) + i*(k+1) + j];
		}
	}
}

void SystemSolver::updateABBDForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& ABBDXsolvers, double const alpha)
{
	std::function<double( double )> alphaF = [ = ]( double x ){ return alpha;};

	Eigen::MatrixXd X( k + 1, k + 1 );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		u.MassMatrix( I, X, alphaF);
		auto ABBDX = ABBDBlocks[i];
		ABBDX.block( k+1, k+1, k+1, k+1 ) += X;
		ABBDXsolvers.emplace_back(ABBDX);
	}
}

void SystemSolver::solveJacEq(N_Vector const& g, N_Vector& delY)
{
	Eigen::VectorXd delLambda( nCells + 1 );
	DGApprox delU(grid,k), delQ(grid,k);
	std::vector< Eigen::VectorXd > g1g2_cellwise;
	K_global.setZero();

	//assemble temp cellwise ABBD blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > ABBDXSolvers{};
	updateABBDForJacSolve(ABBDXSolvers, alpha);

	// Assemble RHS g into cellwise form and solve for QU blocks
	sundialsToDGVecConversion(g, g1g2_cellwise);
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

		Eigen::Matrix2d K_cell = H_cellwise[i] - CG_cellwise[ i ] * QU_0[i];

		//K
		K_global.block( i, i, 2, 2 ) += K_cell;
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nCells + 1 );
	F.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
		F.block<2,1>( i, 0 ) -= CG_cellwise[ i ] * QU_f[ i ];

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	delLambda = lu.solve( F );

	// Now find del u and del q to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd delQU( 2*( k + 1 ) );
		delQU = QU_f[ i ] - QU_0[ i ] * delLambda.block<2,1>( i, 0 );
		delQ.coeffs[ I ] = delQU.block( 0, 0, k + 1, 1 );
		delU.coeffs[ I ] = delQU.block( k + 1, 0, k + 1, 1 );
	}

	DGtoSundialsVecConversion(delQ, delU, delY);
}

void SystemSolver::solveNonIDA(N_Vector Y, N_Vector dYdt, double dt)
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
		RF.block( k + 1, 0, k + 1, 1 ) += (1/dt)*u.coeffs[ I ];
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
		q.coeffs[ I ] = QU.block( 0, 0, k + 1, 1 );
		dudt.coeffs[I] = (QU.block( k + 1, 0, k + 1, 1 ) - u.coeffs[I])/dt;
		u.coeffs[ I ] = QU.block( k + 1, 0, k + 1, 1 );
	}

	DGtoSundialsVecConversion(q, u, Y);
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	auto k = system->k;
	auto grid(system->grid);
	auto nCells = system->nCells;
	auto c_fn = system->getcfn();

	DGApprox tempU(grid, k), tempQ(grid, k), tempdudt(grid, k), temp0(grid, k);
	DGApprox res1(grid, k), res2(grid, k);

	system->sundialsToDGVecConversion(Y, tempU, tempQ);
	system->sundialsToDGVecConversion(dydt, tempdudt, temp0); 

	//Solve for Lambda with Lam = (H^T)^-1*[ -C*Q - G^T*U + L ] 
	Eigen::VectorXd Lambda( nCells + 1 ), CqGuL_global(nCells+1);
	CqGuL_global.setZero();
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::Vector2d Lcell;
		Lcell[0] = system->L_global[i]; Lcell[1] = system->L_global[i+1];

		Eigen::Vector2d CqGuL;
		CqGuL = Lcell - system->C_cellwise[i]*tempQ.coeffs[I] - system->G_cellwise[i]*tempU.coeffs[I];
		CqGuL_global.block(i,0,2,1) += CqGuL;
	} 
	Lambda = system->H_global.solve(CqGuL_global);

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::Vector2d lamCell;
		lamCell[0] = Lambda[i]; lamCell[1] = Lambda[i+1];

		//Res1 = A*Q - B^T*U + C^T*Lam - R
		// length = k+1
		res1.coeffs[ I ] = system->A_cellwise[i]*tempQ.coeffs[I] - system->B_cellwise[i].transpose()*tempU.coeffs[I] + system->C_cellwise[i].transpose()*lamCell - system->RF_cellwise[i].block(0,0,k+1,1);

		//Res2 = B*Q + D*U + X*Udot + E*Lam - F		
		// length = k+1
		res2.coeffs[ I ] = system->B_cellwise[i]*tempQ.coeffs[I] + system->D_cellwise[i]*tempU.coeffs[I] + system->E_cellwise[i]*lamCell - system->RF_cellwise[ i ].block( k + 1, 0, k + 1, 1 ) + tempdudt.coeffs[I];;
	} 
	
	system->DGtoSundialsVecConversion(res1, res2, resval);
	return 0;
}


void SystemSolver::print( std::ostream& out, double t, int nOut )
{
	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs.LowerBound + ( BCs.UpperBound - BCs.LowerBound ) * ( static_cast<double>( i )/( nOut ) );
		out << x << "\t" << EvalCoeffs( u.Basis, u.coeffs, x ) << "\t" << EvalCoeffs( u.Basis, dudt.coeffs, x ) << "\t" << EvalCoeffs( u.Basis, q.coeffs, x ) << std::endl;
	}
}

double SystemSolver::EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x )
{
	for ( auto const & pair : cs )
	{
		if ( pair.first.contains( x ) )
			return B.Evaluate( pair.first, pair.second, x );
	}
	return std::nan( "" );
}
