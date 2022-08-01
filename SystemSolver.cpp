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

	//Solve for Lambda with Lam = (C^T)^-1*[ -A*Q + B^T*U +R ]
	Eigen::VectorXd Lambda( nCells + 1 ), Lambda2( nCells + 1 ), Lambda3( nCells + 1 ), Lambda4( nCells + 1 );
	Eigen::Vector2d lamCell, lamCell3;
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		auto cTInv = Eigen::FullPivLU< Eigen::MatrixXd >(C_cellwise[i].transpose());
		lamCell = cTInv.solve( -A_cellwise[i]*q.coeffs[I] + B_cellwise[i].transpose()*u.coeffs[I] + RF_cellwise[i].block(0,0,k+1,1));
		Lambda[i] = lamCell[0];

		dudt.coeffs[I] = Eigen::FullPivLU< Eigen::MatrixXd >(XMats[i]).solve( -B_cellwise[i]*q.coeffs[I] - D_cellwise[i]*u.coeffs[I] - E_cellwise[i]*lamCell + RF_cellwise[ i ].block( k + 1, 0, k + 1, 1 ) );
		//std::cerr << dudt.coeffs[I] << std::endl << std::endl;

		Lambda2[i] = u_0( I.x_l );

		Eigen::Vector2d Lcell;
		Lcell[0] = L_global[i]; Lcell[1] = L_global[i+1];
		auto HInv = Eigen::FullPivLU< Eigen::MatrixXd >(H_cellwise[i]);
		lamCell3 = HInv.solve(Lcell - C_cellwise[i]*q.coeffs[I] - G_cellwise[i]*u.coeffs[I]);

		//std::cerr << HInv.solve(Lcell) << std::endl << std::endl << HInv.solve(- C_cellwise[i]*q.coeffs[I]) << std::endl << std::endl << HInv.solve(- G_cellwise[i]*u.coeffs[I]) << std::endl << std::endl << lamCell3 << std::endl;
		
		Lambda3[i] += lamCell3[0]/2;
		Lambda3[i+1] = lamCell3[1]/2;
	} 

	Lambda[nCells] = lamCell[1];
	Lambda2[nCells] = u_0( grid.upperBound );
	Lambda3[nCells] = lamCell3[1];

	for(int i = 0; i < Lambda.size(); i++)
	{
		std::cerr << Lambda[i]-Lambda3[i] << "		" << Lambda2[i] << "		" << Lambda3[i] << "		" << Lambda4[i] << std::endl;
	}
	//std::cerr << std::endl;
	//Solve Lambda from u,q and dudt
	/*
	Eigen::VectorXd Lambda( nCells + 1 );
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Lambda[i] = u_0( I.x_l );
	}
	Lambda[nCells] = u_0( grid.upperBound );
	*/
	//std::cerr << Lambda << std::endl << std::endl;


	DGtoSundialsVecConversion(q, u, Lambda2, Y); //To do, initialise lambda from QU
	DGApprox zero{grid,k};

	DGtoSundialsVecConversion(zero, dudt, Eigen::VectorXd::Zero(nCells+1), dYdt);
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

		//std::cerr << ABBD << std::endl << std::endl;

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
		ABCBDECGHMats.emplace_back(ABCBCECGH);
		XMats.emplace_back(X);
	}
	initialised = true;
}

void SystemSolver::clearCellwiseVecs()
{
	ABCBDECGHMats.clear();
	XMats.clear();
	ABBDBlocks.clear();
	CG_cellwise.clear();
	RF_cellwise.clear();
	QU_0_cellwise.clear();
	A_cellwise.clear();
	B_cellwise.clear();
	E_cellwise.clear();
	C_cellwise.clear();
	G_cellwise.clear();
	H_cellwise.clear();
}

void SystemSolver::sundialsToDGVecConversion(N_Vector const& g, std::vector< Eigen::VectorXd >& g1g2_cell, Eigen::VectorXd& g3_globe)
{
	//g is split into 3 parts, each part corresponding to rows of the glodabl matrix. g1g2 refers to the q and u rows. 
	//Here data is parsed into eigen vectors of coefficients, 1 for each cell. g3 refers to the lambda row which will 
	//be stored as a single eigen vector of lambdas at every boundary
	g1g2_cell.clear();
	g3_globe.resize(nCells + 1);

	if( N_VGetLength(g) != 2*nCells*(k+1) + nCells + 1)
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
		g1g2_cell.emplace_back(g1g2Coeffs); //?Missing the nCellth entry maybe??
	}

	//g3 Global vector for [C G H][Q U Gamma]^T = g3
	int j = 0;
	for(int i = 2*nCells*(k+1); i < 2*nCells*(k+1) + nCells + 1; i++)
	{
		g3_globe[j] = gVec[i];
		j++;
	}
}

void SystemSolver::sundialsToDGVecConversion(N_Vector const& Y, DGApprox& U, DGApprox& Q, Eigen::VectorXd& lam)
{
	//g is split into 3 parts, each part corresponding to rows of the glodabl matrix. g1g2 refers to the q and u rows. 
	//Here data is parsed into eigen vectors of coefficients, 1 for each cell. g3 refers to the lambda row which will 
	//be stored as a single eigen vector of lambdas at every boundary

	if( N_VGetLength(Y) != 2*nCells*(k+1) + nCells + 1)
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
	
	int j = 0;
	for(int i = 2*nCells*(k+1); i < 2*nCells*(k+1) + nCells + 1; i++)
	{
		lam[j] = yVec[i];
		j++;
	}
}

void SystemSolver::sundialsToDGVecConversion( N_Vector& Y, std::vector< Eigen::VectorXd >& QULamCellwise)
{
	QULamCellwise.clear();
	std::vector< Eigen::VectorXd > QUCellwise;
	Eigen::VectorXd LamGlobe;
	sundialsToDGVecConversion(Y, QUCellwise, LamGlobe);

	Eigen::VectorXd QULam(2*k+2 + 2);
	for(int i=0; i<nCells; i++)
	{
		QULam.setZero();
		QULam.block(0,0,2*k+2,1) = QUCellwise[i];
		QULam[2*k+2] = LamGlobe[i];
		QULam[2*k+3] = LamGlobe[i+1];
		QULamCellwise.emplace_back(QULam);
	}
}

void SystemSolver::DGtoSundialsVecConversion(DGApprox delQ, DGApprox delU, Eigen::VectorXd delLambda, N_Vector& delY)
{
	if (N_VGetLength(delY) != 2*nCells*(k+1) + nCells + 1)
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
		//std::cerr << qCellCoeffs << std::endl << std::endl;
	}

	for(int i=0; i<nCells+1; i++)
	{
		delYVec[2*nCells*(k+1) + i] = delLambda[i];
	}
}

void SystemSolver::updateUandQ(N_Vector const& Y)
{
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) );


	for(int i=0; i<nCells; i++)
	{
		Interval const& I( grid.gridCells[ i ] );
		for(int j=0; j<k+1; j++)
		{
			q.coeffs[I][j] = yVec[i*(k+1) + j];
			u.coeffs[I][j] = yVec[nCells*(k+1) + i*(k+1) + j];
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
		//std::cerr << "ABBDX =  " << ABBDX << std::endl << std::endl;
	}
}

void SystemSolver::solveJacEq(N_Vector const& g, N_Vector& delY)
{
	Eigen::VectorXd delLambda( nCells + 1 );
	DGApprox delU(grid,k), delQ(grid,k);
	std::vector< Eigen::VectorXd > g1g2_cellwise;
	Eigen::VectorXd g3_global{};
	K_global.setZero();

	//assemble temp cellwise ABBD blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > ABBDXSolvers{};
	updateABBDForJacSolve(ABBDXSolvers, alpha);

	// Assemble RHS g into cellwise form and solve for QU blocks
	sundialsToDGVecConversion(g, g1g2_cellwise, g3_global);
	std::vector< Eigen::VectorXd > QU_f( nCells );
	std::vector< Eigen::MatrixXd > QU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{  
		Interval const& I( grid.gridCells[ i ] );

		//QU_f
		Eigen::VectorXd g1g2 = g1g2_cellwise[ i ];
		//std::cerr << "g1g2 =  " << g1g2 << std::endl << std::endl;
		//g1g2.block( k + 1, 0, k + 1, 1 ) += ( 2.0/dt ) * u.coeffs[ grid.gridCells[ i ] ]; 
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
	F = g3_global;
	for ( unsigned int i=0; i < nCells; i++ )
		F.block<2,1>( i, 0 ) -= CG_cellwise[ i ] * QU_f[ i ];

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	delLambda = lu.solve( F );

	// Now find del u and del q to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd delQU( 2*( k + 1 ) );
		//std::cerr << QU_0[ i ] * delLambda.block<2,1>( i, 0 ) << std::endl << std::endl;
		delQU = QU_f[ i ] - QU_0[ i ] * delLambda.block<2,1>( i, 0 );
		delQ.coeffs[ I ] = delQU.block( 0, 0, k + 1, 1 );
		delU.coeffs[ I ] = delQU.block( k + 1, 0, k + 1, 1 );
	}

	DGtoSundialsVecConversion(delQ, delU, delLambda, delY);

	//VectorWrapper vec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) ); 
	//std::cerr << vec << std::endl << std::endl;

	// Finally Compute dudt (this allows us to do cubic spline fitting to find u(t) for t in between gridpoints)
	/*
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::MatrixXd M( k + 1, k + 1 );
		u.DerivativeMatrix( I, M );
		Eigen::MatrixXd M2( k + 1, k + 1 );
		u.DerivativeMatrix( I, M2, c_fn );
		// Work out du/dt at t=0 by direct evaluation of (29)
		// First (c u + q, w')
		dudt.coeffs[ I ] = 
			RF_cellwise[ i ].block( k + 1, 0, k + 1, 1 ) 
			- M * q.coeffs[ I ] + M2.transpose() * u.coeffs[ I ];

		// Then componentwise add f_i and the boundary terms
		for ( unsigned int j=0; j < k + 1; ++j )
		{
			// Finally minus < (cu)^ . n, w > (but because of the 'n', this leads to
			//  + <eval at lower bound> - <eval at upper bound>
			dudt.coeffs[ I ]( j ) +=
				+ c_fn( I.x_l )*Lambda[ i ]*u.Basis.phi( I, j )( I.x_l )
				- c_fn( I.x_u )*Lambda[ i + 1 ]*u.Basis.phi( I, j )( I.x_u );
		}
	}
	
	t += dt;
	u_of_t[ t ] = std::make_pair( u.coeffs, dudt.coeffs );
	*/
}

void SystemSolver::solveNonIDA(N_Vector Y, N_Vector dYdt, double dt)
{
	std::vector< Eigen::VectorXd > QU_cellwise;
	Eigen::VectorXd tempLam(nCells+1);
	sundialsToDGVecConversion(Y, u, q, tempLam);
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

		std::cerr << QU_f[i] << std::endl << std::endl;

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
	std::cerr << tempLam << std::endl << std::endl;


	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd QU( 2*( k + 1 ) );
		QU = QU_f[ i ] - QU_0[ i ] * tempLam.block<2,1>( i, 0 );
		q.coeffs[ I ] = QU.block( 0, 0, k + 1, 1 );
		dudt.coeffs[I] = (QU.block( k + 1, 0, k + 1, 1 ) - u.coeffs[I])/dt;
		u.coeffs[ I ] = QU.block( k + 1, 0, k + 1, 1 );
	}


	DGtoSundialsVecConversion(q, u, tempLam, Y);
	/*
	// Now find del u and del q to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd QU( 2*( k + 1 ) );
		delQU = QU_f[ i ] + QU_0[ i ] * delLambda.block<2,1>( i, 0 );
		delQ.coeffs[ I ] = delQU.block( 0, 0, k + 1, 1 );
		delU.coeffs[ I ] = delQU.block( k + 1, 0, k + 1, 1 );
	}

	DGtoSundialsVecConversion(delQ, delU, delLambda, delY);

	VectorWrapper vec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) ); 
	*/
	print(std::cout, dt, 10);
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	auto k = system->k;
	auto grid(system->grid);
	auto nCells = system->nCells;
	auto c_fn = system->getcfn();
	DGApprox ResQ(grid,k), ResU(grid,k);

	std::vector<Eigen::VectorXd> QULamCellwise{};		// 2(k+1)+2
	std::vector<Eigen::VectorXd> oUDotoCellwise{};		// 2(k+1)
	std::vector<Eigen::VectorXd> ResCellwise{};			// 2(k+1)

	DGApprox tempU(grid, k), tempQ(grid, k), tempdudt(grid, k), temp0(grid, k);
	Eigen::VectorXd tempLam(nCells+1), tempZeroVec(nCells+1), res3(nCells+1);
	DGApprox res1(grid, k), res2(grid, k);

	system->sundialsToDGVecConversion(Y, tempU, tempQ, tempLam);
	system->sundialsToDGVecConversion(dydt, tempdudt, temp0, tempZeroVec); //?Should be all zeros except for?

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::Vector2d lamCell;
		lamCell[0] = tempLam[i]; lamCell[1] = tempLam[i+1];

		//Res1 = A*Q - B^T*U + C^T*Lam - R
		// length = k+1
		res1.coeffs[ I ] = system->A_cellwise[i]*tempQ.coeffs[I] - system->B_cellwise[i].transpose()*tempU.coeffs[I] + system->C_cellwise[i].transpose()*lamCell - system->RF_cellwise[i].block(0,0,k+1,1);
		//Res2 = B*Q + D*U + X*Udot + E*Lam - F
		//;ength = k+1
		res2.coeffs[I] =  system->B_cellwise[i]*tempQ.coeffs[I] + system->D_cellwise[i]*tempU.coeffs[I] + system->XMats[i]*tempdudt.coeffs[I] + system->E_cellwise[i]*lamCell - system->RF_cellwise[ i ].block( k + 1, 0, k + 1, 1 );
		
		//Res3 = C*Q + G*U + H*Lam - L
		//length = 2
		
		Eigen::Vector2d res3Cell, Lcell;
		Lcell[0] = system->L_global[i]; Lcell[1] = system->L_global[i+1];

		res3Cell = system->C_cellwise[i]*tempQ.coeffs[I] + system->G_cellwise[i]*tempU.coeffs[I] + system->H_cellwise[i]*lamCell - Lcell;
		//res3[i] = res3Cell[0]; 
		res3[i+1] = res3Cell[1];

		auto HInv = Eigen::FullPivLU< Eigen::MatrixXd > (system->H_cellwise[i]);
		res3Cell = HInv.solve(Lcell - system->C_cellwise[i]*tempQ.coeffs[I] - system->G_cellwise[i]*tempU.coeffs[I]);

		//std::cerr << HInv.solve(Lcell) << std::endl << std::endl << HInv.solve(- C_cellwise[i]*q.coeffs[I]) << std::endl << std::endl << HInv.solve(- G_cellwise[i]*u.coeffs[I]) << std::endl << std::endl << lamCell3 << std::endl;
		
		res3[i] += res3Cell[0]/2;
		res3[i+1] = res3Cell[1]/2;

		/*
		//std::cerr << (system->C_cellwise[i]*tempQ.coeffs[I])[0] << "	" << (system->G_cellwise[i]*tempU.coeffs[I])[0] << "	" << (system->H_cellwise[i]*lamCell)[0] << "	" <<  -Lcell[0] << "	" << res3Cell[0]  << "	"  << lamCell[0]  << std::endl;
		//std::cerr << (system->C_cellwise[i]*tempQ.coeffs[I])[1] << "	" << (system->G_cellwise[i]*tempU.coeffs[I])[1] << "	" << (system->H_cellwise[i]*lamCell)[1] << "	" <<  -Lcell[1] << "	" << res3Cell[1]  << "	"  << lamCell[1]   << std::endl << std::endl;
		if(i==0) res3[i] = 0.0;
		else
		{
			auto IL = grid.gridCells[ i-1 ];
			//res3[i] =  -0.5*lamCell[0] - Lcell[0]; //? c.n - Tau need to put in explicitly?
			//res3[i] = system->u.Basis.Evaluate( I, tempQ.coeffs[I], I.x_l ) - system->u.Basis.Evaluate( IL, tempQ.coeffs[IL], IL.x_u );
			//res3[i] -= 0.5*(system->u.Basis.Evaluate( I, tempU.coeffs[I], I.x_l ) - system->u.Basis.Evaluate( IL, tempU.coeffs[IL], IL.x_u ));

			//std::cerr << -0.5*lamCell[0] - Lcell[0] << "	" << system->u.Basis.Evaluate( I, tempQ.coeffs[I], I.x_l ) << "	" << system->u.Basis.Evaluate( IL, tempQ.coeffs[IL], IL.x_u ) << "	" << system->u.Basis.Evaluate( I, tempU.coeffs[I], I.x_l ) << "	" << system->u.Basis.Evaluate( IL, tempU.coeffs[IL], IL.x_u ) << "	" << IL.x_u << "	" << I.x_l << std::endl << std::endl;
			//res3[i] = lamCell[0] - 0.5*(system->u.Basis.Evaluate( I, tempU.coeffs[I], I.x_l ) + system->u.Basis.Evaluate( IL, tempU.coeffs[IL], IL.x_u ));
		}
		*/
	} 
	res3[nCells] = 0.0;
	res3 -= tempLam;
	//res3.setZero();

//---------------------------------------------------------------------------------
	/*
	//Solve for Lambda
	std::vector< Eigen::VectorXd > QU_cellwise;
	system->K_global.setZero();
	//update F
	std::vector< Eigen::VectorXd > QU_f( nCells );
	std::vector< Eigen::MatrixXd > QU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		auto ABBDsolver = Eigen::FullPivLU< Eigen::MatrixXd >(system->ABBDBlocks[i]);
		Interval const& I( grid.gridCells[ i ] );

		//QU_f
		Eigen::VectorXd RF = system->RF_cellwise[ i ];
		RF.block( k + 1, 0, k + 1, 1 ) += tempdudt.coeffs[ I ];
		QU_f[ i ] = ABBDsolver.solve( RF );
		std::cerr << QU_f[i] << std::endl << std::endl;


		//QU_0
		Eigen::MatrixXd CE = system->CEBlocks[ i ];
		QU_0[ i ] = ABBDsolver.solve( CE );

		Eigen::Matrix2d K_cell = system->H_cellwise[i] - system->CG_cellwise[ i ] * QU_0[i];

		//K
		system->K_global.block( i, i, 2, 2 ) += K_cell;
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nCells + 1 );
	F = system->L_global;
	for ( unsigned int i=0; i < nCells; i++ )
		F.block<2,1>( i, 0 ) -= system->CG_cellwise[ i ] * QU_f[ i ];

	Eigen::FullPivLU< Eigen::MatrixXd > lu( system->K_global );
	res3 = tempLam - lu.solve( F );
	std::cerr << tempLam << std::endl << std::endl;
	std::cerr << lu.solve(F) << std::endl << std::endl;
*/
	/*
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd QU( 2*( k + 1 ) );
		QU = QU_f[ i ] - QU_0[ i ] * tempLam.block<2,1>( i, 0 );
		res1.coeffs[I] = tempQ.coeffs[ I ] - QU.block( 0, 0, k + 1, 1 );
		res2.coeffs[I] = tempU.coeffs[ I ] - QU.block( k + 1, 0, k + 1, 1 );
	}
	
	*/
	VectorWrapper Vec( N_VGetArrayPointer( dydt ), N_VGetLength( dydt ) );
	VectorWrapper resVec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
	system->DGtoSundialsVecConversion(res1, res2, res3, resval);
	std::cerr << resVec << std::endl << std::endl;
	//std::cerr << resVec.norm() << std::endl << std::endl;
	return 0;
}


void SystemSolver::print( std::ostream& out, double t, int nOut )
{
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		std::cerr << u.coeffs[I] << std::endl; 
	}
	std::cerr << std::endl;

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
