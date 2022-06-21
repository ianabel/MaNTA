#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c, Fn const& kappa, BoundaryConditions const& boundary )
	: grid(Grid), k(polyNum), nCells(N_cells), dt(Dt), RHS(rhs), tau(Tau), c_fn(c), kappa_fn(kappa), u(grid,k), q(grid,k), dudt(grid,k), BCs(boundary)
{
}

void SystemSolver::setInitialConditions( std::function< double ( double )> u_0 ) {
	// Someone should have initialised us, but still...
	if ( !initialised )
		initialiseMatrices();

	u = u_0;

	// Differentiate u_0 to get q_0
	DGApprox u_0_hr( grid, 2*k, u_0 );
	for ( unsigned int i=0; i < nCells; i++ ) {
		Interval I = grid.gridCells[ i ];
		Eigen::MatrixXd M_hr( 2*k + 1, 2*k + 1 );
		u_0_hr.DerivativeMatrix( I, M_hr );
		q.coeffs[ I ] = -( M_hr * u_0_hr.coeffs[ I ] ).block( 0, 0, k + 1, 1 );
	}

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
				+ ( c_fn( I.x_l )*u_0( I.x_l ) )*u.Basis.phi( I, j )( I.x_l )
				- ( c_fn( I.x_u )*u_0( I.x_u ) )*u.Basis.phi( I, j )( I.x_u );
		}
	}
};

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


		Eigen::MatrixXd ABBD( 2*k + 2, 2*k + 2 );
		ABBD.block( 0, 0, k+1, k+1 ) = A;
		ABBD.block( k+1, 0, k+1, k+1 ) = B;
		ABBD.block( 0, k+1, k+1, k+1 ) = -B.transpose();
		ABBD.block( k+1, k+1, k+1, k+1 ) = D;

		std::cerr << ABBD << std::endl << std::endl;

		// 2*( k + 1) is normally only medium-sized (10 - 20)
		// so just do a full LU factorization to solve 
		ABBDSolvers.emplace_back( ABBD );

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

		E_cellwise[ i ] = E;
		QU_0_cellwise[ i ] = ABBDSolvers.back().solve( CE_vec );

		// To store the RHS
		RF_cellwise[ i ].resize( 2*k + 2 );

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

		CG_cellwise[ i ].resize( 2, 2*( k + 1 ) );
		CG_cellwise[ i ].block( 0, 0, 2, k + 1 ) = C;
		CG_cellwise[ i ].block( 0, k + 1, 2, k + 1 ) = G;


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

		Eigen::Matrix2d K_cell = H - CG_cellwise[ i ] * QU_0_cellwise[ i ];
		K_global.block( i, i, 2, 2 ) += K_cell;

		// Finally fill L
		if ( I.x_l == BCs.LowerBound && /* is b.d. Neumann at lower boundary */ !BCs.isLBoundDirichlet )
			L_global( i )     += BCs.g_N( BCs.LowerBound );
		if ( I.x_u == BCs.UpperBound && /* is b.d. Neumann at upper boundary */ !BCs.isUBoundDirichlet )
			L_global( i + 1 ) += BCs.g_N( BCs.UpperBound );

		//Now build the matrices used in the residual equation. These are largely similar as those for the Jac solve but no inversion is required
		//First we solve for the Q and U resuduals
		// [R_q]       ( A -B^T C^T ) [q]       [R]   ( 0  0  0 ) [  0  ]
		// [R_u]   =   ( B   D   E  ) [u]   -   [F] + ( 0  X  0 ) [du/dt]
		// [R_Lam]     ( C   G   H  ) [Lam]     [L]   ( 0  0  0 ) [  0  ]

		Eigen::MatrixXd ABCBCECGH(2*(k+1) + 2, 2*(k+1) + 2);
		ABCBCECGH.block( 0, 0, 2*k+2, 2*k+2 ) = ABBD;
		ABCBCECGH.block( 0, 2*k+2, 2*k+2, 2) = CE_vec;
		ABCBCECGH.block( 2*k+2, 0, 2, k+1) = C;
		ABCBCECGH.block( 2*k+2, k+1, 2, k+1) = G;
		ABCBCECGH.block( 2*k+2, 2*k+2, 2, 2) = H;

		Eigen::MatrixXd X( k + 1, k + 1 );
		u.MassMatrix( I, X);

		Eigen::MatrixXd CGH(2, 2*k+2 + 2);
		CGH.block( 0, 0, 2, k+1) = C;
		CGH.block( 0, k+1, 2, k+1) = G;
		CGH.block( 0, 2*k+2, 2, 2) = H;

		ABCBDECGHMats.emplace_back(ABCBCECGH);
		XMats.emplace_back(X);
	}
	initialised = true;
}

void SystemSolver::sundialsToDGVecConversion(N_Vector const& g, std::vector< Eigen::VectorXd >& g1g2_cell, Eigen::VectorXd& g3_globe)
{
	//g is split into 3 parts, each part corresponding to rows of the glodabl matrix. g1g2 refers to the q and u rows. 
	//Here data is parsed into eigen vectors of coefficients, 1 for each cell. g3 refers to the lambda row which will 
	//be stored as a single eigen vector of lambdas at every boundary

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
		g1g2_cell[ i ] = g1g2Coeffs; //?Missing the nCellth entry maybe??
	}

	//g3 Global vector for [C G H][Q U Gamma]^T = g3
	int j = 0;
	for(int i = 2*nCells*(k+1); i < 2*nCells*(k+1) + nCells + 1; i++)
	{
		g3_globe[j] = gVec[i];
		j++;
	}
}

void SystemSolver::sundialsToDGVecConversion( N_Vector& Y, std::vector< Eigen::VectorXd >& QULamCellwise)
{
	std::vector< Eigen::VectorXd > QUCellwise;
	Eigen::VectorXd LamGlobe;
	sundialsToDGVecConversion(Y, QUCellwise, LamGlobe);

	Eigen::VectorXd QULam(2*k+2 + 2);
	for(int i=0; i<nCells; i++)
	{
		QULam.setZero();
		QULam.block(0,0,2*k+2,1) = QUCellwise[i];
		QULam[2*k+2] = LamGlobe[2*i];
		QULam[2*k+2] = LamGlobe[2*i+1]; //?make sure this parses correctly?
		QULamCellwise[i] = QULam;
	}
}

void SystemSolver::DGtoSundialsVecConversion(DGApprox delU, DGApprox delQ, Eigen::VectorXd delLambda, N_Vector& delY)
{
	if (N_VGetLength(delY) != 2*nCells*(k+1) + nCells + 1)
		throw std::invalid_argument( "Sundials Vecotor does not match size \n" );

	VectorWrapper delYVec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) );

	Eigen::VectorXd uCellCoeffs(k+1), qCellCoeffs(k+1);
	for(int i=0; i<nCells; i++)
	{
		Interval const& I = grid.gridCells[ i ];
		uCellCoeffs = delU.coeffs[I];
		qCellCoeffs = delQ.coeffs[I];
		for (int j=0; j<k+1; j++)
		{
			delYVec[i*(k+1)+j] = uCellCoeffs[j];
			delYVec[nCells*(k+1) + i*(k+1)+j] = qCellCoeffs[j];
		}
	}

	for(int i=0; i<nCells+1; i++)
	{
		delYVec[2*nCells*(k+1) + i] = delLambda[i];
	}

	//?Make sure that delY is actually being overwritten when we access delY?;
}

void SystemSolver::updateABBDForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& tempABBDBlocks, double const alpha)
{
	tempABBDBlocks = ABBDSolvers;
	std::function<double( double )> alphaF = [ = ]( double x ){ return alpha;};

	Eigen::MatrixXd X( k + 1, k + 1 );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Interval const& I( grid.gridCells[ i ] );
		u.MassMatrix( I, X, alphaF);

		Eigen::MatrixXd ABBD = tempABBDBlocks[i];
		ABBD.block( k+1, k+1, k+1, k+1 ) += X;
	}
}

void SystemSolver::solveJacEq(double const alpha, N_Vector const& g, N_Vector& delY)
{
	Eigen::VectorXd delLambda( nCells + 1 );
	DGApprox delU(grid,k), delQ(grid,k);
	std::vector< Eigen::VectorXd > g1g2_cellwise;
	Eigen::VectorXd g3_global{};

	//assemble temp cellwise ABBD blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > tempABBDBlocks{};
	updateABBDForJacSolve(tempABBDBlocks, alpha);

	// Assemble RHS g into cellwise form and solve for QU_F block
	sundialsToDGVecConversion(g, g1g2_cellwise, g3_global);
	std::vector< Eigen::VectorXd > QU_f( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Eigen::VectorXd g1g2 = g1g2_cellwise[ i ];
		//g1g2.block( k + 1, 0, k + 1, 1 ) += ( 2.0/dt ) * u.coeffs[ grid.gridCells[ i ] ]; 
		QU_f[ i ] = tempABBDBlocks[ i ].solve( g1g2 );
	}

	// Construct the RHS of K Lambda = F
	Eigen::VectorXd F( nCells + 1 );
	F = g3_global;
	for ( unsigned int i=0; i < nCells; i++ )
		F.block<2,1>( i, 0 ) += CG_cellwise[ i ] * QU_f[ i ];//?maybe subtraction?

	Eigen::FullPivLU< Eigen::MatrixXd > lu( K_global );
	delLambda = lu.solve( F );

	// Now find del u and del q to eventually find del Y
	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval const& I = grid.gridCells[ i ];
		Eigen::VectorXd QU( 2*( k + 1 ) );
		QU = QU_f[ i ] + QU_0_cellwise[ i ] * delLambda.block<2,1>( i, 0 );
		delQ.coeffs[ I ] = QU.block( 0, 0, k + 1, 1 );
		delU.coeffs[ I ] = QU.block( k + 1, 0, k + 1, 1 );
	}

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

int SystemSolver::residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	DGApprox ResQ(grid,k), ResU(grid,k);

	std::vector<Eigen::VectorXd> QULamCellwise{};		// 2(k+1)+2
	std::vector<Eigen::VectorXd> oUDotoCellwise{};		// 2(k+1)
	std::vector<Eigen::VectorXd> ResCellwise{};	// 2(k+1)

	sundialsToDGVecConversion(Y, QULamCellwise);
	sundialsToDGVecConversion(dydt, oUDotoCellwise); //?Should be all zeros except for u, possibly set in ID, make sure this is right?

	Eigen::VectorXd resQGlobal((k+1)*nCells), resUGlobal((k+1)*nCells), resLamGlobal(nCells+1);

	for ( unsigned int i = 0; i < nCells; i++ )
	{
		Eigen::VectorXd cellResidual(2*(k+1)+2);
		Eigen::MatrixXd oXoMat(2*(k+1) + 2, 2*(k+1) + 2);
		oXoMat.setZero();
		oXoMat.block(k+1, k+1, k+1, k+1) = XMats[i];
		Eigen::VectorXd RFL(2*(k+1)+2);
		RFL.setZero();
		RFL.block(0,0,2*k+2,1) = RF_cellwise[i];
		RFL[2*k+2] = L_global[i]; RFL[2*k+3] = L_global[i+1];

		cellResidual = ABCBDECGHMats[i]*QULamCellwise[i] + oXoMat*oUDotoCellwise[i] - RFL;
		
		for(int j =0; j<k+1; j++)
		{
			resQGlobal[i*(k+1)+j] = cellResidual[j];
			resUGlobal[nCells*(k+1) + i*(k+1)+j] = cellResidual[k+1 + j];
		}
		resLamGlobal[i] = cellResidual[2*k+2];
		if(i=nCells-1) resLamGlobal[nCells] = cellResidual[2*k+3];
	}

	VectorWrapper resVec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) ); //?make sure this writes into the same memore and doesn't copy
	resVec.block(0,0,nCells*(k+1),1) = resQGlobal;
	resVec.block(nCells*(k+1),0,nCells*(k+1),1) = resUGlobal;
	resVec.block(2*nCells*(k+1),0,nCells+1,1) = resLamGlobal;

	return 0;
}