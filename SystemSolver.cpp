#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <toml.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(Grid const& Grid, unsigned int polyNum, double Dt, TransportSystem *transpSystem )
	: grid(Grid), k(polyNum), nCells(Grid.getNCells()), dt(Dt), problem( transpSystem )
{
	nVar = transpSystem->getNVars();
	initialiseMatrices();
	initialised = true;
}

void SystemSolver::setInitialConditions( N_Vector& Y , N_Vector& dYdt )
{

	y.   Map( N_VGetArrayPointer( Y ) );
	dydt.Map( N_VGetArrayPointer( dYdt ) );

	resetCoeffs();

	y.AssignU( problem->InitialValue );
	y.AssignQ( problem->InitialDerivative );

	y.EvaluateLambda();

	ApplyBCs(y); // If dirichlet, overwrite with those boundary conditions

	y.AssignSigma( problem->SigmaFn );

	for(int var = 0; var < nVar; var++)
	{
		//Solver For dudt with dudt = X^-1( -B*Sig - D*U - E*Lam + F )
		Eigen::Vector2d lamCell;
		for ( unsigned int i=0; i < nCells; i++ )
		{
			Interval I = grid[ i ];

			//Evaluate Source Function
			Eigen::VectorXd S_cellwise(k+1);
			S_cellwise.setZero();

			auto const& x_vals = DGApprox::Integrator().abscissa();
			auto const& x_wgts = DGApprox::Integrator().weights();
			const size_t n_abscissa = x_vals.size();

			S_cellwise.setZero();
			for ( size_t i=0; i < n_abscissa; ++i ) {
				std::vector<double> u_vals( nVar ), q_vals( nVar ), sigma_vals( nVar );
				for ( size_t j = 0 ; j < nVar; ++j ) {
					u_vals( j ) = u[ j ]( x_vals( i ), I );
					q_vals( j ) = q[ j ]( x_vals( i ), I );
					sigma_vals( j ) = sigma[ j ]( x_vals( i ), I );
				}
				double sourceVal = problem->Sources( var, u_vals, q_vals, sigma_vals, x_vals( i ), 0 );
				for ( Eigen::Index j = 0; j < k+1; j++ )
					S_cellwise( j ) += x_wgts( i ) * sourceVal * LegendreBasis::Phi( I, j )( x_vals( i ) );
			}
			
			auto cTInv = Eigen::FullPivLU< Eigen::MatrixXd >(C_cellwise[i].transpose());
			lamCell[0] = lambda.value()[var*(nCells+1) + i]; lamCell[1] = lambda.value()[var*(nCells+1) + i+1];
			//dudt.coeffs[ var ][ i ].second.setZero();
			auto const& sigma_vec = y.sigma[ var ].coeffs[ i ].second;
			auto const& u_vec     = y.u[ var ]    .coeffs[ i ].second;
			dydt.u[ var ].coeffs[ i ].second = 
				XMats[i].block(var*(k+1), var*(k+1), k+1, k+1).inverse()*(
						- B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*sigma_vec
						- D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*u_vec
						- E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell
						+ RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) - S_cellwise);
			dydt.q[ var ].coeffs[ i ].second = 
				<cellwise derivative matrix> * dydt.u[ var ].coeffs[ i ].second;
		}
		auto dSigmaFn_dt = [ problem, &dydt ]( Index i, const Values &u, const Values &q, Position x, Time t ) {
			Values dSigma_du_vals( nVars );
			Values dSigma_dq_vals( nVars );

			problem->dSigmaFn_du( i, dSigmaFn_du_vals, u, q, x, t );
			problem->dSigmaFn_dq( i, dSigmaFn_du_vals, u, q, x, t );

			double sigmaDot = 0;
			for ( Index j=0; j < nVars; ++j ) {
				sigmaDot += dydt.u[ j ]( x ) * dSigmaFn_du_vals( j ) + dydt.q[ j ]( x ) * dSigma_dq_vals( j );
			}

			return sigmaDot;
		};
		dydt.AssignSigma( dSigmaFn_dt );
	}
}

void ApplyBCs( DGSoln &Y )
{
	for ( Index i=0; i < nVars; ++i )
	{
		if ( problem->isLowerBoundaryDirichlet( var ) ) {
			Y.lambda( i )( 0 ) = problem->LowerBoundary( var, 0 );
		}

		if ( problem->isUpperBoundaryDirichlet( var ) ) {
			Y.lambda( i )( grid.getNCells() ) = problem->UpperBoundary( var, 0 );
		}
	}
}

void SystemSolver::seta_fns()
{
	if(plasma)
	{
		for(int var = 0; var < nVar; var++)
		{
			a_fn.push_back(plasma->getVariable(var).a_fn);
		}
	}
	//TO DO: this is to be retired
	else
	{
		for(int var = 0; var < nVar; var++)
		{
			a_fn.push_back([ = ]( double R ){ return 1.0;});
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
		C.setZero();
		D.setZero();
		E.setZero();
		Interval const& I( grid.gridCells[ i ] );
		for(int var=0; var<nVar; var++)
		{
			Avar.setZero();
			Bvar.setZero();
			Dvar.setZero();
			// A_ij = ( phi_j, phi_i )
			u.MassMatrix( I, Avar);
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
			// ( 0    A    B^T )    [ C^T ]
			// ( B    0     D )     [  E  ]
			// ( A   NLu   NLq )^-1 [  0  ]
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
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
					RF_cellwise[ i ]( j + var*(k+1) ) += -u.Basis.phi( I, j )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l, 0.0, var );
					// - < ( c.n - tau ) g_D, w >
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l, 0.0, var );
				}
			}
		}

		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
					RF_cellwise[ i ]( j + var*(k+1) ) += -u.Basis.phi( I, j )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u, 0.0, var );
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u, 0.0, var );
				}
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
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound, 0.0, var );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound, 0.0, var );
		}

		Eigen::MatrixXd X(nVar*(k+1), nVar*(k+1));
		X.setZero();
		for(int var = 0; var < nVar; var++)
		{
			Eigen::MatrixXd Xvar( k + 1, k + 1 );
			u.MassMatrix( I, Xvar, a_fn[var]);
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

// Memory Layout for a sundials Y is, if i indexes the components of u / q / sigma
// Y = [ sigma[ cell0, i=0 ], ..., sigma[ cell0, i= nVar - 1], q[ cell0, i = 0 ], ..., q[ cell0, i = nVar-1 ], u[ cell0, i = 0 ], .. u[ cell0, i = nVar - 1], sigma[ cell1, i=0 ], .... , u[ cellN-1, i = nVar - 1 ], Lambda[ cell0, i=0 ],.. ]

void SystemSolver::mapDGtoSundials(DGVector& sigma, DGVector& q, DGVector& u, Eigen::Map<Eigen::VectorXd>& lam, realtype* Y)
{
	sigma.clear(); sigma.reserve( nVar );
	q.clear();     q.reserve( nVar );
	u.clear();     u.reserve( nVar );
	for(int var = 0; var < nVar; var++)
	{
		sigma[ var ].emplace_back( grid, k, ( Y +                var*(k+1) ), 3*nVar*( k+1 ) );
		q[ var ]    .emplace_back( grid, k, ( Y +   nVar*(k+1) + var*(k+1) ), 3*nVar*( k+1 ) );
		u[ var ]    .emplace_back( grid, k, ( Y + 2*nVar*(k+1) + var*(k+1) ), 3*nVar*( k+1 ) );

		new (&lam) VectorWrapper( Y + nVar*(nCells)*(3*k+3), nVar*(nCells+1) );
	}
}

void SystemSolver::mapDGtoSundials(DGVector& u, realtype* Y)
{

	u.clear(); u.reserve( nVar );
	for(int var = 0; var < nVar; var++)
	{
		u.emplace_back( grid, Y + 2*nVar*( k + 1 ) + var * ( k + 1 ) , 3*nVar*( k+1 ) );
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
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_0 ) * phi_j( x_0 ) * ( n_x = -1 ) 
					RF_cellwise[ i ]( j + var*(k+1) ) += -u.Basis.phi( I, j )( I.x_l ) * ( -1 ) * BCs->g_D( I.x_l, t, var );
					// - < ( c.n - tau ) g_D, w >
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_l ) * ( -c_fn( I.x_l ) - tau( I.x_l ) ) * BCs->g_D( I.x_l, t, var );
				}
			}
		}


		if ( I.x_u == BCs->UpperBound && BCs->isUBoundDirichlet )
		{
			for(int var = 0; var < nVar; var++)
			{
				for ( Eigen::Index j = 0; j < k+1; j++ )
				{
					// < g_D , v . n > ~= g_D( x_1 ) * phi_j( x_1 ) * ( n_x = +1 ) 
					RF_cellwise[ i ]( j+var*(k+1)) += -u.Basis.phi( I, j )( I.x_u ) * ( +1 ) * BCs->g_D( I.x_u, t, var );
					RF_cellwise[ i ]( nVar*(k + 1) + j + var*(k+1) ) -= u.Basis.phi( I, j )( I.x_u ) * (  c_fn( I.x_u ) - tau( I.x_u ) ) * BCs->g_D( I.x_u, t, var );
				}
			}
		}


		for(int var = 0; var < nVar; var++)
		{
			if ( I.x_l == BCs->LowerBound && /* is b.d. Neumann at lower boundary */ !BCs->isLBoundDirichlet )
				L_global( var*(nCells+1) + i )     += BCs->g_N( BCs->LowerBound, t, var );
			if ( I.x_u == BCs->UpperBound && /* is b.d. Neumann at upper boundary */ !BCs->isUBoundDirichlet )
				L_global( var*(nCells+1) + i + 1 ) += BCs->g_N( BCs->UpperBound, t, var );
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
	for(auto term : resTerms)
	{
		res += term;
		for(int i = 0; i< term.size(); i++)
		{
			if( std::abs(term[i]) > maxTerms[i]) maxTerms[i] = std::abs(term[i]);
		}
	}
	for(int i = 0; i< resTerms[0].size(); i++)
	{
		if(res[i] < maxTerms[i]*10e-10) res[i] = 0.0;
	}
	return res;
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

void SystemSolver::updateMForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& MXsolvers, double const alpha, DGApprox& delSig, DGApprox& delQ, DGApprox& delU)
{
	MXsolvers.clear();
	DGApprox newU(grid, k), newQ(grid, k), newSig(grid, k);
	double qMemBlock[nVar*nCells*(k+1)], uMemBlock[nVar*nCells*(k+1)], sigMemBlock[nVar*nCells*(k+1)]; //??need to assign memory block as DGAs don't own memory
	newQ.setCoeffsToArrayMem(qMemBlock, nVar, nCells, grid);
	newU.setCoeffsToArrayMem(uMemBlock, nVar, nCells, grid);
	newSig.setCoeffsToArrayMem(sigMemBlock, nVar, nCells, grid);

	//We want to base our matrix off the current guess
	newQ.sum(q, delQ);
	newU.sum(u, delU);
	newSig.sum(sig, delSig);


	Eigen::MatrixXd X( nVar*(k + 1), nVar*(k + 1) );
	Eigen::MatrixXd NLq(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd NLu(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd Ssig(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd Sq(nVar*(k+1), nVar*(k+1));
	Eigen::MatrixXd Su(nVar*(k+1), nVar*(k+1));
	for ( unsigned int i = 0; i < nCells; i++ )
	{
		X.setZero();
		NLq.setZero();
		NLu.setZero();
		Ssig.setZero();
		Sq.setZero();
		Su.setZero();

		Interval const& I( grid.gridCells[ i ] );
		Eigen::MatrixXd MX(3*nVar*(k+1),3*nVar*(k+1));
		MX.setZero();
		MX = MBlocks[i];
		//X matrix
		for(int var = 0; var < nVar; var++)
		{
		std::function<double( double )> alphaF = [ = ]( double x ){ return alpha*a_fn[var](x);};
			Eigen::MatrixXd Xsubmat( (k + 1), (k + 1) );
			Xsubmat.setZero();
			u.MassMatrix( I, Xsubmat, alphaF);
			X.block(var*(k+1), var*(k+1), k+1, k+1) = Xsubmat;
		}
		MX.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) += X;

		//NLq Matrix
		NLqMat( NLq, newQ, newU, I);
		MX.block( 2*nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1)) = NLq;

		//NLu Matrix
		NLuMat( NLu, newQ, newU, I);
		MX.block( 2*nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1)) = NLu;

		//S_sig Matrix
		sourceObj->setdFdsigMat( Ssig, newSig, newQ, newU, I);
		MX.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = Ssig;

		//S_q Matrix
		sourceObj->setdFdqMat( Sq, newSig, newQ, newU, I);
		MX.block( nVar*(k+1), nVar*(k+1), nVar*(k+1), nVar*(k+1) ) = Sq;

		//S_u Matrix
		sourceObj->setdFduMat( Su, newSig, newQ, newU, I);
		MX.block( nVar*(k+1), 2*nVar*(k+1), nVar*(k+1), nVar*(k+1) ) += Su;

		//if(i==0) std::cerr << MX << std::endl << std::endl;
		//if(i==0)std::cerr << MX.inverse() << std::endl << std::endl;

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

	VectorWrapper delYVec( N_VGetArrayPointer( delY ), N_VGetLength( delY ) );
	mapDGtoSundials(delSig, delQ, delU, delLambda, N_VGetArrayPointer( delY ));
	delYVec.setZero();

	//assemble temp cellwise M blocks
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > factorisedM{};
	updateMForJacSolve(factorisedM, alpha, delSig, delQ, delU);

	// Assemble RHS g into cellwise form and solve for SQU blocks
	mapDGtoSundials(g1g2g3_cellwise, g4, N_VGetArrayPointer( g ));

	std::vector< Eigen::VectorXd > SQU_f( nCells );
	std::vector< Eigen::MatrixXd > SQU_0( nCells );
	for ( unsigned int i = 0; i < nCells; i++ )
	{  
		Interval const& I( grid.gridCells[ i ] );

		//SQU_f
		Eigen::VectorXd g1g2g3 = g1g2g3_cellwise[ i ];

		SQU_f[ i ] = factorisedM[ i ].solve( g1g2g3 );


		//SQU_0
		Eigen::MatrixXd CE = CEBlocks[ i ];
		SQU_0[ i ] = factorisedM[ i ].solve( CE );
		//std::cerr << SQU_0[i] << std::endl << std::endl;
		//std::cerr << CE << std::endl << std::endl;


		Eigen::MatrixXd K_cell(nVar*2,nVar*2);
		K_cell.setZero();
		K_cell = H_cellwise[i] - CG_cellwise[ i ] * SQU_0[i];

		//std::cerr << SQU_f[i] << std::endl << std::endl;
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
	//std::cerr << delLambda << std::endl << std::endl;

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
	std::ofstream dyfile;
	dyfile.open("dyfile.txt");
	print(dyfile, 0.0, 200, 0, delY);
	//delSig.printCoeffs(0);
	//delU.printCoeffs(0);
	//delQ.printCoeffs(0);
	//std::cerr << delLambda << std::endl << std::endl;
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	auto k = system->k;
	auto grid(system->grid);
	auto nCells = system->nCells;
	auto c_fn = system->getcfn();
	auto nVar = system->nVar;

	system->updateBoundaryConditions(tres);

	DGApprox tempSig(grid, k), tempU(grid, k), tempQ(grid, k), tempdudt(grid, k);
	DGApprox tempdSigdt(grid, k), tempdQdt(grid, k);
	DGApprox res1(grid, k), res2(grid, k), res3(grid, k) ;
	DGApprox tempKappa(grid, k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> tempLambda(memBlock, nVar*(nCells+1)), res4(memBlock, nVar*(nCells+1)), lam(memBlock, nVar*(nCells+1)), tempdLamdt(memBlock, nVar*(nCells+1));

	system->mapDGtoSundials(tempSig, tempQ, tempU, tempLambda, N_VGetArrayPointer( Y ));
	//system->mapDGtoSundials(tempdSigdt, tempdQdt, tempdudt, tempdLamdt, N_VGetArrayPointer( Y ));
	system->mapDGtoSundials(tempdudt, N_VGetArrayPointer( dydt )); 
	system->mapDGtoSundials(res1, res2, res3, res4, N_VGetArrayPointer( resval )); 
	VectorWrapper resVec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
	resVec.setZero();

	//Solve for Lambda with Lam = (H^T)^-1*[ -C*Sig - G*U + L ] 
	Eigen::VectorXd CsGuL_global(nVar*(nCells+1));
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
	lam = system->H_global.solve( CsGuL_global );
	for(int var = 0; var < nVar; var++)
	{
		if( problem->isLowerBoundaryDirichlet( var ) ) {
			lam[var*(nCells+1)]        = problem->LowerBoundary( var, static_cast<double>(tres) );
			tempLambda[var*(nCells+1)] = problem->LowerBoundary( var, static_cast<double>(tres) );
		}
		if( problem->isUpperBoundaryDirichlet( var ) ) {
			lam[nCells+var*(nCells+1)]        = problem->UpperBoundary( var, static_cast<double>(tres) );
			tempLambda[nCells+var*(nCells+1)] = problem->UpperBoundary( var, static_cast<double>(tres) );
		}
	}
	res4 = -tempLambda + lam;

	for ( unsigned int i=0; i < nCells; i++ )
	{
		Interval I = grid.gridCells[ i ];
		Eigen::VectorXd lamCell(2*nVar);

		for(int var = 0; var < nVar; var++)
		{
			lamCell[2*var] = tempLambda[var*(nCells+1) + i]; lamCell[2*var + 1] = tempLambda[var*(nCells+1) + i+1];
		}

		//length = nVar*(k+1)
		for(int var = 0; var < nVar; var++)
		{
			std::function< double (double) > kappaFunc = [ = ]( double x ) { return problem->SigmaFn( var, tempU, tempQ, x, 0 ); };
			std::function< double (double) > sourceFunc = [ = ]( double x ) { return system->getSourceObj()->getSourceFunc(var)( x, tempSig, tempQ, tempU); };

			//Evaluate Diffusion Function
			Eigen::VectorXd kappa_cellwise(k+1);
			kappa_cellwise.setZero();
			for ( Eigen::Index j = 0; j < k+1; j++ )
				kappa_cellwise( j ) = tempU.CellProduct( I, kappaFunc, tempU.Basis.phi( I, j ) );

			//Evaluate Source Function
			Eigen::VectorXd S_cellwise(k+1);
			S_cellwise.setZero();
			for ( Eigen::Index j = 0; j < k+1; j++ )
				S_cellwise( j ) = tempU.CellProduct( I, sourceFunc, tempU.Basis.phi( I, j%(k+1) ) );

			res1.coeffs[ var ][ i ].second = -system->A_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempQ.coeffs[var][i].second - system->B_cellwise[i].transpose().block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->C_cellwise[i].transpose().block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( var*(k+1), 0, k + 1, 1 );
			res2.coeffs[ var ][ i ].second = system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempSig.coeffs[ var ][ i ].second + system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second + system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0) - system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ) + S_cellwise + system->XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempdudt.coeffs[ var ][ i ].second;
			res3.coeffs[ var ][ i ].second = tempSig.coeffs[ var ][ i ].second + kappa_cellwise;
			//res1.coeffs[ var ][ i ].second = system->resEval({(-1.0)*system->A_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempQ.coeffs[var][i].second, (-1.0)*system->B_cellwise[i].transpose().block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second, system->C_cellwise[i].transpose().block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0), - system->RF_cellwise[ i ].block( var*(k+1), 0, k + 1, 1 )});
			//res2.coeffs[ var ][ i ].second = system->resEval({system->B_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempSig.coeffs[ var ][ i ].second, system->D_cellwise[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempU.coeffs[ var ][ i ].second, system->E_cellwise[i].block(var*(k+1), var*2, k+1, 2)*lamCell.block<2,1>(var*2,0), - system->RF_cellwise[ i ].block( nVar*(k + 1) + var*(k+1), 0, k + 1, 1 ), S_cellwise, system->XMats[i].block(var*(k+1), var*(k+1), k+1, k+1)*tempdudt.coeffs[ var ][ i ].second});
			//res3.coeffs[ var ][ i ].second = system->resEval({tempSig.coeffs[ var ][ i ].second, kappa_cellwise});
			//if(var == 2) std::cerr << S_cellwise << std::endl << std::endl;
		}
	}

	//system->print(std::cerr, tres, 21, 0);

	VectorWrapper Vec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) );
	VectorWrapper ypVec( N_VGetArrayPointer( dydt ), N_VGetLength( dydt ) );

	//res1.printCoeffs(0);
	//res1.printCoeffs(1);
	//res1.printCoeffs(2);

	//std::cerr << res4 << std::endl << std::endl;

	Eigen::Index maxloc, minloc;
 	std::cerr << Vec.norm() << "	" << "	" << Vec.maxCoeff(&maxloc) << "	" << Vec.minCoeff(&minloc) << "	" << maxloc << "	" << minloc << "	" << tres << std::endl << std::endl;
	system->total_steps++;

	std::ofstream Yfile;
	Yfile.open("yfile.txt");

	if(system->total_steps%100 == 99)
	{
		std::ofstream resfile0;
		std::ofstream resfile1;
		std::ofstream resfile4;
		resfile0.open("res0.txt", std::ofstream::app);
		resfile1.open("res1.txt", std::ofstream::app);
		resfile4.open("res4.txt", std::ofstream::app);
		system->print(resfile0, tres, 200, 0, resval);
		if(nVar > 1) system->print(resfile1, tres, 200, 1, resval);
		resfile4 << res4 << std::endl << std::endl;
	}

	if(system->isTesting())
	{	
		VectorWrapper residualVec( N_VGetArrayPointer( resval ), N_VGetLength( resval ) );
		system->resNorm = residualVec.norm();
	}

	std::ofstream file;
	file.open("time.txt", std::ofstream::app);
	file << tres << std::endl << std::endl;

	return 0;
}

void SystemSolver::print( std::ostream& out, double t, int nOut, int var )
{
	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut-1 ) );
		out << x << "\t" << EvalCoeffs( u.Basis, u.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, q.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, sig.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dudt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dqdt.coeffs, x, var ) << "\t" << EvalCoeffs( u.Basis, dsigdt.coeffs, x, var ) << std::endl;
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
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut-1 ) );
		out << x << "\t" << EvalCoeffs( U.Basis, U.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Q.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Sig.coeffs, x, var ) << std::endl;
	}
	out << std::endl;
}

void SystemSolver::print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY, N_Vector& tempRes )
{
	DGApprox Sig(grid,k), Q(grid,k), U(grid,k);
	DGApprox resSig(grid,k), resQ(grid,k), resU(grid,k);
	double memBlock[nVar*(nCells+1)];
	Eigen::Map<Eigen::VectorXd> Lambda(memBlock, nVar*(nCells+1));
	Eigen::Map<Eigen::VectorXd> resLambda(memBlock, nVar*(nCells+1));

	mapDGtoSundials(Sig, Q, U, Lambda, N_VGetArrayPointer( tempY ));
	mapDGtoSundials(resSig, resQ, resU, Lambda, N_VGetArrayPointer( tempRes ));

	out << "# t = " << t << std::endl;
	for ( int i=0; i<nOut; ++i )
	{
		double x = BCs->LowerBound + ( BCs->UpperBound - BCs->LowerBound ) * ( static_cast<double>( i )/( nOut-1 ) );
		out << x << "\t" << EvalCoeffs( U.Basis, U.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Q.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, Sig.coeffs, x, var )  << "\t" << EvalCoeffs( U.Basis, resU.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, resQ.coeffs, x, var ) << "\t" << EvalCoeffs( U.Basis, resSig.coeffs, x, var ) << std::endl;
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

