
#include <cassert>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Types.hpp"
#include "SystemSolver.hpp"



void SystemSolver::NLqMat( Matrix& NLq, DGSoln const &Y, Interval I ) {
	//	[ dkappa_1dq1    dkappa_1dq2    dkappa_1dq3 ]
	//	[ dkappa_2dq1    dkappa_2dq2    dkappa_2dq3 ]
	//	[ dkappa_3dq1    dkappa_3dq2    dkappa_3dq3 ]

	DerivativeSubMatrix( NLq, &TransportSystem::dSigmaFn_dq, Y, I );
}

void SystemSolver::NLuMat( Matrix& NLu, DGSoln const& Y, Interval I ) {
	//	[ dkappa_1du1    dkappa_1du2    dkappa_1du3 ]
	//	[ dkappa_2du1    dkappa_2du2    dkappa_2du3 ]
	//	[ dkappa_3du1    dkappa_3du2    dkappa_3du3 ]

	DerivativeSubMatrix( NLu, &TransportSystem::dSigmaFn_du, Y, I );
}

// Sets matrices of the form
//	[ dX_1dZ1    dX_1dZ2    dX_1dZ3 ]
//	[ dX_2dZ1    dX_2dZ2    dX_2dZ3 ]
//	[ dX_3dZ1    dX_3dZ2    dX_3dZ3 ]
//
// where X is a sigma function or a source function and Z is one of u, q, or sigma.
 
void SystemSolver::DerivativeSubMatrix( Matrix& mat, void ( TransportSystem::*dX_dZ )( Index, Values&, const Values&, const Values&, Position, double ), DGSoln const& Y, Interval I )
{
	auto const& x_vals = DGApprox::Integrator().abscissa();
	auto const& x_wgts = DGApprox::Integrator().weights();
	const size_t n_abscissa = x_vals.size();

	// ASSERT mat.shape == ( nVars * ( k + 1) , nVars * ( k + 1 ) )
	assert( mat.rows() == nVars * ( k + 1 ) );
	assert( mat.cols() == nVars * ( k + 1 ) );

	mat.setZero();

	// Phi are basis fn's
	// M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

	for ( Index XVar = 0; XVar < nVars; XVar++ )
	{
		std::vector<double> dX_dZ_vals1( nVars );
		std::vector<double> dX_dZ_vals2( nVars );
		for ( size_t i=0; i < n_abscissa; ++i ) {
			// Pull the loop over the gaussian integration points
			// outside so we can evaluate u, q, dX_dZ once and store the values
			
			// All for loops inside here can be parallelised as they all
			// write to separate entries in mat
			
			std::vector<double> u_vals1( nVars ), q_vals1( nVars );
			std::vector<double> u_vals2( nVars ), q_vals2( nVars );

			double wgt = x_wgts[ i ]*( I.h()/2.0 );

			double y_plus  = I.x_l + ( 1.0 + x_vals[ i ] )*( I.h()/2.0 );
			double y_minus = I.x_l + ( 1.0 - x_vals[ i ] )*( I.h()/2.0 );

			for ( size_t j = 0 ; j < nVars; ++j )
			{
				u_vals1[ j ] = Y.u( j )( y_plus, I );
				q_vals1[ j ] = Y.q( j )( y_plus, I );
				u_vals2[ j ] = Y.u( j )( y_minus, I );
				q_vals2[ j ] = Y.q( j )( y_minus, I );
			}

			( problem->*dX_dZ )( XVar, dX_dZ_vals1, u_vals1, q_vals1, y_plus, 0.0 );
			( problem->*dX_dZ )( XVar, dX_dZ_vals2, u_vals2, q_vals2, y_minus, 0.0 );

			for(Index ZVar = 0; ZVar < nVars; ZVar++)
			{
				for ( Index j=0; j < k + 1; ++j )
				{
					for ( Index l=0; l < k + 1; ++l )
					{
						mat( XVar * nVars + j, ZVar * nVars + l ) +=
							wgt * dX_dZ_vals1[ ZVar ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, k, y_plus );
						mat( XVar * nVars + j, ZVar * nVars + l ) +=
							wgt * dX_dZ_vals2[ ZVar ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, k, y_minus );
					}
				}
			}
		}
	}
}

void SystemSolver::dSourcedq_Mat(Eigen::MatrixXd& dSourcedqMatrix, DGSoln const& Y, Interval I)
{
	DerivativeSubMatrix( dSourcedqMatrix, &TransportSystem::dSources_dq, Y, I );
}

void SystemSolver::dSourcedu_Mat(Eigen::MatrixXd& dSourceduMatrix, DGSoln const& Y, Interval I)
{
	DerivativeSubMatrix( dSourceduMatrix, &TransportSystem::dSources_du, Y, I );
}

void SystemSolver::dSourcedsigma_Mat(Eigen::MatrixXd& dSourcedsigmaMatrix, DGSoln const& Y, Interval I )
{
	DerivativeSubMatrix( dSourcedsigmaMatrix, &TransportSystem::dSources_dsigma, Y, I );
}



