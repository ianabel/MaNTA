
#include <Eigen/Dense>
#include <Eigen/Core>

#include "SystemSolver.hpp"

using ValueVector = std::vector<double>;
using Matrix = Eigen::MatrixXd;

void SystemSolver::NLqMat( Matrix& NLq, q, u, Interval I ) {
	//	[ dkappa_1dq1    dkappa_1dq2    dkappa_1dq3 ]
	//	[ dkappa_2dq1    dkappa_2dq2    dkappa_2dq3 ]
	//	[ dkappa_3dq1    dkappa_3dq2    dkappa_3dq3 ]

	auto const& x_vals = DGApprox::Integrator().abscissa();
	auto const& x_wgts = DGApprox::Integrator().weights();
	const size_t n_abscissa = x_vals.size();


	// ASSERT NLq.shape == ( nVar * ( k + 1) , nVar * ( k + 1 ) )
	// std::assert( NLq.rows() == nVar * ( k + 1 ) );
	// std::assert( NLq.cols() == nVar * ( k + 1 ) );

	NLq.setZero();

	// Phi are basis fn's
	// NLq( nVar * K + k, nVar * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

	for(int kappaVar = 0; kappaVar < nVar; kappaVar++)
	{
		std::vector<double> dSigmaFn_dq_vals( nVar );
		for ( size_t i=0; i < n_abscissa; ++i ) {
			// Pull the loop over the gaussian integration points
			// outside so we can evaluate u, q, dSigmaFn_dq once and store the values
			
			// All for loops inside here can be parallelised as they all
			// write to separate entries in NLq
			
			std::vector<double> u_vals( nVar ), q_vals( nVar );

			for ( size_t j = 0 ; j < nVar; ++j )
			{
				u_vals( j ) = u[ j ]( x_vals( i ), I );
				q_vals( j ) = q[ j ]( x_vals( i ), I );
			}

			problem->dSigmaFn_dq( kappaVar, dSigmaFn_dq_vals, u_vals, q_vals, x_vals( i ) );

			for(int qVar = 0; qVar < nVar; qVar++)
			{
				for ( Eigen::Index j=0; j < k + 1; ++j )
				{
					for ( Eigen::Index l=0; l < k + 1; ++l )
					{
						NLq( kappaVar * nVar + j, qVar * nVar + l ) += 
							x_wgts( i ) * dSigmaFn_dq_vals( i ) 
							* DGApprox::Basis().Evaluate( I, j, x_vals( i ) )
							* DGApprox::Basis().Evaluate( I, k, x_vals( i ) );
					}
				}
			}
		}
	}
}

void SystemSolver::NLuMat( Matrix& NLq, q, u, Interval I ) {
	//	[ dkappa_1du1    dkappa_1du2    dkappa_1du3 ]
	//	[ dkappa_2du1    dkappa_2du2    dkappa_2du3 ]
	//	[ dkappa_3du1    dkappa_3du2    dkappa_3du3 ]

	auto const& x_vals = DGApprox::Integrator().abscissa();
	auto const& x_wgts = DGApprox::Integrator().weights();
	const size_t n_abscissa = x_vals.size();


	// ASSERT NLu.shape == ( nVar * ( k + 1) , nVar * ( k + 1 ) )
	// std::assert( NLu.rows() == nVar * ( k + 1 ) );
	// std::assert( NLu.cols() == nVar * ( k + 1 ) );

	NLu.setZero();

	// Phi are basis fn's
	// NLu( nVar * K + k, nVar * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

	for(int kappaVar = 0; kappaVar < nVar; kappaVar++)
	{
		std::vector<double> dSigmaFn_du_vals( nVar );
		for ( size_t i=0; i < n_abscissa; ++i ) {
			// Pull the loop over the gaussian integration points
			// outside so we can evaluate u, q, dSigmaFn_du once and store the values
			
			// All for loops inside here can be parallelised as they all
			// write to separate entries in NLu
			
			std::vector<double> u_vals( nVar ), q_vals( nVar );

			for ( size_t j = 0 ; j < nVar; ++j )
			{
				u_vals( j ) = u[ j ]( x_vals( i ), I );
				q_vals( j ) = q[ j ]( x_vals( i ), I );
			}

			problem->dSigmaFn_du( kappaVar, dSigmaFn_du_vals, u_vals, q_vals, x_vals( i ) );

			for(int uVar = 0; uVar < nVar; uVar++)
			{
				for ( Eigen::Index j=0; j < k + 1; ++j )
				{
					for ( Eigen::Index l=0; l < k + 1; ++l )
					{
						NLu( kappaVar * nVar + j, uVar * nVar + l ) += 
							x_wgts( i ) * dSigmaFn_du_vals( i ) 
							* DGApprox::Basis().Evaluate( I, j, x_vals( i ) )
							* DGApprox::Basis().Evaluate( I, k, x_vals( i ) );
					}
				}
			}
		}
	}
}


