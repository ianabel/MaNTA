
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
 
void SystemSolver::DerivativeSubMatrix( Matrix& mat, void ( TransportSystem::*dX_dZ )( Index, Values&, const State&, Position, double ), DGSoln const& Y, Interval I )
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
		Values dX_dZ_vals1( nVars );
		Values dX_dZ_vals2( nVars );
		for ( size_t i=0; i < n_abscissa; ++i ) {
			// Pull the loop over the gaussian integration points
			// outside so we can evaluate u, q, dX_dZ once and store the values
			
			// All for loops inside here can be parallelised as they all
			// write to separate entries in mat
			
			double wgt = x_wgts[ i ]*( I.h()/2.0 );

			double y_plus  = I.x_l + ( 1.0 + x_vals[ i ] )*( I.h()/2.0 );
			double y_minus = I.x_l + ( 1.0 - x_vals[ i ] )*( I.h()/2.0 );

			State Y_plus = Y.eval( y_plus ), Y_minus = Y.eval( y_minus );

			( problem->*dX_dZ )( XVar, dX_dZ_vals1, Y_plus, y_plus, 0.0 );
			( problem->*dX_dZ )( XVar, dX_dZ_vals2, Y_minus, y_minus, 0.0 );

			for(Index ZVar = 0; ZVar < nVars; ZVar++)
			{
				for ( Index j=0; j < k + 1; ++j )
				{
					for ( Index l=0; l < k + 1; ++l )
					{
						mat( XVar * ( k + 1 ) + j, ZVar * ( k + 1 ) + l ) +=
							wgt * dX_dZ_vals1[ ZVar ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, l, y_plus );
						mat( XVar * ( k + 1 ) + j, ZVar * ( k + 1 ) + l ) +=
							wgt * dX_dZ_vals2[ ZVar ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, l, y_minus );
					}
				}
			}
		}
	}
}

void SystemSolver::dSourcedq_Mat( Matrix& dSourcedqMatrix, DGSoln const& Y, Interval I)
{
	DerivativeSubMatrix( dSourcedqMatrix, &TransportSystem::dSources_dq, Y, I );
}

void SystemSolver::dSourcedu_Mat( Matrix& dSourceduMatrix, DGSoln const& Y, Interval I)
{
	DerivativeSubMatrix( dSourceduMatrix, &TransportSystem::dSources_du, Y, I );
}

void SystemSolver::dSourcedsigma_Mat( Matrix& dSourcedsigmaMatrix, DGSoln const& Y, Interval I )
{
	DerivativeSubMatrix( dSourcedsigmaMatrix, &TransportSystem::dSources_dsigma, Y, I );
}

void SystemSolver::dSources_dScalars_Mat( Matrix& mat, DGSoln const& Y, Interval I )
{
	auto const& x_vals = DGApprox::Integrator().abscissa();
	auto const& x_wgts = DGApprox::Integrator().weights();
	const size_t n_abscissa = x_vals.size();

	// ASSERT mat.shape == ( nVars * ( k + 1) , nScalars )
	assert( mat.rows() == nVars * ( k + 1 ) );
	assert( mat.cols() == nScalars );

	mat.setZero();

	// Phi are basis fn's
	// M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

	for ( Index XVar = 0; XVar < nVars; XVar++ )
	{
		Values dSdS_vals1( nVars );
		Values dSdS_vals2( nVars );
		for ( size_t i=0; i < n_abscissa; ++i ) {
			// Pull the loop over the gaussian integration points
			// outside so we can evaluate u, q, dX_dZ once and store the values
			
			// All for loops inside here can be parallelised as they all
			// write to separate entries in mat
			
			double wgt = x_wgts[ i ]*( I.h()/2.0 );

			double y_plus  = I.x_l + ( 1.0 + x_vals[ i ] )*( I.h()/2.0 );
			double y_minus = I.x_l + ( 1.0 - x_vals[ i ] )*( I.h()/2.0 );

			State Y_plus = Y.eval( y_plus ), Y_minus = Y.eval( y_minus );

			problem->dSources_dScalars( XVar, dSdS_vals1, Y_plus, y_plus, jt );
			problem->dSources_dScalars( XVar, dSdS_vals2, Y_minus, y_minus, jt );

			for(Index iScalar = 0; iScalar < nScalars; iScalar++)
			{
				for ( Index j=0; j < k + 1; ++j )
				{
					mat( XVar * ( k + 1 ) + j, iScalar ) +=
						wgt * dSdS_vals1[ iScalar ] * LegendreBasis::Evaluate( I, j, y_plus );
					mat( XVar * ( k + 1 ) + j, iScalar ) +=
						wgt * dSdS_vals2[ iScalar ] * LegendreBasis::Evaluate( I, j, y_minus );
				}
			}
		}
	}
}

void SystemSolver::dSourcedPhi_Mat( Matrix& mat, DGSoln const& Y, Interval I )
{
	auto const& x_vals = DGApprox::Integrator().abscissa();
	auto const& x_wgts = DGApprox::Integrator().weights();
	const size_t n_abscissa = x_vals.size();

	// ASSERT mat.shape == ( nVars * ( k + 1) , nAux * ( k + 1 ) )
	assert( mat.rows() == nVars * ( k + 1 ) );
	assert( mat.cols() == nAux  * ( k + 1 ) );

	mat.setZero();

	// Phi are basis fn's
	// M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

	for ( Index Var = 0; Var < nVars; Var++ )
	{
		Values dS_dPhi_vals1( nVars );
		Values dS_dPhi_vals2( nVars );
		for ( size_t i=0; i < n_abscissa; ++i ) {
			// Pull the loop over the gaussian integration points
			// outside so we can evaluate u, q, dX_dZ once and store the values
			
			// All for loops inside here can be parallelised as they all
			// write to separate entries in mat
			
			double wgt = x_wgts[ i ]*( I.h()/2.0 );

			double y_plus  = I.x_l + ( 1.0 + x_vals[ i ] )*( I.h()/2.0 );
			double y_minus = I.x_l + ( 1.0 - x_vals[ i ] )*( I.h()/2.0 );

			State Y_plus = Y.eval( y_plus ), Y_minus = Y.eval( y_minus );

			( problem->dSources_dPhi )( Var, dS_dPhi_vals1, Y_plus, y_plus, jt );
			( problem->dSources_dPhi )( Var, dS_dPhi_vals2, Y_minus, y_minus, jt );

			for(Index Aux = 0; Aux < nAux; Aux++)
			{
				for ( Index j=0; j < k + 1; ++j )
				{
					for ( Index l=0; l < k + 1; ++l )
					{
						mat( Var * ( k + 1 ) + j, Aux * ( k + 1 ) + l ) +=
							wgt * dS_dPhi_vals1[ Aux ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, l, y_plus );
						mat( Var * ( k + 1 ) + j, Aux * ( k + 1 ) + l ) +=
							wgt * dS_dPhi_vals2[ Aux ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, l, y_minus );
					}
				}
			}
		}
	}
}

void SystemSolver::dAux_Mat( Eigen::Ref<Matrix> mat, DGSoln const& Y, Interval I )
{
  auto const& x_vals = DGApprox::Integrator().abscissa();
  auto const& x_wgts = DGApprox::Integrator().weights();
  const size_t n_abscissa = x_vals.size();

  // Assert Mat.shape == ( nAux * ( k + 1 ), ( 3 * nVars + nAux ) * ( k + 1 ) )
  assert( mat.rows() == nAux * ( k + 1 ) );
  assert( mat.cols() == ( 3*nVars + nAux ) * ( k + 1 ) );

  mat.setZero();

  for ( Index Aux = 0; Aux < nAux; Aux++ )
  {
    Values dG_du_vals1( nVars );
    Values dG_du_vals2( nVars );
    Values dG_dq_vals1( nVars );
    Values dG_dq_vals2( nVars );
    Values dG_dsigma_vals1( nVars );
    Values dG_dsigma_vals2( nVars );
    Values dG_dPhi_vals1( nAux );
    Values dG_dPhi_vals2( nAux );
    for ( size_t i=0; i < n_abscissa; ++i ) {
      // Pull the loop over the gaussian integration points
      // outside so we can evaluate u, q, dX_dZ once and store the values

      // All for loops inside here can be parallelised as they all
      // write to separate entries in mat

      double wgt = x_wgts[ i ]*( I.h()/2.0 );

      double y_plus  = I.x_l + ( 1.0 + x_vals[ i ] )*( I.h()/2.0 );
      double y_minus = I.x_l + ( 1.0 - x_vals[ i ] )*( I.h()/2.0 );

      State Y_plus = Y.eval( y_plus ), Y_minus = Y.eval( y_minus );

      State dG1( nVars, nScalars, nAux ),dG2( nVars, nScalars, nAux );
      ( problem->AuxGPrime )( Aux, dG1, Y_plus, y_plus, jt );
      dG_du_vals1 = dG1.Variable;
      dG_dq_vals1 = dG1.Derivative;
      dG_dsigma_vals1 = dG1.Flux;
      dG_dPhi_vals1 = dG1.Aux;

      ( problem->AuxGPrime )( Aux, dG2, Y_minus, y_minus, jt );
      dG_du_vals2 = dG2.Variable;
      dG_dq_vals2 = dG2.Derivative;
      dG_dsigma_vals2 = dG2.Flux;
      dG_dPhi_vals2 = dG2.Aux;



      for(Index Var = 0; Var < nVars; Var++)
      {
        for ( Index j=0; j < k + 1; ++j )
        {
          for ( Index l=0; l < k + 1; ++l )
          {
            mat( Aux * ( k + 1 ) + j, Var * ( k + 1 ) + l ) +=
              wgt * dG_dsigma_vals1[ Var ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, Var * ( k + 1 ) + l ) +=
              wgt * dG_dsigma_vals2[ Var ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, l, y_minus );
            
            mat( Aux * ( k + 1 ) + j, nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_dq_vals1[ Var ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_dq_vals2[ Var ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, l, y_minus );
            
            mat( Aux * ( k + 1 ) + j, 2 * nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_du_vals1[ Var ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, 2 * nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_du_vals2[ Var ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, l, y_minus );
          }
        }
      }
      for(Index A2 = 0; A2 < nAux; A2++)
      {
        for ( Index j=0; j < k + 1; ++j )
        {
          for ( Index l=0; l < k + 1; ++l )
          {
            mat( Aux * ( k + 1 ) + j, SQU_DOF + A2 * ( k + 1 ) + l ) +=
              wgt * dG_dPhi_vals1[ A2 ] * LegendreBasis::Evaluate( I, j, y_plus ) * LegendreBasis::Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, SQU_DOF + A2 * ( k + 1 ) + l ) +=
              wgt * dG_dPhi_vals2[ A2 ] * LegendreBasis::Evaluate( I, j, y_minus ) * LegendreBasis::Evaluate( I, l, y_minus );
          }
        }
      }
    }
  }
}

