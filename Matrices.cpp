
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Types.hpp"
#include "SystemSolver.hpp"
// SystemSolver.hpp only forward-declares FieldModel; the A1 assembly below calls
// dGeometry_dpsi through it, so this translation unit needs the definition.
#include "FieldModel.hpp"



void SystemSolver::NLqMat( Matrix& NLq, DGSoln const &Y, Index intervalIndex ) {
	//	[ dkappa_1dq1    dkappa_1dq2    dkappa_1dq3 ]
	//	[ dkappa_2dq1    dkappa_2dq2    dkappa_2dq3 ]
	//	[ dkappa_3dq1    dkappa_3dq2    dkappa_3dq3 ]

	DerivativeSubMatrix( NLq, &TransportSystem::dSigmaFn_dq, Y, intervalIndex );
}

void SystemSolver::NLuMat( Matrix& NLu, DGSoln const& Y, Index intervalIndex ) {
	//	[ dkappa_1du1    dkappa_1du2    dkappa_1du3 ]
	//	[ dkappa_2du1    dkappa_2du2    dkappa_2du3 ]
	//	[ dkappa_3du1    dkappa_3du2    dkappa_3du3 ]

	DerivativeSubMatrix( NLu, &TransportSystem::dSigmaFn_du, Y, intervalIndex );
}

void SystemSolver::NLphiMat( Matrix& M, DGSoln const& Y, Index intervalIndex ) {
 return;
}

// Sets matrices of the form
//	[ dX_1dZ1    dX_1dZ2    dX_1dZ3 ]
//	[ dX_2dZ1    dX_2dZ2    dX_2dZ3 ]
//	[ dX_3dZ1    dX_3dZ2    dX_3dZ3 ]
//
// where X is a sigma function or a source function and Z is one of u, q, or sigma.
// dX_dZ contains data on nodes in given cell
void SystemSolver::DerivativeSubMatrix(Matrix &mat, std::vector<Eigen::Ref<Matrix>> const dX_dZ, DGSoln const & Y, Index intervalIndex)
{
	// ASSERT mat.shape == ( nVars * ( k + 1) , nVars * ( k + 1 ) )
	assert(mat.rows() == nVars * (k + 1));
	assert(mat.cols() == nVars * (k + 1));

	mat.setZero();

	// With interpolation we have Mass * diagonal( F'(nodes) ) (c.f. https://arxiv.org/pdf/1811.09667 eq 3.16ff)

	for (Index XVar = 0; XVar < nVars; XVar++)
	{
		Matrix M = Y.getBasis().MassMatrix(grid[intervalIndex]);
		for (Index j = 0; j < k + 1; ++j)
		{
			for (Index ZVar = 0; ZVar < nVars; ZVar++)
			{
				mat(XVar * (k + 1) + j, ZVar * (k + 1) + j) = dX_dZ[XVar](ZVar, j);
			}
		}
		for (Index ZVar = 0; ZVar < nVars; ZVar++)
		{
			mat.block(XVar * (k + 1), ZVar * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
		}
	}
}

// See the declaration in SystemSolver.hpp for what the chain matrix is and why
// the q column needs two calls.
void SystemSolver::accumulateStarBlocks(MatrixRef mat,
										std::vector<Eigen::Ref<Matrix>> const &dX_dZ,
										Matrix const &chain, Index nX, Index nZ,
										Index intervalIndex) const
{
	assert(mat.rows() == nX * (k + 1));
	assert(mat.cols() == nZ * (k + 1));
	assert(chain.rows() == k + 2);
	assert(chain.cols() == k + 1);

	Matrix const &A9 = postprocessor->A9(intervalIndex);

	for (Index XVar = 0; XVar < nX; XVar++)
	{
		for (Index ZVar = 0; ZVar < nZ; ZVar++)
		{
			// Materialised because asDiagonal() on a transposed row of an
			// Eigen::Ref is fragile to alias analysis, and this is k+2 doubles.
			const Vector d = dX_dZ[XVar].row(ZVar).transpose();
			mat.block(XVar * (k + 1), ZVar * (k + 1), k + 1, k + 1) +=
				(A9 * d.asDiagonal()) * chain;
		}
	}
}

void SystemSolver::DerivativeSubMatrix( Matrix& mat, void ( TransportSystem::*dX_dZ )( Index, VectorRef, const State&, Position, double ), DGSoln const& Y, Index intervalIndex )
{
	// ASSERT mat.shape == ( nVars * ( k + 1) , nVars * ( k + 1 ) )
	assert( mat.rows() == nVars * ( k + 1 ) );
	assert( mat.cols() == nVars * ( k + 1 ) );

	mat.setZero();

    // With interpolation we have Mass * diagonal( F'(nodes) ) (c.f. https://arxiv.org/pdf/1811.09667 eq 3.16ff)


    for ( Index XVar = 0; XVar < nVars; XVar++ )
    {
        Matrix M = Y.getBasis().MassMatrix( grid[ intervalIndex ] );
        for ( Index j=0; j < k + 1; ++j )
        {
            Vector vals( nVars );
            vals.setZero();
            double xi = grid[ intervalIndex ].fromRef( Y.getBasis().Nodes( j ) );
            State s = Y.evalOnNode( intervalIndex, j );
            ( problem->*dX_dZ )( XVar, vals, s, xi, jt );
            for(Index ZVar = 0; ZVar < nVars; ZVar++)
            {
                mat( XVar * ( k + 1 ) + j, ZVar * ( k + 1 ) + j ) = vals[ ZVar ];
            }
        }
        for(Index ZVar = 0; ZVar < nVars; ZVar++) {
            mat.block( XVar * ( k + 1 ), ZVar * ( k + 1 ), k + 1, k + 1 ).applyOnTheLeft( M );
        }
    }
}

void SystemSolver::dSourcedq_Mat( Matrix& dSourcedqMatrix, DGSoln const& Y, Index I)
{
	DerivativeSubMatrix( dSourcedqMatrix, &TransportSystem::dSources_dq, Y, I );
}

void SystemSolver::dSourcedu_Mat( Matrix& dSourceduMatrix, DGSoln const& Y, Index I)
{
	DerivativeSubMatrix( dSourceduMatrix, &TransportSystem::dSources_du, Y, I );
}

void SystemSolver::dSourcedsigma_Mat( Matrix& dSourcedsigmaMatrix, DGSoln const& Y, Index I )
{
	DerivativeSubMatrix( dSourcedsigmaMatrix, &TransportSystem::dSources_dsigma, Y, I );
}

void SystemSolver::dSources_dScalars_Mat( Matrix& mat, DGSoln const& Y, Index intervalIndex, Time tEval )
{
	Interval const &I( grid[ intervalIndex ] );

	// ASSERT mat.shape == ( nVars * ( k + 1) , nScalars )
	assert( mat.rows() == nVars * ( k + 1 ) );
	assert( mat.cols() == nScalars );

	mat.setZero();

	// This is the derivative of the source term as the *residual* forms it, so
	// it has to be built the same way the residual builds it. residual() uses
	//
	//     res.u += ... - InterpolateOntoBasis( I, S( nodes ) )
	//
	// i.e. the projection of the *interpolant* of S, so the derivative is the
	// projection of the interpolant of dS/dmu -- the same
	// `Mass * (values at nodes)` form every other Jacobian block in this file
	// uses (c.f. https://arxiv.org/pdf/1811.09667 eq 3.16ff).
	//
	// This used to integrate dS/dmu exactly by Gauss quadrature instead. The two
	// agree only when dS/dmu is a polynomial the basis represents: for
	// ScalarTestLD3, whose dS/dJ is a narrow Gaussian, they differed by 7% of
	// the residual at k = 2 on 4 cells (falling to 6e-9 by k = 6 on 32 cells, as
	// the interpolation error dies away). The Jacobian is never assembled, so
	// the only symptom was degraded Newton convergence.
	Values dSdS( nScalars );
	Matrix nodal( nScalars, k + 1 );

	for ( Index XVar = 0; XVar < nVars; XVar++ )
	{
		for ( Index j = 0; j < k + 1; ++j )
		{
			dSdS.setZero();
			double x_j = I.fromRef( Y.getBasis().Nodes( j ) );
			State s = Y.evalOnNode( intervalIndex, j );
			problem->dSources_dScalars( XVar, dSdS, s, x_j, tEval );
			nodal.col( j ) = dSdS;
		}

		for ( Index iScalar = 0; iScalar < nScalars; ++iScalar )
		{
			Vector vals = nodal.row( iScalar ).transpose();
			mat.block( XVar * ( k + 1 ), iScalar, k + 1, 1 ) =
				Y.getBasis().InterpolateOntoBasis( I, vals );
		}
	}
}

void SystemSolver::dSources_dScalars_StarMat(Matrix &mat, GlobalState const &states,
											 std::vector<Position> const &points,
											 Index intervalIndex, Time tEval)
{
	assert(mat.rows() == nVars * (k + 1));
	assert(mat.cols() == nScalars);

	mat.setZero();

	const Index nStar = k + 2;
	Matrix const &A9 = postprocessor->A9(intervalIndex);

	Values dSdS(nScalars);
	Matrix nodal(nScalars, nStar);

	for (Index XVar = 0; XVar < nVars; XVar++)
	{
		for (Index m = 0; m < nStar; ++m)
		{
			dSdS.setZero();
			const Index g = intervalIndex * nStar + m;
			problem->dSources_dScalars(XVar, dSdS, states[g], points[g], tEval);
			nodal.col(m) = dSdS;
		}

		for (Index iScalar = 0; iScalar < nScalars; ++iScalar)
			mat.block(XVar * (k + 1), iScalar, k + 1, 1) =
				A9 * Vector(nodal.row(iScalar).transpose());
	}
}

// ------------------------------------------------- the field coupling, A1 --

SystemSolver::GeometryResponse SystemSolver::geometryResponse(PhysicsNodes const &nodes, Time tEval)
{
	if (problem->spec().geometryDerivatives == GeometryDerivatives::Differenced)
		return differencedGeometryResponse(nodes, tEval);

	const Index nNodes = static_cast<Index>(nodes.points.size());
	GeometryResponse r;
	r.sigma.assign(nVars, Matrix(nGeom, nNodes));
	r.sources.assign(nVars, Matrix(nGeom, nNodes));
	r.aux.assign(nAux, Matrix(nGeom, nNodes));

	Values dXdG(nGeom);
	for (Index j = 0; j < nNodes; ++j)
	{
		const State s = nodes.states[j];

		// Each arrives zeroed, so a case that does not read geometry -- and so
		// does not override the hook at all -- contributes an identically zero
		// column, which is exactly right: it does not couple.
		auto take = [&](Matrix &into, auto hook, Index i)
		{
			dXdG.setZero();
			(problem->*hook)(i, dXdG, s, nodes.points[j], tEval);
			into.col(j) = dXdG;
		};
		for (Index v = 0; v < nVars; ++v)
		{
			take(r.sigma[v], &TransportSystem::dSigmaFn_dGeometry, v);
			take(r.sources[v], &TransportSystem::dSources_dGeometry, v);
		}
		for (Index a = 0; a < nAux; ++a)
			take(r.aux[a], &TransportSystem::dAuxG_dGeometry, a);
	}
	return r;
}

// A forward difference, one geometry slot at a time across every node at once,
// so each perturbation is one batched call. The step is relative to the slot's
// size at the node, floored at its largest size anywhere on the grid: a slot
// that vanishes somewhere -- <|grad x|^2> on the axis -- is still stepped on the
// scale it has elsewhere, rather than by sqrt(eps) in absolute terms, which
// means nothing for a quantity of arbitrary units.
SystemSolver::GeometryResponse SystemSolver::differencedGeometryResponse(PhysicsNodes const &nodes,
																		 Time tEval)
{
	const Index nNodes = static_cast<Index>(nodes.points.size());
	GeometryResponse r;
	r.sigma.assign(nVars, Matrix(nGeom, nNodes));
	r.sources.assign(nVars, Matrix(nGeom, nNodes));
	r.aux.assign(nAux, Matrix(nGeom, nNodes));

	const double root = std::sqrt(std::numeric_limits<double>::epsilon());
	Matrix const &g = nodes.states.GeometryMatrix();

	std::vector<PhysicsOutput> perturbed;
	perturbed.reserve(nGeom);
	std::vector<Vector> steps;
	steps.reserve(nGeom);
	for (Index slot = 0; slot < nGeom; ++slot)
	{
		const double floor = g.row(slot).cwiseAbs().maxCoeff();
		Vector h(nNodes);
		for (Index j = 0; j < nNodes; ++j)
		{
			const double scale = std::max(std::abs(g(slot, j)), floor);
			h(j) = root * (scale > 0.0 ? scale : 1.0);
			// The step actually taken, which rounding may have moved.
			h(j) = (g(slot, j) + h(j)) - g(slot, j);
		}

		GlobalState shifted = nodes.states;
		shifted.GeometryMatrix().row(slot) += h.transpose();
		perturbed.push_back(problem->ComputePhysics(shifted, nodes.points, tEval));
		steps.push_back(std::move(h));
	}

	// Last, so the case's source cache is left at the state itself.
	const PhysicsOutput base = problem->ComputePhysics(nodes.states, nodes.points, tEval);

	for (Index slot = 0; slot < nGeom; ++slot)
	{
		PhysicsOutput const &p = perturbed[slot];
		Vector const &h = steps[slot];
		for (Index v = 0; v < nVars; ++v)
		{
			r.sigma[v].row(slot) = ((p[0][v] - base[0][v]).array() / h.array()).matrix().transpose();
			r.sources[v].row(slot) = ((p[1][v] - base[1][v]).array() / h.array()).matrix().transpose();
		}
		for (Index a = 0; a < nAux; ++a)
			r.aux[a].row(slot) = ((p[2][a] - base[2][a]).array() / h.array()).matrix().transpose();
	}
	return r;
}

std::vector<Matrix> SystemSolver::geometryJacobianOnNodes(Vector const &psi,
														  std::vector<Position> const &points,
														  Index intervalIndex, Index nNodes, Time tEval)
{
	std::vector<Matrix> dGdPsi(nNodes, Matrix::Zero(nGeom, nField));
	for (Index j = 0; j < nNodes; ++j)
		fieldModel->dGeometry_dpsi(dGdPsi[j], psi, points[intervalIndex * nNodes + j], tEval);
	return dGdPsi;
}

void SystemSolver::fieldChainOnNodes(Matrix &nodal, Matrix const &dXdG,
									 std::vector<Matrix> const &dGdPsi, Index intervalIndex,
									 Index nNodes)
{
	assert(nodal.cols() == nNodes);
	for (Index j = 0; j < nNodes; ++j)
	{
		const Vector col = dXdG.col(intervalIndex * nNodes + j);
		nodal.col(j) = dGdPsi[j].transpose() * col;
	}
}

// See the declaration for the shape.
//
// The sign of the u block is the one thing here that cannot be read off the
// chain rule. residual() forms that row as
//
//     res.u = ... - InterpolateOntoBasis( I, S( nodes ) )
//
// so d(res.u)/d(psi) carries the same minus sign, exactly as assembleCellMatrix
// *subtracts* every source block from MX. Getting it wrong flips the sign of one
// third of A1, which the Jacobian being unassembled would hide completely: the
// answer would still be right and only the Newton iteration count would move.
void SystemSolver::dPhysics_dField_Mat(Matrix &mat, DGSoln const &Y, GeometryResponse const &response,
									   std::vector<Position> const &points, Index intervalIndex,
									   Time tEval)
{
	Interval const &I(grid[intervalIndex]);

	assert(mat.rows() == (3 * nVars + nAux) * (k + 1));
	assert(mat.cols() == nField);

	mat.setZero();

	const std::vector<Matrix> dGdPsi =
		geometryJacobianOnNodes(Vector(Y.getField()), points, intervalIndex, k + 1, tEval);
	Matrix nodal(nField, k + 1);

	// The projection the residual applies to a physics value: the interpolatory
	// mass-matrix form of arXiv:1811.09667, i.e. the projection of the
	// *interpolant*. Not an exact quadrature of the derivative -- the two agree
	// only when the integrand is a polynomial the basis represents, which is the
	// trap dSources_dScalars_Mat records above.
	auto project = [&](Index rowOffset, double scale)
	{
		for (Index m = 0; m < nField; ++m)
		{
			Vector vals = nodal.row(m).transpose();
			mat.block(rowOffset, m, k + 1, 1) =
				scale * Y.getBasis().InterpolateOntoBasis(I, vals);
		}
	};

	for (Index XVar = 0; XVar < nVars; XVar++)
	{
		// sigma rows: res.sigma = A sigma + Pi( sigma_hat ).
		fieldChainOnNodes(nodal, response.sigma[XVar], dGdPsi, intervalIndex, k + 1);
		project(XVar * (k + 1), 1.0);

		// u rows: res.u = ... - Pi( S ). See the note on the sign above.
		fieldChainOnNodes(nodal, response.sources[XVar], dGdPsi, intervalIndex, k + 1);
		project(2 * nVars * (k + 1) + XVar * (k + 1), -1.0);
	}

	// aux rows: res.Aux = Pi( G ), the constraint imposed by projection.
	for (Index aux = 0; aux < nAux; aux++)
	{
		fieldChainOnNodes(nodal, response.aux[aux], dGdPsi, intervalIndex, k + 1);
		project(3 * nVars * (k + 1) + aux * (k + 1), 1.0);
	}
}

void SystemSolver::dPhysics_dField_StarMat(Matrix &mat, DGSoln const &Y,
										   GeometryResponse const &response,
										   std::vector<Position> const &points,
										   Index intervalIndex, Time tEval)
{
	assert(mat.rows() == (3 * nVars + nAux) * (k + 1));
	assert(mat.cols() == nField);

	mat.setZero();

	const Index nStar = k + 2;
	Matrix const &A9 = postprocessor->A9(intervalIndex);

	const std::vector<Matrix> dGdPsi =
		geometryJacobianOnNodes(Vector(Y.getField()), points, intervalIndex, nStar, tEval);
	Matrix nodal(nField, nStar);

	auto project = [&](Index rowOffset, double scale)
	{
		for (Index m = 0; m < nField; ++m)
			mat.block(rowOffset, m, k + 1, 1) =
				scale * (A9 * Vector(nodal.row(m).transpose()));
	};

	for (Index XVar = 0; XVar < nVars; XVar++)
	{
		fieldChainOnNodes(nodal, response.sigma[XVar], dGdPsi, intervalIndex, nStar);
		project(XVar * (k + 1), 1.0);

		fieldChainOnNodes(nodal, response.sources[XVar], dGdPsi, intervalIndex, nStar);
		project(2 * nVars * (k + 1) + XVar * (k + 1), -1.0);
	}

	for (Index aux = 0; aux < nAux; aux++)
	{
		fieldChainOnNodes(nodal, response.aux[aux], dGdPsi, intervalIndex, nStar);
		project(3 * nVars * (k + 1) + aux * (k + 1), 1.0);
	}
}

void SystemSolver::dPhi_Mat(Matrix &mat, std::vector<Eigen::Ref<Matrix>> const dX_dZ, DGSoln const &Y, Index intervalIndex )
{
  // ASSERT mat.shape == ( nVars * ( k + 1) , nVars * ( k + 1 ) )
	assert(mat.rows() == nVars * (k + 1));
	assert(mat.cols() == nAux * (k + 1));

	mat.setZero();

	// With interpolation we have Mass * diagonal( F'(nodes) ) (c.f. https://arxiv.org/pdf/1811.09667 eq 3.16ff)

	for (Index XVar = 0; XVar < nVars; XVar++)
	{
		Matrix M = Y.getBasis().MassMatrix(grid[intervalIndex]);
		for (Index j = 0; j < k + 1; ++j)
		{
			for (Index ZVar = 0; ZVar < nAux; ZVar++)
			{
				mat(XVar * (k + 1) + j, ZVar * (k + 1) + j) = dX_dZ[XVar](ZVar, j);
			}
		}
		for (Index ZVar = 0; ZVar < nAux; ZVar++)
		{
			mat.block(XVar * (k + 1), ZVar * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
		}
	}
}
void SystemSolver::dSourcedPhi_Mat( Matrix& mat, DGSoln const& Y, Index intervalIndex )
{
    Interval const &I( grid[ intervalIndex ] );

	auto const& x_vals = y.getBasis().abscissae();
	auto const& x_wgts = y.getBasis().weights();
	const size_t n_abscissa = x_vals.size();

	// ASSERT mat.shape == ( nVars * ( k + 1) , nAux * ( k + 1 ) )
	assert( mat.rows() == nVars * ( k + 1 ) );
	assert( mat.cols() == nAux  * ( k + 1 ) );

	mat.setZero();

	// Phi are basis fn's
	// M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

	for ( Index Var = 0; Var < nVars; Var++ )
	{
		Values dS_dPhi_vals1( nAux );
		Values dS_dPhi_vals2( nAux );
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
							wgt * dS_dPhi_vals1[ Aux ] * y.getBasis().Evaluate( I, j, y_plus ) * y.getBasis().Evaluate( I, l, y_plus );
						mat( Var * ( k + 1 ) + j, Aux * ( k + 1 ) + l ) +=
							wgt * dS_dPhi_vals2[ Aux ] * y.getBasis().Evaluate( I, j, y_minus ) * y.getBasis().Evaluate( I, l, y_minus );
					}
				}
			}
		}
	}
}
void SystemSolver::dAux_Mat(Eigen::Ref<Matrix> mat, GlobalStateMatrix& dAux, DGSoln const &Y, Index intervalIndex)
{
  Interval const &I( grid[intervalIndex] );
  // Assert Mat.shape == ( nAux * ( k + 1 ), ( 3 * nVars + nAux ) * ( k + 1 ) )
  assert( mat.rows() == nAux * ( k + 1 ) );
  assert( mat.cols() == ( 3*nVars + nAux ) * ( k + 1 ) );

	mat.setZero();

	// With interpolation we have Mass * diagonal( F'(nodes) ) (c.f. https://arxiv.org/pdf/1811.09667 eq 3.16ff)
	//
	// The column layout is the one MX uses throughout: [ sigma | q | u | phi ],
	// each of the first three nVars*(k+1) wide and phi nAux*(k+1) wide. See
	// updateMatricesForJacSolve, which writes NLq at column nVars*(k+1) and NLu
	// at 2*nVars*(k+1), and the sibling overload below, which integrates the
	// same quantities against the same layout.
	//
	// This used to write dG/du into column j, dG/dq into ZVar*(k+1)+j and
	// dG/dsigma into 2*ZVar*(k+1)+j -- three different derivatives piled into
	// the sigma column block, with dG/du assigned rather than accumulated so
	// only the last variable's value survived. For a case like AuxVarTest,
	// whose only nonzero aux derivatives are dG/du and dG/dphi, that dropped
	// dG/du from the Jacobian entirely. The residual was unaffected, so the
	// answer stayed correct and only Newton convergence suffered -- which is
	// why no regression test noticed. The M application below was also missing
	// the q and u column blocks, and its inner loop shadowed `Aux`.
	const auto dG_du = dAux.Variable(intervalIndex);
	const auto dG_dq = dAux.Derivative(intervalIndex);
	const auto dG_dsigma = dAux.Flux(intervalIndex);
	const auto dG_dphi = dAux.Aux(intervalIndex);

	const Index sigmaBlock = 0;
	const Index qBlock = nVars * (k + 1);
	const Index uBlock = 2 * nVars * (k + 1);
	const Index phiBlock = 3 * nVars * (k + 1);

	const Matrix M = Y.getBasis().MassMatrix(grid[intervalIndex]);

	for (Index Aux = 0; Aux < nAux; Aux++)
	{
		for (Index j = 0; j < k + 1; ++j)
		{
			for (Index ZVar = 0; ZVar < nVars; ZVar++)
			{
				mat(Aux * (k + 1) + j, sigmaBlock + ZVar * (k + 1) + j) += dG_dsigma[Aux](ZVar, j);
				mat(Aux * (k + 1) + j, qBlock + ZVar * (k + 1) + j) += dG_dq[Aux](ZVar, j);
				mat(Aux * (k + 1) + j, uBlock + ZVar * (k + 1) + j) += dG_du[Aux](ZVar, j);
			}
			for (Index A2 = 0; A2 < nAux; A2++)
				mat(Aux * (k + 1) + j, phiBlock + A2 * (k + 1) + j) += dG_dphi[Aux](A2, j);
		}

		for (Index ZVar = 0; ZVar < nVars; ZVar++)
		{
			mat.block(Aux * (k + 1), sigmaBlock + ZVar * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
			mat.block(Aux * (k + 1), qBlock + ZVar * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
			mat.block(Aux * (k + 1), uBlock + ZVar * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
		}
		for (Index A2 = 0; A2 < nAux; A2++)
			mat.block(Aux * (k + 1), phiBlock + A2 * (k + 1), k + 1, k + 1).applyOnTheLeft(M);
	}
}
void SystemSolver::dAux_Mat( Eigen::Ref<Matrix> mat, DGSoln const& Y, Index intervalIndex )
{
  Interval const &I( grid[intervalIndex] );
  auto const& x_vals = y.getBasis().abscissae();
  auto const& x_wgts = y.getBasis().weights();
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
      dG_du_vals1 = dG1.u();
      dG_dq_vals1 = dG1.q();
      dG_dsigma_vals1 = dG1.sigma();
      dG_dPhi_vals1 = dG1.phi();

      ( problem->AuxGPrime )( Aux, dG2, Y_minus, y_minus, jt );
      dG_du_vals2 = dG2.u();
      dG_dq_vals2 = dG2.q();
      dG_dsigma_vals2 = dG2.sigma();
      dG_dPhi_vals2 = dG2.phi();



      for(Index Var = 0; Var < nVars; Var++)
      {
        for ( Index j=0; j < k + 1; ++j )
        {
          for ( Index l=0; l < k + 1; ++l )
          {
            mat( Aux * ( k + 1 ) + j, Var * ( k + 1 ) + l ) +=
              wgt * dG_dsigma_vals1[ Var ] * y.getBasis().Evaluate( I, j, y_plus ) * y.getBasis().Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, Var * ( k + 1 ) + l ) +=
              wgt * dG_dsigma_vals2[ Var ] * y.getBasis().Evaluate( I, j, y_minus ) * y.getBasis().Evaluate( I, l, y_minus );
            
            mat( Aux * ( k + 1 ) + j, nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_dq_vals1[ Var ] * y.getBasis().Evaluate( I, j, y_plus ) * y.getBasis().Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_dq_vals2[ Var ] * y.getBasis().Evaluate( I, j, y_minus ) * y.getBasis().Evaluate( I, l, y_minus );
            
            mat( Aux * ( k + 1 ) + j, 2 * nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_du_vals1[ Var ] * y.getBasis().Evaluate( I, j, y_plus ) * y.getBasis().Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, 2 * nVars * ( k + 1 ) + Var * ( k + 1 ) + l ) +=
              wgt * dG_du_vals2[ Var ] * y.getBasis().Evaluate( I, j, y_minus ) * y.getBasis().Evaluate( I, l, y_minus );
          }
        }
      }
      for(Index A2 = 0; A2 < nAux; A2++)
      {
        for ( Index j=0; j < k + 1; ++j )
        {
          for ( Index l=0; l < k + 1; ++l )
          {
            mat( Aux * ( k + 1 ) + j, SQU_DOF * nVars + A2 * ( k + 1 ) + l ) +=
              wgt * dG_dPhi_vals1[ A2 ] * y.getBasis().Evaluate( I, j, y_plus ) * y.getBasis().Evaluate( I, l, y_plus );
            mat( Aux * ( k + 1 ) + j, SQU_DOF * nVars + A2 * ( k + 1 ) + l ) +=
              wgt * dG_dPhi_vals2[ A2 ] * y.getBasis().Evaluate( I, j, y_minus ) * y.getBasis().Evaluate( I, l, y_minus );
          }
        }
      }
    }
  }
}

