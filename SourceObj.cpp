#include "SourceObj.hpp"

SourceObj::SourceObj(int k_, int nVar_)
	: k(k_), nVar(nVar_)
{}

void SourceObj::setdFdqMat(Eigen::MatrixXd& dFdqMatrix, DGApprox q, DGApprox u, Interval I)
{
//	[ dF_1dq1    dF_1dq2    dF_1dq3 ]
//	[ dF_2dq1    dF_2dq2    dF_2dq3 ]
//	[ dF_3dq1    dF_3dq2    dF_3dq3 ]

	dFdqMatrix.setZero();
	for(int FVar = 0; FVar < nVar; FVar++)
	{
		for(int qVar = 0; qVar < nVar; qVar++)
		{
			Eigen::MatrixXd blockMat (k+1, k+1);
			dFdqSubMat(blockMat, FVar, qVar, q, u, I);
			dFdqMatrix.block(FVar*(k+1), qVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void SourceObj::setdFduMat(Eigen::MatrixXd& dFduMatrix, DGApprox q, DGApprox u, Interval I)
{
//	[ dF_1du1    dF_1du2    dF_1du3 ]
//	[ dF_2du1    dF_2du2    dF_2du3 ]
//	[ dF_3du1    dF_3du2    dF_3du3 ]

	dFduMatrix.setZero();
	for(int FVar = 0; FVar < nVar; FVar++)
	{
		for(int uVar = 0; uVar < nVar; uVar++)
		{
			Eigen::MatrixXd blockMat (k+1, k+1);
			dFduSubMat(blockMat, FVar, uVar, q, u, I);
			dFduMatrix.block(FVar*(k+1), uVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void SourceObj::dFdqSubMat(Eigen::MatrixXd& dFdqSubMatrix, int F_var, int q_var, DGApprox q, DGApprox u, Interval I)
{
	dFdqSubMatrix.setZero();
	std::function< double (double)> dFdqFunc = [ = ]( double x ){ return dFdqFuncs[F_var][q_var](x, q, u);};
	u.MassMatrix( I, dFdqSubMatrix, dFdqFunc);
}

void SourceObj::dFduSubMat(Eigen::MatrixXd& dFduSubMatrix, int F_var, int u_var, DGApprox q, DGApprox u, Interval I)
{
	dFduSubMatrix.setZero();
	std::function< double (double)> dFduFunc = [ = ]( double x ){ return dFduFuncs[F_var][u_var](x, q, u);};
	u.MassMatrix( I, dFduSubMatrix, dFduFunc);
}

void SourceObj::clear()
{
	sourceFuncs.clear();
	dFdqFuncs.clear();
	dFdqFuncs.clear();
}