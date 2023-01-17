#include "SourceObj.hpp"

SourceObj::SourceObj(int k_, int nVar_)
	: k(k_), nVar(nVar_)
{}

SourceObj::SourceObj(int k_, int nVar_, std::string reactionCase)
	: k(k_), nVar(nVar_)
{
	if(reactionCase == "1dLinearTest") buildSingleVariableLinearTest();
	else if(reactionCase == "3VarNoSource") build3VariableLinearTest();
	else if(reactionCase == "1DFisher") build1DFisherSource();

	else throw std::logic_error( "Source Case provided does not exist" );
}

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

//---------------Reaction cases-------------------
void SourceObj::buildSingleVariableLinearTest()
{
	auto nVar = 1;

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  0.0;};
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	dFdqFuncs.resize(nVar);
	dFduFuncs.resize(nVar);

	dFdqFuncs[0].push_back(dF_0dq_0);
	dFduFuncs[0].push_back(dF_0du_0);
}

void SourceObj::build3VariableLinearTest()
{
	if(nVar != 3) throw std::runtime_error("check your source build, you did it wrong.");

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  0.0;};
	sourceFuncs.push_back(F_0);
	sourceFuncs.push_back(F_0);
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	dFdqFuncs.resize(nVar);
	dFduFuncs.resize(nVar);

	dFdqFuncs[0].push_back(dF_0dq_0);
	dFdqFuncs[0].push_back(dF_0dq_0);
	dFdqFuncs[0].push_back(dF_0dq_0);
	dFdqFuncs[1].push_back(dF_0dq_0);
	dFdqFuncs[1].push_back(dF_0dq_0);
	dFdqFuncs[1].push_back(dF_0dq_0);
	dFdqFuncs[2].push_back(dF_0dq_0);
	dFdqFuncs[2].push_back(dF_0dq_0);
	dFdqFuncs[2].push_back(dF_0dq_0);


	dFduFuncs[0].push_back(dF_0du_0);
	dFduFuncs[0].push_back(dF_0du_0);
	dFduFuncs[0].push_back(dF_0du_0);
	dFduFuncs[1].push_back(dF_0du_0);
	dFduFuncs[1].push_back(dF_0du_0);
	dFduFuncs[1].push_back(dF_0du_0);
	dFduFuncs[2].push_back(dF_0du_0);
	dFduFuncs[2].push_back(dF_0du_0);
	dFduFuncs[2].push_back(dF_0du_0);
}

void SourceObj::build1DFisherSource()
{
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");
	clear();

	double beta = 1.0;
	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  -u(x,0)*beta*(1.0-u(x,0));};
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};
	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -beta + 2.0*beta*u(x,0);};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	dFdqFuncs.resize(nVar);
	dFduFuncs.resize(nVar);
	dFdqFuncs[0].push_back(dF_0dq_0);
	dFduFuncs[0].push_back(dF_0du_0);
}