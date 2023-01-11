#include "DiffusionObj.hpp"
#include <boost/functional/hash.hpp>

DiffusionObj::DiffusionObj(int k_, int nVar_)
	: k(k_), nVar(nVar_)
{}

DiffusionObj::DiffusionObj(int k_, int nVar_, std::string diffCase)
	: k(k_), nVar(nVar_)
{
	if(diffCase == "1dLinearTest") buildSingleVariableLinearTest();
	else if(diffCase == "3VarLinearTest") build3VariableLinearTest();

	else throw std::logic_error( "Diffusion Case provided does not exist" );
}


void DiffusionObj::NLqMat(Eigen::MatrixXd& NLq, DGApprox q, DGApprox u, Interval I)
{
//	[ dkappa_1dq1    dkappa_1dq2    dkappa_1dq3 ]
//	[ dkappa_2dq1    dkappa_2dq2    dkappa_2dq3 ]
//	[ dkappa_3dq1    dkappa_3dq2    dkappa_3dq3 ]

	NLq.setZero();
	for(int kappaVar = 0; kappaVar < nVar; kappaVar++)
	{
		for(int qVar = 0; qVar < nVar; qVar++)
		{
			Eigen::MatrixXd blockMat (k+1, k+1);
			delqKappaMat(blockMat, kappaVar, qVar, q, u, I);
			NLq.block(kappaVar*(k+1), qVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void DiffusionObj::NLuMat(Eigen::MatrixXd& NLu, DGApprox q, DGApprox u, Interval I)
{
//	[ dkappa_1du1    dkappa_1du2    dkappa_1du3 ]
//	[ dkappa_2du1    dkappa_2du2    dkappa_2du3 ]
//	[ dkappa_3du1    dkappa_3du2    dkappa_3du3 ]

	NLu.setZero();
	for(int kappaVar = 0; kappaVar < nVar; kappaVar++)
	{
		for(int uVar = 0; uVar < nVar; uVar++)
		{
			Eigen::MatrixXd blockMat (k+1, k+1);
			deluKappaMat(blockMat, kappaVar, uVar, q, u, I);
			NLu.block(kappaVar*(k+1), uVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void DiffusionObj::deluKappaMat(Eigen::MatrixXd& dukappaMat, int kappa_var, int u_var, DGApprox q, DGApprox u, Interval I)
{
	dukappaMat.setZero();
	std::function< double (double)> dukappaFunc = [ = ]( double x ){ return deluKappaFuncs[kappa_var][u_var](x, q, u);};
	u.MassMatrix( I, dukappaMat, dukappaFunc);
}

void DiffusionObj::delqKappaMat(Eigen::MatrixXd& dqkappaMat, int kappa_var, int q_var, DGApprox q, DGApprox u, Interval I)
{
	dqkappaMat.setZero();
	std::function< double (double)> dqkappaFunc = [ = ]( double x ){ return delqKappaFuncs[kappa_var][q_var](x, q, u);};
	u.MassMatrix( I, dqkappaMat, dqkappaFunc);
}

void DiffusionObj::clear()
{
	kappaFuncs.clear();
	delqKappaFuncs.clear();
	deluKappaFuncs.clear();
}



//----------------------------------------Diffusion Cases----------------------------------------

void DiffusionObj::buildSingleVariableLinearTest()
{
	auto nVar = 1;
	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0*q(x,0);};
	kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	delqKappaFuncs.resize(nVar);
	deluKappaFuncs.resize(nVar);

	delqKappaFuncs[0].push_back(dkappa0dq0);
	deluKappaFuncs[0].push_back(dkappa0du0);
}

void DiffusionObj::build3VariableLinearTest()
{
	auto nVar = 3;
	delqKappaFuncs.resize(nVar);
	deluKappaFuncs.resize(nVar);

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0*q(x,0) + 0.1*q(x,1);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0*q(x,1) + 0.1*q(x,2);};
	std::function<double( double, DGApprox, DGApprox )> kappa2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0*q(x,2) + 0.1*q(x,0);};
	kappaFuncs.push_back(kappa0);
	kappaFuncs.push_back(kappa1);
	kappaFuncs.push_back(kappa2);

	//-----------dk/dq---------------------

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.1;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	delqKappaFuncs[0].push_back(dkappa0dq0);
	delqKappaFuncs[0].push_back(dkappa0dq1);
	delqKappaFuncs[0].push_back(dkappa0dq2);

	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.1;};

	delqKappaFuncs[1].push_back(dkappa1dq0);
	delqKappaFuncs[1].push_back(dkappa1dq1);
	delqKappaFuncs[1].push_back(dkappa1dq2);

	std::function<double( double, DGApprox, DGApprox )> dkappa2dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.1;};
	std::function<double( double, DGApprox, DGApprox )> dkappa2dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa2dq2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0;};

	delqKappaFuncs[2].push_back(dkappa2dq0);
	delqKappaFuncs[2].push_back(dkappa2dq1);
	delqKappaFuncs[2].push_back(dkappa2dq2);

	//----------dk/du--------------

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	deluKappaFuncs[0].push_back(dkappa0du0);
	deluKappaFuncs[0].push_back(dkappa0du1);
	deluKappaFuncs[0].push_back(dkappa0du2);

	std::function<double( double, DGApprox, DGApprox )> dkappa1du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	deluKappaFuncs[1].push_back(dkappa1du0);
	deluKappaFuncs[1].push_back(dkappa1du1);
	deluKappaFuncs[1].push_back(dkappa1du2);

	std::function<double( double, DGApprox, DGApprox )> dkappa2du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa2du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa2du2 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	deluKappaFuncs[2].push_back(dkappa2du0);
	deluKappaFuncs[2].push_back(dkappa2du1);
	deluKappaFuncs[2].push_back(dkappa2du2);
}