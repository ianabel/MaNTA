#include "DiffusionObj.hpp"
#include <boost/functional/hash.hpp>

DiffusionObj::DiffusionObj(int k_, int nVar_)
	: k(k_), nVar(nVar_)
{}

DiffusionObj::DiffusionObj(int k_, int nVar_, std::string diffCase)
	: k(k_), nVar(nVar_)
{
	if(diffCase == "1dLinearTest") buildSingleVariableLinearTest();

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
	std::cerr << dqkappaFunc(1.0) << std::endl << std::endl;
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
