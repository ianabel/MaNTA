#include "SourceObj.hpp"
#include "Constants.hpp"

SourceObj::SourceObj(int k_, int nVar_)
	: k(k_), nVar(nVar_)
{}

SourceObj::SourceObj(int k_, int nVar_, std::string reactionCase)
	: k(k_), nVar(nVar_)
{
	if(reactionCase == "1DSourceless") buildSingleVariableLinearTest();
	else if(reactionCase == "2DSourceless") build2DSourceless();
	else if(reactionCase == "3DSourceless") build3VariableLinearTest();
	else if(reactionCase == "1DFisher") build1DFisherSource();
	else if(reactionCase == "1DConstSource") build1DConstSource();
	else if(reactionCase == "CylinderPlasmaConstDensity") buildCylinderPlasmaConstDensitySource();

	else throw std::logic_error( "Source Case provided does not exist" );
}

void SourceObj::setdFdqMat(Eigen::MatrixXd& dFdqMatrix, DGApprox sig, DGApprox q, DGApprox u, Interval I)
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
			dFdqSubMat(blockMat, FVar, qVar, sig, q, u, I);
			dFdqMatrix.block(FVar*(k+1), qVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void SourceObj::setdFduMat(Eigen::MatrixXd& dFduMatrix, DGApprox sig, DGApprox q, DGApprox u, Interval I)
{
//	[ dS_1du1    dS_1du2    dS_1du3 ]
//	[ dS_2du1    dS_2du2    dS_2du3 ]
//	[ dS_3du1    dS_3du2    dS_3du3 ]

	dFduMatrix.setZero();
	for(int SVar = 0; SVar < nVar; SVar++)
	{
		for(int uVar = 0; uVar < nVar; uVar++)
		{
			Eigen::MatrixXd blockMat (k+1, k+1);
			dFduSubMat(blockMat, SVar, uVar, sig, q, u, I);
			dFduMatrix.block(SVar*(k+1), uVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void SourceObj::setdFdsigMat(Eigen::MatrixXd& dFdsigMatrix, DGApprox sig, DGApprox q, DGApprox u, Interval I)
{
//	[ dS_1dsig1    dS_1dsig2    dS_1dsig3 ]
//	[ dS_2dsig1    dS_2dsig2    dS_2dsig3 ]
//	[ dS_3dsig1    dS_3dsig2    dS_3dsig3 ]

	dFdsigMatrix.setZero();
	for(int SVar = 0; SVar < nVar; SVar++)
	{
		for(int sigVar = 0; sigVar < nVar; sigVar++)
		{
			Eigen::MatrixXd blockMat (k+1, k+1);
			dFdsigSubMat(blockMat, SVar, sigVar, sig, q, u, I);
			dFdsigMatrix.block(SVar*(k+1), sigVar*(k+1), k+1, k+1) = blockMat;
		}
	}
}

void SourceObj::dFdqSubMat(Eigen::MatrixXd& dFdqSubMatrix, int F_var, int q_var, DGApprox sig, DGApprox q, DGApprox u, Interval I)
{
	dFdqSubMatrix.setZero();
	std::function< double (double)> dSdqFunc = [ = ]( double x ){ return delqSourceFuncs[F_var][q_var](x, sig, q, u);};
	u.MassMatrix( I, dFdqSubMatrix, dSdqFunc);
}

void SourceObj::dFduSubMat(Eigen::MatrixXd& dFduSubMatrix, int F_var, int u_var, DGApprox sig, DGApprox q, DGApprox u, Interval I)
{
	dFduSubMatrix.setZero();
	std::function< double (double)> dSduFunc = [ = ]( double x ){ return deluSourceFuncs[F_var][u_var](x, sig, q, u);};
	u.MassMatrix( I, dFduSubMatrix, dSduFunc);
}

void SourceObj::dFdsigSubMat(Eigen::MatrixXd& dFdsigSubMatrix, int F_var, int sig_var, DGApprox sig, DGApprox q, DGApprox u, Interval I)
{
	dFdsigSubMatrix.setZero();
	std::function< double (double)> dSdsigFunc = [ = ]( double x ){ return delsigSourceFuncs[F_var][sig_var](x, sig, q, u);};
	sig.MassMatrix( I, dFdsigSubMatrix, dSdsigFunc);
}

void SourceObj::clear()
{
	sourceFuncs.clear();
	delqSourceFuncs.clear();
	delqSourceFuncs.clear();

	delqSourceFuncs.resize(nVar);
	deluSourceFuncs.resize(nVar);
}

//---------------Reaction cases-------------------
void SourceObj::buildSingleVariableLinearTest()
{
	auto nVar = 1;

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return  0.0;};
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	delqSourceFuncs.resize(nVar);
	deluSourceFuncs.resize(nVar);

	delqSourceFuncs[0].push_back(dF_0dq_0);
	deluSourceFuncs[0].push_back(dF_0du_0);
}

void SourceObj::build2DSourceless()
{
	if(nVar != 2) throw std::runtime_error("check your source build, you did it wrong.");

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return  0.0;};
	sourceFuncs.push_back(F_0);
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	delqSourceFuncs.resize(nVar);
	deluSourceFuncs.resize(nVar);

	delqSourceFuncs[0].push_back(dF_0dq_0);
	delqSourceFuncs[0].push_back(dF_0dq_0);
	delqSourceFuncs[1].push_back(dF_0dq_0);
	delqSourceFuncs[1].push_back(dF_0dq_0);

	deluSourceFuncs[0].push_back(dF_0du_0);
	deluSourceFuncs[0].push_back(dF_0du_0);
	deluSourceFuncs[1].push_back(dF_0du_0);
	deluSourceFuncs[1].push_back(dF_0du_0);
}

void SourceObj::build3VariableLinearTest()
{
	if(nVar != 3) throw std::runtime_error("check your source build, you did it wrong.");

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return  0.0;};
	sourceFuncs.push_back(F_0);
	sourceFuncs.push_back(F_0);
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	delqSourceFuncs.resize(nVar);
	deluSourceFuncs.resize(nVar);

	delqSourceFuncs[0].push_back(dF_0dq_0);
	delqSourceFuncs[0].push_back(dF_0dq_0);
	delqSourceFuncs[0].push_back(dF_0dq_0);
	delqSourceFuncs[1].push_back(dF_0dq_0);
	delqSourceFuncs[1].push_back(dF_0dq_0);
	delqSourceFuncs[1].push_back(dF_0dq_0);
	delqSourceFuncs[2].push_back(dF_0dq_0);
	delqSourceFuncs[2].push_back(dF_0dq_0);
	delqSourceFuncs[2].push_back(dF_0dq_0);


	deluSourceFuncs[0].push_back(dF_0du_0);
	deluSourceFuncs[0].push_back(dF_0du_0);
	deluSourceFuncs[0].push_back(dF_0du_0);
	deluSourceFuncs[1].push_back(dF_0du_0);
	deluSourceFuncs[1].push_back(dF_0du_0);
	deluSourceFuncs[1].push_back(dF_0du_0);
	deluSourceFuncs[2].push_back(dF_0du_0);
	deluSourceFuncs[2].push_back(dF_0du_0);
	deluSourceFuncs[2].push_back(dF_0du_0);
}

void SourceObj::build1DFisherSource()
{
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");
	clear();

	double beta = 1.0;
	std::function<double( double, DGApprox, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return  -u(x,0)*beta*(1.0-u(x,0));};
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 2*beta;};
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return -beta + 2.0*beta*u(x,0);};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	delqSourceFuncs.resize(nVar);
	deluSourceFuncs.resize(nVar);
	delqSourceFuncs[0].push_back(dF_0dq_0);
	deluSourceFuncs[0].push_back(dF_0du_0);
}

void SourceObj::build1DConstSource()
{
	auto nVar = 1;

	clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u )
	{
		return -1.0;
	//	if(x>0.5) return  -2.0;
	//	else return 0.0;
	};
	sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	delqSourceFuncs.resize(nVar);
	deluSourceFuncs.resize(nVar);

	delqSourceFuncs[0].push_back(dF_0dq_0);
	deluSourceFuncs[0].push_back(dF_0du_0);
}

void SourceObj::buildCylinderPlasmaConstDensitySource()
{
	//Constants
	double n = 1.0;
	double Te = 30;

	if(nVar != 2) throw std::runtime_error("check your source build, you did it wrong.");

	clear();
	double gamma = 1e-6;
	double I_r = 1e-2;
	
	//Label variables to corespond to specific channels
	int P = 0;
	int omega = 1;

	auto lambda = [n, Te](){
		if(Te<50) return 23.4 - 1.15*::log(n) + 3.45*::log(Te);
		else return 25.3 - 1.15*::log(n) + 2.3*::log(Te);
	};
	std::function<double (double)> tau = [ = ](double Ps){
		return 3.0e9/lambda()*::sqrt(ionMass/(2*protonMass))*::pow(Ps,3/2)/::pow(n,5/2);
	};
	std::function<double (double)> dtaudP = [ = ](double Ps){return 4.5e9/lambda()*::sqrt(ionMass/(2*protonMass))*::pow(Ps,1/2)/::pow(n,5/2);};

	std::function<double( double, DGApprox, DGApprox, DGApprox )> S_P = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return -gamma*R*R*R*u(R,P)*q(R,omega)*q(R,omega)/tau(u(R,P));};
	std::function<double( double, DGApprox, DGApprox, DGApprox )> S_omega = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return  -I_r;};
	sourceFuncs.push_back(S_P);
	sourceFuncs.push_back(S_omega);

	//----------dS/du--------------

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_PdP = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return -gamma*R*R*R*q(R,omega)*q(R,omega)/tau(u(R,P)) + gamma*R*R*R*u(R,P)*q(R,omega)*q(R,omega)/(tau(u(R,P))*tau(u(R,P)))*dtaudP(u(R,P)) ;};
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_Pdomega = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	deluSourceFuncs[0].push_back(dS_PdP);
	deluSourceFuncs[0].push_back(dS_Pdomega);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_omegadP = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_omegadomega = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};


	deluSourceFuncs[1].push_back(dS_omegadP);
	deluSourceFuncs[1].push_back(dS_omegadomega);

	//----------dS/dq--------------

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_PddP = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_Pddomega = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return -2*gamma*R*R*R*u(R,P)*q(R,omega)/tau(u(R,P));};
	
	delqSourceFuncs[0].push_back(dS_PddP);
	delqSourceFuncs[0].push_back(dS_Pddomega);

	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_omegaddP = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_omegaddomega = [ = ]( double R,DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};
	
	delqSourceFuncs[1].push_back(dS_omegaddP);
	delqSourceFuncs[1].push_back(dS_omegaddomega);
}