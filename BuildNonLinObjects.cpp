#include "DiffusionObj.hpp"
#include "SourceObj.hpp"

//----------------------Diffusion Obects----------------

//This is user input. Eventually this will be put in an input file format, but for now its hard coded.
/*
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double kappa_const = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (7/4)*q(x,0) - (::sqrt(3)/4)*q(x,1);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -(::sqrt(3)/4)*q(x,0) + (7/4)*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (7/4);};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -(::sqrt(3)/4);};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -(::sqrt(3)/4);};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return (7/4);};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);
	diffobj->delqKappaFuncs[0].push_back(dkappa0dq1);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq0);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq1);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
	diffobj->deluKappaFuncs[0].push_back(dkappa0du1);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du0);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du1);
}

void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double kappa_const = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5))*q(x,0);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5))*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5));};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5));};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);
	diffobj->delqKappaFuncs[0].push_back(dkappa0dq1);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq0);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq1);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
	diffobj->deluKappaFuncs[0].push_back(dkappa0du1);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du0);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du1);
}
*/

//2 variable non-linear case
/*
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + u(x,0)*u(x,1))*q(x,0);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + u(x,1)*u(x,0))*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + u(x,0)*u(x,1));};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + u(x,1)*u(x,0));};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*u(x,1)*q(x,0);};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*u(x,0)*q(x,0);};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*u(x,1)*q(x,1);};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*u(x,0)*q(x,1);};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);
	diffobj->delqKappaFuncs[0].push_back(dkappa0dq1);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq0);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq1);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
	diffobj->deluKappaFuncs[0].push_back(dkappa0du1);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du0);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du1);
}
*/

//2-Dimensional Linear
/*
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*q(x,0);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);
	diffobj->delqKappaFuncs[0].push_back(dkappa0dq1);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq0);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq1);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
	diffobj->deluKappaFuncs[0].push_back(dkappa0du1);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du0);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du1);
}
*/

//1 Variable non-linear
/*
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + ::pow(u(x,0),5))*q(x,0);};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + ::pow(u(x,0),5));};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 5*beta*::pow(u(x,0),4)*q(x,0);};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}
*/

/*
//PTransp
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;
	double alpha = 0.5;
	double kap = 10;
	double qc = 0.5;
	double xi_o = 1.0;

	auto xi = [ ]( double q_ )
	{
		if( ::abs(q_) > 0.5) return 10*::pow( ::abs(q_) - 0.5, 0.5) + 1.0;
		else return 1.0;
	};

	auto dxidq = [ ]( double q_ )
	{
		if( ::abs(q_) > 0.5) return 5.0*q_/(::abs(q_)*::pow( ::abs(q_) - 0.5, 0.5));
		else return 0.0;
	};


	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  x*xi(q(x,0))*q(x,0);};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return x*xi(q(x,0)) + x*q(x,0)*dxidq(q(x,0));};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}
*/



//Shestakov case
/*
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u )
	{
		double u_ = std::max(0.1, u(x,0));
		//if(::pow(q(x,0) / u_,2)*q(x,0) > 100.0 ) return 100.0;
		//else if(::pow(q(x,0) / u_,2)*q(x,0) < -100.0 ) return -100.0;
		return ::pow(q(x,0) / u_,2)*q(x,0);
	};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u )
	{
		double u_ = std::max(0.1, u(x,0));
		//if(3*::pow(q(x,0) / u_,2) > 100.0) return 100.0;
		//else if(3*::pow(q(x,0) / u_,2) < -100.0) return -100.0;
		return 3*::pow(q(x,0) / u_,2);
	};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u )
	{
		double u_ = std::max(0.1, u(x,0));
		//if(-2*::pow(q(x,0) / u_,3) > 100.0) return 100.0;
		//else if(-2*::pow(q(x,0) / u_,3) < -100.0) return -100.0;
		return -2*::pow(q(x,0) / u_,3);
	};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}
*/

//Single variable linear case
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  1.0*q(x,0);};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}


//----------------------Source Obects----------------
//Sourceless case - 2 channel
/*
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  0.0;};
	sourceobj->sourceFuncs.push_back(F_0);
	sourceobj->sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);
	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);
	sourceobj->dFdqFuncs[1].push_back(dF_0dq_0);
	sourceobj->dFdqFuncs[1].push_back(dF_0dq_0);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
	sourceobj->dFduFuncs[1].push_back(dF_0du_0);
	sourceobj->dFduFuncs[1].push_back(dF_0du_0);
}
*/


// 1D sourceless
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  0.0;};
	sourceobj->sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
}

//1D Fisher's Equation
/*
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  -u(x,0)*beta*(1.0-u(x,0));};
	sourceobj->sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -beta + 2.0*beta*u(x,0);};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
}
*/

//2D Fisher's Equation
/*
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  -u(x,0)*beta*(1.0-u(x,0));};
	std::function<double( double, DGApprox, DGApprox )> F_1 = [ = ]( double x, DGApprox q, DGApprox u ){ return  -2.0*u(x,1)*beta*(1.0-u(x,1));};
	sourceobj->sourceFuncs.push_back(F_0);
	sourceobj->sourceFuncs.push_back(F_1);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dF_0dq_1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dF_1dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dF_1dq_1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -beta + 2.0*beta*u(x,0);};
	std::function<double( double, DGApprox, DGApprox )> dF_0du_1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dF_1du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dF_1du_1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -2.0*beta + 4.0*beta*u(x,1);};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);
	sourceobj->dFdqFuncs[0].push_back(dF_0dq_1);
	sourceobj->dFdqFuncs[1].push_back(dF_1dq_0);
	sourceobj->dFdqFuncs[1].push_back(dF_1dq_1);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
	sourceobj->dFduFuncs[0].push_back(dF_0du_1);
	sourceobj->dFduFuncs[1].push_back(dF_1du_0);
	sourceobj->dFduFuncs[1].push_back(dF_1du_1);
}
*/

//Fisher's Equation - exact sol
/*
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  -u(x,0)*(0.5-u(x,0));};
	sourceobj->sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -0.5 + 2.0*u(x,0);};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
}
*/

//Shestakov case
/*
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();
	double So = -1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  x < 0.1 ? So : 0.0;};
	sourceobj->sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
}
*/
