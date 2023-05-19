#include "pouseille.hpp"
#include "../Constants.hpp"

void Pouseille::pickVariables()
{
	//Add your variables here
	addVariable("velocity");
}

void Pouseille::seta_fns()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//Assign a_fn
	vel.a_fn = [ = ]( double R ){ return R/nu;};

}

void Pouseille::setKappas()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//assign kappa funtions
	vel.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return R*q(R,vel.index);};
}

void Pouseille::setSources()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//Assign Sources
	vel.sourceFunc = [ = ]( double R, DGApprox sig, DGApprox q, DGApprox u ){return -R*G/nu;};
}

void Pouseille::setdudKappas()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//Derivatives must be added for each kappa function and each variable
	//Therefore the number of functions to be defined and added = (# of channels)^2
	//example here is for 2 variables P_ion and omega
	
	//----------------velocity----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappavdv = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	vel.addDeluKappaFunc(vel.index, dkappavdv);

}

void Pouseille::setdqdKappas()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//We also need to define derivatives wrt the spacial derivative of each variable:
	//Same example as above
	//----------------velocity----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappavddv = [ = ]( double R, DGApprox q, DGApprox u ){ return R;};

	vel.addDelqKappaFunc(vel.index, dkappavddv);
}

void Pouseille::setdudSources()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//The same is done for our source functions
	//example:
	//----------------velocity----------------------
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_vdv = [ = ]( double R, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0 ;};

	vel.addDeluSourceFunc(vel.index, dS_vdv);
}

void Pouseille::setdqdSources()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//And one last set of funcitons
	//example:
	//----------------velocity----------------------
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_vddv = [ = ]( double R, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	vel.addDelqSourceFunc(vel.index, dS_vddv);
}

void Pouseille::setdsigdSources()
{
	//Declare variables here
	auto& vel = variables.at("velocity");

	//And one last set of funcitons
	//example:
	//----------------velocity----------------------
	std::function<double( double, DGApprox, DGApprox, DGApprox )> dS_vdsig_v = [ = ]( double R, DGApprox sig, DGApprox q, DGApprox u ){ return 0.0;};

	vel.addDelsigSourceFunc(vel.index, dS_vdsig_v);
}
