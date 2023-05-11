#include "Template.hpp"
#include "../Constants.hpp"

void PlasmaTemplate::pickVariables()
{
	//Add your variables here
	//example: addVariable("P_ion");
}

void PlasmaTemplate::seta_fns()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//Assign a_fn
	//example: P_ion.a_fn = [ = ]( double R ){ return R;};

}

void PlasmaTemplate::setKappas()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//assign kappa funtions
	//example: P_ion.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return beta(R)*R*u(R,P_ion.index)*q(R,P_ion.index)/tauI(u(R,P_ion.index),R);};
}

void PlasmaTemplate::setSources()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//Assign Sources
	//example: P_ion.sourceFunc = [ = ]( double R, DGApprox q, DGApprox u ){return -3.0/(10.0*Om*Om)*R*R*R*u(R,P_ion.index)*q(R,omega.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R);};
}

void PlasmaTemplate::setdudKappas()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//Derivatives must be added for each kappa function and each variable
	//Therefore the number of functions to be defined and added = (# of channels)^2
	//example here is for 2 variables P_ion and omega
	/*
	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPdP = [ = ]( double R, DGApprox q, DGApprox u ){ return beta(R)*R*q(R,P_ion.index)/tauI(u(R,P_ion.index),R) - beta(R)*R*u(R,P_ion.index)*q(R,P_ion.index)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP(u(R,P_ion.index),R) ;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPdOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_ion.addDeluKappaFunc(P_ion.index, dkappaPdP);
	P_ion.addDeluKappaFunc(omega.index, dkappaPdOmega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadP = [ = ]( double R, DGApprox q, DGApprox u ){ return (3.0/10.0*R*R*R*q(R,omega.index)/(Om*Om))*(1/tauI(u(R,P_ion.index),R) - u(R,P_ion.index)*dtauIdP(u(R,P_ion.index),R)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))); };
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDeluKappaFunc(P_ion.index, dkappaOmegadP);
	omega.addDeluKappaFunc(omega.index, dkappaOmegadOmega);
	*/
}

void PlasmaTemplate::setdqdKappas()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//We also need to define derivatives wrt the spacial derivative of each variable:
	//Same example as above
	/*
	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPddP = [ = ]( double R, DGApprox q, DGApprox u ){ return beta(R)*R*u(R,P_ion.index)/tauI(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dkappaPddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_ion.addDelqKappaFunc(P_ion.index, dkappaPddP);
	P_ion.addDelqKappaFunc(omega.index, dkappaPddOmega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*R*u(R,P_ion.index)/(Om*Om*tauI(u(R,P_ion.index),R));};

	omega.addDelqKappaFunc(P_ion.index, dkappaOmegaddP);
	omega.addDelqKappaFunc(omega.index, dkappaOmegaddOmega);
	*/
}

void PlasmaTemplate::setdudSources()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//The same is done for our source functions
	//example:
	/*
	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PdP = [ = ]( double R, DGApprox q, DGApprox u ){ return -3.0/(10.0*Om*Om)*R*R*R*q(R,omega.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R) + 3.0/(10.0*Om*Om)*R*R*R*u(R,P_ion.index)*q(R,omega.index)*q(R,omega.index)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP(u(R,P_ion.index),R) ;};
	std::function<double( double, DGApprox, DGApprox )> dS_Pdomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};


	P_ion.addDeluSourceFunc(P_ion.index, dS_PdP);
	P_ion.addDeluSourceFunc(omega.index, dS_Pdomega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_omegadP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegadomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDeluSourceFunc(P_ion.index, dS_omegadP);
	omega.addDeluSourceFunc(omega.index, dS_omegadomega);
	*/
}

void PlasmaTemplate::setdqdSources()
{
	//Declare variables here
	//example: auto& P_ion = variables.at("P_ion");

	//And one last set of funcitons
	//example:
	/*
	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PddP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_Pddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return -2*3.0/(10.0*Om*Om)*R*R*R*u(R,P_ion.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R);};

	P_ion.addDelqSourceFunc(P_ion.index, dS_PddP);
	P_ion.addDelqSourceFunc(omega.index, dS_Pddomega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDelqSourceFunc(P_ion.index, dS_omegaddP);
	omega.addDelqSourceFunc(omega.index, dS_omegaddomega);
	*/
}

