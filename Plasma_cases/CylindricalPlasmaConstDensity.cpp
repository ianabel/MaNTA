#include "CylindricalPlasmaConstDensity.hpp"
#include "../Constants.hpp"

void CylindricalPlasmaConstDensity::pickVariables()
{
	addVariable("P_ion");
	addVariable("omega");
}

void CylindricalPlasmaConstDensity::seta_fns()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	P_ion.a_fn = [ = ]( double R ){ return 1.0;};
	omega.a_fn = [ = ]( double R ){ return 1.0;};
}

void CylindricalPlasmaConstDensity::setKappas()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	P_ion.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return beta(R)*u(R,P_ion.index)*q(R,P_ion.index)/tauI(u(R,P_ion.index),R);};
	omega.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*u(R,P_ion.index)*q(R,omega.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R));};
}

void CylindricalPlasmaConstDensity::setSources()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	std::function<double( double, DGApprox, DGApprox )> sourceP_ion = [ = ]( double R, DGApprox q, DGApprox u ){ return -gamma*R*R*u(R,P_ion.index)*q(R,omega.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> sourceOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return  -I_r(R);};

	P_ion.setSourceFunc(sourceP_ion);
	omega.setSourceFunc(sourceOmega);
}

void CylindricalPlasmaConstDensity::setdudKappas()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPdP = [ = ]( double R, DGApprox q, DGApprox u ){ return beta(R)*q(R,P_ion.index)/tauI(u(R,P_ion.index),R) - beta(R)*u(R,P_ion.index)*q(R,P_ion.index)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP(u(R,P_ion.index),R) ;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPdOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_ion.addDeluKappaFunc(P_ion.index, dkappaPdP);
	P_ion.addDeluKappaFunc(omega.index, dkappaPdOmega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadP = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*q(R,omega.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R)) - 3.0/10.0*R*R*u(R,P_ion.index)*q(R,omega.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDeluKappaFunc(P_ion.index, dkappaOmegadP);
	omega.addDeluKappaFunc(omega.index, dkappaOmegadOmega);
}

void CylindricalPlasmaConstDensity::setdqdKappas()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPddP = [ = ]( double R, DGApprox q, DGApprox u ){ return beta(R)*u(R,P_ion.index)/tauI(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dkappaPddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_ion.addDelqKappaFunc(P_ion.index, dkappaPddP);
	P_ion.addDelqKappaFunc(omega.index, dkappaPddOmega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*u(R,P_ion.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R));};

	omega.addDelqKappaFunc(P_ion.index, dkappaOmegaddP);
	omega.addDelqKappaFunc(omega.index, dkappaOmegaddOmega);
}

void CylindricalPlasmaConstDensity::setdudSources()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PdP = [ = ]( double R, DGApprox q, DGApprox u ){ return -gamma*R*R*q(R,omega.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R) + gamma*R*R*u(R,P_ion.index)*q(R,omega.index)*q(R,omega.index)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP(u(R,P_ion.index),R) ;};
	std::function<double( double, DGApprox, DGApprox )> dS_Pdomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};


	P_ion.addDeluSourceFunc(P_ion.index, dS_PdP);
	P_ion.addDeluSourceFunc(omega.index, dS_Pdomega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_omegadP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegadomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDeluSourceFunc(P_ion.index, dS_omegadP);
	omega.addDeluSourceFunc(omega.index, dS_omegadomega);
}

void CylindricalPlasmaConstDensity::setdqdSources()
{
	auto& P_ion = variables.at("P_ion");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PddP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_Pddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return -2*gamma*R*R*u(R,P_ion.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R);};

	P_ion.addDelqSourceFunc(P_ion.index, dS_PddP);
	P_ion.addDelqSourceFunc(omega.index, dS_Pddomega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddP = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDelqSourceFunc(P_ion.index, dS_omegaddP);
	omega.addDelqSourceFunc(omega.index, dS_omegaddomega);
}

double CylindricalPlasmaConstDensity::tauI(double Pi, double R)
{
 	if(Pi>0)return 3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(J_eV(Pi),3.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me));
	else return 3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(n(R),3.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me)); //if we have a negative temp just treat it as 1eV
	
	//return 3.0e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,3/2)/::pow(n,5/2);
}

double CylindricalPlasmaConstDensity::dtauIdP(double Pi, double R)
{
	if(Pi>0)return (3.0/2.0)*3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(J_eV(Pi),1.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me));
	else return (3.0/2.0)*3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(n(R),1.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me));
	//return 4.5e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,1/2)/::pow(n,5/2);
}

double CylindricalPlasmaConstDensity::lambda(double R)
{
	//std::cerr << 23.4 - 1.15*::log10(n) + 3.45*::log10(J_eV(Te)) << "	" << J_eV(Te) << std::endl << std::endl;

	if(Te<eV_J(50.0)) return 23.4 - 1.15*::log10(n(R)) + 3.45*::log10(J_eV(Te));
	else return 25.3 - 1.15*::log10(n(R)) + 2.3*::log10(J_eV(Te));
	//return 18.4-1.15*::log10(n)+2.3*::log10(J_eV(Te));
}
