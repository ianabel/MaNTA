#include "Cylinder3Var.hpp"
#include "../Constants.hpp"

#include <math.h>

void Cylinder3Var::pickVariables()
{
	addVariable("P_ion");
	addVariable("P_e");
	addVariable("omega");
}

void Cylinder3Var::seta_fns()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	P_ion.a_fn = [ = ]( double R ){ return R;};
	P_e.a_fn = [ = ]( double R ){ return R;};
	omega.a_fn = [ = ]( double R ){ return J(R)*R;};
}

void Cylinder3Var::setKappas()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	P_ion.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return (2.0/3.0)*2.0/(Om*Om*mi*n(R))*R*u(R,P_ion.index)*q(R,P_ion.index)/tauI(u(R,P_ion.index),R);};
	P_e.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return (2.0/3.0)*4.66/(Om*Om*me*n(R))*R*u(R,P_e.index)*q(R,P_e.index)/tauE(u(R,P_e.index),R);};
	omega.kappaFunc = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*R*u(R,P_ion.index)*q(R,omega.index)/(Om*Om*tauI(u(R,P_ion.index),R));};
}

void Cylinder3Var::setSources()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	std::function<double( double, DGApprox, DGApprox )> sourceP_ion = [ = ]( double R, DGApprox q, DGApprox u ){ return -3.0/(10.0*Om*Om)*R*R*u(R,P_ion.index)*q(R,omega.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R) - Ci(u(R,P_ion.index), u(R,P_e.index), R);};
	std::function<double( double, DGApprox, DGApprox )> sourceP_e = [ = ]( double R, DGApprox q, DGApprox u ){ return -Ce(u(R,P_ion.index), u(R,P_e.index), R);};
	std::function<double( double, DGApprox, DGApprox )> sourceOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return  -I_r(R)*R*R*B_mid;};

	P_ion.setSourceFunc(sourceP_ion);
	P_e.setSourceFunc(sourceP_e);
	omega.setSourceFunc(sourceOmega);
}

void Cylinder3Var::setdudKappas()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPidPi = [ = ]( double R, DGApprox q, DGApprox u ){ return (2.0/3.0)*2.0/(Om*Om*mi*n(R))*q(R,P_ion.index)/tauI(u(R,P_ion.index),R) - (2.0/3.0)*2.0/(Om*Om*mi*n(R))*u(R,P_ion.index)*q(R,P_ion.index)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP_i(u(R,P_ion.index),R) ;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPidPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPidOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_ion.addDeluKappaFunc(P_ion.index, dkappaPidPi);
	P_ion.addDeluKappaFunc(P_e.index, dkappaPidPe);
	P_ion.addDeluKappaFunc(omega.index, dkappaPidOmega);

	//----------------P_e----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPedPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPedPe = [ = ]( double R, DGApprox q, DGApprox u ){ return (2.0/3.0)*4.66/(Om*Om*mi*n(R))*q(R,P_e.index)/tauE(u(R,P_e.index),R) - (2.0/3.0)*4.66/(Om*Om*mi*n(R))*u(R,P_e.index)*q(R,P_e.index)/(tauE(u(R,P_e.index),R)*tauE(u(R,P_e.index),R))*dtauEdP_e(u(R,P_e.index),R) ;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPedOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_e.addDeluKappaFunc(P_ion.index, dkappaPedPi);
	P_e.addDeluKappaFunc(P_e.index, dkappaPedPe);
	P_e.addDeluKappaFunc(omega.index, dkappaPedOmega);


	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*q(R,omega.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R)) - 3.0/10.0*R*R*u(R,P_ion.index)*q(R,omega.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP_i(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegadOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDeluKappaFunc(P_ion.index, dkappaOmegadPi);
	omega.addDeluKappaFunc(P_e.index, dkappaOmegadPe);
	omega.addDeluKappaFunc(omega.index, dkappaOmegadOmega);
}

void Cylinder3Var::setdqdKappas()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPiddPi = [ = ]( double R, DGApprox q, DGApprox u ){ return (2.0/3.0)*2.0/(Om*Om*mi*n(R))*u(R,P_ion.index)/tauI(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dkappaPiddPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPiddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_ion.addDelqKappaFunc(P_ion.index, dkappaPiddPi);
	P_ion.addDelqKappaFunc(P_e.index, dkappaPiddPe);
	P_ion.addDelqKappaFunc(omega.index, dkappaPiddOmega);

	//----------------P_e----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaPeddPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaPeddPe = [ = ]( double R, DGApprox q, DGApprox u ){ return (2.0/3.0)*4.66/(Om*Om*mi*n(R))*u(R,P_e.index)/tauE(u(R,P_ion.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dkappaPeddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_e.addDelqKappaFunc(P_ion.index, dkappaPeddPi);
	P_e.addDelqKappaFunc(P_e.index, dkappaPeddPe);
	P_e.addDelqKappaFunc(omega.index, dkappaPeddOmega);


	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddPe  = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappaOmegaddOmega = [ = ]( double R, DGApprox q, DGApprox u ){ return 3.0/10.0*R*R*u(R,P_ion.index)/(J(R)*Om*Om*tauI(u(R,P_ion.index),R));};

	omega.addDelqKappaFunc(P_ion.index, dkappaOmegaddPi);
	omega.addDelqKappaFunc(P_e.index, dkappaOmegaddPe);
	omega.addDelqKappaFunc(omega.index, dkappaOmegaddOmega);
}

void Cylinder3Var::setdudSources()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PidPi = [ = ]( double R, DGApprox q, DGApprox u ){ return -3.0/(10.0*Om*Om)*R*R*q(R,omega.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R) + 3.0/(10.0*Om*Om)*R*R*u(R,P_ion.index)*q(R,omega.index)*q(R,omega.index)/(tauI(u(R,P_ion.index),R)*tauI(u(R,P_ion.index),R))*dtauIdP_i(u(R,P_ion.index),R) - dCidPi(u(R,P_ion.index),u(R,P_e.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dS_PidPe = [ = ]( double R, DGApprox q, DGApprox u ){ return -dCidPe(u(R,P_ion.index),u(R,P_e.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dS_Pidomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};


	P_ion.addDeluSourceFunc(P_ion.index, dS_PidPi);
	P_ion.addDeluSourceFunc(P_e.index, dS_PidPe);
	P_ion.addDeluSourceFunc(omega.index, dS_Pidomega);

	//----------------P_e----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PedPi = [ = ]( double R, DGApprox q, DGApprox u ){ return -dCedPi(u(R,P_ion.index),u(R,P_e.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dS_PedPe = [ = ]( double R, DGApprox q, DGApprox u ){ return -dCedPe(u(R,P_ion.index),u(R,P_e.index),R);};
	std::function<double( double, DGApprox, DGApprox )> dS_Pedomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};


	P_e.addDeluSourceFunc(P_ion.index, dS_PedPi);
	P_e.addDeluSourceFunc(P_e.index, dS_PedPe);
	P_e.addDeluSourceFunc(omega.index, dS_Pedomega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_omegadPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegadPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegadomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDeluSourceFunc(P_ion.index, dS_omegadPi);
	omega.addDeluSourceFunc(P_e.index, dS_omegadPe);
	omega.addDeluSourceFunc(omega.index, dS_omegadomega);
}

void Cylinder3Var::setdqdSources()
{
	auto& P_ion = variables.at("P_ion");
	auto& P_e = variables.at("P_e");
	auto& omega = variables.at("omega");

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PiddPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_PiddPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_Piddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return -2*3.0/(10.0*Om*Om)*R*R*u(R,P_ion.index)*q(R,omega.index)/tauI(u(R,P_ion.index),R);};

	P_ion.addDelqSourceFunc(P_ion.index, dS_PiddPi);
	P_ion.addDelqSourceFunc(P_e.index, dS_PiddPe);
	P_ion.addDelqSourceFunc(omega.index, dS_Piddomega);

	//----------------P_ion----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_PeddPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_PeddPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_Peddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	P_e.addDelqSourceFunc(P_ion.index, dS_PeddPi);
	P_e.addDelqSourceFunc(P_e.index, dS_PeddPe);
	P_e.addDelqSourceFunc(omega.index, dS_Peddomega);

	//----------------omega----------------------
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddPi = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddPe = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dS_omegaddomega = [ = ]( double R, DGApprox q, DGApprox u ){ return 0.0;};

	omega.addDelqSourceFunc(P_ion.index, dS_omegaddPi);
	omega.addDelqSourceFunc(P_e.index, dS_omegaddPe);
	omega.addDelqSourceFunc(omega.index, dS_omegaddomega);
}

double Cylinder3Var::tauI(double Pi, double R)
{
 	if(Pi>0)return 3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(J_eV(Pi),3.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me));
	else return 3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(n(R),3.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me)); //if we have a negative temp just treat it as 1eV
	
	//return 3.0e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,3/2)/::pow(n,5/2);
}

double Cylinder3Var::dtauIdP_i(double Pi, double R)
{
	if(Pi>0)return (3.0/2.0)*3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(J_eV(Pi),1.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me));
	else return (3.0/2.0)*3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(n(R),1.0/2.0))*(1.0/lambda(R))*(::sqrt(mi/me));
	//return 4.5e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,1/2)/::pow(n,5/2);
}

double Cylinder3Var::lambda(double R)
{
	return 23.4 - 1.15*::log10(n(R)) + 3.45*::log10(40);

	//return 18.4-1.15*::log10(n)+2.3*::log10(J_eV(Te));
}

double Cylinder3Var::tauE(double Pe, double R)
{
 	if(Pe>0)return 3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(J_eV(Pe),3.0/2.0))*(1.0/lambda(R));
	else return 3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(n(R),3.0/2.0))*(1.0/lambda(R)); //if we have a negative temp just treat it as 1eV
	
	//return 3.0e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,3/2)/::pow(n,5/2);
}

double Cylinder3Var::dtauEdP_e(double Pe, double R)
{
	if(Pe>0)return (3.0/2.0)*3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(J_eV(Pe),1.0/2.0))*(1.0/lambda(R));
	else return (3.0/2.0)*3.44e11*(1.0/::pow(n(R),5.0/2.0))*(::pow(n(R),1.0/2.0))*(1.0/lambda(R));
	//return 4.5e9/lambda()*::sqrt(mi/(2*mp))*::pow(Pi,1/2)/::pow(n,5/2);
}

double Cylinder3Var::nu(double Pi, double Pe, double R)
{
	return 0.0;
	double alpha = ::pow(2,5/2)*::sqrt(M_PI)*::pow(e_charge,4)*n(R)*n(R)*::sqrt(mi*me)*lambda(R);
	return alpha/n(R)*::pow(mi*Pe+me*Pi, -3.0/2.0);
}

double Cylinder3Var::dnudPi(double Pi, double Pe,  double R)
{
	return 0.0;
	double alpha = ::pow(2,5/2)*::sqrt(M_PI)*::pow(e_charge,4)*n(R)*n(R)*::sqrt(mi*me)*lambda(R);
	return 1.5*alpha*me/n(R)*::pow(mi*Pe+me*Pi, -5.0/2.0);
}

double Cylinder3Var::dnudPe(double Pi, double Pe,  double R)
{
	return 0.0;
	double alpha = ::pow(2,5/2)*::sqrt(M_PI)*::pow(e_charge,4)*n(R)*n(R)*::sqrt(mi*me)*lambda(R);
	return 1.5*alpha*mi/n(R)*::pow(mi*Pe+me*Pi, -5.0/2.0);
}

double Cylinder3Var::Ce(double Pi, double Pe, double R)
{
	return R*nu(Pi, Pe, R)*(Pi-Pe)/n(R);
}

double Cylinder3Var::dCedPe(double Pi, double Pe, double R)
{
	return -R*nu(Pi,Pe,R)/n(R) + R*dnudPe(Pi,Pe,R)*(Pi-Pe)/n(R);
}

double Cylinder3Var::dCedPi(double Pi, double Pe, double R)
{
	return R*nu(Pi,Pe,R)/n(R) + R*dnudPi(Pi,Pe,R)*(Pi-Pe)/n(R);
}

double Cylinder3Var::Ci(double Pi, double Pe, double R)
{
	return -Ce(Pi, Pe, R);
}

double Cylinder3Var::dCidPe(double Pi, double Pe, double R)
{
	return -dCedPe(Pi, Pe, R);
}

double Cylinder3Var::dCidPi(double Pi, double Pe, double R)
{
	return -dCedPi(Pi, Pe, R);
}