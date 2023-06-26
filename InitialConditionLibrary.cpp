#include "InitialConditionLibrary.hpp"
#include "Plasma_cases/Plasma.hpp"
#include "Constants.hpp"
#include <cmath>
#include <stdexcept>

std::function<double( double, int )> InitialConditionLibrary::getqInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double, int )> { [=]( double y, int var ){ return -2.0*4.0*(y - 5.0)*::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); }};
	else if(initialCondition == "Sech2") return std::function<double( double, int )> { [=]( double y, int var ){ return -20.0*::tanh(10.0*y)/(::cosh(10.0*y)*::cosh(10.0*y)); } };
	else if(initialCondition == "Sech2_2") return std::function<double( double, int )> { [=]( double y, int var ){ return -6.0*::tanh(3.0*y)/(::cosh(3.0*y)*::cosh(3.0*y)); } };
	else if(initialCondition == "Linear") return std::function<double( double, int )> { [=]( double y, int var ){ return -1.0; } };
	else if(initialCondition == "Const") return std::function<double( double, int )> { [=]( double y, int var ){ return 0.0; } };
	else if(initialCondition == "Zero") return std::function<double( double, int )> { [=]( double y, int var ){ return 0.0; } };
	else if(initialCondition == "Step") return std::function<double( double, int )> { [=]( double y, int var ){ return 0.0; } };
	if(initialCondition == "Test") return std::function<double( double, int )> { [=]( double y, int var )
	{
		if(var == 0) return -2.0*400.0*(y - 0.5)*::exp( -400.0*( y - 0.5 )*( y - 0.5 ) );
		//if(var == 0) return 0.0;
		else return 0.0;
	}};
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

std::function<double( double, int )> InitialConditionLibrary::getuInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double, int )>{ [=]( double y, int var ){ return ::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); } };
	else if(initialCondition == "Sech2") return std::function<double( double, int )> { [=]( double y, int var ){ return 1.0/(::cosh(10.0*y)*::cosh(10.0*y)); } };
	else if(initialCondition == "Sech2_2") return std::function<double( double, int )> { [=]( double y, int var ){ return 1.0/(::cosh(3.0*y)*::cosh(3.0*y)) - 1.0/(::cosh(3.0)*::cosh(3.0)); } };
	else if(initialCondition == "Linear") return std::function<double( double, int )> { [=]( double y, int var ){ return 1.0 - 1.0*y; } };
	else if(initialCondition == "Const") return std::function<double( double, int )> { [=]( double y, int var ){ 
			if( var == 2) return 10000.0;
			else return 10.0; } };
	else if(initialCondition == "Zero") return std::function<double( double, int )> { [=]( double y, int var ){ return 0.0; } };
	else if(initialCondition == "Step") return std::function<double( double, int )> { [=]( double y, int var ){ if(y < 0.5) return 0.0; else return 10.0; } };
	if(initialCondition == "Test") return std::function<double( double, int )>{ [=]( double y, int var )
	{
		//if(var == 0 && y > 0.35 && y < 0.65) return 10.0;
		if(var == 0) return ::exp( -400.0*( y - 0.5 )*( y - 0.5 ));
		else return 0.0; }
	};
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

std::function<double( double, int )> InitialConditionLibrary::getSigInitial()
{
	//??TO DO: everything from here on out should be removed along with the functions in diffObj and sourceObj
	if(diffusionCase == "1DLinearTest" || diffusionCase == "2DLinear") return std::function<double( double, int )> { [=]( double y, int var ){ return -1.0*getqInitial()(y, var); } };
	if(diffusionCase == "3VarLinearTest") return std::function<double( double, int )> { [=]( double y, int var ){ return -1.1*getqInitial()(y, var); } };
	if(diffusionCase == "1DCriticalDiffusion")
	{
		auto xi = [ ]( double q_ )
		{
			if( ::abs(q_) > 0.5) return 10*::pow( ::abs(q_) - 0.5, 0.5) + 1.0;
			else return 1.0;
		};
		return std::function<double( double, int )> { [=]( double y, int var ){ return -1.0*y*xi(getqInitial()(y, var))*getqInitial()(y, var); } };
	}
	if(diffusionCase == "CylinderPlasmaConstDensity")
	{
		//Label variables to corespond to specific channels
		int P = 0;
		int omega = 1;
		auto nVar = 2;
		double beta = 1.0;

		std::function<double (double)> tau = [ = ](double Ps){return 1.0;};
		std::function<double (double)> dtaudP = [ = ](double Ps){return 0.0;};

		std::function<double( double, int )> sig_0 = [ = ]( double R, int var )
		{
			if(var=P) return -beta*R*getuInitial()(R,P)*getqInitial()(R,P)/tau(getuInitial()(R,P));
			else if(var=omega) return -3.0/10.0*R*R*getuInitial()(R,P)*getqInitial()(R,omega)/tau(getuInitial()(R,P));
			else throw std::logic_error( "function not defined for given channel" );
		};
		return sig_0;
	}
	else return std::function<double( double, int )> { [=]( double y, int var ){ return -1.0*getqInitial()(y, var); } };
	//else throw std::logic_error( "Diffusion Case provided does not exist" );
}