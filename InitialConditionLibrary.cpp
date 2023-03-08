#include "InitialConditionLibrary.hpp"
#include <cmath>
#include <stdexcept>

std::function<double( double, int )> InitialConditionLibrary::getqInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double, int )> { [=]( double y, int var ){ return -2.0*4.0*(y - 5.0)*::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); }};
	else if(initialCondition == "Sech2") return std::function<double( double, int )> { [=]( double y, int var ){ return -20.0*::tanh(10.0*y)/(::cosh(10.0*y)*::cosh(10.0*y)); } };
	else if(initialCondition == "Sech2_2") return std::function<double( double, int )> { [=]( double y, int var ){ return -6.0*::tanh(3.0*y)/(::cosh(3.0*y)*::cosh(3.0*y)); } };
	else if(initialCondition == "Linear") return std::function<double( double, int )> { [=]( double y, int var ){ return -1.0; } };
	else if(initialCondition == "Const") return std::function<double( double, int )> { [=]( double y, int var ){ return 0.0; } };
	if(initialCondition == "Test") return std::function<double( double, int )> { [=]( double y, int var )
	{
		if(var == 0) return -2.0*4.0*(y - 5.0)*::exp( -4.0*( y - 5.0 )*( y - 5.0 ) );
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
	else if(initialCondition == "Const") return std::function<double( double, int )> { [=]( double y, int var ){ return 0.5; } };
	if(initialCondition == "Test") return std::function<double( double, int )>{ [=]( double y, int var )
	{
		if(var == 0) return ::exp( -4.0*( y - 5.0 )*( y - 5.0 ) );
		else return 0.5; }
	};
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

std::function<double( double, int )> InitialConditionLibrary::getSigInitial()
{
	if(diffusionCase == "1DLinearTest") return std::function<double( double, int )> { [=]( double y, int var ){ return -1.0*getqInitial()(y, var); } };
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
	/*
	if(diffusionCase == "CylinderPlasmaConstDensity")
	{
		//P = 0, Omega = 1
		if()
	}
	*/
	else throw std::logic_error( "Diffusion Case provided does not exist" );
}