#include "InitialConditionLibrary.hpp"
#include <cmath>
#include <stdexcept>

Fn InitialConditionLibrary::getqInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double )> { [=]( double y ){ return -2.0*4.0*(y - 5.0)*::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); }};
	else if(initialCondition == "Sech2") return std::function<double( double )> { [=]( double y ){ return -20.0*::tanh(10.0*y)/(::cosh(10.0*y)*::cosh(10.0*y)); } };
	else if(initialCondition == "Sech2_2") return std::function<double( double )> { [=]( double y ){ return -6.0*::tanh(3.0*y)/(::cosh(3.0*y)*::cosh(3.0*y)); } };
	else if(initialCondition == "Linear") return std::function<double( double )> { [=]( double y ){ return -1.0; } };
	else if(initialCondition == "Const") return std::function<double( double )> { [=]( double y ){ return 0.0; } };
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

Fn InitialConditionLibrary::getuInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double )>{ [=]( double y ){ return ::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); } };
	else if(initialCondition == "Sech2") return std::function<double( double )> { [=]( double y ){ return 1.0/(::cosh(10.0*y)*::cosh(10.0*y)); } };
	else if(initialCondition == "Sech2_2") return std::function<double( double )> { [=]( double y ){ return 1.0/(::cosh(3.0*y)*::cosh(3.0*y)) - 1.0/(::cosh(3.0)*::cosh(3.0)); } };
	else if(initialCondition == "Linear") return std::function<double( double )> { [=]( double y ){ return 1.0 - 1.0*y; } };
	else if(initialCondition == "Const") return std::function<double( double )> { [=]( double y ){ return 0.5; } };
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

Fn InitialConditionLibrary::getSigInitial()
{
	if(diffusionCase == "1DLinearTest") return std::function<double( double )> { [=]( double y ){ return -1.0*getqInitial()(y); } };
	if(diffusionCase == "3VarLinearTest") return std::function<double( double )> { [=]( double y ){ return -1.1*getqInitial()(y); } };
	if(diffusionCase == "1DCriticalDiffusion")
	{
		auto xi = [ ]( double q_ )
		{
			if( ::abs(q_) > 0.5) return 10*::pow( ::abs(q_) - 0.5, 0.5) + 1.0;
			else return 1.0;
		};
		return std::function<double( double )> { [=]( double y ){ return -1.0*y*xi(getqInitial()(y))*getqInitial()(y); } };
	}
	else throw std::logic_error( "Diffusion Case provided does not exist" );
}