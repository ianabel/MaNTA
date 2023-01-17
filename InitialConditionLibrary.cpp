#include "InitialConditionLibrary.hpp"
#include <cmath>
#include <stdexcept>

Fn InitialConditionLibrary::getqInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double )> { [=]( double y ){ return -2.0*4.0*(y - 5.0)*::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); }};
	else if(initialCondition == "Sech2") return std::function<double( double )> { [=]( double y ){ return -20*::tanh(10*y)/(::cosh(10*y)*::cosh(10*y)); } };
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

Fn InitialConditionLibrary::getuInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double )>{ [=]( double y ){ return ::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); } };
	else if(initialCondition == "Sech2") return std::function<double( double )> { [=]( double y ){ return 1/(::cosh(10*y)*::cosh(10*y)); } };
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

Fn InitialConditionLibrary::getSigInitial()
{
	if(diffusionCase == "1dLinearTest") return std::function<double( double )> { [=]( double y ){ return -1.0*getqInitial()(y); } };
	if(diffusionCase == "3VarLinearTest") return std::function<double( double )> { [=]( double y ){ return -1.1*getqInitial()(y); } };
	else throw std::logic_error( "Diffusion Case provided does not exist" );
}