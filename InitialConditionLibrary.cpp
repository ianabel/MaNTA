#include "InitialConditionLibrary.hpp"
#include <cmath>
#include <stdexcept>

Fn InitialConditionLibrary::getqInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double )> { [=]( double y ){ return -2.0*4.0*(y - 5.0)*::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); }};
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

Fn InitialConditionLibrary::getuInitial()
{
	if(initialCondition == "Gaussian") return std::function<double( double )>{ [=]( double y ){ return ::exp( -4.0*( y - 5.0 )*( y - 5.0 ) ); } };
	else throw std::logic_error( "Initial Condition provided does not exist" );
}

Fn InitialConditionLibrary::getSigInitial()
{
	if(diffusionCase == "1dLinearTest") return std::function<double( double )> { [=]( double y ){ return -1.0*getqInitial()(y); } };
	else throw std::logic_error( "Diffusion Case provided does not exist" );
}