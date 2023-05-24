#include "Diagnostic.hpp"

double Diagnostic::Voltage() const
{
	double B_mid = 0.3;
	//Calculates the approximate integral of omega over the domain
	int omegaIndex;
	try 
	{
		omegaIndex = plasma->getVariable("omega").index;
	}
	catch (const std::out_of_range& e) { return -1.0; }

	double voltage = 0.0;
	double dx = ( system->BCs->UpperBound - system->BCs->LowerBound ) /( 199 );

	for ( int i=0; i<199; ++i )
	{
		//Uses a simple parallelogram rule
		double x = system->BCs->LowerBound + ( system->BCs->UpperBound - system->BCs->LowerBound ) * ( static_cast<double>( i )/( 199 ) ) + 0.5*dx;
		voltage += B_mid*x*system->EvalCoeffs( system->u.Basis, system->u.coeffs, x, omegaIndex )*dx;
	}
	return voltage;
}