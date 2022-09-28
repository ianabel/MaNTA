#include "ErrorTester.hpp"

void ErrorTester::L2Norm(int k, int nCells, int nVar, SystemSolver& system)
{
	if(L2vals.size() != nVar) L2vals.resize(nVar);

	int nOut = (k+1)*nCells;
	double h = (xMax - xMin)/nOut;
	for(int var = 0; var < nVar; var++)
	{
		double L2norm = 0.0;
		for ( int i=0; i<nOut; ++i )
		{
			double x = xMin + ( xMax - xMin ) * ( static_cast<double>( i )/( nOut ) );
			double uNum = system.EvalCoeffs( system.u.Basis, system.u.coeffs, x, var );
			double uAna = uExact(x,tEnd);
			L2norm += (uNum - uAna)*(uNum - uAna)*h;
		}
		L2norm = ::sqrt(L2norm);
		L2vals[var].emplace((k+1)*nCells, L2norm);
	}
}

void ErrorTester::H1SemiNorm(int k, int nCells, int nVar, SystemSolver& system)
{
	if(H1vals.size() != nVar) H1vals.resize(nVar);

	int nOut = (k+1)*nCells;
	double H1snorm = 0.0;
	double h = (xMax - xMin)/nOut;
	for(int var = 0; var < nVar; var++)
	{
		for ( int i=1; i<nOut-1; ++i )
		{
			double x = xMin + ( xMax - xMin ) * ( static_cast<double>( i )/( nOut ) );
			double qNum = system.EvalCoeffs( system.u.Basis, system.q.coeffs, x, var );
			double qAna = qExact(x,tEnd);
			H1snorm += (qNum - qAna)*(qNum - qAna)*h;
		}
		H1snorm = ::sqrt(H1snorm);
		H1vals[var].emplace(nOut, H1snorm);
	}

}