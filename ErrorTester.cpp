#include "ErrorTester.hpp"

void ErrorTester::L2Norm(int k, int nCells, SystemSolver& system)
{
	int nOut = (k+1)*nCells;
	double L2norm = 0.0;
	double h = (xMax - xMin)/nOut;
	for ( int i=0; i<nOut; ++i )
	{
		double x = xMin + ( xMax - xMin ) * ( static_cast<double>( i )/( nOut ) );
		double uNum = system.EvalCoeffs( system.u.Basis, system.u.coeffs, x );
		double uAna = uExact(x,tEnd);
		L2norm += (uNum - uAna)*(uNum - uAna)*h;
		//std::cerr << uAna << "	" << uNum << "	" << L2norm << std::endl;
	}
	L2norm = ::sqrt(L2norm);

	L2vals.emplace((k+1)*nCells, L2norm);
}

void ErrorTester::H1SemiNorm(int k, int nCells, SystemSolver& system)
{
	int nOut = (k+1)*nCells;
	double H1snorm = 0.0;
	double h = (xMax - xMin)/nOut;
	for ( int i=1; i<nOut-1; ++i )
	{
		double x = xMin + ( xMax - xMin ) * ( static_cast<double>( i )/( nOut ) );
		double qNum = system.EvalCoeffs( system.u.Basis, system.q.coeffs, x );
		double qAna = qExact(x,tEnd);
		H1snorm += (qNum - qAna)*(qNum - qAna)*h;
		//std::cerr << qAna << "	" << qNum << "	" << H1snorm << std::endl;
	}

	H1vals.emplace(nOut, H1snorm);
}