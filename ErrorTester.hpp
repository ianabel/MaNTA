#pragma once
#include <functional>
#include "SystemSolver.hpp"

class ErrorTester
{
public:
	ErrorTester( std::function<double( double, double )> uExact_, std::function<double( double, double )> qExact_, double tEnd_)
		: uExact(uExact_), qExact(qExact_), tEnd (tEnd_) {}

	void setBounds(double lBound, double uBound) {xMin = lBound; xMax = uBound;}

	void L2Norm(int k, int nCells, int nVar, SystemSolver& systemm);
	void H1SemiNorm(int k, int nCells, int nVar, SystemSolver& system);

	std::vector<std::map<int, double>> L2vals; //(k+1)*nCells, L2norm
	std::vector<std::map<int, double>> H1vals; //(k+1)*nCells, H1seminorm
private:
	std::function<double( double, double )> uExact, qExact; // (x,t)
	double tEnd;
	double xMin, xMax;
};