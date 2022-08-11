#include "SystemSolver.hpp"
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "gridStructures.hpp"

SystemSolver::SystemSolver(std::shared_ptr<Grid> Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Vars, double Dt, Fn const& rhs, 
	Fn const& Tau, Fn const& c, BoundaryConditions const& boundary )
	: nVars(N_Vars), nCells(N_cells), k(polyNum)
{
	for ( int i = 0; i<nVars; i++)
		variables.emplace(i, Variable(Grid, polyNum, N_cells, Dt, rhs, Tau, c, boundary));
}

void SystemSolver::initialiseVariables(std::function< double (double)> u_0, N_Vector Y, N_Vector dYdt)
{
	if( N_VGetLength(Y) != nVars*2*nCells*(k+1))
		throw std::invalid_argument( "Sundials Vector does not match size \n" );
	if( N_VGetLength(dYdt) != nVars*2*nCells*(k+1))
		throw std::invalid_argument( "Sundials Vector does not match size \n" );

	int i = 0;
	for(auto & [key, var] : variables)
	{
		realtype* Yptr =  N_VGetArrayPointer( Y )+i*2*nCells*(k+1);
		realtype* dYdtptr =  N_VGetArrayPointer( dYdt )+i*2*nCells*(k+1);
		var.setInitialConditions(u_0, Yptr, dYdtptr);
		i++;
	}
}

void SystemSolver::initialiseMatrices()
{
	for(auto & [key, var] : variables)
	{
		var.initialiseMatrices();
	}
}

void SystemSolver::initialiseKappa(double kappaConst)
{
	for (int i = 0; i < nVars; i++)
	{
		std::vector<Fn> row;
		for (int j = 0; j < nVars; j++)
		{
			if(i == j) row.emplace_back(std::function<double( double )> {[ = ]( double x ){ return kappaConst;}});
			else row.emplace_back(std::function<double( double )> {[ = ]( double x ){ return 0.0;}});
		}
		Kappa.emplace_back(row);
	}

	int i = 0;
	for(auto & [key, var] : variables)
	{
		var.setKappa_ii(Kappa[i][i]);
		i++;
	}
}

void SystemSolver::setAlpha(double const a)
{
	for(auto & [key, var] : variables)
	{
		var.setAlpha(a);
	}
}

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data)
{
	auto system = static_cast<UserData*>(user_data)->system;
	int nCells = system->getCellCount(), k = system->getPolyNum();

	int i = 0;
	for(auto & [key, var] : system->variables)
	{
		realtype* varYptr =  N_VGetArrayPointer( Y )+i*2*nCells*(k+1);
		realtype* vardYdtptr =  N_VGetArrayPointer( dydt )+i*2*nCells*(k+1);
		realtype* varResvalprt =  N_VGetArrayPointer( resval )+i*2*nCells*(k+1);
		var.residual(varYptr, vardYdtptr, varResvalprt);
		i++;
	}

	return 0;
}

void SystemSolver::solveJacEq(N_Vector const& G, N_Vector& delY)
{
	int i = 0;
	for(auto & [key, var] : variables)
	{
		realtype* varGptr =  N_VGetArrayPointer( G )+i*2*nCells*(k+1);
		realtype* varDelYptr =  N_VGetArrayPointer( delY )+i*2*nCells*(k+1);
		var.solveJacEq(varGptr, varDelYptr);
		i++;
	}
}

void SystemSolver::print(double t, int nOut)
{
	int i=0;
	for(auto & [key, var] : variables)
	{
		std::string filename = "u_t_" + std::to_string(i) + ".plot";
		if(firstPrint)
		{
			std::ofstream out (filename);
			var.print(out, t, nOut);
		}
		else
		{
			std::ofstream out (filename, std::ios::app);
			var.print(out, t, nOut);
		}
		i++;
	}
	firstPrint = false;
}