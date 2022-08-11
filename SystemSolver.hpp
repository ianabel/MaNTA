#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>

#include "gridStructures.hpp"
#include "Variable.hpp"


class SystemSolver
{
/*This is essentially the encapsulation class of all the different variable. 
Any method here that is replicated in the variable class just calls the variable method for all variables in the system
*/
public:
	SystemSolver(std::shared_ptr<Grid> Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Vars, double Dt, Fn const& rhs, 
					Fn const& Tau, Fn const& c, BoundaryConditions const& boundary );
	~SystemSolver() = default;

	//Functions that just do the stated thing but for every variable
	void initialiseVariables(std::function< double (double)> u_0, N_Vector Y, N_Vector dYdt);
	void initialiseMatrices();
	void setAlpha(double const a);

	//Solves the Jy = -G equation
	void solveJacEq(N_Vector const& G, N_Vector& delY);

	//Initialises kappa as identity
	void initialiseKappa(double kappaConst);

	void print(double t, int nOut);

	int getPolyNum() const {return k;}
	int getCellCount() const {return nCells;}
	int getVariableCount() const {return nVars;}

	std::map<int, Variable> variables;
	std::vector<std::vector<Fn>> Kappa; //Length nVars x nVars. stored as a vector of row vectors
private:
	double dt;
	double t;
	bool initialised = false, firstPrint = true;
	int nVars, nCells, k;
};

struct UserData {
	SystemSolver* system;
};