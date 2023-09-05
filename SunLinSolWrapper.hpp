#pragma once

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_types.h>
#include <memory>

#include "SystemSolver.hpp"

class SunLinSolWrapper
{
public:
	SunLinSolWrapper( SystemSolver *sysSolver )
		: solver(sysSolver) {}
	~SunLinSolWrapper() = default;

	int Setup( SUNMatrix M );  
	int Solve( SUNMatrix A, N_Vector x, N_Vector b );

	//Sun linear solver operations
	static SUNLinearSolver_Type LSGetType( SUNLinearSolver LS );
	static SUNLinearSolver_ID LSGetID( SUNLinearSolver /* LS */ );
	static int LSinitialize(SUNLinearSolver /* LS */);
	static int LSsetup(SUNLinearSolver LS, SUNMatrix M );
	static int LSsolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x, N_Vector b, realtype);
	static int LSfree(SUNLinearSolver LS);
	static SUNLinearSolver SunLinSol( SystemSolver* solver, void *mem, SUNContext ctx );

private:
	SystemSolver *solver;
};



