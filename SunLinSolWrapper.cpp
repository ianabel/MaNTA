#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_types.h>

#include "SunLinSolWrapper.hpp"
#include "SystemSolver.hpp"

#define LSWrapper( ls ) reinterpret_cast<SunLinSolWrapper*>( LS->content )

SUNLinearSolver_Type LSGetType( SUNLinearSolver LS )
{
	return SUNLINEARSOLVER_DIRECT;
}

SUNLinearSolver_ID LSGetID( SUNLinearSolver /* LS */ )
{
	return SUNLINEARSOLVER_CUSTOM;
}

int LSinitialize(SUNLinearSolver /* LS */)
{
	return SUNLS_SUCCESS;
}

int LSsetup(SUNLinearSolver LS, SUNMatrix M )
{
	int err = LSWrapper( LS )->Setup( M );
	return err;
}

int LSsolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x, N_Vector b, realtype)
{
	int err = LSWrapper( LS )->Solve( M, x, b );
	return err;
}

int LSfree(SUNLinearSolver LS)
{
	delete LSWrapper( LS );
	return SUNLS_SUCCESS;
}

typedef int (*IDALsJacTimesVecFn)(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, N_Vector v, N_Vector Jv, realtype c_j, void *user_data, N_Vector tmp1, N_Vector tmp2)
{
	solver.setVectors(N_vector yy, N_vector yp);
	solver.solve(c_j);
	
}