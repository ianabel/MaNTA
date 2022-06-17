#include "SunLinSolWrapper.hpp"

int SunLinSolWrapper::Solve( SUNMatrix A, N_Vector x, N_Vector b )
{
	solver.solveJacEq(alpha, b, x);
	return 0;
}

int SunLinSolWrapper::Setup( SUNMatrix mat)
{
	return 0;
}

#define LSWrapper( ls ) reinterpret_cast<SunLinSolWrapper*>( LS->content )

SUNLinearSolver_Type SunLinSolWrapper::LSGetType( SUNLinearSolver LS )
{
	return SUNLINEARSOLVER_DIRECT;
}

SUNLinearSolver_ID SunLinSolWrapper::LSGetID( SUNLinearSolver /* LS */ )
{
	return SUNLINEARSOLVER_CUSTOM;
}

int SunLinSolWrapper::LSinitialize(SUNLinearSolver /* LS */)
{
	return SUNLS_SUCCESS;
}

int SunLinSolWrapper::LSsetup(SUNLinearSolver LS, SUNMatrix M )
{
	int err = LSWrapper( LS )->Setup( M );
	return err;
}

int SunLinSolWrapper::LSsolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x, N_Vector b, realtype)
{
	int err = LSWrapper( LS )->Solve( M, x, b );
	return err;
}

int SunLinSolWrapper::LSfree(SUNLinearSolver LS)
{
	delete LSWrapper( LS );
	return SUNLS_SUCCESS;
}

struct _generic_SUNLinearSolver_Ops LSOps = 
{
	.gettype = SunLinSolWrapper::LSGetType,
	.getid = SunLinSolWrapper::LSGetID,
	.setatimes = nullptr,
	.setpreconditioner = nullptr,
	.setscalingvectors = nullptr,
	.initialize = SunLinSolWrapper::LSinitialize,
	.setup = SunLinSolWrapper::LSsetup,
	.solve = SunLinSolWrapper::LSsolve,
	.numiters = nullptr,
	.resnorm = nullptr,
	.lastflag = nullptr,
	.space = nullptr,
	.resid = nullptr,
	.free = SunLinSolWrapper::LSfree,
};

SUNLinearSolver SunLinSolWrapper::SunLinSol(SystemSolver& solver)
{
	SUNLinearSolver LS = SUNLinSolNewEmpty();
	LS->content = new SunLinSolWrapper(solver);
	LS->ops = &LSOps;
	return LS;
}
