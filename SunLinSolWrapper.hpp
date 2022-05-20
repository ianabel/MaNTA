#pragma once

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_types.h>

#include "SystemSolver.hpp"

class SunLinSolWrapper
{
public:
	SunLinSolWrapper();
	~SunLinSolWrapper();

	int Setup( SUNMatrix M ) ;
	int Solve( SUNMatrix A, N_Vector x, N_Vector b );

private:
	SystemSolver* solver;
};

struct _generic_SUNLinearSolver_Ops LSOps {
	.gettype = LSGetType,
	.getid = LSGetID,
	.setatimes = nullptr,
	.setpreconditioner = nullptr,
	.setscalingvectors = nullptr,
	.initialize = LSinitialize,
	.setup = LSsetup,
	.solve = LSsolve,
	.numiters = nullptr,
	.resnorm = nullptr,
	.lastflag = nullptr,
	.space = nullptr,
	.resid = nullptr,
	.free = LSfree,
};