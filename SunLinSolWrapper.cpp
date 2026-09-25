#include "SunLinSolWrapper.hpp"
#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type sunrealtype          */
#include <exception>
#include <memory>
#include <print>

// SUNDIALS calls this through a C function pointer, so an escaping C++ exception
// is undefined behaviour -- the same hazard static_residual exists to close, and
// closed the same way. A positive return is a *recoverable* linear solver
// failure: IDA cuts the step, re-forms the Jacobian and tries again, which is the
// right response to the one exception this can now see.
//
// That exception is new. The banded trace solve reports a singular K_global,
// where the dense FullPivLU it replaced silently returned the particular solution
// with the free components zeroed (see CLAUDE.md, "The trace solve"). Nothing in
// the tree is expected to reach it -- the Dirichlet rows that made K genuinely
// singular are imposed explicitly now -- but "not expected" is not "cannot", and
// the alternative is unwinding through SUNDIALS.
//
// It also closes a hole that was always here: solveJacEq allocates, and a
// std::bad_alloc, or anything a field model threw, would have done the same.
int SunLinSolWrapper::Solve( SUNMatrix A, N_Vector x, N_Vector b )
{
	try
	{
		solver->solveJacEq( b, x);
	}
	catch ( std::exception &e )
	{
		std::println("Caught exception in the linear solve : {} ; Retrying. ", e.what());
		return 1;
	}
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
	return SUN_SUCCESS;
}

int SunLinSolWrapper::LSsetup(SUNLinearSolver LS, SUNMatrix M )
{
	int err = LSWrapper( LS )->Setup( M );
	return err;
}

int SunLinSolWrapper::LSsolve(SUNLinearSolver LS, SUNMatrix M, N_Vector x, N_Vector b, sunrealtype)
{
	int err = LSWrapper( LS )->Solve( M, x, b );
	return err;
}

int SunLinSolWrapper::LSfree(SUNLinearSolver LS)
{
	delete LSWrapper( LS );
	LS->ops = nullptr;
	SUNLinSolFreeEmpty( LS );
	return SUN_SUCCESS;
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

SUNLinearSolver SunLinSolWrapper::SunLinSol( SystemSolver* solver, void *mem, SUNContext ctx )
{
	SUNLinearSolver LS = SUNLinSolNewEmpty(ctx);
	LS->content = new SunLinSolWrapper(solver);
	LS->ops = &LSOps;
	return LS;
}
