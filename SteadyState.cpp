// Pseudo-transient continuation to a steady state.
//
// The solver's usual route to a steady state is to integrate until dY/dt is
// small: TerminateOnSteadyState is a stopping test on the time loop in
// Solver.cpp, and IDA sizes every step from a local error estimate on a
// transient nobody wants. Measured on the three benchmarks under
// python-examples/, that costs 113-375 evaluations of the physics per point,
// where the schemes in refs/ reach the same state in 10-15.
//
// Pseudo-transient continuation (Kelley & Keyes) keeps a backward-Euler mass
// term purely as a damping device and sizes dt from the *residual* rather than
// from an error estimate:
//
//     F(u; dt) = residual( u, dYdt = id * (u - u_prev)/dt ) = 0
//     dt <- dt * ||F_prev|| / ||F_now||                      (SER)
//
// As dt grows the damping vanishes and the iteration becomes Newton's method on
// the steady problem. dt = infinity from the outset is exactly that, which is
// why SteadyMode::Newton needs no separate code here.
//
// Two pieces of the existing solver make this cheap. setAlpha already scales the
// mass term in the u row -- IDA's cj for the forward solve, and 0 where dF/dy
// alone is wanted, which computeAlgebraicTimeDerivatives has relied on for as
// long as it has existed -- so alpha = 1/dt and alpha = 0 are both already
// supported and exercised. And SunLinSolWrapper is solver-agnostic: its Setup is
// a no-op and its Solve calls solveJacEq, so KINSOL drives the same static
// condensation IDA does.
//
// Why KIN_NONE and not KIN_LINESEARCH: in the Kelley-Keyes formulation the 1/dt
// term *is* the globalisation, so a line search on top is redundant, and dt is
// then the only step control -- which is what makes the SER schedule below mean
// anything.
//
// This was very nearly justified on a different and false ground. CLAUDE.md
// describes the Dirichlet constraints as "imposed inside the linear solve",
// which would make ||F|| blind to a Dirichlet violation and rule out any merit
// function built on it. The code says otherwise: the block in solveJacEq that
// would force del_y.lambda to (boundary value - current lambda) is commented
// out, above the words "We really should do something here", and the boundary
// data reaches the residual instead -- Dirichlet through RF_cellwise into the
// sigma and q rows, Neumann through L_global into the lambda row. So ||F|| does
// see the boundary conditions, and KIN_LINESEARCH is a legitimate thing to try
// later. It is simply not needed here.

#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>

#include "ErrorChecker.hpp"
#include "Logging.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
#include "SystemSolver.hpp"

namespace
{
    // KINSOL solves F(u) = 0 and knows nothing about dt; the damping term and
    // the matching alpha are set by the outer loop before each KINSol call.
    int SteadyResidual(N_Vector u, N_Vector fval, void *user_data)
    {
        return reinterpret_cast<SystemSolver *>(user_data)->steadyResidual(u, fval);
    }

    int SteadyJacSetup(N_Vector u, N_Vector /* fu */, SUNMatrix /* J */,
                       void *user_data, N_Vector /* tmp1 */, N_Vector /* tmp2 */)
    {
        reinterpret_cast<SystemSolver *>(user_data)->steadyJacSetup(u);
        return 0;
    }

    // How many KINSol calls before giving up. Each is a full Newton solve, so a
    // healthy run uses ten or so; this is a runaway backstop, not a budget.
    constexpr int MaxContinuationSteps = 200;
} // namespace

int SystemSolver::steadyResidual(N_Vector u, N_Vector fval)
{
    if (std::isfinite(ptcStep))
    {
        // ptcDYdt = id * (u - uPrev)/dt. Multiplying by `id` is what keeps the
        // damping off the algebraic rows: sigma, q, lambda and phi have no time
        // derivative in the residual, and adding one would change the equations
        // rather than damp them.
        N_VLinearSum(1.0 / ptcStep, u, -1.0 / ptcStep, uPrev, ptcDYdt);
        N_VProd(ptcDYdt, id, ptcDYdt);
    }
    else
    {
        N_VConst(0.0, ptcDYdt);
    }

    return residual(t0, u, ptcDYdt, fval);
}

void SystemSolver::steadyJacSetup(N_Vector u)
{
    setJacTime(t0);
    setAlpha(std::isfinite(ptcStep) ? 1.0 / ptcStep : 0.0);
    setJacEvalY(u, ptcDYdt);
    updateBoundaryConditions(t0);
    updateMatricesForJacSolve();
}

void SystemSolver::solveSteadyState()
{
    if (!initialised)
        throw std::runtime_error("solveSteadyState called before initialize()");

    // Scratch vectors live as long as the solver so a second call reuses them,
    // which is what keeps repeated configure/run cycles cheap.
    if (uPrev == nullptr)
    {
        uPrev = N_VClone(Y);
        ptcDYdt = N_VClone(Y);
        kinScale = N_VClone(Y);
        if (uPrev == nullptr || ptcDYdt == nullptr || kinScale == nullptr)
            throw std::runtime_error("N_VClone failed in solveSteadyState");
    }
    N_VConst(1.0, kinScale); // no scaling; the DOFs are already commensurate

    if (kin_mem == nullptr)
    {
        kin_mem = KINCreate(ctx);
        if (ErrorChecker::check_retval(kin_mem, "KINCreate", 0))
            throw std::runtime_error("KINCreate failed");

        int retval = KINSetUserData(kin_mem, static_cast<void *>(this));
        if (ErrorChecker::check_retval(&retval, "KINSetUserData", 1))
            throw std::runtime_error("KINSetUserData failed");

        retval = KINInit(kin_mem, SteadyResidual, Y);
        if (ErrorChecker::check_retval(&retval, "KINInit", 1))
            throw std::runtime_error("KINInit failed");

        // Its own wrapper over the same solveJacEq; sunMat is stateless and is
        // shared with IDA.
        kinLS = SunLinSolWrapper::SunLinSol(this, kin_mem, ctx);
        retval = KINSetLinearSolver(kin_mem, kinLS, sunMat);
        if (ErrorChecker::check_retval(&retval, "KINSetLinearSolver", 1))
            throw std::runtime_error("KINSetLinearSolver failed");

        retval = KINSetJacFn(kin_mem, SteadyJacSetup);
        if (ErrorChecker::check_retval(&retval, "KINSetJacFn", 1))
            throw std::runtime_error("KINSetJacFn failed");
    }

    // Each inner solve only has to make progress, not converge to the eventual
    // tolerance: SER re-damps and tries again. Asking for the final tolerance at
    // a small dt would burn Newton iterations chasing a heavily damped problem.
    KINSetFuncNormTol(kin_mem, steady_state_tol);
    KINSetNumMaxIters(kin_mem, 20);

    // Take KINSOL's step clamp out of the picture. Its default maximum Newton
    // step is 1000*||u_0||, which is *zero* when the initial condition is zero
    // -- and a zero initial state is entirely ordinary here; Park's benchmark
    // starts from u = 0. Every step is then clamped to nothing: KINSol returns
    // KIN_MXNEWT_5X_EXCEEDED (-7) on a linear problem that should converge in
    // one step, and pseudo-transient continuation crawls, because SER only ever
    // sees the tiny residual reduction a clamped step produces. dt is the step
    // control here; there is no second one wanted.
    KINSetMaxNewtonStep(kin_mem, 1.0e10);

    // PseudoTransientInitialStep, then initialTimestep, then delta_t. dt0 is
    // *not* a safe fallback on its own: `initialTimestep` defaults to zero,
    // meaning "let IDA choose", and zero here makes 1/dt infinite, the damping
    // term NaN and every continuation step a no-op -- which is exactly what it
    // did, silently, until the residual trace showed dt = 0.
    const double fallback = (dt0 > 0.0) ? dt0 : dt;
    ptcStep = (steadyMode == SteadyMode::Newton)
                  ? std::numeric_limits<double>::infinity()
                  : (ptcInitialStep > 0.0 ? ptcInitialStep : fallback);

    if (!(ptcStep > 0.0))
        throw std::runtime_error(
            "Pseudo-transient continuation needs a positive initial step and "
            "could not find one: set PseudoTransientInitialStep, initialTimestep "
            "or delta_t.");

    N_VScale(1.0, Y, uPrev);

    // The steady residual, which is what convergence is measured against
    // throughout -- not the damped one KINSOL sees, which vanishes at any dt
    // simply by taking a small enough step.
    auto steadyNorm = [&]() -> double
    {
        N_VConst(0.0, ptcDYdt);
        residual(t0, Y, ptcDYdt, res);
        return std::sqrt(N_VDotProd(res, res));
    };

    double Fprev = steadyNorm();
    logmsg<LOG_LEVEL::INFO>("Steady solve: initial ||F|| = {:g}, dt = {:g}", Fprev, ptcStep);

    if (Fprev < steady_state_tol)
    {
        logmsg<LOG_LEVEL::INFO>("Steady solve: initial state already converged");
        setJacEvalY(Y, ptcDYdt);
        return;
    }

    int step = 0;
    for (; step < MaxContinuationSteps; ++step)
    {
        // uPrev is both the backward-Euler anchor for this attempt and the state
        // to fall back to if the attempt makes things worse.
        N_VScale(1.0, Y, uPrev);

        const int retval = KINSol(kin_mem, Y, KIN_NONE, kinScale, kinScale);

        // Only a genuinely broken solve is fatal. "Ran out of iterations"
        // (KIN_MAXITER_REACHED) and "the step stopped moving"
        // (KIN_STEP_LT_STPTOL) are the ordinary way an attempt at too large a dt
        // ends, and answering them by damping is the entire point of pseudo-
        // transient continuation. Treating them as failures is what made the
        // Jardin and Shestakov benchmarks throw at dt = 1000 rather than back
        // off to a dt they could solve.
        if (retval < 0 && retval != KIN_MAXITER_REACHED && retval != KIN_STEP_LT_STPTOL)
        {
            throw std::runtime_error(std::format(
                "Steady solve failed: KINSol returned {} at continuation step {} "
                "with dt = {:g} and ||F|| = {:g}. Consider SteadyStateSolver = "
                "\"TimeMarch\", or a smaller PseudoTransientInitialStep.",
                retval, step, ptcStep, Fprev));
        }

        const double Fnow = steadyNorm();
        logmsg<LOG_LEVEL::INFO>("Steady solve: step {}, dt = {:g}, ||F|| = {:g}",
                                step, ptcStep, Fnow);

        if (Fnow < steady_state_tol)
        {
            logmsg<LOG_LEVEL::INFO>("Steady state reached in {} continuation steps", step + 1);
            // yJac is the copy that outlives the solve, and the adjoint solve
            // and the output path both read it.
            N_VConst(0.0, ptcDYdt);
            setJacEvalY(Y, ptcDYdt);

            // dYdt is IDA's derivative vector, and nothing in this function has
            // touched it -- the damping above runs on the scratch ptcDYdt. So on
            // return it still holds whatever IDACalcIC left at t0, which for a
            // converged steady state is simply wrong: the defining property of
            // the answer is that dy/dt vanishes. Two things read it afterwards
            // and both were getting the t0 derivative -- WriteRestartFile
            // (Solver.cpp), so a restart resumed from a state whose y was the
            // steady one and whose y' was not, and writeDiagnostics, which a
            // physics case is entitled to differentiate. Measured before the
            // fix on AdjointPoster: ||dYdt|| = 103.4 at convergence.
            //
            // Zeroed here rather than at either reader, because integrate()
            // ends with its own setJacEvalY(Y, dYdt) (Solver.cpp) -- so the
            // zero setJacEvalY just put in dydtJac was overwritten on the way
            // out with the stale value anyway. Fixing the vector is what makes
            // all three agree.
            N_VConst(0.0, dYdt);
            return;
        }

        if (Fnow < Fprev)
        {
            // Switched Evolution Relaxation, with a floor on the growth that SER
            // on its own badly needs: the ratio is only as large as the residual
            // reduction, and the residual reduction is only as large as dt
            // allows, so a conservative dt0 is self-perpetuating. Measured on
            // Park's benchmark from dt0 = 1e-2, plain SER grew dt by 4% a step
            // and took 62 continuation steps; doubling on any step that made
            // progress brings that to 15, and costs nothing in safety because a
            // step that fails is rejected outright below.
            if (std::isfinite(ptcStep))
            {
                ptcStep *= std::max(Fprev / Fnow, 2.0);
                if (ptcStep > ptcMaxStep)
                    ptcStep = ptcMaxStep;
            }
            Fprev = Fnow;
        }
        else
        {
            // Rejected. Put the state back and damp hard: an accepted step is
            // the only thing that may move Y, so a bad dt costs one solve rather
            // than corrupting the iterate. In Newton mode there is no dt to
            // damp, so drop to a finite one and continue as pseudo-transient --
            // which is the honest thing to do when the undamped step failed.
            N_VScale(1.0, uPrev, Y);
            ptcStep = std::isfinite(ptcStep) ? ptcStep * 0.25 : fallback;
        }
    }

    throw std::runtime_error(std::format(
        "Steady solve did not converge in {} continuation steps: ||F|| = {:g} "
        "against a tolerance of {:g}, with dt = {:g}. The residual is not "
        "falling; SteadyStateSolver = \"TimeMarch\" is the fallback.",
        MaxContinuationSteps, Fprev, steady_state_tol, ptcStep));
}
