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
#include <kinsol/kinsol_ls.h>   /* KINGetNumJacEvals */
#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <string_view>

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

// Read what the KINSol call that just returned did. Must be called after *every*
// KINSol, because KINSOL zeroes these at the top of each call (KINSolInit) --
// reading them once at the end reports the final inner solve and nothing else.
// That per-call reset is also what makes these numbers per-step for free: there
// is nothing to difference, and a difference would in fact be wrong.
//
// A read that fails leaves the field at zero rather than taking the run down:
// this is diagnostics, and a counter is not worth an exception.
void SystemSolver::readKinStats(SteadyStepStats &s) const
{
    if (kin_mem == nullptr)
        return;

    long v = 0;
    if (KINGetNumNonlinSolvIters(kin_mem, &v) == KIN_SUCCESS)
        s.newtonIters = v;
    if (KINGetNumFuncEvals(kin_mem, &v) == KIN_SUCCESS)
        s.kinFuncEvals = v;
    if (KINGetNumJacEvals(kin_mem, &v) == KIN_SUCCESS)
        s.kinJacEvals = v;
}

// One line per continuation step, printed as the step finishes rather than
// collected and dumped at the end -- a solve that is going to fail is one whose
// trace you want *while* it runs, and the failure paths below throw.
//
// The header is printed by the caller, once, so this stays a single row and the
// column widths live in one place. Widths are sized for the numbers a healthy
// solve produces and are allowed to overflow rather than truncate: a step that
// takes 1000 residual evaluations has told you something, and eliding a digit
// to keep the table straight would be the one case where the alignment costs
// more than it is worth.
void SystemSolver::reportSteadyStep(SteadyStepStats const &s) const
{
    if (!steadyStepDiagnostics)
        return;

    std::println("  {:>4}  {:>10.3e}  {:>10.3e}  {:>5}  {:>5}  {:>5}  {:>6}  {}",
                 s.step, s.dt, s.residualNorm, s.newtonIters, s.residualEvals,
                 s.jacBuilds, s.jacSolves,
                 s.kinRetval < 0 && s.kinRetval != KIN_MAXITER_REACHED &&
                         s.kinRetval != KIN_STEP_LT_STPTOL
                     ? std::format("FAILED ({})", s.kinRetval)
                 : s.accepted ? "accepted"
                              : "rejected");
}

void SystemSolver::reportSteadyStats(std::string_view outcome, SteadyStats const &s) const
{
    if (!steadyDiagnostics)
        return;

    std::println("Steady solve statistics -- {}", outcome);
    std::println("  continuation steps      : {}  ({} rejected)", s.steps, s.rejected);
    std::println("  KINSOL Newton iterations: {}", s.newtonIters);
    std::println("  residual evaluations    : {}  (of which KINSOL: {})",
                 s.residualEvals, s.kinFuncEvals);
    std::println("  Jacobian builds         : {}  (KINSOL asked for {})",
                 s.jacBuilds, s.kinJacEvals);
    std::println("  Jacobian solves         : {}", s.jacSolves);
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

    // What this call costs. MaNTA's counters are monotonic over the solver --
    // IDA writes to them too -- so they are differenced against here, which is
    // also what makes a second solve on one object report its own cost.
    //
    // Snapshotted *before* the first steadyNorm() below, or that evaluation goes
    // unreported: the merit function is part of what a steady solve pays for,
    // and it costs one residual per continuation step plus this one.
    SteadyStats stats;
    const long residualEvals0 = nResidualEvals, jacBuilds0 = nJacBuilds,
               jacSolves0 = nJacSolves;

    // Cleared, not appended to: this describes one solve, and PyRunner runs many
    // on one solver.
    steadyStepStats.clear();

    double Fprev = steadyNorm();

    auto finish = [&](std::string_view outcome, int steps, int rejected)
    {
        stats.steps = steps;
        stats.rejected = rejected;
        stats.residualEvals = nResidualEvals - residualEvals0;
        stats.jacBuilds = nJacBuilds - jacBuilds0;
        stats.jacSolves = nJacSolves - jacSolves0;
        steadyStats = stats;
        reportSteadyStats(outcome, stats);
    };

    // Entering the loop. Unconditional and on stdout, because the equivalent for
    // TimeMarch -- "Writing output at ...", then the three IDA totals -- is, and
    // a steady run used to print nothing at all between "Done." and whatever the
    // physics case logged. The logmsg calls through the loop stay at INFO, which
    // is compile-time gated (Logging.hpp: WARNING unless VERBOSE or DEBUG), so
    // they are not a substitute for this.
    std::println("Steady solve: {} on {} cells at k = {}, tolerance {:g}",
                 steadyMode == SteadyMode::Newton ? "Newton" : "PseudoTransient",
                 nCells, k, steady_state_tol);
    std::println("  initial ||F|| = {:g}, dt = {:g}, SER rate {:g}, floor {:g}, "
                 "max step {:g}",
                 Fprev, ptcStep, ptcSERRate, ptcSERFloor, ptcMaxStep);
    logmsg<LOG_LEVEL::INFO>("Steady solve: initial ||F|| = {:g}, dt = {:g}", Fprev, ptcStep);

    if (Fprev < steady_state_tol)
    {
        logmsg<LOG_LEVEL::INFO>("Steady solve: initial state already converged");
        std::println("  the initial state is already converged; nothing to do.");
        setJacEvalY(Y, ptcDYdt);
        finish("converged (no continuation steps needed)", 0, 0);
        return;
    }

    // The per-step table's header, once. Printed here rather than from
    // reportSteadyStep so that the widths and the labels are declared together
    // and a row cannot drift from its column.
    //
    // "res" is MaNTA's count for the whole step, so it includes the merit
    // function's one evaluation and KINSOL's count is therefore one lower --
    // that offset is deliberate and is what the totals' two residual numbers
    // separate. dt is the step the call was damped with, ||F|| the *steady*
    // residual afterwards, which is not the norm KINSol converged: KINSol sees
    // the damped residual, which any small enough dt makes small.
    if (steadyStepDiagnostics)
        std::println("  {:>4}  {:>10}  {:>10}  {:>5}  {:>5}  {:>5}  {:>6}  {}",
                     "step", "dt", "||F||", "iters", "res", "jac", "solves",
                     "outcome");

    int step = 0;
    int rejected = 0;
    for (; step < MaxContinuationSteps; ++step)
    {
        // What this one continuation step costs. MaNTA's counters are monotonic,
        // so they are differenced across the whole step body -- not just across
        // KINSol -- which is what puts the merit evaluation below into the step
        // that paid for it. KINSOL's own counters need no snapshot: it zeroes
        // them in KINSolInit, so they are already per-call.
        SteadyStepStats rec;
        rec.step = step;
        rec.dt = ptcStep;
        const long stepResidualEvals0 = nResidualEvals, stepJacBuilds0 = nJacBuilds,
                   stepJacSolves0 = nJacSolves;

        // Bring the record up to date with everything counted so far. Called at
        // each exit from the step, because two of the three leave through a
        // `return` or a `throw` and a record completed only at the bottom of the
        // loop would be missing exactly the steps worth looking at.
        auto closeRecord = [&](double Fnow, bool accepted)
        {
            rec.residualNorm = Fnow;
            rec.accepted = accepted;
            rec.residualEvals = nResidualEvals - stepResidualEvals0;
            rec.jacBuilds = nJacBuilds - stepJacBuilds0;
            rec.jacSolves = nJacSolves - stepJacSolves0;
            stats.add(rec);
            steadyStepStats.push_back(rec);
            reportSteadyStep(rec);
        };

        // uPrev is both the backward-Euler anchor for this attempt and the state
        // to fall back to if the attempt makes things worse.
        N_VScale(1.0, Y, uPrev);

        const int retval = KINSol(kin_mem, Y, KIN_NONE, kinScale, kinScale);
        rec.kinRetval = retval;

        // Immediately: KINSOL zeroes its counters at the top of each KINSol.
        readKinStats(rec);

        // Only a genuinely broken solve is fatal. "Ran out of iterations"
        // (KIN_MAXITER_REACHED) and "the step stopped moving"
        // (KIN_STEP_LT_STPTOL) are the ordinary way an attempt at too large a dt
        // ends, and answering them by damping is the entire point of pseudo-
        // transient continuation. Treating them as failures is what made the
        // Jardin and Shestakov benchmarks throw at dt = 1000 rather than back
        // off to a dt they could solve.
        if (retval < 0 && retval != KIN_MAXITER_REACHED && retval != KIN_STEP_LT_STPTOL)
        {
            // NaN rather than Fprev: no steady residual was evaluated after this
            // call, and reporting the previous step's norm as this one's would
            // be a plausible number that is not a measurement.
            closeRecord(std::numeric_limits<double>::quiet_NaN(), false);
            finish(std::format("FAILED: KINSol returned {}", retval), step, rejected);
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

            // Accepted by construction: the loop only reaches here with
            // Fnow < steady_state_tol, and Fprev is at least the tolerance or
            // the early return above would have taken it.
            closeRecord(Fnow, true);

            std::println("  converged: ||F|| = {:g} after {} continuation steps.",
                         Fnow, step + 1);
            finish("converged", step + 1, rejected);
            return;
        }

        // Closed here, before Fprev moves: "accepted" is the same test the
        // branch below makes, and taking it after the assignment would record
        // every step as accepted.
        closeRecord(Fnow, Fnow < Fprev);

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
            //
            // Both numbers are configurable -- PseudoTransientSERRate and
            // PseudoTransientSERFloor -- because what they trade off is
            // problem-dependent: the measurement above is Park's, and
            // Shestakov's degenerate flux is the case where growing dt fast is
            // exactly what makes the inner solve start rejecting steps.
            // Defaults 1.0 and 2.0, which is what this line has always done.
            if (std::isfinite(ptcStep))
            {
                // pow(x, 1.0) is exact for the default, so the ordinary path is
                // bit for bit what the plain ratio gave.
                const double growth = ptcSERRate == 1.0
                                          ? Fprev / Fnow
                                          : std::pow(Fprev / Fnow, ptcSERRate);
                ptcStep *= std::max(growth, ptcSERFloor);
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
            ++rejected;
        }
    }

    finish("FAILED: ran out of continuation steps", step, rejected);

    throw std::runtime_error(std::format(
        "Steady solve did not converge in {} continuation steps: ||F|| = {:g} "
        "against a tolerance of {:g}, with dt = {:g}. The residual is not "
        "falling; SteadyStateSolver = \"TimeMarch\" is the fallback.",
        MaxContinuationSteps, Fprev, steady_state_tol, ptcStep));
}
