#ifndef SYSTEMSOLVER_HPP
#define SYSTEMSOLVER_HPP

#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of sunrealtype, sunindextype  */
#include <nvector/nvector_serial.h>
#include <filesystem>

#include "Types.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Logging.hpp"
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "gridStructures.hpp"
#include "TransportSystem.hpp"
#include "DGSoln.hpp"
#include "NetCDFIO.hpp"
#include "AdjointProblem.hpp"
#include "Postprocessing.hpp"

// Unit tests exercise the HDG block assembly, the static-condensation solve and
// the adjoint vectors directly -- all private. The previous scheme befriended
// one struct per Boost test case (BOOST_AUTO_TEST_CASE generates a struct), so
// every new test that touched private state needed both a forward declaration
// and a friend line added to this header. That does not scale to the current
// suite, so a TEST build simply widens access instead. Release builds are
// unaffected: MANTA_TEST_PRIVATE is plain `private` unless -DTEST is set.
#ifdef TEST
#define MANTA_TEST_PRIVATE public
#else
#define MANTA_TEST_PRIVATE private
#endif

// Forward-declared rather than included: FieldModel.hpp pulls in toml11 for the
// registry at its foot, and SystemSolver.hpp reaches 25 translation units that
// have no other reason to parse it. A shared_ptr to an incomplete type is fine
// as long as its destructor is instantiated where the type is complete, which
// is ~SystemSolver in SystemSolver.cpp.
class FieldModel;

class SystemSolver
{
    public:
        SystemSolver(Grid const &Grid, unsigned int polyNum, TransportSystem *pProblem);
        SystemSolver(const SystemSolver &) = delete; // Best practice to define this as deleted. We can't copy this class.
        ~SystemSolver();

        void setOutputCadence(double Dt)
        {
            if (Dt < 0)
                throw std::logic_error("Output cadence cannot be negative.");
            dt = Dt;
        };
        void setInitialTimestep(double Dt0) { dt0 = Dt0; };
        void setInitialTime(double T) { t0 = T; };
        void setSteadyStateTolerance(double ss_tol)
        {
            if (ss_tol <= 0)
                throw std::logic_error("Tolerance for steady-state termination cannot be zero or negative.");
            steady_state_tol = ss_tol;
            TerminateOnSteadyState = true;
        };

        // Ask for a steady solve without naming a tolerance, leaving whatever
        // setSteadyStateTolerance last set -- or steady_state_tol's own default
        // if nothing did, which is the same 1e-3 run_ss() falls back to.
        //
        // Arming and choosing a tolerance were one operation, so the only way to
        // ask for a steady solve was to have an opinion about how tight it
        // should be. That is why SteadyStateTolerance's *presence* was the
        // signal, and why a configuration naming SteadyStateSolver but omitting
        // the tolerance quietly time-marched.
        void setSteadyStateTermination(bool on) { TerminateOnSteadyState = on; };

        // How a steady state is reached. TimeMarch is the original behaviour --
        // integrate until dY/dt is small -- and is kept because it is the only
        // one that picks a branch by physics rather than by wherever Newton
        // lands. The other two are one algorithm: pseudo-transient continuation
        // keeps the 1/dt mass term purely as damping and grows dt from the
        // residual, and Newton is its dt = infinity limit, taken from the start.
        //
        // Both freeze explicitly time-dependent data -- boundary values, sources
        // -- at t_initial. There is no time axis to evaluate them on.
        enum class SteadyMode
        {
            TimeMarch,
            PseudoTransient,
            Newton,
        };

        // The time a converged steady state is stamped with in the output, on
        // the two modes that do not integrate. It is a label, not a physical
        // time: PseudoTransient and Newton drive the residual to zero without
        // advancing anything, so there is no elapsed time to report, and every
        // time-dependent input was frozen at t_initial (see SteadyMode above).
        //
        // 1.0 rather than t0 or tFinal, and the choice is forced. tret never
        // leaves t0 on this path, so stamping with it would give the file two
        // slices both at t0 -- and tFinal is no better, because PyRunner's
        // run_ss() calls runSolver(0), which is the path every steady run in
        // this tree takes. A fixed label separates the converged state from
        // the initial condition in every case, and reads as the flag it is.
        //
        // So a steady run's .nc holds exactly two slices: t = 0, the initial
        // condition, and t = 1, the answer. docs/running.rst says so.
        static constexpr double STEADY_STATE_TIME = 1.0;

        void setSteadyMode(SteadyMode mode) { steadyMode = mode; };
        SteadyMode getSteadyMode() const { return steadyMode; };
        void setPseudoTransientInitialStep(double dt) { ptcInitialStep = dt; };
        // Whether a finished steady solve estimates the objective. One residual,
        // one Jacobian build and one solve each time, which is nothing against a
        // whole continuation but is charged *per solve* -- so a solve driven in
        // slices pays it per slice, and a five-step slice of a fifteen-step
        // continuation spends a third as much again on estimating as on solving.
        // A driver that only wants the estimate at the end turns it off for the
        // slices in between.
        void setEstimateObjectiveOnFinish(bool on) { estimateObjectiveOnFinish = on; };
        bool getEstimateObjectiveOnFinish() const { return estimateObjectiveOnFinish; };

        // The pseudo-time step the continuation has climbed to, which is what a
        // resumed solve picks up. Infinite in Newton mode.
        double getPseudoTransientStep() const { return ptcStep; };

        void setMaxContinuationSteps(long steps)
        {
            if (steps < 1)
                throw std::logic_error("MaxContinuationSteps must be at least 1: a steady solve that may take no continuation steps cannot make progress.");
            maxContinuationSteps = steps;
        };
        long getMaxContinuationSteps() const { return maxContinuationSteps; };
        void setPseudoTransientMaxStep(double dt) { ptcMaxStep = dt; };

        // The SER schedule on an accepted step:
        //
        //     dt <- dt * max( (||F_prev|| / ||F_now||)^rate , floor )
        //
        // rate = 1 and floor = 2 are what this has always done; both defaults
        // are repeated in ConfigSchema.cpp and must move together.
        //
        // rate is the exponent, so it says how hard dt leans on the residual
        // reduction: 0 ignores it and grows at the floor alone, 1 is plain SER,
        // above 1 is more aggressive. Negative would shrink dt as the residual
        // falls, which is backwards, so it is refused.
        //
        // floor is the least dt may grow on a step that made progress. It is
        // the reason plain SER is not enough here -- see solveSteadyState. A
        // value below 1 can never bind (this branch runs only when the residual
        // fell, so the ratio already exceeds 1) and is refused rather than
        // silently ignored; 1 exactly is "no floor", which is plain SER.
        void setPseudoTransientSERRate(double rate)
        {
            if (rate < 0.0)
                throw std::logic_error("The SER rate cannot be negative: dt would shrink as the residual falls.");
            ptcSERRate = rate;
        };
        void setPseudoTransientSERFloor(double floorValue)
        {
            if (floorValue < 1.0)
                throw std::logic_error("The SER floor cannot be below 1: it is a growth factor on an accepted step, and would never bind.");
            ptcSERFloor = floorValue;
        };

        // How KINSOL is driven inside a steady solve. These apply to the inner
        // Newton solve in *both* SteadyStateSolver modes, not only to "Newton":
        // PseudoTransient is Newton on a damped residual, so it runs the same
        // KINSol with a 1/dt term added. The one mode they do not touch is
        // TimeMarch, which never builds a KINSOL object at all.
        //
        // Everything here has a default that reproduces what the code did when
        // these were hardcoded, so adding them changes no existing run.

        // How many Newton iterations one KINSol call may take before it gives up
        // and hands back to the continuation loop.
        //
        // The default is **20 against KINSOL's own 200**, and that gap is
        // deliberate rather than an oversight: an inner solve here only has to
        // make progress, because SER re-damps and tries again from a better dt.
        // Letting it run to 200 spends iterations converging a heavily damped
        // problem that is about to be replaced. Raise it for SteadyStateSolver =
        // "Newton", where there is no outer loop to fall back on and giving up
        // early is simply a failure.
        void setNewtonMaxIterations(long iterations)
        {
            if (iterations < 1)
                throw std::logic_error("NewtonMaxIterations must be at least 1: a KINSol call that may take no iterations cannot make progress.");
            newtonMaxIters = iterations;
        };

        // How many Newton iterations may reuse one Jacobian factorisation
        // (KINSOL's msbset). 1 is full Newton -- rebuild every iteration -- and
        // larger values are modified Newton.
        //
        // This is the setting the per-step diagnostics above *measure*: the
        // "jac" and "solves" columns are builds against reuses, and their ratio
        // is what this controls. AdjointPoster at k = 3 pays 7 builds for 35
        // solves at the default; that is this number, not a property of the
        // problem.
        //
        // The three costs are not comparable, and that ordering is why this is a
        // key rather than a constant. A Jacobian *solve* is always cheap -- a
        // static condensation against a factorisation that already exists. A
        // Jacobian *assembly* is at least as expensive as a residual evaluation,
        // and how much more is a property of the physics case rather than of the
        // solver: a differentiable flux (hand-written derivatives,
        // AutodiffTransportSystem, JAX) pays value-and-gradient against value,
        // which is more but not by much, while a Jacobian finite-differenced from
        // expensive flux calls costs many evaluations per assembly and can
        // dominate the run outright.
        //
        // So raising this trades assemblies away for extra Newton iterations, and
        // each of those costs a residual evaluation plus a cheap solve. Which side
        // wins depends on how the case is differentiated, which is exactly why
        // there is no setting that is right for every case. docs/running.rst has a
        // measurement at the cheap-Jacobian end and a warning against reading it
        // as a recommendation.
        //
        // Note KINSOL may rebuild sooner than this on its own: residual
        // monitoring and a large step both force a setup (kinsol.c:1890, 1897),
        // so this is a ceiling on reuse rather than a schedule.
        void setNewtonJacobianReuse(long iterations)
        {
            if (iterations < 1)
                throw std::logic_error("NewtonJacobianReuse must be at least 1; 1 means rebuild the Jacobian every iteration.");
            newtonJacReuse = iterations;
        };

        // KINSOL's scaled-step stopping test (scsteptol): a KINSol whose step
        // falls below this returns KIN_STEP_LT_STPTOL.
        //
        // Worth knowing what that return *means here*, because it is not a
        // failure: solveSteadyState treats it, like KIN_MAXITER_REACHED, as the
        // ordinary way an attempt at too large a dt ends, and answers it by
        // damping. So raising this makes inner solves give up sooner and the
        // outer loop re-damp more eagerly; lowering it does the reverse. It is a
        // continuation-schedule control wearing a tolerance's clothing.
        //
        // Zero means "leave KINSOL's default", which is uround^(2/3) ~ 3.7e-11 --
        // machine-dependent, which is why the default is a sentinel rather than a
        // number. KINSetScaledStepTol restores the default when handed zero, so
        // it is passed through unconditionally.
        void setNewtonStepTolerance(double tol)
        {
            if (tol < 0.0)
                throw std::logic_error("NewtonStepTolerance cannot be negative; use zero for KINSOL's default.");
            newtonStepTol = tol;
        };

        // What KINSOL's scaling vectors hold. Unit is what this has always used.
        //
        // KINSOL's convergence tests are on *scaled* quantities, so with unit
        // scaling they are dimensional: on a case carrying densities near 1e19
        // beside temperatures near 1e3, one SteadyStateTolerance means something
        // different for each variable, and the largest simply dominates.
        // ErrorWeights fills the vectors from getErrorWeights -- the same
        // 1/(rtol|y| + atol) weights IDA's WRMS norm uses -- which makes the test
        // relative and puts the variables on comparable footing.
        //
        // Off by default because it changes what convergence *means* for every
        // existing steady run, not because it is worse. Note one honest
        // limitation: KINSOL takes separate u_scale and f_scale, and both get the
        // same vector here, as they already did when both were ones. The residual
        // does not carry the solution's units, so a properly derived f_scale
        // would be a different vector; this is an improvement on unit scaling
        // rather than the last word on it.
        enum class NewtonScaling { Unit, ErrorWeights };
        void setNewtonScaling(NewtonScaling scaling) { newtonScaling = scaling; };

        // Report the work a steady solve did: continuation steps, KINSOL Newton
        // iterations, residual evaluations, Jacobian builds and Jacobian solves.
        // Off by default, unlike the time loop's equivalent summary, which is
        // unconditional -- a steady solve is often run in a loop by an
        // optimisation driver, where one block per solve is noise rather than
        // information.
        void setSteadyStateDiagnostics(bool on) { steadyDiagnostics = on; };

        // Report each KINSol invocation as it returns, one line per continuation
        // step, rather than only the total. Independent of the summary above --
        // they compose, and either can be had on its own -- because they answer
        // different questions. A total says what the solve cost; the per-step
        // trace says *where*, which is the only way to tell a run that took
        // twenty cheap steps from one that took three expensive ones. The two
        // have the same cost when nothing is wrong and diverge sharply when
        // something is: a solve whose dt has outrun the problem spends its
        // iterations in steps that are then rejected, and the total cannot show
        // that.
        void setSteadyStateStepDiagnostics(bool on) { steadyStepDiagnostics = on; };

        // What one KINSol invocation cost -- one continuation step. KINSOL's own
        // counters are *already* per-call, since it zeroes them in KINSolInit,
        // so these are read straight out rather than differenced; MaNTA's
        // monotonic counters are differenced across the step. The two do not
        // measure the same span deliberately: `kinFuncEvals` is what KINSOL
        // asked for, `residualEvals` is what the step actually paid, and the
        // difference is the merit function -- one steady-residual evaluation per
        // step, which KINSOL never sees because it is evaluated at dt = infinity
        // rather than at the damped dt KINSol is solving.
        // What a converged objective is worth, per objective.
        //
        // A solve stops when ||F|| is small, not when G is: an optimisation sweep
        // comparing G at two parameter points needs to know how much of the
        // difference is the answer moving and how much is each solve stopping
        // short. Both quantities that answer it are already assembled -- dG/dy is
        // G_y, which initializeMatricesForAdjointSolve builds per objective, and
        // the Newton step to the solution is J^-1 F from the matrix the solve has
        // already factorised -- so
        //
        //     corrected   = value - (dG/dy) . J^-1 F
        //     uncertainty = ||dG/dy|| ||J^-1 F||
        //
        // the first a first-order extrapolation to the fixed point, the second a
        // Cauchy-Schwarz bound on what is left. Measured by replaying every
        // continuation step of a PseudoTransient solve, the bound holds at all of
        // them -- 1.04x the true error at its tightest, 2.3x at its loosest -- and
        // the correction is worth two to four orders of magnitude: a state whose
        // raw G is 3% out reports 0.04% once corrected.
        //
        // It bounds *solver* error only. It says how far this solve stopped short
        // of its own fixed point, not how far that fixed point is from the
        // continuum, so it compares two runs at one discretisation and nothing
        // else. That is the sweep's question.
        // Why a steady solve stopped. The loop has three ways out and two of
        // them throw, so without this a caller that catches has no way to tell a
        // solve that ran out of continuation steps -- a partial answer, and often
        // a usable one -- from one KINSol abandoned outright.
        enum class SteadyOutcome
        {
            NotRun,
            Converged,
            OutOfSteps,
            SolverFailed,
        };

        struct ObjectiveEstimate
        {
            Vector value;        // G as the run actually converged it
            Vector corrected;    // G extrapolated to the fixed point
            Vector uncertainty;  // bound on |corrected - value|, and on the error left
            bool valid = false;  // false when there is no AdjointProblem to ask
        };

        struct SteadyStepStats
        {
            int  step = 0;          // continuation step index, from zero
            int  kinRetval = 0;     // what KINSol returned; see kinsol.h
            bool accepted = false;  // did the step reduce ||F||, or was it rolled back
            double dt = 0.0;        // the pseudo-time step this call was damped with
            double residualNorm = 0.0; // steady ||F|| after the call; NaN if it failed
            long newtonIters = 0;   // KINSOL Newton iterations, this call
            long kinFuncEvals = 0;  // residual evaluations KINSOL made, this call
            long kinJacEvals = 0;   // Jacobian setups KINSOL asked for, this call
            long residualEvals = 0; // every residual call the step made, merit included
            long jacBuilds = 0;     // updateMatricesForJacSolve calls, this step
            long jacSolves = 0;     // solveJacEq calls, this step
        };

        // What one steady solve cost. The two halves are gathered differently
        // and it matters which is which:
        //
        //  * MaNTA's counters (nResidualEvals and friends) are monotonic over
        //    the solver's lifetime, and IDA writes to them too, so these fields
        //    are filled by differencing against a snapshot taken on entry.
        //  * **KINSOL resets its own counters at the start of every KINSol
        //    call**, so they are per-call, not per-solver. Differencing them
        //    across the continuation loop reports the last inner solve alone --
        //    which on a converged run is a plausible-looking 1 Newton iteration
        //    for 35 Jacobian solves. They are summed from the per-step records
        //    instead, which is now the only place they are read at all.
        struct SteadyStats
        {
            int  steps = 0;         // continuation steps taken
            int  rejected = 0;      // of those, rejected and damped
            // The *steady* residual the solve ended on -- not the damped one
            // KINSol converged, which any small enough dt makes small. This is
            // the number the tolerance is tested against, and the only measure
            // of progress a caller driving the solve in slices can see.
            double residualNorm = std::numeric_limits<double>::quiet_NaN();
            long newtonIters = 0;   // KINSOL Newton iterations, summed
            long kinFuncEvals = 0;  // residual evaluations KINSOL made, summed
            long kinJacEvals = 0;   // Jacobian setups KINSOL asked for, summed
            long residualEvals = 0; // every residual call, KINSOL's and the merit function's
            long jacBuilds = 0;     // updateMatricesForJacSolve calls
            long jacSolves = 0;     // solveJacEq calls

            // Fold one finished continuation step into the totals. The KINSOL
            // fields are the only ones that must come through here: the three
            // MaNTA counters are differenced over the whole solve as well, so
            // they have an independent value to check these against, and
            // `steady_step_stats_sum_to_the_totals` does exactly that.
            void add(SteadyStepStats const &s)
            {
                newtonIters += s.newtonIters;
                kinFuncEvals += s.kinFuncEvals;
                kinJacEvals += s.kinJacEvals;
            }
        };
        void readKinStats(SteadyStepStats &s) const;
        void reportSteadyStep(SteadyStepStats const &s) const;
        void reportSteadyStats(std::string_view outcome, SteadyStats const &s) const;

        // What the last steady solve cost, whether or not it was printed and
        // whether or not it converged -- a failed solve fills this in before
        // throwing. Zeroed only by construction, so it survives the throw for a
        // caller that wants the numbers rather than the log line.
        SteadyStats lastSteadyStats() const { return steadyStats; };

        // The estimate at the state the solver is in now. One residual, one
        // Jacobian build and one solve, plus a dot product per objective; the
        // Jacobian work is at alpha = 0, the steady operator, whatever the solve
        // was damped with. Leaves alpha and the factorised blocks as it found
        // them, so it is safe to call from inside a continuation loop.
        ObjectiveEstimate estimateObjective();

        // What estimateObjective() last reported, filled at the end of a steady
        // solve. Invalid when the run had no AdjointProblem.
        ObjectiveEstimate lastObjectiveEstimate() const { return objectiveEstimate; };

        // Why the last steady solve stopped. Set before either failure path
        // throws, so it survives being caught.
        SteadyOutcome lastSteadyOutcome() const { return steadyOutcome; };

        // The same, one entry per KINSol invocation, in the order they ran.
        // Filled whether or not the per-step lines were printed, so a driver can
        // have the trace without the output -- and, like the totals, it survives
        // a failed solve, whose last entry is the call that failed. Cleared at
        // the top of every solveSteadyState, so it describes one solve rather
        // than the solver's history.
        std::vector<SteadyStepStats> const &lastSteadyStepStats() const { return steadyStepStats; };

        // Drive the state to a steady one without integrating to it. Assumes
        // initialize() has run, so Y/dYdt/LS/sunMat exist and Y holds a
        // consistent initial condition. Leaves the converged state in Y and in
        // yJac, which is what the adjoint solve and the output path read.
        void solveSteadyState(bool resume = false);

        // Resume a solve that stopped on its step budget rather than on its
        // tolerance, from the state and the pseudo-transient step it left
        // behind. A fresh solveSteadyState() re-enters at
        // PseudoTransientInitialStep and re-climbs the SER ramp; this does not,
        // which is the whole saving on a short run driven in slices.
        //
        // The solver must still be initialised -- this is a phase of one run,
        // not a second run -- so a caller wanting it drives initialize() and
        // solveSteadyState() itself rather than going through runSolver(),
        // which frees the state on the way out of a failed solve.
        void continueSteadyState() { solveSteadyState(true); };

        // The two halves of what integrate() does once a steady solve has an
        // answer in Y. Public because a sliced solve drives the phases itself
        // and has to reach them; integrate() calls the same two, so the sliced
        // and unsliced paths cannot drift apart.
        //
        // finishRun() is shared with the time-marching branch. Call it once per
        // run: it closes the output files, so a second call writes nothing.
        void writeSteadyState();
        void finishRun();

        // Close every output file the run opened. Idempotent, and called on the
        // failure paths too -- a run that dies still has to leave a readable
        // .nc behind.
        void closeOutputFiles();

        // Copy the current state into yJac, which is where "the solution" is
        // read from: `y` is a non-owning view over Y and dangles after
        // destroySundials(). finishRun() does this at the end of a run; a
        // sliced solve does it per slice, so a driver looking between slices
        // sees the state it has reached rather than the initial condition.
        void captureState();

        // The two KINSOL callbacks, public because the C shims in
        // SteadyState.cpp reach them through the user_data pointer.
        int steadyResidual(N_Vector u, N_Vector fval);
        void steadyJacSetup(N_Vector u);

        void setNOutput(int nO)
        {
            if (nO <= 0)
                throw std::logic_error("Number of output grid points cannot be zero or negative.");
            nOut = nO;
        };
        void setMinStepSize(double dt_min)
        {
            if (dt_min <= 0)
                throw std::logic_error("Minimum delta t cannot be zero or negative.");
            min_step_size = dt_min;
        };

        void setTolerances(std::vector<double> a, double r)
        {
            if (r <= 0)
                throw std::logic_error("Cannot set tolerance to non-positive value");
            atol = a;
            rtol = r;
        };

        // Initialises u, q and lambda to satisfy residual equation at t=0
        void setInitialConditions(N_Vector &Y, N_Vector &dYdt);

        void ApplyDirichletBCs(DGSoln &);

        // Builds initial matrices
        void initialiseMatrices();

        void clearCellwiseVecs();

        void resetCoeffs();

        // Creates the MX cellwise matrices used at each Jacobian iteration
        // Factorization of these matrices is done here
        void updateMatricesForJacSolve();

        // Solves the Jy = g equation. Dispatches on whether a field model is
        // attached: without one this *is* solveTransportJac, bit for bit.
        void solveJacEq(N_Vector g, N_Vector delY);

        // The uncoupled transport operator: HDG static condensation plus the
        // scalar bordering, and nothing about the field.
        //
        // Kept separate from solveJacEq because solveCoupledJacExact applies it
        // nField + 1 times as its *inner* solve. Anything that wrote the field
        // block in here -- as the block-Jacobi psi solve that preceded this
        // split did -- would corrupt every one of those. See the definition.
        void solveTransportJac(N_Vector g, N_Vector delY);

        // Exact Schur complement onto psi. See the definition; costs one
        // transport solve per field degree of freedom, so it is a verification
        // tool rather than a production path.
        void solveCoupledJacExact(N_Vector g, N_Vector delY);

        // Block Gauss-Seidel between the transport and field blocks, with
        // Irons-Tuck acceleration: one transport solve and one field solve per
        // sweep, against the exact path's nField + 1 transport solves. Stops
        // once the relative change in psi between sweeps is below
        // FieldSolveTolerance, up to FieldSolveMaxSweeps.
        //
        // Reaching that cap **escalates to solveCoupledJacExact** rather than
        // returning the last iterate, so this mode can no longer be wrong, only
        // slower -- which is what makes it a safe default. The escalation is
        // counted in fieldSweepFallbacks and reported once per run.
        void solveCoupledJacIterative(N_Vector g, N_Vector delY);

        // Solves the HDG part of Jy = g
        void solveHDGJac(N_Vector g, N_Vector delY);

        void setAlpha(double const a) { alpha = a; }

        // print current output for u and q to output file
        void print(std::ostream &out, double t, int nOut, bool printSources = false);
        void print(std::ostream &out, double t, int nOut, N_Vector const &tempY, bool printSources = false);
        void printOnNodes(std::ostream &out, double t, N_Vector const &tempY, bool printSources = false);
        double getdt() const { return dt; }

        void setTesting(bool t) { testing = t; }
        bool isTesting() const { return testing; }

        void updateBoundaryConditions(double t);


        void mapDGtoSundials(std::vector<VectorWrapper> &SQU_cell, VectorWrapper &lam, sunrealtype *const &Y) const;

        // The run lifecycle, in three phases.
        //
        //   initialize()       allocate the SUNDIALS objects, build the initial
        //                      condition, open the output files, and -- only for
        //                      a run that will time-march -- run IDACalcIC
        //   integrate(tFinal)  the time loop, then the adjoint solve and the
        //                      final netCDF / restart output
        //   destroySundials()  free everything initialize() allocated
        //
        // runSolver() composes the three and is what the standalone binary and
        // the tests call; behaviour through that entry point is unchanged, except
        // that cleanup now happens even when the time loop throws.
        //
        // They are separate so that a caller can allocate, look at the state,
        // integrate and free as distinct steps. PyRunner::G() is the motivating
        // case: it wants the objective without also paying for a gradient.
        //
        // destroySundials() nulls what it frees, so calling it twice -- or
        // without a preceding initialize() -- is safe. initialize() after a
        // destroySundials() starts a fresh run on the same object.
        void initialize();
        void integrate(double tFinal);
        void destroySundials();
        void runSolver(double tFinal);

        void setAdjointProblem(AdjointProblem *ap) { adjointProblem = ap; };
        void runAdjointSolve();

        void setJacTime(double tt) { jt = tt; };
        void setTime(double tt) { t = tt; };
        void setTau(double tau) { tauc = tau; };

        void setInputFile(std::string const &fn) { inputFilePath = fn; };

        void setZeroFlux(bool in) { zeroFlux = in; };

        // Switch the residual and Jacobian to the superconvergent interpolatory
        // scheme of Chen, Cockburn, Singler & Zhang (J Sci Comput 81:2188): the
        // physics is evaluated on the k+2 nodes of the degree-(k+1) basis with
        // the postprocessed u* in place of u_h, and interpolated into P_{k+1}
        // rather than P_k. Off by default -- with it off the solver is the
        // interpolatory HDG method of arXiv:1811.09667, exactly as before.
        //
        // The postprocessed u* is reconstructed and written to the output either
        // way; this flag controls only whether the *method* uses it.
        void setSuperconvergent(bool in) { superconvergent = in; };
        bool isSuperconvergent() const { return superconvergent; };

        // Null when k = 0, where the degree-0 NodalBasis cannot be evaluated
        // off-node and there is nothing to reconstruct from.
        Postprocessor const *getPostprocessor() const { return postprocessor.get(); };

        // The polynomial degree this solver was built at. Fixed for its
        // lifetime -- DGSolnImpl holds k by value and the basis with it, so
        // changing it means a new solver. That is what runAdaptiveDegree does,
        // and its caller needs this to find out where it landed.
        unsigned int getOrder() const { return k; };

        // Whether a run will take the steady path rather than the time loop.
        //
        // Two conditions, and the pairing is easy to get wrong: SteadyStateSolver
        // names the *mode*, but the mode is only consulted once termination is
        // armed, and arming happens through the presence of SteadyStateTolerance.
        // So a configuration naming PseudoTransient -- which is the default, and
        // therefore what a config that says nothing gets -- still time-marches
        // unless a tolerance was given. Checking the mode alone reads as a steady
        // solve and is not one.
        //
        // Solver.cpp's branch and every caller that needs to know go through
        // this, so the rule is written once.
        bool solvesForSteadyState() const
        {
            return TerminateOnSteadyState && steadyMode != SteadyMode::TimeMarch;
        };

        // How well resolved the run's answer is, for one variable, from the gap
        // between u_h and its own postprocessing u*. Reconstructs u* from yJac
        // first, so this is valid after destroySundials() -- yJac owns its
        // memory and the postprocessor is a member, where Y does not and is not.
        //
        // A method rather than something a caller assembles, because assembling
        // it means reaching yJac and the postprocessor, both of which are
        // private for good reason.
        AccuracyEstimate accuracyEstimate(Index var);

        // The converged state as plain vectors, independent of SUNDIALS and of
        // this object's lifetime. This is what TransportSystem::setRestartValues
        // wants, and taking a copy is what lets a caller destroy this solver
        // before building the next one -- which anything driving a sequence of
        // solves must do, since Integrator's weight cache is a process global
        // keyed on (order, grid).
        std::vector<double> stateVector() const;
        std::vector<double> derivativeVector() const;

        // Gates the netCDF output and the restart file -- <stem>.nc and
        // <stem>.restart.nc. The .dat flags below are deliberately *not* nested
        // under this one: they are opt-in already, so folding them in would
        // change what a configuration setting only WriteDatFile does.
        void setWriteOutput(bool in) { writeOutput = in; };

        // The plain-text .dat files are a gnuplot convenience, not the primary
        // output -- netCDF is. Both default to off so a run writes only its
        // .nc; ask for them explicitly when you want to plot.
        void setWriteDatFile(bool in) { writeDatFile = in; };
        // <stem>.dydt.dat and <stem>.res.dat. Additionally require a
        // PHYSICS_DEBUG build, since that is what computes the residual and
        // error weights they report.
        void setWriteDebugDatFiles(bool in) { writeDebugDatFiles = in; };

        // Let IDA grow the step by up to 10x between steps instead of the
        // default 2x. Worth it when the transient is short relative to the run
        // and the interesting part is the steady state -- an optimisation driver
        // calling run_ss() in a loop, for instance. It makes IDA more likely to
        // overshoot and have to retry, so it is off by default.
        void setAggressiveTimesteps(bool in) { aggressiveTimesteps = in; };

        // IDASetSuppressAlg: take sigma, q, lambda and phi out of IDA's local
        // error test, leaving only u and the differential scalars in it. Off by
        // default, and the default is load-bearing rather than conservative --
        // turning it on is measurably not answer-preserving. See
        // setSuppressAlgebraicError's use in Solver.cpp and docs/running.rst.
        void setSuppressAlgebraicError(bool in) { suppressAlgebraicError = in; };

        // Run IDACalcIC on a run that would otherwise skip it.
        //
        // Two skip by default, for different reasons, and neither is about
        // saving work:
        //
        //   * a **steady solve** never takes an IDA step, so solveSteadyState
        //     drives the whole residual to zero from Y with KINSOL and whatever
        //     IDACalcIC computed is discarded by the first accepted continuation
        //     step. Worse, IDACalcIC fails on states a steady solve handles
        //     without difficulty -- python-examples/jardin-critical-gradient
        //     returns IDA_CONV_FAIL from the *exact* steady state, the one
        //     initial condition a steady solve would have accepted instantly.
        //
        //   * a **restart** resumes from a state the previous run had already
        //     driven onto the constraint manifold, so there is nothing to
        //     correct. getInitialResidualNorm() reports how true that was on the
        //     run in hand.
        //
        // A cold time-marching run always runs IDACalcIC and there is no way to
        // turn that off. It needs a consistent initial condition and its guess is
        // not one: IDA_ERR_FAIL (-3) on the first step is what an inconsistent
        // state looks like, and a local error estimate that will not shrink with
        // h is not something an option should be able to opt into. A caller who
        // does not care about the transient wants SteadyStateSolver =
        // PseudoTransient or Newton, not an uncorrected time march.
        //
        // So this key only ever *adds* IDACalcIC. It is the escape hatch for a
        // restart that is not as consistent as a restart is supposed to be -- a
        // file written by a different discretisation, say, where
        // setInitialConditions projects rather than copies and builds the trace
        // by averaging.
        //
        // Cost, for scale. IDACalcIC has no cheap path: its convergence test is
        // on the *Newton step* -- IDANewtonIC calls lsolve and only then tests
        // ||J^-1 F|| against epsNewt (ida_ic.c:404-417) -- and IDAnlsIC calls
        // lsetup unconditionally before that (ida_ic.c:345). The outer loop then
        // runs the whole thing twice on success, to refresh the error weights at
        // the converged state (ida_ic.c:232). So the floor is two Jacobian builds
        // and two Jacobian solves *however consistent the state already is*:
        // measured on four fixtures handed a state CalcIC had just converged to
        // (||F|| between 6e-15 and 7e-12), it costs 2 residual evaluations, 2
        // builds and 2 solves with zero Newton iterations, every time. For MaNTA
        // a build is updateMatricesForJacSolve() -- assemble and factorise every
        // per-cell MX. On an already consistent AuxVarTest warm start at rtol
        // 1e-6 that floor is the whole of the saving: 2 residual evaluations of
        // 89 and 2 builds of 21.
        //
        // **Note what this deliberately is not: a residual threshold.** It
        // replaced ConsistentICTolerance, which skipped when the initial weighted
        // residual fell below a number the caller supplied. What IDACalcIC tests
        // is ||J^-1 F||, a correction to y, and the two are related by a per-row
        // amplification s_i = ||J^-1 e_i||_wrms that is nowhere near proportional
        // to the error weights. Measured as s_i/ewt_i on LinearDiffusion, MatTest
        // and AuxVarTest:
        //
        //   sigma, q                 0.6 - 10      about right, and uniform
        //   u                        2.3e-4 - 2.0  over-weighted up to ~4000x
        //   lambda, Dirichlet ends   exactly 0     largest weight in the vector,
        //                                          on rows residual never writes
        //   aux                      0.9 - 39      under-weighted up to ~10x vs sigma
        //
        // The u rows are the differential ones, whose residual IDA absorbs into
        // u'. The Dirichlet trace rows are imposed inside the linear solve, so
        // J^-1 e_i is identically zero there and they can only dilute the mean.
        // Over six AuxVarTest warm-start states -- three tolerances, corrected and
        // not -- ||J^-1 F|| / ||F|| ran from 15 to 187, for one problem at one
        // discretisation. There is no number to pick, so the decision is made from
        // what the run *is* rather than from what its residual measures.
        void setForceConsistentIC(bool force) { forceConsistentIC = force; };
        bool getForceConsistentIC() const { return forceConsistentIC; };

        // The weighted residual norm initialize() measured at the initial state,
        // and whether it then ran IDACalcIC. NaN and false when nothing was
        // measured -- a steady solve, which skips CalcIC outright.
        double getInitialResidualNorm() const { return initial_residual_norm; };
        bool initialConditionWasCorrected() const { return calcICRan; };

        void setJacEvalY( N_Vector, N_Vector );
        int residual(sunrealtype, N_Vector, N_Vector, N_Vector);

        // Couple this solver to a magnetic-field model, whose unknowns join the
        // solution vector after the global scalars and whose geometry reaches
        // the physics through State::geom.
        //
        // Must be called before initialize(), and refuses afterwards: it
        // reshapes the five DGSoln members and reallocates the three that own
        // their memory, and there is no way to do that safely to a live run.
        // Passing nullptr detaches, which is what every existing run already is.
        void setFieldModel(std::shared_ptr<FieldModel> model);

        FieldModel *getFieldModel() const { return fieldModel.get(); };
        Index getFieldDOF() const { return nField; };
        Index getGeometrySlots() const { return nGeom; };

        // How the coupled Jacobian is solved once a field model is attached.
        // Iterative is block Gauss-Seidel and is the default; Exact is the
        // Schur complement onto psi, and is a verification tool rather than a
        // production path -- it is the oracle the iterative path is compared
        // against, and what makes the coupled system checkable by SolveJacTests'
        // method. See solveCoupledJacIterative and solveCoupledJacExact above
        // for what each costs.
        //
        // Consulted by initialize(), which says once per run what the choice
        // costs, and by solveJacEq.
        enum class FieldSolveMode
        {
            Iterative,
            Exact,
        };

        void setFieldSolveMode(FieldSolveMode m) { fieldSolveMode = m; };
        FieldSolveMode getFieldSolveMode() const { return fieldSolveMode; };

        void setFieldSolveTolerance(double tol)
        {
            if (tol <= 0)
                throw std::logic_error("Field solve tolerance cannot be zero or negative.");
            fieldSolveTolerance = tol;
        };
        double getFieldSolveTolerance() const { return fieldSolveTolerance; };

        void setFieldSolveMaxSweeps(int n)
        {
            if (n < 1)
                throw std::logic_error("Field solve sweep cap must be at least one.");
            fieldSolveMaxSweeps = n;
        };
        int getFieldSolveMaxSweeps() const { return fieldSolveMaxSweeps; };

        // The adjoint sweep's own cap, separate from the forward one and larger.
        // The transposed iteration has the same spectrum -- it is the transpose --
        // but always runs at cj = 0, where rho is largest, so it is strictly the
        // harder direction and inheriting the forward cap would under-serve it.
        void setFieldSolveMaxAdjointSweeps(int n)
        {
            if (n < 1)
                throw std::logic_error("Field solve adjoint sweep cap must be at least one.");
            fieldSolveMaxAdjointSweeps = n;
        };
        int getFieldSolveMaxAdjointSweeps() const { return fieldSolveMaxAdjointSweeps; };

        // What the coupled sweeps cost, and whether either had to escalate.
        // Zeroed per run by initialize(); nothing here feeds the answer.
        struct FieldSweepStats
        {
            long solves, iterations, fallbacks, adjointSweeps;
            bool adjointFellBack;
        };
        FieldSweepStats getFieldSweepStats() const
        {
            return {fieldSweepSolves, fieldSweepIterations, fieldSweepFallbacks,
                    fieldAdjointSweeps, fieldAdjointFellBack};
        };

        // The solution as it stands. `y` is a non-owning view over memory
        // SUNDIALS owns and dangles after destroySundials(), so yJac is the only
        // copy that outlives a run; initialize() seeds it with the initial
        // condition, so this is meaningful between the two phases as well.
        DGSoln const &getSolution() const { return yJac; };

        // Fill the geometry rows of `states` from the field model, at the points
        // those states were sampled on. A no-op with no model attached.
        //
        // Called once per residual and once per Jacobian update, never once per
        // variable: geometry does not depend on which equation is being
        // assembled.
        //
        // With Superconvergent = true the points are the k+2 star nodes rather
        // than the k+1 basis nodes, which needs no special case here -- geometry
        // is a function of (psi, x) and star nodes are just more x.
        void evaluateGeometry(DGSoln const &Y, std::vector<Position> const &points,
                              GlobalState &states, Time t);
        // ||F(t, Y, dYdt)|| in the WRMS norm with this solver's own error
        // weights -- the same measure the WriteDebugDatFiles output reports, so
        // the number a user sees in the .res.dat is the number the skip above is
        // decided on. Costs one residual evaluation and nothing else.
        double weightedResidualNorm(double t, N_Vector Y_in, N_Vector dYdt_in);

        // Adjoints
        void setSolveAdjoint(bool a) { solveAdjoint = a; }

        void initializeMatricesForAdjointSolve(Index gIndex = 0);

        // Solve J^T z = dG/dy at the state the matrices above were built from.
        // Dispatches on whether a field model is attached, exactly as solveJacEq
        // does forwards: without one this *is* solveTransportAdjoint.
        void solveAdjointState();

        // One objective's contribution to G_p, at whatever adjoint state and
        // adjoint matrices are currently in place. See the definition for why it
        // is callable separately from computeAdjointGradients().
        void accumulateAdjointGradients(Index gIndex);

        // The transpose of solveCoupledJacExact. The block elimination runs the
        // other way round, so the Schur complement onto psi is
        //
        //     ( B^T - A1^T A^-T A2^T ) z_psi = G_psi - A1^T A^-T G_y
        //     A^T z_x                        = G_y - A2^T z_psi
        //
        // which is why FieldModel declares applyBTranspose and solveBTranspose
        // beside the forward pair: a model that supplied only one direction
        // cannot be silently accommodated here.
        //
        // Costs nField + 1 transposed transport solves, the same as the forward
        // exact path, and is a verification tool for the same reason.
        void solveCoupledAdjointExact();

        // The transposed block Gauss-Seidel sweep, Irons-Tuck accelerated like
        // its forward twin, and capped by FieldSolveMaxAdjointSweeps rather than
        // FieldSolveMaxSweeps. Reaching that cap escalates to
        // solveCoupledAdjointExact -- it used to throw, and escalating is
        // strictly stronger: the caller gets a correct gradient rather than an
        // exception. Warned once, and recorded in fieldAdjointFellBack.
        void solveCoupledAdjointIterative();

        void computeAdjointGradients();

        void PrintDebugInfo();

        friend class PyRunner; // We need to be able to access private variables for the Python runner class

    MANTA_TEST_PRIVATE:
        Grid grid;
        unsigned int k;		   // polynomial degree per cell
        unsigned int nCells;   // Total cell count
        unsigned int nVars;	   // Total number of variables
        unsigned int nScalars; // Any global scalars
        unsigned int nAux;	   // Any auxiliary constraints

        // The field model's unknowns, and the geometry slots derived from them.
        // Both are zero until setFieldModel attaches a model, which is every
        // existing run. Declared here, ahead of the DGSoln members below,
        // because they are arguments to those members' constructors and member
        // initialisation follows declaration order.
        Index nField = 0;
        Index nGeom = 0;

        unsigned int nP;       // Number of parameters to compute for adjoint sensitivity problem

        using EigenCellwiseSolver = Eigen::FullPivLU<Matrix>;
        using EigenGlobalSolver = Eigen::FullPivLU<Matrix>;

        std::vector<Matrix> XMats;
        std::vector<Matrix> MBlocks;

        std::vector<Matrix> CEBlocks;
        Matrix K_global;
        Vector L_global;
        Matrix H_global_mat;
        Eigen::FullPivLU<Matrix> H_global;
        std::vector<Vector> RF_cellwise;
        std::vector<Matrix> CG_cellwise;
        std::vector<Matrix> A_cellwise, B_cellwise, D_cellwise, E_cellwise, C_cellwise, G_cellwise, H_cellwise, Csigma_cellwise, Cq_cellwise;

        // Adjoint vectors
        std::vector<Matrix> adjoint_CEBlocks;
        std::vector<Matrix> adjoint_CGBlocks;
        std::vector<Vector> G_y;
        Vector adjoint_lambdas;
        std::vector<Vector> adjoint_squ;

        // The right-hand-side-independent half of the transposed HDG solve:
        // M^-T CG^T per cell, and the condensed trace operator factorised.
        //
        // Hoisted out of solveAdjointState because the coupled paths apply A^-T
        // repeatedly -- nField + 1 times for the exact Schur complement, once
        // per sweep for the iterative one -- and neither of these depends on the
        // right-hand side. Filled by initializeMatricesForAdjointSolve, which is
        // also the only place MXSolvers holds M^T.
        std::vector<Matrix> adjoint_SQU_0;
        EigenGlobalSolver adjoint_K;

        // The coupling blocks as the adjoint needs them: transposed, stored,
        // and used only from here.
        //
        //   A1_transpose_cellwise[i]  (nField, localDOF)  -- A1_cellwise[i]^T
        //   A2_transpose_cellwise[i]  (localDOF, nField)  -- column f is cell
        //                             i's [ sigma | q | u | aux ] segment of
        //                             the A2 row a2[f]
        //
        // Materialised rather than transposed at each use so that a test can
        // zero one of them and require the gradient to go wrong: without that
        // guard a gradient check passes on an objective that never sees the
        // coupling. Stored transposed for the same reason M is -- the adjoint
        // operator *is* the transpose, so keeping the two shapes side by side
        // is what makes a missing block visible in review.
        std::vector<Matrix> A1_transpose_cellwise;
        std::vector<Matrix> A2_transpose_cellwise;

        // The field block of the adjoint state, z_psi, and of the adjoint
        // right-hand side, dG/dpsi.
        //
        // G_field is identically zero and that is a *limit*, not a fact about
        // the discretisation: AdjointProblem reports dg/du, dg/dq, dg/dsigma and
        // dg/dphi and has no geometry hook, so an objective whose integrand
        // reads State::geom directly loses that term. It cannot be detected from
        // here -- the same standing limit AdjointVectors.cpp records for the
        // absent dgFn_dscalars -- so it is named here and in TODO rather than
        // left as an unremarked zero.
        Vector adjoint_field;
        Vector G_field;

        SUNContext ctx;
        N_Vector *v, *w;

        // The field coupling.
        //
        //   A1_cellwise[i]  one cell's ( (3 nVars + nAux)(k+1), nField ) block of
        //                   d(transport residual)/d(psi), in MX's row order
        //                   [ sigma | q | u | aux ]. Sized in initialiseMatrices,
        //                   filled by assembleFieldCoupling.
        //   a2[f]           field row f of d(field residual)/d(transport DOF),
        //                   as a full-length vector so it contracts with a
        //                   solution vector by a plain dot product -- the shape
        //                   the scalar bordering's `w` already uses.
        //
        // a2 is allocated by setFieldModel rather than by the constructor,
        // because that is where nField -- and so both the count and the length
        // of these vectors -- becomes known. Null with no model attached.
        std::vector<Matrix> A1_cellwise;
        N_Vector *a2 = nullptr;

        // ---- state of one run, owned between initialize() and destroySundials()
        //
        // These were locals of runSolver(). They are members so the three phases
        // can be called separately; destroySundials() nulls each one, which is
        // both what makes it idempotent and what lets initialize() be called
        // again afterwards.
        //
        // `ctx` above is deliberately *not* one of them: it belongs to the
        // SystemSolver, created in the constructor and freed in the destructor.
        // destroySundials() must not touch it -- freeing it per-run is what used
        // to make a second runSolver() call on the same object fail at IDACreate.
        void *IDA_mem = nullptr;      // IDA memory structure
        SUNLinearSolver LS = nullptr; // linear solver memory structure
        SUNMatrix sunMat = nullptr;   // the deliberately-empty matrix IDA needs

        // Pseudo-transient continuation. kinLS is a second wrapper over the same
        // solveJacEq rather than a share of LS, so the two owners never argue
        // over one object's lifetime; sunMat is stateless and is shared.
        void *kin_mem = nullptr;
        SUNLinearSolver kinLS = nullptr;
        N_Vector uPrev = nullptr;    // previous PTC iterate
        N_Vector ptcDYdt = nullptr;  // id * (u - uPrev)/dt, the damping term
        N_Vector kinScale = nullptr; // unit scaling; KINSol requires a vector
        SteadyMode steadyMode = SteadyMode::TimeMarch;
        double ptcInitialStep = 0.0; // 0 means "use dt0"
        // How many KINSol calls before giving up. Each is a full Newton solve,
        // so a healthy run uses ten or so, and the default is a runaway
        // backstop rather than a budget. Lowering it deliberately is what makes
        // a solve stop early enough to be looked at and then resumed.
        long maxContinuationSteps = 200;
        bool estimateObjectiveOnFinish = true;
        double ptcMaxStep = std::numeric_limits<double>::infinity();
        double ptcStep = 0.0;        // the current dt; infinite in Newton mode
        double ptcSERRate = 1.0;     // exponent on the residual ratio
        double ptcSERFloor = 2.0;    // least growth on an accepted step

        // KINSOL's own settings. Every default here is what the code hardcoded
        // before they were configurable, so an unconfigured run is unchanged.
        long newtonMaxIters = 20;    // KINSOL's default is 200; see the setter
        long newtonJacReuse = 10;    // KINSOL's msbset default
        double newtonStepTol = 0.0;  // 0 = leave KINSOL's uround^(2/3)
        NewtonScaling newtonScaling = NewtonScaling::Unit;

        // Work counters, monotonic over the solver's lifetime. solveSteadyState
        // snapshots them on entry and reports the difference, which is what
        // makes them meaningful when one solver is run more than once -- the
        // pattern PyRunner depends on. IDA has its own equivalents
        // (IDAGetNumResEvals and friends) and the time loop prints those; these
        // count MaNTA's own entry points, so they cover both drivers and they
        // separate the two costs a steady solve actually pays: building the
        // per-cell blocks and factorising them (nJacBuilds) against the static
        // condensation that reuses them (nJacSolves).
        long nResidualEvals = 0;
        long nJacBuilds = 0;
        long nJacSolves = 0;
        bool steadyDiagnostics = false;
        bool steadyStepDiagnostics = false;
        SteadyStats steadyStats;
        ObjectiveEstimate objectiveEstimate;
        SteadyOutcome steadyOutcome = SteadyOutcome::NotRun;
        std::vector<SteadyStepStats> steadyStepStats;
        N_Vector Y = nullptr;         // solution
        N_Vector dYdt = nullptr;      // time derivative of the solution
        N_Vector constraints = nullptr;
        N_Vector id = nullptr;        // which components are differential
        N_Vector res = nullptr;       // residual
        N_Vector absTolVec = nullptr;
        sunrealtype tout = 0.0, tret = 0.0;

        std::ofstream out0, dydt_out, res_out;
        // writeDebugDatFiles && physics_debug. Computed once in initialize()
        // because the time loop and the teardown both need it.
        bool debugDat = false;

        std::vector<Matrix> W_cellwise;
        Matrix N_global; // Scalar-scalar coupling matrix

        //?Point the duplicated matrices to the same place?
        std::vector<EigenCellwiseSolver> MXSolvers;

        DGSoln y, dydt; // memory owned by SUNDIALS

        double *yJacMem = nullptr;
        double *dydtJacMem = nullptr;

        DGSoln yJac; // memory owned by us
        DGSoln dydtJac; // memory owned by us

        // Built in initialiseMatrices(), once the polynomial degree and grid are
        // fixed. Non-copyable and holds a reference to `grid`, hence the pointer.
        std::unique_ptr<Postprocessor> postprocessor;
        bool superconvergent = false;

        Matrix G_p; // gradients computed by adjoint state method

        // Where the physics derivatives were evaluated: the k+2 star nodes with
        // the superconvergent scheme, the k+1 cell nodes otherwise. Returned
        // together because the two must agree -- the scalar columns are built on
        // the same node set as the rest of the Jacobian, and re-deriving it at
        // each use is how the two would drift apart.
        struct PhysicsNodes
        {
            std::vector<Position> points;
            GlobalState states;
        };

        // Size and fill the three derivative blocks at the state Y and time
        // tEval, and report the nodes they were evaluated on.
        PhysicsNodes evaluatePhysicsDerivatives(DGSoln const &Y, Time tEval,
                                                GlobalStateMatrix &dSigma_vals,
                                                GlobalStateMatrix &dSource_vals,
                                                GlobalStateMatrix &dAux_vals);

        // One cell's Jacobian block, [ sigma | q | u | aux ] by
        // [ sigma | q | u | aux ], from derivative blocks evaluatePhysicsDerivatives
        // has filled.
        //
        // alphaValue scales the mass term in the u row -- IDA's cj for the
        // forward solve, and 0 where dF/dy alone is wanted, which is what makes
        // it shareable with anything that needs dF/dy alone. It is the *only*
        // place this block layout is written down for the forward direction;
        // initializeMatricesForAdjointSolve holds the transposed copy and has to
        // be kept in step with it block for block.
        Matrix assembleCellMatrix(Index i, DGSoln const &Y,
                                  GlobalStateMatrix &dSigma_vals,
                                  GlobalStateMatrix &dSource_vals,
                                  GlobalStateMatrix &dAux_vals, double alphaValue);

        // The scalar coupling: v (how the HDG rows depend on the scalars) and w
        // (how the scalar constraints depend on the HDG unknowns), plus the
        // scalar-scalar matrix N. Written through the caller's storage rather
        // than into the members, so that a second consumer can assemble its own
        // copy without disturbing the forward solve's.
        void assembleScalarCoupling(DGSoln const &Y, DGSoln const &Ydot,
                                    PhysicsNodes const &nodes, Time tEval,
                                    double alphaValue, std::vector<DGSoln> &v_map,
                                    std::vector<DGSoln> &w_map, Matrix &N_out);

        // The three field blocks, from one FieldResidualPrime call.
        //
        //   A1 (per cell, into A1_cellwise) -- how the transport rows see psi,
        //      by the chain rule through the case's geometry hooks and the
        //      model's dGeometry/dpsi.
        //   A2 (into a2)                    -- how the field rows see the
        //      transport unknowns, weighted by alpha exactly as the scalar `w`
        //      vectors are.
        //   B  (into the model)             -- dR/dpsi + alpha dR/d(psi'), which
        //      the model assembles and factorises for itself.
        //
        // One call is deliberate: a model that solves a coupled system
        // internally reports every row at once, and is entitled to be asked
        // once per Jacobian. This replaced updateFieldBlock, which made the
        // same call, threw dR and dRdot away, and left the Jacobian block
        // diagonal.
        void assembleFieldCoupling(DGSoln const &Y, DGSoln const &Ydot,
                                   PhysicsNodes const &nodes, Time tEval,
                                   double alphaValue);

        // Column m of A1, scattered into a full-length solution vector: each
        // cell's block at its own offset, zero everywhere else -- including the
        // field block, so that the transport solve it is fed to cannot mistake
        // it for a right-hand side for psi.
        void scatterA1Column(Index m, N_Vector out) const;

        // Apply A^-T: the transpose of the uncoupled transport operator, by the
        // same static condensation solveHDGJac performs forwards.
        //
        // `rhs` is one vector per cell in the [ sigma | q | u | aux ] ordering
        // and the trace rows' right-hand side is identically zero. That is a
        // property of everything that reaches here rather than a simplification:
        // the objective has no trace dependence (AdjointProblem::dg reports four
        // blocks, none of them lambda), and neither has A2 -- a field residual
        // is handed a GlobalState, which has no trace slot at all.
        // initializeMatricesForAdjointSolve *checks* the second of those rather
        // than assuming it.
        //
        // Needs adjoint_SQU_0 and adjoint_K, so only meaningful after
        // initializeMatricesForAdjointSolve.
        void solveTransportAdjoint(std::vector<Vector> const &rhs,
                                   std::vector<Vector> &squOut, Vector &lambdaOut);

        // Fill adjoint_SQU_0 and factorise adjoint_K -- the part of the
        // transposed solve that does not depend on the right-hand side.
        void factoriseAdjointTrace();

        // Transpose A1 and A2 into A1_transpose_cellwise / A2_transpose_cellwise,
        // and refuse an A2 row with anything outside the cellwise blocks.
        void transposeFieldCoupling();

        // work's cellwise [sigma | q | u | aux] segment gets A1_cellwise[i]*dpsi
        // subtracted, cell by cell -- the A1 dpsi term of the block
        // Gauss-Seidel sweep in solveCoupledJacIterative. A1_cellwise already
        // carries d(res)/d(psi) with its own sign baked in (see
        // dPhysics_dField_Mat), so this is a plain subtraction with no sign of
        // its own; the lambda, scalar and field segments of work are untouched.
        void subtractA1Times(Vector const &dpsi, N_Vector work) const;

        // Allocate (or reallocate) the two buffers yJac and dydtJac map. Called
        // from the constructor, and again from setFieldModel, which changes how
        // long they have to be.
        void allocateJacobianStorage();

        // Free the a2 vectors, using the *current* nField as the count -- so
        // call it before changing nField, not after.
        void freeFieldWorkVectors();

        // The whole Jacobian, densely, in the solution vector's own ordering:

        // The central-difference step: cbrt(eps) scaled by |t|. That is the
        // exponent that balances a *central* difference's truncation against its
        // round-off; sqrt(eps) is the one-sided choice and costs 2.5 decimal
        // places here. See the note on the definition.
        static double timeDifferenceStep(Time tEval);

        // dF/dt at fixed state, by central difference of residual() in t alone.
        // This is the whole right-hand side of the algebraic-derivative solve, and
        // it is the only place the explicit time dependence of the boundary data,
        // the flux, the sources and the aux constraint enters -- none of which has
        // an analytic derivative anywhere in the tree.
        //
        // Exactly zero, bit for bit, for an autonomous case: residual() is a
        // function of t only through those, so the two evaluations return
        // identical vectors rather than nearly identical ones.
        //
        // Puts RF_cellwise and L_global back at tEval however it returns -- both
        // residual() calls leave them at tEval - h, and they are what the forward
        // residual reads.
        Vector differenceResidualInTime(Time tEval, double h);

        void NLqMat(Matrix &, DGSoln const &, Index);
        void NLuMat(Matrix &, DGSoln const &, Index);
        void NLphiMat(Matrix &, DGSoln const &, Index);

        void dSourcedu_Mat(Matrix &, DGSoln const &, Index);
        void dSourcedq_Mat(Matrix &, DGSoln const &, Index);
        void dSourcedsigma_Mat(Matrix &, DGSoln const &, Index);

        void DerivativeSubMatrix(Matrix &mat, void (TransportSystem::*dX_dZ)(Index, VectorRef, const State &, Position, Time), DGSoln const &Y, Index I);

        void DerivativeSubMatrix(Matrix &mat, std::vector<Eigen::Ref<Matrix>> const dX_dZ, DGSoln const &, Index intervalIndex);

        // The superconvergent counterpart of DerivativeSubMatrix, and the only
        // place the chain rule through the postprocessing lives.
        //
        // With the star scheme a physics value X is evaluated at the k+2 star
        // nodes with u* in place of u_h, and the resulting P_{k+1} interpolant is
        // projected onto the P_k test space by A9. So for a cell dof vector Z,
        //
        //     d/dZ ( X, phi_i )_K  =  A9 . diag( dX/dW ) . dW/dZ
        //
        // where W is whichever field X was differentiated with respect to and the
        // trailing `chain` is dW/dZ evaluated at the star nodes:
        //
        //     Z = u coefficients      chain = B12   (u* depends on them)
        //     Z = q coefficients      chain = V     for dX/dq, and additionally
        //                                    B11    for dX/du, since u* depends
        //                                           on q as well
        //     Z = sigma or phi        chain = V     (simply sampled there)
        //
        // Accumulates rather than assigns, so the two contributions to the q
        // column can be added in turn. dX_dZ[XVar](WVar, m) is dX_XVar/dW_WVar at
        // star node m, the same indexing DerivativeSubMatrix uses.
        void accumulateStarBlocks(MatrixRef mat,
                                  std::vector<Eigen::Ref<Matrix>> const &dX_dZ,
                                  Matrix const &chain, Index nX, Index nZ,
                                  Index intervalIndex) const;

        // Takes the evaluation time rather than reading `jt`: this is now called
        // from the algebraic-derivative solve as well, which is not a Jacobian
        // evaluation and so does not set it.
        void dSources_dScalars_Mat(Matrix &, DGSoln const &, Index, Time );

        // Superconvergent counterpart. The scalars do not enter the
        // postprocessing, so there is no chain matrix -- only the star nodes and
        // A9 in place of the mass matrix. Takes the states and positions the
        // caller already has rather than re-deriving them from a DGSoln, which it
        // could not do for the star nodes anyway.
        void dSources_dScalars_StarMat(Matrix &, GlobalState const &,
                                       std::vector<Position> const &, Index, Time);

        // One cell's block of A1: d(sigma, u and aux residual rows)/d(psi), by
        // the chain rule
        //
        //     d(row)/d(psi_m) = sum_g d(row)/d(geometry_g) . d(geometry_g)/d(psi_m)
        //
        // The first factor is the case's, the second the field model's. The q
        // rows and the trace rows have no geometry dependence and stay zero.
        //
        // Shape ( (3 nVars + nAux)(k+1), nField ), laid out [ sigma | q | u | aux ]
        // to match assembleCellMatrix's rows.
        //
        // `states` must be the ones evaluatePhysicsDerivatives filled -- they
        // carry geometry, and a hook may read it (d/dg of g^2 q is 2 g q). That
        // is why this takes them rather than calling DGSoln::evalOnNode the way
        // dSources_dScalars_Mat does: a State built from a DGSoln has no
        // geometry rows at all.
        void dPhysics_dField_Mat(Matrix &mat, DGSoln const &Y, GlobalState const &states,
                                 std::vector<Position> const &points, Index intervalIndex,
                                 Time tEval);

        // Superconvergent counterpart: the k+2 star nodes and A9 in place of
        // InterpolateOntoBasis. There is no chain matrix -- geometry is a
        // function of (psi, x), and u* does not enter it.
        void dPhysics_dField_StarMat(Matrix &mat, DGSoln const &Y, GlobalState const &states,
                                     std::vector<Position> const &points, Index intervalIndex,
                                     Time tEval);

        // d(one physics value)/d(psi_m) at each node of one cell, (nField, nNodes):
        // the case's dX/dgeometry there contracted with the model's
        // dGeometry/dpsi. The part the two functions above share; they differ
        // only in how they project the result onto the test space.
        void fieldChainOnNodes(Matrix &nodal, Index XVar,
                               void (TransportSystem::*dX_dGeom)(Index, VectorRef, const State &,
                                                                 Position, Time),
                               Vector const &psi, GlobalState const &states,
                               std::vector<Position> const &points, Index intervalIndex,
                               Index nNodes, Time tEval);

        void dSourcedPhi_Mat(Matrix &, DGSoln const &, Index );
        void dPhi_Mat(Matrix &, std::vector<Eigen::Ref<Matrix>> const dX_dZ, DGSoln const &, Index );

        void dAux_Mat(Eigen::Ref<Matrix>, DGSoln const &, Index );
        void dAux_Mat(Eigen::Ref<Matrix>, GlobalStateMatrix&, DGSoln const &, Index );

        // Takes the nodal values of one dg/dZ, batched, and weights them. The
        // pointwise sibling that took a member-function pointer and integrated it
        // with the basis's Gauss rule is gone, along with the dGdu_Vec/dGdq_Vec/
        // dGdsigma_Vec wrappers over it: it computed Int dg/dZ phi_i dx, which is
        // the derivative of Int g dx and not of the sum_m w_m g_m that GFn
        // actually reports, and no solve ever called it -- only a pair of
        // "does not throw" assertions did.
        void DerivativeSubVector(Index, Vector &, Eigen::Ref<Matrix> const dX_dZ, DGSoln const &, Index intervalIndex);

        // The same operator over the auxiliary variables: separate only because
        // there are nAux of these where DerivativeSubVector's loop runs to nVars.
        // The overload that integrated the pointwise dgFn_dphi on the basis's
        // Gauss rule -- the last of the family above, and the last differentiating
        // Int g dx rather than what GFn reports -- is gone with them.
        void dGdaux_Vec(Index, Vector &, Eigen::Ref<Matrix> const dX_dZ, DGSoln const &, Index intervalIndex);

        double resNorm = 0.0; // Exclusively for unit testing purposes

        double dt;
        double t0, t, jt;

        // Really we should do init in the constructor and not need this flag. TODO
        bool initialised = false;

        // Which quantity a *Neumann* end constrains: q (false) or sigma (true).
        // Read in exactly one place, effectiveBoundaryCondition below, which
        // expresses it as the equivalent Mixed coefficients.
        bool zeroFlux = false;

        // The boundary condition an end is *assembled* as, which is not always the
        // one the case declared: a Neumann end is a Mixed one with b = 1, or with
        // d = 1 when zeroFlux is set, that flag's entire meaning. Returning the
        // coefficients rather than branching on the kind is what gives the
        // assembly one path for both, so a fix to one cannot miss the other.
        //
        // Dirichlet is passed through unchanged and handled separately: it is not
        // a Mixed row at all but an identically zero one, with the datum
        // substituted into the cell rows instead.
        BoundaryCondition effectiveBoundaryCondition(BoundaryCondition const &declared) const
        {
                if (declared.kind != BoundaryKind::Neumann)
                        return declared;
                return zeroFlux ? BoundaryCondition::mixed(0.0, 0.0, 1.0)
                                : BoundaryCondition::mixed(0.0, 1.0, 0.0);
        }
        BoundaryCondition effectiveLowerBoundary(Index var) const
        {
                return effectiveBoundaryCondition(problem->lowerBoundaryCondition(var));
        }
        BoundaryCondition effectiveUpperBoundary(Index var) const
        {
                return effectiveBoundaryCondition(problem->upperBoundaryCondition(var));
        }

        // Text output is opt-in; netCDF is what a run produces by default.
        bool writeOutput = true;
        bool writeDatFile = false;
        bool writeDebugDatFiles = false;

        // IDASetEtaMax(10.0) rather than IDA's default 2.0. See
        // setAggressiveTimesteps.
        bool aggressiveTimesteps = false;

        // IDASetSuppressAlg. See setSuppressAlgebraicError.
        bool suppressAlgebraicError = false;

        double alpha = 1.0;
        bool testing = false;

        // Why do we need to know? Surely everything is encoded in the construction of the Grid, which is done elsewhere?
        bool highGridBoundary = true;

        bool solveAdjoint = false; 

        // Hide all physics-specific info in here
        TransportSystem *problem = nullptr;

        // Null when no field model is configured, which is every existing run.
        // Held by shared_ptr because the adjoint solve and the Python layer both
        // need to reach it and neither owns the solver.
        std::shared_ptr<FieldModel> fieldModel = nullptr;

        // Iterative to match the FieldSolve default, so there is one default
        // rather than two that can drift. Read by solveJacEq and by
        // initialize(), which reports what the choice costs.
        FieldSolveMode fieldSolveMode = FieldSolveMode::Iterative;
        double fieldSolveTolerance = 1e-8;
        int fieldSolveMaxSweeps = 20;
        // Larger than the forward cap, and for the reason the setter gives: the
        // adjoint runs at cj = 0, where the coupling is stiffest. One default,
        // matching ConfigSchema's, rather than two that can drift.
        int fieldSolveMaxAdjointSweeps = 100;

        // Diagnostics for the coupled sweeps, zeroed per run. Nothing here feeds
        // the answer, so none of it can disturb the bit-for-bit reuse invariant
        // that a_second_integration_on_one_solver_matches_a_fresh_one pins -- but
        // it is zeroed per run all the same, because a cumulative count reported
        // as a per-run one is a lie a second run would tell silently.
        long fieldSweepSolves = 0;
        long fieldSweepIterations = 0;
        long fieldSweepFallbacks = 0;
        long fieldAdjointSweeps = 0;
        bool fieldAdjointFellBack = false;

        // Irons-Tuck vector acceleration: Aitken's Delta^2 generalised to
        // vectors.
        //
        // Both coupled sweeps are affine fixed-point iterations on the field
        // block alone. Writing G for one sweep, G(p) = c + M p with
        // M = B^-1 A2 A^-1 A1 forwards and its transpose backwards, so the plain
        // sweep converges only for rho(M) < 1 -- and rho is a property of the
        // coupling rather than of the time step: 1.611 at cj = 0 against 1.571
        // at cj = 1e8 for RichGeometricDiffusion.
        //
        // Given the two most recent increments D_k = G(p_k) - p_k and D_{k-1}:
        //
        //     mu  = D_k . (D_k - D_{k-1}) / |D_k - D_{k-1}|^2
        //     p*  = G(p_k) - mu D_k
        //
        // which is the secant (rank-one quasi-Newton) step on F(p) = G(p) - p.
        // For nField == 1 and an affine G it lands on the fixed point *exactly*,
        // in one step, for every m -- including m > 1, where no relaxation
        // parameter helps at all: D_k = m D_{k-1} gives mu = m/(m-1) and
        // p* = p_k + D_k/(1-m), and D_k = (1-m)(p_fix - p_k), so p* = p_fix.
        // That exactness for a divergent scalar map is why this rather than SOR,
        // whose eigenvalues 1 - w + w*lambda cannot be brought inside the unit
        // circle by any w > 0 when lambda > 1.
        //
        // For nField > 1 it is a rank-one approximation: it removes the dominant
        // eigendirection and leaves the rest, which is why the caller still needs
        // the exact fallback. Anderson acceleration -- equivalently GMRES on the
        // Schur complement, since the map is affine -- is the depth-m
        // generalisation and is in TODO.
        static Vector ironsTuck(const Vector &g, const Vector &delta, const Vector &deltaPrev);

        AdjointProblem *adjointProblem = nullptr;

        // Tau
        double tauc;
        double tau(double x) const { return tauc; };

        double rtol;
        std::vector<double> atol;

        NetCDFIO nc_output;
        NetCDFIO restart_file;
        void initialiseNetCDF(std::string const &fname, size_t nOut);
        void WriteTimeslice(double tNew);
        void WriteRestartFile(std::string const &fname, N_Vector const &Y, N_Vector const &dYdt, size_t nOut);
        void WriteAdjoints();

        // The field model's own netCDF group: one time series per FieldDOF, one
        // spatial variable per FieldSlot, and the spec's `label` as an attribute
        // so a run records what its x meant. No-ops with no model attached, and
        // reached only from the three functions above -- all of which Solver.cpp
        // already gates on writeOutput.
        //
        // Both take the file, because the same group is written to <stem>.nc and
        // to <stem>.restart.nc, and both take the time, because geometry is a
        // function of (psi, x, t) and the two files are written at different
        // times: the t0 slice against t0, a timeslice against its own tNew, the
        // restart file against the time the run reached.
        void initialiseFieldOutput(NetCDFIO &file, Time tEval);
        void writeFieldTimeslice(NetCDFIO &file, size_t tIndex, Time tEval);

        size_t S_DOF,
        U_DOF, Q_DOF, AUX_DOF, SQU_DOF;
        size_t localDOF;

        bool TerminateOnSteadyState = false;

        bool forceConsistentIC = false;

        // Set by setInitialConditions: true when a restart had to be *projected*
        // onto a different discretisation rather than copied. The skip in
        // initialize() is conditional on it -- see setForceConsistentIC.
        bool restartWasProjected = false;
        double initial_residual_norm = std::numeric_limits<double>::quiet_NaN();
        bool calcICRan = false;
        double steady_state_tol = 1e-3;

#ifdef PHYSICS_DEBUG
        constexpr static bool physics_debug = true;
#else
        constexpr static bool physics_debug = false;
#endif

        std::filesystem::path inputFilePath;
        double dt0 = 0.0; // initial dt for CalcIC
        int nOut;
        double min_step_size;

        int getErrorWeights( N_Vector y, N_Vector ewt );
        static int getErrorWeights_static( N_Vector, N_Vector, void * );

        // Allocated by initialize() only on the debug-.dat path, so it has to
        // start null: destroySundials() frees it if it is non-null, and an
        // uninitialised pointer there is a segfault on every ordinary run.
        N_Vector wgt = nullptr;
};

#endif // SYSTEMSOLVER_HPP
