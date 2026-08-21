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
        void solveSteadyState();

        // The two KINSOL callbacks, public because the C shims in
        // SteadyState.cpp reach them through the user_data pointer.
        int steadyResidual(N_Vector u, N_Vector fval);
        void steadyJacSetup(N_Vector u);

        // Arm the dG/dt early-exit gate: after the initial condition is built,
        // abandon the run rather than integrate it if the objective is already
        // getting worse. For an optimisation sweep that turns a wasted transport
        // solve into the cost of initialisation alone.
        //
        // An absolute threshold on a dimensional quantity has no sensible
        // default, so the gate is off until this is called -- like
        // setSteadyStateTolerance above, which this deliberately mirrors.
        void setObjectiveDecreaseTolerance(double dGdt_tol)
        {
            if (dGdt_tol <= 0)
                throw std::logic_error("Tolerance for objective-decrease termination cannot be zero or negative.");
            objective_decrease_tol = dGdt_tol;
            CheckObjectiveDecrease = true;
        };
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

        // Fill the algebraic blocks of dydtComplete -- q', sigma', phi' and
        // lambda' -- by differentiating the constraints that define them.
        //
        // IDA never computes them: IDA_YA_YDP_INIT produces algebraic *values*
        // and differential *derivatives*, so at t0 those blocks of its dYdt are
        // identically zero and anything differentiating the solution in time sees
        // only the u term. Differentiating the algebraic residual rows gives
        // dF/dy . ydot = -dF/dt, which is a linear system in exactly those
        // unknowns once u' -- which IDA does have -- is treated as data.
        //
        // Reads Y and dYdt, so it is only meaningful after initialize(). Writes
        // dydtComplete and nothing else; IDA's own dYdt is the state it takes its
        // first step from and must not be touched.
        void computeAlgebraicTimeDerivatives();

        // Solves the Jy = g equation
        void solveJacEq(N_Vector g, N_Vector delY);
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

        // The dG/dt gate, asked between initialize() and integrate() -- which is
        // the reason the split has to exist for it. Returns false when the gate
        // is disarmed, so an unconfigured caller sees no behaviour change.
        //
        // Only meaningful after initialize(): it reads y and dydt, which map the
        // live SUNDIALS vectors, and it needs the derivative initialize() left
        // there -- IDACalcIC's on a time-marching run, and setInitialConditions'
        // guess on a steady one, which skips CalcIC (Solver.cpp). Before
        // initialize() there is nothing mapped; after destroySundials() they
        // dangle.
        bool objectiveIsDecreasing();

        // Whether the gate rejected the run, i.e. runSolver() skipped the time
        // loop. Cleared at the top of every initialize().
        bool wasRejected() const { return objective_rejected; };

        // The dG/dt values the last objectiveIsDecreasing() computed, one per
        // objective. For diagnostics and for the tests.
        Vector const &lastDGdt() const { return last_dGdt; };

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

        void setJacEvalY( N_Vector, N_Vector );
        int residual(sunrealtype, N_Vector, N_Vector, N_Vector);

        // Adjoints
        void setSolveAdjoint(bool a) { solveAdjoint = a; }

        void initializeMatricesForAdjointSolve();

        void solveAdjointState(Index i);

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

        SUNContext ctx;
        N_Vector *v, *w;

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
        // The time derivative with its algebraic blocks filled in.
        //
        // IDA's dYdt has zeros in q, sigma and phi at t0: IDA_YA_YDP_INIT
        // computes algebraic *values* and differential *derivatives*, so there
        // is no y' for them to fetch. computeAlgebraicTimeDerivatives() solves
        // the differentiated constraints for them and writes the answer here.
        //
        // Here rather than into dYdt because dYdt is the state IDA takes its
        // first step from: changing its algebraic entries after IDACalcIC would
        // alter the integration, and the symptom would be a step-size failure
        // somewhere later rather than anything pointing back here.
        double *dydtCompleteMem = nullptr;

        DGSoln yJac; // memory owned by us
        DGSoln dydtJac; // memory owned by us
        DGSoln dydtComplete; // memory owned by us; see dydtCompleteMem above

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
        // this shareable with computeAlgebraicTimeDerivatives(). It is the *only*
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

        // The whole Jacobian, densely, in the solution vector's own ordering:
        // [ sigma | q | u | aux ] per cell, then all of lambda, then mu. Built
        // from the same blocks the forward solve applies without ever forming --
        // assembleCellMatrix, CEBlocks, CG_cellwise, H_cellwise and the scalar
        // coupling -- so it cannot drift from them.
        //
        // Only computeAlgebraicTimeDerivatives() and the tests want this; the
        // forward path never assembles a Jacobian and never should.
        Matrix assembleDenseJacobian(DGSoln const &Y, DGSoln const &Ydot, Time tEval,
                                     double alphaValue);

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

        // The time derivative of the objective, by the chain rule over the four
        // vectors above. See AdjointVectors.cpp for why it is assembled here
        // rather than asked of AdjointProblem.
        Value dGdt(Index gIndex, DGSoln const &Y, DGSoln const &Ydot);
        Value dGdt(Index gIndex) { return dGdt(gIndex, y, dydt); };

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

        size_t S_DOF,
        U_DOF, Q_DOF, AUX_DOF, SQU_DOF;
        size_t localDOF;

        bool TerminateOnSteadyState = false;
        double steady_state_tol = 1e-3;

        // Off unless setObjectiveDecreaseTolerance arms it. There is no default
        // worth having: dG/dt carries the units of the objective over time, so
        // any number here would be meaningful for one case and nonsense for the
        // next.
        bool CheckObjectiveDecrease = false;
        double objective_decrease_tol = 0.0;
        bool objective_rejected = false;
        Vector last_dGdt;
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
