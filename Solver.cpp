#include <ida/ida.h>				  /* prototypes for IDA fcts., consts.    */
#include <kinsol/kinsol.h>			  /* KINFree, for the steady-state path   */
#include <nvector/nvector_serial.h>	  /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <toml.hpp>
#include <exception>
#include <fstream>
#include <limits>
#include <print>
#include <memory>

#include "Types.hpp"
#include "FieldModel.hpp"
#include "SystemSolver.hpp"
#include "gridStructures.hpp"
// The field hooks are handed the same quadrature weights the scalar ones are,
// so the model does not have to pick a rule of its own; they are cached here.
#include "PyIntegrator.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
#include "ErrorChecker.hpp"

// Unadvertised, but in the library
extern "C"
{
	int IDAEwtSet(N_Vector, N_Vector, void *);
}

int static_residual(sunrealtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int JacSetup(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

// runSolver() is the whole run: allocate, integrate, free. It exists so that the
// standalone binary and the tests have one call to make, and so that the free
// happens even if the time loop throws -- which it previously did not, since
// every N_VDestroy sat after the loop.
//
// destroySundials() is idempotent, so the catch-and-rethrow below cannot double
// free.
void SystemSolver::runSolver(double tFinal)
{
	initialize();

	try
	{
		integrate(tFinal);
	}
	catch (...)
	{
		destroySundials();
		throw;
	}
	destroySundials();
}

void SystemSolver::initialize()
{
	int retval;

	// Whatever the field model cached about the last run does not apply to this
	// one. This has to be here rather than in initialiseMatrices(), which is
	// skipped entirely when `initialised` is already set: that is the
	// RF_cellwise trap, where the second run on a reused solver solved its
	// initial dydt out of the previous run's final-time boundary data.
	if (fieldModel)
		fieldModel->resetForRun();

	// ...and neither do the sweep counts. Here, in the unconditional part of
	// initialize() and beside resetForRun() for the same reason: a cumulative
	// count reported as a per-run one is a lie a second run would tell silently.
	fieldSweepSolves = 0;
	fieldSweepIterations = 0;
	fieldSweepFallbacks = 0;
	fieldAdjointSweeps = 0;
	fieldAdjointFellBack = false;

	if (!initialised)
		initialiseMatrices();

	//-------------------------------------System Design----------------------------------------------

	IDA_mem = IDACreate(ctx);
	if (ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0))
		throw std::runtime_error("Sundials Initialization Error");

	retval = IDASetUserData(IDA_mem, static_cast<void *>(this));
	if (ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
		throw std::runtime_error("Sundials Initialization Error");

	//-----------------------------Initial conditions-------------------------------

	// Set original vector lengths
	// The field model's unknowns go last, after the scalars, so nothing before
	// them moves. nField is zero unless setFieldModel has attached a model.
	Y = N_VNew_Serial(nVars * 3 * nCells * (k + 1) + nVars * (nCells + 1) + nScalars + nAux * nCells * (k + 1) + nField, ctx);
	if (ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0))
		throw std::runtime_error("Sundials Initialization Error");

	dYdt = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0))
		throw std::runtime_error("Sundials Initialization Error");

	// Initialise Y and dYdt
	setInitialConditions(Y, dYdt);

	// Seed yJac/dydtJac with the initial condition. `y` and `dydt` are
	// non-owning views over memory SUNDIALS owns, so yJac is the only copy of the
	// state that survives destroySundials(), and it is what PyRunner::getSolution
	// and PyRunner::G read. Until the end of integrate() it otherwise holds
	// uninitialised memory, so anything inspecting the solver after initialize()
	// alone -- which the three-phase lifecycle now makes possible -- read
	// garbage.
	//
	// Here rather than at the end of setInitialConditions because setJacEvalY
	// asserts the vector length matches the full DoF, and setInitialConditions is
	// also called directly by tests that size their own N_Vectors.
	setJacEvalY(Y, dYdt);

	// A field DOF declared differential whose residual carries no d/dt is a row
	// every unknown of which IDA_YA_YDP_INIT holds fixed: no Newton direction
	// touches it, so the backtracking loop runs to exhaustion and IDA reports
	// IDA_LINESEARCH_FAIL (-13) -- a message about the linesearch for a defect in
	// the declaration. That is exactly what kept python-physics/mirror-plasma's
	// voltage controller from ever starting, and the residual there was 4.3e-6:
	// irreducible beats small, so there is no threshold to test against. Ask
	// instead which unknowns each row can reach, here, where the answer can name
	// the DOF.
	//
	// After setJacEvalY, because that is what puts the initial condition into
	// yJac and dydtJac, and before IDACalcIC, which is what would otherwise fail.
	if (fieldModel)
	{
		GlobalStateMatrix dR(nField), dRdot(nField);
		for (Index f = 0; f < nField; ++f)
		{
			dR.add(nCells, k, nVars, nScalars, nAux);
			dRdot.add(nCells, k, nVars, nScalars, nAux);
		}
		Matrix dRdpsi = Matrix::Zero(nField, nField);
		Matrix dRddpsidt = Matrix::Zero(nField, nField);

		fieldModel->FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt,
									   Vector(yJac.getField()), Vector(dydtJac.getField()),
									   yJac.evalOnNodes(), yJac.getPoints(),
									   Integrator::getIntegrationWeights(yJac.getBasis(), grid),
									   t0);

		for (Index f = 0; f < nField; ++f)
			if (fieldModel->isFieldDOFDifferential(f) && dRddpsidt.row(f).isZero(0.0))
				throw std::invalid_argument(
					"Field DOF '" + fieldModel->getSpec().dofs[f].name +
					"' is declared differential but its residual row carries no time "
					"derivative. IDACalcIC holds every differential value fixed, so this row "
					"is irreducible and the initialisation would fail with "
					"IDA_LINESEARCH_FAIL.");

		// What the chosen coupled solve costs, said once per run rather than
		// once per Jacobian. Here rather than in applySolverConfig because
		// nField is only known once a model is attached, and warning about the
		// cost of a solve that will never happen -- FieldSolve set on a run with
		// no field model -- is noise.
		//
		// **The two levels differ, and deliberately.** Exact is a WARNING: the
		// user asked for a verification tool and is about to pay nField + 1
		// transport solves per Jacobian solve for it in what may be a production
		// run, which is a choice worth interrupting. Iterative is INFO, because
		// it describes what the *default* does -- it fired on every coupled run
		// including every unconfigured one, and a warning that always fires
		// teaches a reader to skip warnings, which is a real cost given the
		// genuine one this function's caller prints at the end of a run when
		// fallbacks > 0. INFO is compiled out below WARNING (Logging.hpp), so
		// this text is reachable on a VERBOSE or DEBUG build; the permanent
		// homes for it are docs/running.rst and docs/field_coupling.rst, and
		// what a release build reports about the sweep is the "Coupled field
		// sweeps" line, which is measurement rather than description.
		if (fieldSolveMode == FieldSolveMode::Exact)
			logmsg<LOG_LEVEL::WARNING>(
				"FieldSolve = exact forms the Schur complement onto the field block, which "
				"costs one full transport solve per field degree of freedom: {} transport "
				"solves per Jacobian solve where the iterative path costs one. It is a "
				"verification tool and is not intended for production runs.",
				nField + 1);
		else
			logmsg<LOG_LEVEL::INFO>(
				"FieldSolve = iterative: block Gauss-Seidel between the transport and field "
				"blocks with Irons-Tuck acceleration, one transport solve per sweep against "
				"exact's {} per Jacobian solve. Stops once the relative change in psi is below "
				"FieldSolveTolerance = {}, up to FieldSolveMaxSweeps = {} sweeps ({} for the "
				"adjoint); a sweep that reaches its cap falls back to the exact solve, so this "
				"mode costs more than exact in the worst case and never less accuracy. It is "
				"only *cheaper* than exact when the sweep converges in fewer than {} sweeps, "
				"which no test fixture in this tree manages -- it is a bet on a field block "
				"far larger than the transport one, not a free improvement. Watch the "
				"\"Coupled field sweeps\" line at the end of the run.",
				nField + 1, fieldSolveTolerance, fieldSolveMaxSweeps, fieldSolveMaxAdjointSweeps,
				nField + 1);
	}

	// ----------------- Allocate and initialize all other sun-vectors. -------------
	//
	// Note the `throw` on each of these checks. They used to construct a
	// std::runtime_error and discard it -- a no-op -- so every SUNDIALS failure
	// from here to the end of this function was silently ignored and the run
	// carried on with a null vector or an unconfigured IDA.

	res = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)res, "N_VClone", 0))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");
	// sunrealtype tRes;

	// No constraints are imposed as negative coefficients may allow for a better fit across a cell
	constraints = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");

	// Specify only u as differential
	id = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)id, "N_VClone", 0))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");

	// setConstant, not Constant. Eigen's Constant is a *static factory*: calling it
	// through an instance is legal, builds a fresh constant expression and discards
	// it, so the line this replaces did nothing at all and `id` kept the zeros
	// zeroCoeffs() left. N_VL1Norm(id) was 0 here, which told IDA the entire system
	// was algebraic -- so IDASetId, and therefore the IDA_YA_YDP_INIT call below,
	// have been solving a different initialisation problem from the intended one
	// for as long as this code has existed. Nothing warns: Constant is not
	// [[nodiscard]] and the statement declares no unused variable.
	DGSoln isDifferential(nVars, grid, k, nScalars, nAux, nField);
	isDifferential.Map(N_VGetArrayPointer(id));
	isDifferential.zeroCoeffs();
	for (Index v = 0; v < nVars; ++v)
		for (Index i = 0; i < nCells; ++i)
			isDifferential.u(v).getCoeff(i).second.setConstant(1.0);

	for (Index s = 0; s < nScalars; ++s)
	{
		if (problem->isScalarDifferential(s))
		{
			isDifferential.Scalar(s) = 1.0;
		}
	}

	// nField is only ever nonzero with a model attached, but the guard keeps that
	// invariant visible where the pointer is dereferenced rather than three
	// hundred lines away in setFieldModel.
	if (fieldModel)
		for (Index f = 0; f < nField; ++f)
		{
			if (fieldModel->isFieldDOFDifferential(f))
				isDifferential.Field(f) = 1.0;
		}

	retval = IDASetId(IDA_mem, id);
	if (ErrorChecker::check_retval(&retval, "IDASetId", 1))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");

	// Optionally take the algebraic rows out of IDA's local error test. `id`
	// above is what makes this meaningful: u and the differential scalars carry
	// 1.0, everything else 0.0, and IDASetSuppressAlg drops exactly the zeros.
	//
	// It is off by default because it is *not* answer-preserving, however much
	// the direct-run numbers suggest otherwise. On a single run to a steady
	// state it reproduces the same answer to five significant figures for
	// 13-44% fewer calls into the physics, and it dissolves the
	// Absolute_tolerance cliff that otherwise makes atol <= 1e-7 fail on the
	// first step. But sigma, q, lambda and phi are then controlled only by the
	// Newton tolerance, and two things read them: a restart file serialises the
	// whole DOF vector, and phi is a physics quantity in its own right when
	// nAux > 0. Measured, a restart round trip degrades from 1.9e-6 to 8.6e-4,
	// and on AuxVarTest q and sigma -- the fields the flag drops from the error
	// test -- land 1.0e-6 from a converged solution against 4.1e-7 with it off.
	// (This used to cite a 1.0% drift past a 0.84% tolerance on that case. It
	// was measured when the case ran at rtol = atol = 1e-2, where its own answer
	// is 4.1% from converged, so the drift was step-sequence noise rather than
	// the flag.) Turning it on is a trade, not an improvement.
	if (suppressAlgebraicError)
	{
		retval = IDASetSuppressAlg(IDA_mem, SUNTRUE);
		if (ErrorChecker::check_retval(&retval, "IDASetSuppressAlg", 1))
			throw std::runtime_error("Sundials initialization Error, run in debug to find");
	}

	// Initialise IDA
	retval = IDAInit(IDA_mem, static_residual, t0, Y, dYdt);
	if (ErrorChecker::check_retval(&retval, "IDAInit", 1))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");

	// Set tolerances
	absTolVec = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)absTolVec, "N_VClone", 0))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");
	VectorWrapper absTolVals(N_VGetArrayPointer(absTolVec), N_VGetLength(absTolVec));
	absTolVals.setZero();

	DGSoln tolerances(nVars, grid, k, nScalars, nAux, nField);
	tolerances.Map(N_VGetArrayPointer(absTolVec));
	for (Index i = 0; i < nCells; ++i)
	{
		for (Index v = 0; v < nVars; ++v)
		{
			if (atol.size() == 1)
			{
				double absTol = atol[0];
				tolerances.u(v).getCoeff(i).second.setConstant(absTol);
				tolerances.q(v).getCoeff(i).second.setConstant(absTol);
				tolerances.sigma(v).getCoeff(i).second.setConstant(absTol);
				tolerances.lambda(v).setConstant(absTol);
			}
			else if (atol.size() == nVars)
			{
				double absTolU, absTolQ, absTolSigma;
				absTolU = atol[v];
				absTolQ = atol[v];
				absTolSigma = atol[v];
				tolerances.u(v).getCoeff(i).second.setConstant(absTolU);
				tolerances.q(v).getCoeff(i).second.setConstant(absTolQ);
				tolerances.sigma(v).getCoeff(i).second.setConstant(absTolSigma);
				tolerances.lambda(v).setConstant(absTolU);
			}
		}

		for (Index a = 0; a < nAux; ++a)
		{
			tolerances.Aux(a).getCoeff(i).second.setConstant(atol[0]);
		}
	}

	for (Index i = 0; i < nScalars; ++i)
		tolerances.Scalar(i) = atol[0];

	for (Index i = 0; i < nField; ++i)
		tolerances.Field(i) = atol[0];

	retval = IDAWFtolerances(IDA_mem, SystemSolver::getErrorWeights_static);
	if (ErrorChecker::check_retval(&retval, "IDAWFtolerances", 1))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");

	//--------------set up user-built objects------------------

	// Use empty SunMatrix Object
	sunMat = SunMatrixNew(ctx);

	// The only linear solver wrapper ever constructed from this object, so we can
	// give it a pointer to 'this': it does not outlive the SystemSolver.
	LS = SunLinSolWrapper::SunLinSol(this, IDA_mem, ctx);

	if (IDASetLinearSolver(IDA_mem, LS, sunMat) != SUN_SUCCESS)
		throw std::runtime_error("Error in IDASetLinearSolver");

	IDASetJacFn(IDA_mem, JacSetup);

	IDASetMaxNonlinIters(IDA_mem, 10);

	// filename(), not stem(): inputFilePath now holds the configuration's
	// OutputFilename, which is already a base name -- the TOML source seeds it
	// from the config file's stem. Stemming it a second time would turn a
	// config named run.v2.conf into output called run.nc. filename() also keeps
	// the long-standing behaviour that output lands in the current directory
	// rather than beside any path given in OutputFilename.
	std::string baseName = inputFilePath.filename().string();

	// The .dat files are opt-in; netCDF below is what a run produces by
	// default. Nothing is opened unless asked for, so a plain run leaves no
	// text output behind at all.
	if (writeDatFile)
	{
		out0.open(baseName + ".dat");
		std::println(out0, "# Time indexes blocks. ");
		std::println(out0, "# Columns Headings: ");
		std::print(out0, "# x");
		for (Index v = 0; v < nVars; ++v)
			std::print(out0, "\tvar{0} u\tvar{0} q\tvar{0} sigma\tvar{0} u_star\tvar{0} source", v);
		std::println(out0, "");
	}

	// The diagnostic .dat files need both the option and a PHYSICS_DEBUG build:
	// the residual norms and error weights they report are only computed on
	// that path, and `wgt` is only allocated there. One flag for both because
	// they are always written together.
	debugDat = physics_debug && writeDebugDatFiles;

	if (debugDat)
	{
		// "pre-calcIC" only when there is going to be one. A steady solve skips
		// IDACalcIC (see below), so on that path this block and its partner after
		// the solve would bracket nothing, and the labels would name a correction
		// that never happened.
		const char *const icStage =
			solvesForSteadyState() ? "initial guess" : "pre-calcIC";

		wgt = N_VClone(res);
		dydt_out.open(baseName + ".dydt.dat");
		std::println(dydt_out, "# dydt at the {}", icStage);
		printOnNodes(dydt_out, t0, dYdt);
		res_out.open(baseName + ".res.dat");
		residual(t0, Y, dYdt, res);
		getErrorWeights(Y, wgt);
		double residual_val = N_VWrmsNorm(res, wgt);
		std::println(res_out, "# Residual norm at t = {:g} ({}) is {:g}", t0, icStage, residual_val);
		printOnNodes(res_out, t0, res);
		if (writeDatFile)
		{
			std::println(out0, "# t = {:g} ({}) ", t0, icStage);
			print(out0, t0, nOut, true);
		}
	}

	//------------------------------Solve------------------------------
	// Update initial solution to be within tolerance of the residual equation

	// ...but only for a run that will time-march. IDACalcIC exists to hand IDA a
	// state consistent with the algebraic constraints before it takes its first
	// step, and a steady solve never takes one: solveSteadyState drives the *whole*
	// residual to zero from Y with KINSOL, so whatever inconsistency the guess
	// carries is removed by the answer rather than before it, and everything CalcIC
	// computes is overwritten by the first accepted continuation step.
	//
	// Being wasted is the smaller half. IDACalcIC is itself a damped Newton solve,
	// and it fails on states a steady solve handles without difficulty --
	// python-examples/jardin-critical-gradient records a case where starting from
	// the *exact* steady state makes it return IDA_CONV_FAIL (-4), which is the one
	// initial condition a steady solve would have accepted instantly. So requiring
	// it ahead of a solve that does not need it converts runs that would have
	// converged into runs that never start, and reports the failure as if the
	// answer were unreachable.
	//
	// The gate is solvesForSteadyState(), i.e. `TerminateOnSteadyState && steadyMode
	// != TimeMarch`, so a plain transient and an explicit SteadyStateSolver =
	// "TimeMarch" both keep the call unchanged. Both integrate, and IDA's first step
	// is only as good as the state it starts from.
	//
	// One thing reads the initial condition rather than the answer, and on the
	// steady path it now sees the guess setInitialConditions built: the t0
	// timeslice in the netCDF and .dat output. That is the state the run actually
	// started from, so it is if anything the more honest report.
	//
	// ...and not for a restart either, which resumes from a state the previous run
	// had already driven onto the constraint manifold. IDACalcIC cannot find that
	// out cheaply -- its convergence test is on the Newton step rather than on the
	// residual, so it does a Jacobian setup and solve before it can discover it had
	// nothing to do, and repeats the whole thing to refresh the error weights.
	// Measured floor, on a state it had itself just converged to: two residual
	// evaluations, two Jacobian builds and two Jacobian solves, zero Newton
	// iterations. See setForceConsistentIC for the source references and for why
	// this is decided from what the run *is* rather than from a residual threshold.
	//
	// A cold time-marching run is the case that is left, and it always runs
	// IDACalcIC. There is deliberately no way to turn that off: its guess is not a
	// consistent state, IDA_ERR_FAIL on the first step is what starting from one
	// looks like, and a caller who does not care about the transient wants
	// SteadyStateSolver = PseudoTransient or Newton rather than an uncorrected
	// time march. ForceConsistentIC only ever adds the call back.
	//
	// The norm is still *reported* on every time-marching run, armed or not,
	// because it is how a caller finds out whether the restart they resumed from
	// was as consistent as a restart is supposed to be.
	initial_residual_norm = std::numeric_limits<double>::quiet_NaN();
	calcICRan = false;

	// A restart skips only on the *copy* path. The claim behind the default is
	// that a restart resumes from a state the previous run had already driven onto
	// the constraint manifold, and that is true only when the discretisation
	// matches: a restart at a different degree is projected, which transfers u, q,
	// aux and the scalars and then rebuilds sigma and the trace, so what it hands
	// IDA is a guess like any other. Measured on AuxVarTest resuming at a lower
	// degree, skipping there fails with IDA_ERR_FAIL where running IDACalcIC
	// completes the run -- so the carve-out is not tidiness.
	const bool restarting = problem && problem->isRestarting();
	const bool restartIsConsistent = restarting && !restartWasProjected;
	bool wouldSkip = solvesForSteadyState() || restartIsConsistent;

	// One-directional: it adds IDACalcIC where the run would have skipped, and
	// cannot remove it from the cold time-marching run that needs it.
	if (forceConsistentIC)
		wouldSkip = false;

	if (!solvesForSteadyState())
	{
		initial_residual_norm = weightedResidualNorm(t0, Y, dYdt);

		logmsg<LOG_LEVEL::INFO>(
			"Initial state has a weighted residual of {:g}, so IDACalcIC {}.",
			initial_residual_norm,
			wouldSkip	 ? "is skipped (this run resumes from a restart file)"
			: !restarting	 ? "will run"
			: forceConsistentIC ? "will run (ForceConsistentIC)"
							 : "will run (this restart was projected onto a different "
							   "discretisation, so it is not a consistent state)");
	}

	calcICRan = !wouldSkip;

	if (calcICRan)
	{
		// The `retval = 0` that used to sit between these two lines made the check
		// below unreachable: IDACalcIC's status was overwritten before it was read, so
		// a failed initial-condition calculation carried on silently into the time
		// loop with whatever partial state IDA had reached. The name passed to
		// check_retval said "IDASolve" too, so even the message would have pointed at
		// the wrong call.
		//
		// tout1 is an absolute *time* -- "the first value of t at which a solution will
		// be requested" -- and is what IDA takes the direction and rough scale of the
		// independent variable from. This used to pass the *interval*,
		// `dt0 > 0 ? dt0 : dt`, which is the same number only when t0 is zero.
		//
		// Every fixture in the tree starts at zero, so nothing noticed. Set t_initial
		// equal to delta_t and the run dies before evaluating a single residual:
		//
		//     t_initial = 0.1, delta_t = 0.1   ->  tout1 == t0, IDA_ILL_INPUT (-22),
		//                                          "tout1 too close to t0", and the
		//                                          throw below kills the run
		//
		// which is a plain configuration, not a corner. A restart is the common way to
		// reach it, since it resumes at the time the file was written. Other values of
		// t_initial are wrong in a quieter way -- t_initial > delta_t hands IDA a tout1
		// *behind* t0, i.e. the wrong direction of integration, which IDA does not
		// reject.
		//
		// The first time integrate() actually asks for is t0 + dt: it sets `tout = t0`
		// and the loop does `tout += dt` before the first IDASolve.
		retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, t0 + (dt0 > 0.0 ? dt0 : dt));
		if (ErrorChecker::check_retval(&retval, "IDACalcIC", 1))
		{
			throw std::runtime_error("IDACalcIC could not complete");
		}
	}

	if (calcICRan)
	{

		long int nresevals = 0;
		IDAGetNumResEvals(IDA_mem, &nresevals);
		logmsg<LOG_LEVEL::INFO>("Number of Residual Evaluations due to IDACalcIC: {}", nresevals);

		if (nresevals > 10)
			logmsg<LOG_LEVEL::WARNING>("IDACalcIC required {} residual evaluations. Check settings in {}", nresevals, std::string(inputFilePath));

		// Take IDACalcIC's result. It keeps the corrected initial condition inside
		// IDA and hands it over only on request, so without this Y and dYdt still
		// hold the state that was *fed* to CalcIC rather than the one it computed.
		//
		// Here, once, before anything reads Y or writes output. Everything
		// downstream then means the same thing by "the initial condition": the t0
		// timeslice initialiseNetCDF writes below, the t0 block of the .dat file,
		// and the residual the debug path evaluates. It used to be fetched inside
		// the debugDat branch instead, which meant the t0 output reported the
		// pre-CalcIC state on an ordinary run and the corrected one under
		// WriteDebugDatFiles, so a discrepancy could reproduce only with debug
		// output switched on.
		//
		// Note what this does *not* fix. dYdt's algebraic blocks stay zero: q, sigma
		// and phi are algebraic, and IDA_YA_YDP_INIT computes algebraic values and
		// differential derivatives, not the other way round. That is structural, not
		// the old wrong id vector -- see at_t0_only_the_differential_part_of_dydt_exists
		// -- and it stays that way, because dYdt is the state IDA takes its first
		// step from.
		//
		// Checked, unlike every other use of it in this file's history: it fails
		// with IDA_ILL_INPUT if IDA has already taken a step, and on failure it
		// leaves Y and dYdt holding their *pre*-CalcIC values rather than reporting
		// anything. That is the same silent-failure shape as the `retval = 0`
		// described above, and it would show up as a run whose initial condition is
		// quietly the uncorrected one.
		retval = IDAGetConsistentIC(IDA_mem, Y, dYdt);
		if (ErrorChecker::check_retval(&retval, "IDAGetConsistentIC", 1))
			throw std::runtime_error("Could not retrieve the corrected initial condition");
	}

	if (writeDatFile)
		print(out0, t0, nOut, true);
	if (debugDat)
	{
		residual(t0, Y, dYdt, res);
		std::println(dydt_out, "# dydt at the initial condition the run starts from");
		printOnNodes(dydt_out, t0, dYdt);

		IDAEwtSet(Y, wgt, IDA_mem);

		std::println(res_out, "# Residual norm at t = {:g} (initial condition) is {:g}", t0,
					 N_VWrmsNorm(res, wgt));
		printOnNodes(res_out, t0, res);
	}

	// This also writes the t0 timeslice -- the corrected one, per the fetch above
	//
	// writeOutput gates every netCDF and restart write in this file. nc_output is
	// never opened when it is false, and NcFile::close() on an unopened file is a
	// no-op -- the destructor's `filename != ""` guard relies on the same thing --
	// but the Close() calls are guarded too, so the reason is visible where it
	// applies rather than resting on netCDF's behaviour. The .dat flags are
	// separate and stay separate: they are opt-in already.
	if (writeOutput)
		initialiseNetCDF(baseName + ".nc", nOut);

	IDASetMaxNumSteps(IDA_mem, 50000);

	IDASetMinStep(IDA_mem, min_step_size);

	t = t0;
	tout = t0;
	tret = t0;

	if (problem->isRestarting()) // If restarting, try to continue at same delta t
	{
		IDASetInitStep(IDA_mem, dt);
	}
	if (dt0 > 0.0)
		IDASetInitStep(IDA_mem, dt0);

	// Let IDA grow the step faster than its default factor of 2. See
	// setAggressiveTimesteps.
	if (aggressiveTimesteps)
		IDASetEtaMax(IDA_mem, 10.0);
}

// ||F(t, Y, dYdt)|| in the WRMS norm with this solver's own error weights.
//
// The same measure the WriteDebugDatFiles blocks print, deliberately: the number
// the initial-condition skip is decided on is then a number a user can already
// see in the .res.dat rather than one only this function knows.
//
// The weights are the state's, not the residual's, which is the scaling IDA
// itself uses for the quantity this stands in for -- IDACalcIC tests
// ||J^-1 F||_WRMS, a correction to y, against epsNewt. So the threshold means
// the same thing at every tolerance setting: tightening Absolute_tolerance
// tightens what counts as a consistent initial state, exactly as it tightens
// everything else.
//
// One residual evaluation and one error-weight fill; no Jacobian work at all,
// which is the entire point.
double SystemSolver::weightedResidualNorm(double t, N_Vector Y_in, N_Vector dYdt_in)
{
	if (res == nullptr)
		throw std::logic_error("weightedResidualNorm needs the SUNDIALS vectors initialize() allocates");

	N_Vector wgtLocal = N_VClone(Y_in);
	if (ErrorChecker::check_retval((void *)wgtLocal, "N_VClone", 0))
		throw std::runtime_error("Sundials initialization Error, run in debug to find");

	residual(t, Y_in, dYdt_in, res);
	getErrorWeights(Y_in, wgtLocal);
	const double norm = N_VWrmsNorm(res, wgtLocal);

	N_VDestroy(wgtLocal);
	return norm;
}

void SystemSolver::integrate(double tFinal)
{
	int retval;

	if (IDA_mem == nullptr)
		throw std::logic_error("integrate() called before initialize()");

	// Steady-state stopping conditions
	const sunrealtype dydt_rel_tol = steady_state_tol;
	const sunrealtype dydt_abs_tol = 1e-3;

	if (t0 > tFinal)
	{
    logmsg<LOG_LEVEL::ERROR>("Initial time t = {} is after the end of the simulation at t = {}", t0, tFinal);
		throw std::runtime_error("Simulation ends before it begins.");
	}

	// Two routes to a final state. TimeMarch integrates to it and is the only one
	// that selects a branch by physics rather than by wherever Newton lands; the
	// other two drive the residual to zero directly. Both leave the answer in Y
	// and yJac, so everything below this block -- the adjoint solve, the output,
	// the restart file -- is common.
	//
	// Gated on TerminateOnSteadyState, not on steadyMode alone. SteadyStateSolver
	// defaults to PseudoTransient, and without this a plain run(tFinal) -- a
	// transient, where the whole point is the path -- would jump to the end state
	// and report it as the answer at tFinal.
	if (solvesForSteadyState())
	{
		try
		{
			solveSteadyState();
		}
		catch (...)
		{
			// A failed steady solve is exactly when the state is worth looking
			// at, and until now it was the one case that produced nothing: the
			// throw propagates to runSolver, which frees everything. The time
			// loop already does this for a failed IDASolve, and this is the same
			// bargain -- write what there is, close the files, then rethrow.
			//
			// Y holds the last iterate. On the "ran out of continuation steps"
			// path that is the last *accepted* one, since a rejected step
			// restores uPrev before damping; on a hard KINSol failure it is
			// whatever KINSOL left, which is the honest thing to show. dYdt is
			// still the t0 derivative either way -- solveSteadyState only zeroes
			// it on success -- so a diagnostic hook differentiating it here is
			// reading the initial condition's rate of change, not this state's.
			//
			// Stamped STEADY_STATE_TIME like a converged one. Nothing in the
			// file distinguishes the two; the exception and the exit status do.
			logmsg<LOG_LEVEL::ERROR>("Steady solve failed; writing the last state reached to the output.");
			if (writeDatFile)
				print(out0, STEADY_STATE_TIME, nOut, Y, true);
			if (writeOutput)
				WriteTimeslice(STEADY_STATE_TIME);
			closeOutputFiles();
			throw;
		}

		writeSteadyState();
	}
	else
	{
	// Solving Loop
	while (tFinal - tret > min_step_size || TerminateOnSteadyState)
	{
		tout += dt;
		if (tout > tFinal && !TerminateOnSteadyState)
			tout = tFinal; // Never ask for results beyond tFinal
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
		{
			// try to emit final data
			if (writeDatFile)
				print(out0, tret, nOut, true);
			if (debugDat)
      {
	      residual(tret, Y, dYdt, res);
        IDAEwtSet(Y, wgt, IDA_mem);
        std::println(res_out, "# Residual norm at t = {:g} is {:g}", tret,
                     N_VWrmsNorm(res, wgt));
        printOnNodes(res_out, tret, res);
        printOnNodes(dydt_out, tret, dYdt);
      }
			if (writeOutput)
				WriteTimeslice(tret);
			if (writeDatFile)
				out0.close();
			if (writeOutput)
				nc_output.Close();

			throw std::runtime_error("IDASolve could not complete");
		}

		long int nstep_tmp;
		IDAGetNumSteps(IDA_mem, &nstep_tmp);
		std::println("Writing output at {:g} ( {} timesteps )", tret, nstep_tmp);
		if (writeDatFile)
			print(out0, tret, nOut, Y, true);
		if (debugDat)
		{
			printOnNodes(dydt_out, tret, dYdt);
			residual(tret, Y, dYdt, res);
			IDAEwtSet(Y, wgt, IDA_mem);
			std::println(res_out, "# Residual norm at t = {:g} is {:g}", tret,
						 N_VWrmsNorm(res, wgt));
			printOnNodes(res_out, tret, res);
		}
		if (writeOutput)
			WriteTimeslice(tret);

		// Check if steady-state is achieved (test the lambda points)
		if (TerminateOnSteadyState)
		{
			sunrealtype dydt_norm = 0.0;
			for (Index i = 0; i < nCells; i++)
				for (Index v = 0; v < nVars; v++)
				{
					sunrealtype xi = dydt.lambda(v)[i] * dt;
					sunrealtype wi = 1.0 / (y.lambda(v)[i] * dydt_rel_tol + dydt_abs_tol);
					dydt_norm += xi * xi * wi * wi;
				}
			dydt_norm = sqrt(dydt_norm);
			if (physics_debug)
				std::println(" dy/dt norm inferred from lambdas is {:g}", dydt_norm);
			if (dydt_norm < 1.0)
			{
				std::println("Steady State achieved at time t = {:g}", tret);
				break;
			}
		}

		// Diagnostics go here
	}

	long int nsteps, nresevals, njacevals;
	IDAGetNumSteps(IDA_mem, &nsteps);
	IDAGetNumResEvals(IDA_mem, &nresevals);
	IDAGetNumLinSolvSetups(IDA_mem, &njacevals);

	std::println("Total Number of Timesteps             :{}", nsteps);
	std::println("Total Number of Residual Evaluations  :{}", nresevals);
	std::println("Total Number of Jacobian Computations :{}", njacevals);

	if (nField > 0 && getFieldSolveMode() == SystemSolver::FieldSolveMode::Iterative)
	{
		auto fs = getFieldSweepStats();
		std::println("Coupled field sweeps                  :{} over {} solves ({} exact fallbacks)",
					 fs.iterations, fs.solves, fs.fallbacks);
		// This is the only signal a user has that the coupling is not converging.
		// The run is still correct -- the fallback is the exact solve -- so this is
		// a cost report, not an error, and it says what to do about it.
		if (fs.fallbacks > 0)
			logmsg<LOG_LEVEL::WARNING>(
				"{} of {} coupled Jacobian solves exhausted FieldSolveMaxSweeps = {} and fell "
				"back to the exact Schur solve, at {} transport solves each. The answers are "
				"correct; the run is paying for both. Raise FieldSolveMaxSweeps, or set "
				"FieldSolve = exact and skip the sweeps.",
				fs.fallbacks, fs.solves, fieldSolveMaxSweeps, nField + 1);
	}
	}

	finishRun();
}

// The converged state of a steady solve, written at STEADY_STATE_TIME.
//
// A physics case's writeDiagnostics is called from WriteTimeslice and nowhere
// else, so this is also what gives a steady run its diagnostics rather than
// leaving initialiseDiagnostics and finaliseDiagnostics to bracket nothing.
//
// STEADY_STATE_TIME, not tret: see its definition. Deliberately no
// IDAGetNumSteps report -- IDA never ran.
void SystemSolver::writeSteadyState()
{
	if (writeDatFile)
		print(out0, STEADY_STATE_TIME, nOut, Y, true);
	if (debugDat)
	{
		printOnNodes(dydt_out, STEADY_STATE_TIME, dYdt);
		residual(t0, Y, dYdt, res);
		IDAEwtSet(Y, wgt, IDA_mem);
		std::println(res_out, "# Residual norm at steady state is {:g}",
					 N_VWrmsNorm(res, wgt));
		printOnNodes(res_out, STEADY_STATE_TIME, res);
	}
	if (writeOutput)
		WriteTimeslice(STEADY_STATE_TIME);
}

// Everything a run does once its final state is in Y, whichever route reached
// it: the adjoint solve, the output files, the restart file, and the copy into
// yJac. Shared by integrate() and by a sliced steady solve, so the two cannot
// drift apart on the sequencing -- which matters most for the last of them.
void SystemSolver::finishRun()
{
	// filename(), not stem(): inputFilePath holds the configuration's
	// OutputFilename, which is already a base name -- the TOML source seeds it
	// from the config file's stem. Stemming it a second time would turn a config
	// named run.v2.conf into output called run.nc. filename() also keeps the
	// long-standing behaviour that output lands in the current directory rather
	// than beside any path given in OutputFilename.
	const std::string baseName = inputFilePath.filename().string();

	// The adjoint solve is allowed to fail, and its failure must not take the
	// forward run's output with it.
	//
	// The coupled adjoint sweep no longer throws on non-convergence -- it
	// escalates to the exact transposed Schur solve, which is strictly stronger
	// than refusing, since the caller gets a correct gradient rather than an
	// exception. What reaches this catch is therefore a throw out of the *field
	// model* itself: from solveCoupledAdjointExact, or from the sweep's own
	// solveBTranspose before it. But this call sits *before* finaliseDiagnostics,
	// closeOutputFiles() and WriteRestartFile, and runSolver's catch(...)
	// rethrows, so an unguarded throw here destroyed the netCDF and the restart
	// file of a run that had integrated perfectly. The gradient is the optional
	// half of the run; the solution is not, and losing hours of transport solve
	// because a Schur sweep would not converge is a worse failure than the one
	// being reported.
	//
	// Held as an exception_ptr rather than by moving the call after the output
	// block, because captureState() below must still run after the adjoint solve:
	// the gradients are defined at the state the adjoint matrices were built
	// from. See its own comment.
	std::exception_ptr adjointFailure;
	if (solveAdjoint)
	{
		try
		{
			runAdjointSolve();
			// WriteAdjoints();
		}
		catch (std::exception const &e)
		{
			adjointFailure = std::current_exception();
			logmsg<LOG_LEVEL::ERROR>(
				"The adjoint solve failed; the gradients are unavailable. The forward "
				"solution and its output files are unaffected and are being written now.\n  {}",
				e.what());
		}
	}

	if (writeOutput)
		problem->finaliseDiagnostics(nc_output);
	closeOutputFiles();

	if (writeOutput)
		WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut);

	// Leave yJac holding the *final* solution. It is the only copy that outlives
	// destroySundials() -- `y` is a non-owning view over Y -- and it is what
	// PyRunner::getSolution, getAdjointGradients and G read.
	//
	// Deliberately after runAdjointSolve(): the adjoint solve above is defined
	// at the state its matrices were built from, so moving this earlier would
	// change the gradients.
	captureState();

	if (writeOutput)
		nc_output.Close();

	// Now that everything is on disk. G_p holds whatever the failed solve left
	// there, which is why this rethrows rather than returning quietly: a caller
	// that went on to read getAdjointGradients() would get a plausible matrix
	// that is not the gradient of anything.
	if (adjointFailure)
		std::rethrow_exception(adjointFailure);
}

void SystemSolver::closeOutputFiles()
{
	if (writeDatFile)
		out0.close();
	if (debugDat)
	{
		dydt_out.close();
		res_out.close();
	}
	if (writeOutput)
		nc_output.Close();
}

void SystemSolver::captureState()
{
	setJacEvalY(Y, dYdt);
}

void SystemSolver::destroySundials()
{
	// Everything here is nulled after being freed, so a second call is a no-op
	// and so is a call with no preceding initialize(). runSolver() relies on
	// that: it calls this from both the normal and the exceptional path.

	// No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
	if (LS)
	{
		SUNLinSolFree(LS);
		LS = nullptr;
	}

	// Before sunMat, which KINSOL shares rather than owns.
	if (kinLS)
	{
		SUNLinSolFree(kinLS);
		kinLS = nullptr;
	}

	if (kin_mem)
	{
		KINFree(&kin_mem); // as IDAFree, nulls its argument
		kin_mem = nullptr; // ... belt and braces, since we test the pointer
	}

	if (sunMat)
	{
		MatDestroy(sunMat);
		sunMat = nullptr;
	}

	if (IDA_mem)
		IDAFree(&IDA_mem); // IDAFree nulls its argument itself

	// Free the raw data buffers allocated by SUNDIALS

	// Guarded on the pointer, not on debugDat: a second run with the option
	// turned off would otherwise leak the previous run's vector.
	if (wgt)
	{
		N_VDestroy(wgt);
		wgt = nullptr;
	}

	for (N_Vector *vec : {&Y, &dYdt, &constraints, &id, &res, &absTolVec,
	                      &uPrev, &ptcDYdt, &kinScale})
	{
		if (*vec)
		{
			N_VDestroy(*vec);
			*vec = nullptr;
		}
	}

	// The output streams too: initialize() opens them, so this is where they are
	// closed. std::ofstream::open on an already-open stream sets failbit rather
	// than reopening, so leaving them open here would silently discard the .dat
	// output of every run after the first.
	for (std::ofstream *stream : {&out0, &dydt_out, &res_out})
		if (stream->is_open())
			stream->close();

	// And the netCDF file, for the same reason and with a sharper symptom.
	// initialize() opens it via initialiseNetCDF; only integrate() used to close
	// it, so any path that allocates and then does not complete a time loop left
	// it open -- integrate() throwing and runSolver catching, for instance. The next run in the same process then fails inside
	// netCDF trying to create a file this process still holds, and reports
	// "Permission denied", which reads like a filesystem problem rather than a
	// handle we never released. Close() just clears the name and closes the file,
	// so calling it here as well as at the end of integrate() is harmless.
	if (writeOutput)
		nc_output.Close();

	// `ctx` is deliberately NOT freed here. It belongs to the SystemSolver: it is
	// created in the constructor (SystemSolver.cpp:18) and freed in the
	// destructor (:65). Freeing it per-run left the member dangling, so a
	// *second* run on the same object failed at IDACreate with "Sundials
	// Initialization Error" -- which is why PyRunner::run used to work only once
	// per configure(), even though it goes to the trouble of clearing
	// TerminateOnSteadyState for a repeat call. The standalone binary never
	// noticed because it runs once and exits.
}

void SystemSolver::runAdjointSolve()
{
	// This used to refuse a field model outright, because the adjoint matrices
	// carried neither geometry nor a transpose of the coupling and so would have
	// returned a silently wrong gradient beside a perfectly good G. Both are now
	// here: initializeMatricesForAdjointSolve fills geometry before it evaluates
	// anything and stores A1^T and A2^T beside M^T, and solveAdjointState
	// eliminates them exactly (FieldSolve = exact) or sweeps to a checked
	// backward error that *throws* if it is not reached.
	//
	// Two limits survive the lifting and are deliberate rather than overlooked.
	// An objective whose integrand reads State::geom directly loses its dG/dpsi
	// term, because AdjointProblem reports four state derivatives and geometry
	// is not among them; and a FieldModel cannot depend on an adjoint parameter
	// at all, so dR/dp is zero by construction rather than by assumption. Both
	// are recorded in SystemSolver.hpp beside G_field and in TODO.

	if (solveAdjoint)
	{
    logmsg<LOG_LEVEL::INFO>("Computing adjoints");
		// computeAdjointGradients assembles and solves the adjoint itself, once per
		// objective. Both halves belong to *one* objective -- G_y is dG/dy for it and
		// is the solve's right-hand side -- so doing either here would give every
		// objective the first one's gradient.
		computeAdjointGradients();
	}
	else
	{
    logmsg<LOG_LEVEL::ERROR>("Error: runAdjointSolve called but \"solveAdjoint\" was set to false");
	}
}
/*
 * SUNDIALS Calls this function to recompute the local Jacobian
 * This is the function that should set the point at which the sub-matrices for the Jacobian solve are evaluated
 */
int JacSetup(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	// Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian.
	// We use this function to capture t and cj for the solve.
	auto System = reinterpret_cast<SystemSolver *>(user_data);
	System->setJacTime(tt);
	System->setAlpha(cj);
	System->setJacEvalY(yy, yp);
	System->updateBoundaryConditions(tt);
	System->updateMatricesForJacSolve();
	return 0;
}
