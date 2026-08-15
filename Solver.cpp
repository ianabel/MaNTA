#include <ida/ida.h>				  /* prototypes for IDA fcts., consts.    */
#include <kinsol/kinsol.h>			  /* KINFree, for the steady-state path   */
#include <nvector/nvector_serial.h>	  /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <toml.hpp>
#include <fstream>
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

	// The dG/dt gate, which is why the three phases are separate: the objective's
	// time derivative is a property of the initial condition, so it can be asked
	// after initialize() and answered before paying for integrate(). Disarmed
	// unless setObjectiveDecreaseTolerance has been called, in which case this is
	// false and the run proceeds exactly as before.
	if (objectiveIsDecreasing())
	{
		destroySundials();
		return;
	}

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

// Does the objective get worse from here?
//
// Every objective must clear the bar: one that is falling faster than the
// tolerance rejects the step even if the others improve, which is the
// all-must-improve rule origin/optimize-mode used and worth keeping -- a sweep
// that accepts a step because two of three objectives improved is not maximising
// anything in particular.
//
// The sign convention is that optimisation *maximises* G, so a decrease is the
// bad direction. The tolerance is one-sided slack on that: dG/dt may dip by up to
// objective_decrease_tol before the step is called bad, which leaves room for a
// transient that recovers, and keeps quadrature noise about zero from rejecting a
// run that is really flat.
bool SystemSolver::objectiveIsDecreasing()
{
	if (!CheckObjectiveDecrease)
		return false;

	if (!adjointProblem)
		throw std::logic_error("ObjectiveDecreaseTolerance is set but no AdjointProblem is; there is no objective to test. Set solveAdjoint, or drop the tolerance.");

	const Index ng = adjointProblem->getNg();
	last_dGdt.resize(ng);

	bool decreasing = false;
	for (Index gIndex = 0; gIndex < ng; ++gIndex)
	{
		// dydtComplete, not IDA's dydt. At t0 the latter's q, sigma and phi
		// blocks are identically zero, so three of dGdt's four terms would
		// multiply by nothing and the objective would be judged on its u
		// dependence alone -- an objective depending only on q would score
		// exactly zero. computeAlgebraicTimeDerivatives() fills them in.
		last_dGdt(gIndex) = dGdt(gIndex, y, dydtComplete);
		if (last_dGdt(gIndex) < -objective_decrease_tol)
			decreasing = true;
	}

	if (decreasing)
		logmsg<LOG_LEVEL::WARNING>("Objective is decreasing at t = {}: dG/dt = {}, tolerance {}. Abandoning this run without integrating.",
								   t0, last_dGdt(0), objective_decrease_tol);
	else
		logmsg<LOG_LEVEL::INFO>("dG/dt gate passed at t = {}: dG/dt = {}.", t0, last_dGdt(0));

	objective_rejected = decreasing;
	return decreasing;
}

void SystemSolver::initialize()
{
	int retval;

	// A fresh run: whatever the gate concluded about the last one does not apply.
	objective_rejected = false;

	// ...and whatever the field model cached about it does not either. This has
	// to be here rather than in initialiseMatrices(), which is skipped entirely
	// when `initialised` is already set: that is the RF_cellwise trap, where the
	// second run on a reused solver solved its initial dydt out of the previous
	// run's final-time boundary data.
	if (fieldModel)
		fieldModel->resetForRun();

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
		if (fieldSolveMode == FieldSolveMode::Exact)
			logmsg<LOG_LEVEL::WARNING>(
				"FieldSolve = exact forms the Schur complement onto the field block, which "
				"costs one full transport solve per field degree of freedom: {} transport "
				"solves per Jacobian solve where the iterative path costs one. It is a "
				"verification tool and is not intended for production runs.",
				nField + 1);
		else
			logmsg<LOG_LEVEL::WARNING>(
				"FieldSolve = iterative is not implemented yet; falling back to the exact "
				"Schur complement, which costs {} transport solves per Jacobian solve.",
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
	// nAux > 0. Measured, a restart round trip degrades from 1.9e-6 to 8.6e-4
	// and the AuxVarTest regression case drifts 1.0% against a 0.84% tolerance.
	// Turning it on is a trade, not an improvement.
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
		wgt = N_VClone(res);
		dydt_out.open(baseName + ".dydt.dat");
		std::println(dydt_out, "# dydt before CalcIC");
		printOnNodes(dydt_out, t0, dYdt);
		res_out.open(baseName + ".res.dat");
		residual(t0, Y, dYdt, res);
		getErrorWeights(Y, wgt);
		double residual_val = N_VWrmsNorm(res, wgt);
		std::println(res_out, "# Residual norm at t = {:g} (pre-calcIC) is {:g}", t0, residual_val);
		printOnNodes(res_out, t0, res);
		if (writeDatFile)
		{
			std::println(out0, "# t = {:g} (pre-calcIC) ", t0);
			print(out0, t0, nOut, true);
		}
	}

	//------------------------------Solve------------------------------
	// Update initial solution to be within tolerance of the residual equation

	// The `retval = 0` that used to sit between these two lines made the check
	// below unreachable: IDACalcIC's status was overwritten before it was read, so
	// a failed initial-condition calculation carried on silently into the time
	// loop with whatever partial state IDA had reached. The name passed to
	// check_retval said "IDASolve" too, so even the message would have pointed at
	// the wrong call.
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, dt0 > 0.0 ? dt0 : dt);
	if (ErrorChecker::check_retval(&retval, "IDACalcIC", 1))
	{
		throw std::runtime_error("IDACalcIC could not complete");
	}

	long int nresevals = 0;
	IDAGetNumResEvals(IDA_mem, &nresevals);
  logmsg<LOG_LEVEL::INFO>("Number of Residual Evaluations due to IDACalcIC: {}", nresevals);

	if (nresevals > 10)
    logmsg<LOG_LEVEL::WARNING>("IDACalcIC required {} residual evaluations. Check settings in {}", nresevals, std::string(inputFilePath));

	// Take IDACalcIC's result. It keeps the corrected initial condition inside IDA
	// and hands it over only on request, so without this Y and dYdt still hold the
	// state that was *fed* to CalcIC rather than the one it computed.
	//
	// Here, once, before anything reads Y or writes output. Everything downstream
	// then means the same thing by "the initial condition": the t0 timeslice
	// initialiseNetCDF writes below, the t0 block of the .dat file, the residual the
	// debug path evaluates, and the state the dG/dt gate differentiates. It used to
	// be fetched in two places for two reasons -- inside the debugDat branch, and
	// again for the gate when armed -- which meant the t0 output reported the
	// pre-CalcIC state on an ordinary run and the corrected one under
	// WriteDebugDatFiles, so a discrepancy could reproduce only with debug output
	// switched on.
	//
	// Note what this does *not* fix. dYdt's algebraic blocks stay zero: q, sigma and
	// phi are algebraic, and IDA_YA_YDP_INIT computes algebraic values and
	// differential derivatives, not the other way round. That is structural, not the
	// old wrong id vector -- see at_t0_only_the_differential_part_of_dydt_exists --
	// and it stays that way, because dYdt is the state IDA takes its first step
	// from. The gate reads dydtComplete instead, which the two blocks below seed
	// from this and then fill in.
	//
	// Checked, unlike every other use of it in this file's history: it fails with
	// IDA_ILL_INPUT if IDA has already taken a step, and on failure it leaves Y and
	// dYdt holding their *pre*-CalcIC values rather than reporting anything. That is
	// the same silent-failure shape as the `retval = 0` described above, and it
	// would show up as a run whose initial condition is quietly the uncorrected one.
	retval = IDAGetConsistentIC(IDA_mem, Y, dYdt);
	if (ErrorChecker::check_retval(&retval, "IDAGetConsistentIC", 1))
		throw std::runtime_error("Could not retrieve the corrected initial condition");

	// Seed the complete derivative from IDA's. Its algebraic blocks are zero at
	// this point; computeAlgebraicTimeDerivatives() fills them when the gate is
	// armed, and nothing else reads them.
	//
	// Here rather than beside setJacEvalY above, because until the fetch on the
	// line before this dYdt still holds the *guess* setInitialConditions built
	// rather than the derivative IDACalcIC corrected it to. Seeding from the guess
	// left dydtComplete's u block disagreeing with the state it is meant to
	// describe by a fraction of a percent -- small enough to look like round-off
	// and quite large enough to matter to anything differentiating the solution.
	{
		DGSoln idaDerivative(nVars, grid, k, nScalars, nAux, nField);
		idaDerivative.Map(N_VGetArrayPointer(dYdt));
		dydtComplete.copy(idaDerivative);
	}

	// Only when the gate is armed: this is a dense assembly and factorisation of
	// the whole system, and nothing but the gate reads the algebraic blocks. A
	// run with the gate disarmed pays nothing and is unchanged.
	if (CheckObjectiveDecrease)
		computeAlgebraicTimeDerivatives();

	if (writeDatFile)
		print(out0, t0, nOut, true);
	if (debugDat)
	{
		residual(t0, Y, dYdt, res);
		std::println(dydt_out, "# After CalcIC ");
		printOnNodes(dydt_out, t0, dYdt);

	

		IDAEwtSet(Y, wgt, IDA_mem);



		std::println(res_out, "# Residual norm at t = {:g} (post-CalcIC) is {:g}", t0,
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

void SystemSolver::integrate(double tFinal)
{
	int retval;
	// filename(), not stem(): inputFilePath now holds the configuration's
	// OutputFilename, which is already a base name -- the TOML source seeds it
	// from the config file's stem. Stemming it a second time would turn a
	// config named run.v2.conf into output called run.nc. filename() also keeps
	// the long-standing behaviour that output lands in the current directory
	// rather than beside any path given in OutputFilename.
	std::string baseName = inputFilePath.filename().string();

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
	if (TerminateOnSteadyState && steadyMode != SteadyMode::TimeMarch)
	{
		solveSteadyState();
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
	}

	if (solveAdjoint)
	{
		runAdjointSolve();
		// WriteAdjoints();
	}

	if (writeOutput)
		problem->finaliseDiagnostics(nc_output);
	if (writeDatFile)
		out0.close();
	if (debugDat)
	{
		dydt_out.close();
		res_out.close();
	}
	if (writeOutput)
		nc_output.Close();

	if (writeOutput)
		WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut);

	// Leave yJac holding the *final* solution. It is the only copy that outlives
	// destroySundials() -- `y` is a non-owning view over Y -- and it is what
	// PyRunner::getSolution, getAdjointGradients and G read. Until now it held
	// whatever state IDA last evaluated a Jacobian at, which can be several steps
	// stale, so a caller asking for "the solution" got a slightly earlier one.
	//
	// Deliberately after runAdjointSolve(): the adjoint solve above is defined
	// at the state its matrices were built from, so moving this earlier would
	// change the gradients.
	setJacEvalY(Y, dYdt);

	if (writeOutput)
		nc_output.Close();
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
	// it open -- the dG/dt gate rejecting a run, or integrate() throwing and
	// runSolver catching. The next run in the same process then fails inside
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
	// The third of the three "a later task owns this" refusals, beside the ones
	// in setInitialConditions (restart) and computeAlgebraicTimeDerivatives (the
	// dG/dt gate). Uniformity is the point: an unguarded hole beside two guarded
	// ones is worse than either.
	//
	// initializeMatricesForAdjointSolve and computeAdjointGradients both build
	// their GlobalState from y.evalOnNodes()/evalOnStarNodes and hand it to
	// ComputePhysicsDerivatives without evaluateGeometry, so the *case's* own
	// dSigmaFn_dq -- not merely a dg hook -- would read index 0 of a zero-length
	// geometry vector. Filling it would not be enough on its own: the adjoint
	// operator is the transpose of the forward one block for block, and the
	// dGeometry/dpsi coupling is not in either yet, so a filled-but-untransposed
	// adjoint would return a silently wrong gradient with a perfectly good G.
	// That is the failure mode CLAUDE.md records for this file, and refusing is
	// the only honest answer until the transposes land.
	if (fieldModel)
		throw std::logic_error(
			"The adjoint solve is not supported with a field model attached: the adjoint "
			"matrices carry no geometry and no transpose of the field coupling, so the "
			"gradient would be wrong rather than absent.");

	if (solveAdjoint)
	{
    logmsg<LOG_LEVEL::INFO>("Computing adjoints");
		initializeMatricesForAdjointSolve();
		solveAdjointState(0);
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
