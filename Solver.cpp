#include <ida/ida.h>				  /* prototypes for IDA fcts., consts.    */
#include <kinsol/kinsol.h>			  /* KINFree, for the steady-state path   */
#include <nvector/nvector_serial.h>	  /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <toml.hpp>
#include <algorithm>
#include <exception>
#include <format>
#include <fstream>
#include <limits>
#include <print>
#include <memory>

#include "Types.hpp"
#include "FieldModel.hpp"
#include "EquilibriumModel.hpp"
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

	// The same for a tier-0 equilibrium, and for the solver's own cache of it,
	// which is also the next Solve's warm start: a reused solver starts from the
	// cold equilibrium a fresh one would.
	equilibriumCache = {};
	equilibriumSolves = 0;
	// Tier 1's blocks hold an equilibrium of the last run's; nothing may read
	// them before this run's first Jacobian build.
	reducedBlocks = {};
	adjointReducedBlocks = {};
	if (equilibriumModel)
	{
		equilibriumModel->resetForRun();

		// No adjoint through the field at tier 0. Run as a frozen-field gradient
		// it would be wrong by the field's share -- a third of the closed-form
		// gradient on the theta pinch -- so it is refused rather than run. Tier
		// 1 transposes its response (applyResponseTranspose), so it has one.
		if (!reducedCoupling() && (adjointProblem != nullptr || steadyTest == SteadyTest::Objective))
		{
			throw std::invalid_argument(
				"An adjoint cannot be solved with FieldCoupling = \"consistent\": the equilibrium "
				"supplies no derivatives, and a gradient with the field frozen would be wrong by "
				"its share. Couple the field model to get gradients.");
		}

		// Well below the run's own tolerances: an equilibrium's error goes into
		// every residual, where IDA reads it as local error.
		const double absolute = 1e-3 * *std::min_element(atol.begin(), atol.end());
		if (!equilibriumModel->setTolerances(1e-3 * rtol, absolute))
			logmsg<LOG_LEVEL::WARNING>(
				"The equilibrium model '{}' cannot resolve its geometry to {:g} relative, {:g} "
				"absolute, a thousandth of the run's tolerances; its error will reach every residual.",
				equilibriumModel->getSpec().name, 1e-3 * rtol, absolute);
	}

	// The case's route to geometry outside the residual (its diagnostics),
	// installed by the solver that runs it and cleared by one with no field
	// model, so a case reused under a different solver never keeps a stale one.
	// Here rather than in setFieldModel or the constructor: the discretisation
	// estimate builds a second solver on the same case that never initialises.
	if (fieldModel || equilibriumModel)
	{
		std::vector<std::string> slots;
		for (FieldSlot const &g : equilibriumModel ? equilibriumModel->getSpec().geometry
		                                           : fieldModel->getSpec().geometry)
			slots.push_back(g.name);
		problem->setGeometryProvider(std::move(slots), [this](VectorRef out, DGSoln const &Y, Position x, Time t)
		                             { geometryAt(out, Y, x, t); });
	}
	else
		problem->setGeometryProvider({}, {});

	// Nor does the kappa a held nodal tau took from the last run's final
	// Jacobian build.
	haveJacobianKappa = false;

	// Nor the derivatives the last run's Jacobian builds left for an adjoint
	// solve to reuse, nor the adjoint states solved from them: a reused solver
	// may be running a case whose parameters have changed, at a state that
	// happens to match.
	jacobianDerivatives.valid = false;
	adjointDerivatives.valid = false;
	fieldBlocksHoldAdjoint = false;
	adjointSolutions.clear();
	adjointSolutionState.clear();

	// A tau frozen per continuation step is set by the continuation loop, and a
	// time march has none: tau would stay at the initial condition's for the
	// whole run.
	if (tauFrozenPerStep() && !solvesForSteadyState())
		throw std::invalid_argument(
			"tauUpdate = \"ContinuationStep\" or \"JacobianBuild\" needs a steady solve with "
			"SteadyStateSolver = \"PseudoTransient\" or \"Newton\"; a time march "
			"would freeze tau at the initial condition. Use tauUpdate = \"Residual\".");

	// ...and neither do the sweep counts. Here, in the unconditional part of
	// initialize() and beside resetForRun() for the same reason: a cumulative
	// count reported as a per-run one is a lie a second run would tell silently.
	fieldSweepSolves = 0;
	fieldSweepIterations = 0;
	fieldSweepFallbacks = 0;
	fieldAdjointSweeps = 0;
	fieldAdjointFellBack = false;
	adjointKrylovIterations = 0;
	adjointResponses = 0;
	reducedResponses = 0;
	krylovSolves = 0;
	krylovIterations = 0;

	// Where and how this run will evaluate the case, said before the first
	// evaluation -- and initialiseMatrices below is that, on a solver's first
	// run, because it integrates aFn into the mass matrix. Offered on every run,
	// because the case may have been run by another solver in between; the case
	// is told only when the plan differs from the last one it was told, and a
	// changed plan is refused here for a case that does not support regridding
	// (see TransportSystem::deliverEvaluationPlan). The plan reads no state and
	// the case's hook writes none of the solver's, so a reused solver stays bit
	// for bit a fresh one.
	//
	// MaxPhysicsBatch is a promise to the case about the largest batch it will be
	// handed on every residual and Jacobian build, so a level that breaks it is
	// refused here, before the case hears of it. The adaptation controllers choose
	// no such level; reaching this means the configured one is too large.
	auto plan = std::make_shared<const EvaluationPlan>(evaluationPlan());
	if (maxPhysicsBatch > 0 && plan->largestRecurringBatch() > maxPhysicsBatch)
		throw std::invalid_argument(std::format(
			"{} cells at k = {} hand the physics batches of {} points on every residual or "
			"Jacobian build, past MaxPhysicsBatch = {}. Use fewer cells or a lower degree.",
			nCells, k, plan->largestRecurringBatch(), maxPhysicsBatch));
	// The discretisation estimate, checked here rather than at the end of the
	// first solve it would make: what it refuses is fixed by now, and its batch,
	// once a steady solve rather than every residual, is still a batch the case
	// was promised it would not get.
	if (estimateDiscretisation && adjointProblem != nullptr)
	{
		if (nScalars > 0 || fieldModel || equilibriumModel || k < 1 || tauScaling != TauScaling::Constant)
			throw std::invalid_argument(
				"EstimateDiscretisationError cannot estimate this run: it needs "
				"PolynomialDegree >= 1, a constant tau, and no global scalars or field model.");
		for (auto const &site : plan->sitesOf(EvaluationKind::DiscretisationEstimate))
			if (maxPhysicsBatch > 0 && site.batchSize() > maxPhysicsBatch)
				throw std::invalid_argument(std::format(
					"EstimateDiscretisationError hands the physics a batch of {} points at the "
					"end of each steady solve, past MaxPhysicsBatch = {}.",
					site.batchSize(), maxPhysicsBatch));
	}
	problem->deliverEvaluationPlan(plan);

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

	// Set original vector lengths. The layout is DGSoln's, so the length is too.
	// The field model's unknowns go last, after the scalars, so nothing before
	// them moves; nField is zero unless setFieldModel has attached a model.
	Y = N_VNew_Serial(DGSoln::getDoF(nVars, nCells, k, nScalars, nAux, nField), ctx);
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
	// Same shape of check, same reason, same place: a source that reads du/dt can
	// cancel its own variable's mass term, which turns a differential row
	// algebraic without saying so. A no-op unless a variable declared it.
	checkEffectiveMassMatrix(yJac, dydtJac, t0);

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
									   fieldStatesAt(yJac, dydtJac), yJac.getPoints(),
									   integrator.integrationWeights(yJac.getBasis(), grid),
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
		if (fieldSolveMode == FieldSolveMode::Krylov)
			logmsg<LOG_LEVEL::INFO>(
				"FieldSolve = krylov: GMRES on the coupled Jacobian with the field eliminated, one "
				"transport solve per iteration, stopping where FieldSolve = iterative would "
				"(FieldSolveTolerance = {}); FieldSolveMaxSweeps = {} is its Krylov dimension. One "
				"that does not converge falls back to the exact solve.",
				fieldSolveTolerance, fieldSolveMaxSweeps);
		else if (fieldSolveMode == FieldSolveMode::Exact)
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
				"exact's {} per Jacobian solve. Stops once a sweep changes the increment by less "
				"than FieldSolveTolerance = {} in the error-weighted norm (or a millionth of the "
				"increment), up to FieldSolveMaxSweeps = {} sweeps ({} for the adjoint); a sweep that reaches its cap falls back to the exact solve, so this "
				"mode costs more than exact in the worst case and never less accuracy. It is "
				"only *cheaper* than exact when the sweep converges in fewer than {} sweeps, "
				"which depends on the coupling: the theta pinch does it in ~3.7 at any "
				"resolution, while a strongly divergent coupling may not at all. Watch the "
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

	// Take the algebraic rows out of IDA's local error test, by default for a
	// run with a field and otherwise only when asked. `id` above is what makes
	// this meaningful: u and the differential scalars carry 1.0, everything
	// else 0.0, and IDASetSuppressAlg drops exactly the zeros. sigma, q, lambda
	// and phi stay in the Newton test, so they are converged at every step;
	// what goes is the check of their local truncation error.
	//
	// Neither choice is answer-preserving, which is why the default depends on
	// the run. Measured against runs 1000x tighter (docs/running.rst has the
	// tables): at matched tolerance, suppression gives 1-5x the time error on
	// most of the regression configs, for 4-52% less work; tightened tenfold to
	// cost about the same, it is the more accurate on 11 of 15. Where it loses
	// -- Nonlin2, nonlin, PIDTest, ld's q -- its error stops following the
	// tolerance. And it costs restart fidelity on plain diffusion: a restart
	// reads sigma and q back from the file, and LinearDiffusion's q lands 100x
	// further from the uninterrupted run.
	//
	// A field changes the balance. With the algebraic rows in the test, IDA's
	// order selection on the coupled theta pinch falls into a trap at order 2
	// for 120-200 steps or does not, as round-off decides: 149-290 steps over a
	// +-8% tolerance sweep, under either FieldSolve. Suppressed, it takes 99-101
	// at every one of those tolerances, both field solves agree to ten digits,
	// and its 5x larger time error is a millionth of the spatial one; restarts
	// of both theta-pinch regression configs are as faithful either way.
	if (suppressesAlgebraicError())
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
	idaResidualEvalsBeforeReinit = 0;
	idaJacobianSetupsBeforeReinit = 0;

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
		// Full Newton steps, with no line search, when the field is an
		// equilibrium. The Jacobian then carries no coupling, so its Newton
		// direction need not descend ||F|| at all where the coupling is strong,
		// and IDACalcIC's backtracking cannot recover from that -- it fails with
		// IDA_LINESEARCH_FAIL. Measured on the theta pinch at T = 0.9, 5 and 32
		// cells: the line search fails, and full steps converge in IDACalcIC's
		// default 10 iterations. At T = 0.3, where it succeeds, the full step
		// was accepted every time and the run is bit for bit the same either way.
		// Tier 1's Jacobian does carry the coupling, so it keeps the line
		// search: on the theta pinch at T = 0.3 and 0.9, 5 and 32 cells, the
		// runs take the same steps and residuals with it on or off.
		if (equilibriumModel && !reducedCoupling())
			IDASetLineSearchOffIC(IDA_mem, SUNTRUE);
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
		// differential derivatives, not the other way round. They are solved for
		// below, on every path that ran IDACalcIC except a copied restart.
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

		// The zero algebraic y' above is wrong wherever the algebraic components
		// move, and IDA's predictor then carries an error of h |y'| in sigma, q,
		// lambda and phi -- first order, where the error test expects second. A
		// projected restart resumes mid-run with its first step at the output
		// cadence, and paid ten error-test failures, i.e. the run. A cold start
		// paid in step size instead: at a tight tolerance the first step had to
		// fall below MinStepSize's default before dsm, falling in proportion to h,
		// passed -- 9 of the 17 regression configs at 1000x their own tolerance.
		// So both get the exact algebraic derivatives. A copied restart takes
		// y' from the file instead (setInitialConditions).
		if (!restarting || restartWasProjected)
		{
			consistentAlgebraicDerivatives(Y, dYdt, t0);
			IDAGetNumResEvals(IDA_mem, &idaResidualEvalsBeforeReinit);
			IDAGetNumLinSolvSetups(IDA_mem, &idaJacobianSetupsBeforeReinit);
			retval = IDAReInit(IDA_mem, t0, Y, dYdt);
			if (ErrorChecker::check_retval(&retval, "IDAReInit", 1))
				throw std::runtime_error("Could not hand IDA the consistent algebraic derivatives");
		}
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

	// A restart's first step is a guess at the output cadence. It is not a step
	// the previous run took -- the file records none -- and a first-order step
	// that long usually fails the error test a few times before IDA has cut it to
	// size. That is cheap now that a copied restart resumes a consistent y': the
	// error estimate falls like h^2, a factor of 16 per cut, so even a large
	// misjudgement is a handful of residuals. Measured against leaving the first
	// step to IDA, whose own estimate starts far smaller and climbs, this is the
	// cheaper of the two -- 70 residuals against 78 and 8 Jacobian builds
	// against 13 on AuxVarTest, 20 against 32 and 4 against 8 on MatTest.
	if (problem->isRestarting())
		IDASetInitStep(IDA_mem, dt);
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
// the same thing at every tolerance setting: tightening AbsoluteTolerance
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

// The algebraic components of y' at a consistent state. IDA_YA_YDP_INIT
// computes algebraic values and differential derivatives and leaves the
// algebraic derivatives where it found them, so a state IDACalcIC has just made
// consistent still has z' = 0 -- which tells IDA's predictor those components
// are not moving, and makes the first step's error estimate h |z'| rather than
// h^2.
//
// What fixes z' is that every row of the residual stays zero along the
// trajectory: d/dt F(t, y, y') = F_y y' + F_y' y'' + F_t = 0. The differential
// derivatives u' are given, and nothing reads an algebraic y', so F_y' y'' is
// F_y' u'' and the unknowns are u'' and z'. Where no algebraic row reads y' --
// every case but a scalar constraint on du/dt -- the algebraic rows alone give
// G_z z' = -G_t - G_u u', and u'' is a by-product. Where one does, as
// ScalarTestLD3's controller reads dM/dt, its row reads u'' too, and u'' is
// pinned by the differentiated differential rows; leaving them out, or setting
// u'' to zero, gives a y' that contradicts its own constraint.
//
// The coupled Jacobian solve J(alpha) = F_y + alpha F_y' solves exactly this
// system, up to O(1/alpha), with x_d = u' + u'' / alpha:
//
//   J(alpha) x = alpha F_y' u' - F_t + F_y P_d (x_d - u'),
//
// P_d keeping the differential components. The last term is what the O(1/alpha)
// is, and it is known once x is, so it is iterated: each pass takes one residual
// difference for it and one solve, and contracts the error by about s / alpha, s
// the fastest rate in the differential rows. F_y' u' is F(t, Y, u') - F(t, Y, 0),
// exact for a residual affine in y'. F_t is the case's own explicit time
// derivative on the algebraic rows (explicitTimeDerivative), checked against the
// residual, and a forward difference of the residual on the differential ones,
// where it enters only through u''. Only the algebraic components are taken; the
// differential ones stay as IDACalcIC computed them.
void SystemSolver::consistentAlgebraicDerivatives(N_Vector Yc, N_Vector dYdtc, Time tc)
{
	N_Vector ud = N_VClone(Yc), zero = N_VClone(Yc), F0 = N_VClone(Yc), Fa = N_VClone(Yc),
			 Fv = N_VClone(Yc), base = N_VClone(Yc), rhs = N_VClone(Yc), x = N_VClone(Yc),
			 xPrevious = N_VClone(Yc), d = N_VClone(Yc), Yp = N_VClone(Yc), minusFt = N_VClone(Yc);

	double const *isDiff = N_VGetArrayPointer(id);
	const sunindextype n = N_VGetLength(Yc);

	// u': the differential components of dYdt, the algebraic ones zero.
	N_VProd(dYdtc, id, ud);
	N_VConst(0.0, zero);
	residual(tc, Yc, zero, F0);
	residual(tc, Yc, ud, Fa);
	// F_y' u', on every row: M u' on the differential rows, and the du/dt an
	// algebraic row reads, where one does.
	N_VLinearSum(1.0, Fa, -1.0, F0, base);
	double const *fy = N_VGetArrayPointer(base);
	double readsYdot = 0.0;
	for (sunindextype i = 0; i < n; ++i)
		if (isDiff[i] == 0.0)
			readsYdot = std::max(readsYdot, std::abs(fy[i]));

	// -F_t: the case's own d/dt hooks on the algebraic rows...
	N_VConst(0.0, minusFt);
	{
		DGSoln Yh(nVars, grid, k, N_VGetArrayPointer(Yc), nScalars, nAux, nField);
		DGSoln dYh(nVars, grid, k, N_VGetArrayPointer(ud), nScalars, nAux, nField);
		DGSoln Gt(nVars, grid, k, N_VGetArrayPointer(minusFt), nScalars, nAux, nField);
		explicitTimeDerivative(Yh, dYh, tc, Gt);
	}

	// ...checked against the residual itself. A case may make its data depend on
	// t without declaring the d/dt hooks -- their defaults are zero -- and then
	// G_t says the data stand still and y' contradicts them: MatrixDiffusionTest
	// did, and at 1000x its tolerances a cold start that ran with z' = 0 failed
	// its first step with this one. A forward difference in t at fixed (Y, u')
	// is G_t to about 1e-8 for any case, so where the hooks disagree with it by
	// more than that could explain, the difference is used and the case is
	// named. Forward, so nothing is evaluated before t0. The differential rows
	// take the difference whatever the hooks say: they reach z' only through u''
	// and only where an algebraic row reads du/dt, and a source's explicit time
	// dependence has no hook to compare with.
	{
		const double e = 1e-7 * std::max(1.0, std::abs(tc));
		N_Vector Fb = N_VClone(Yc);
		residual(tc + e, Yc, ud, Fb);
		double *h = N_VGetArrayPointer(minusFt);
		double const *fa = N_VGetArrayPointer(Fa), *fb = N_VGetArrayPointer(Fb);
		double gap = 0.0, scale = 0.0, size = 0.0;
		for (sunindextype i = 0; i < n; ++i)
		{
			size = std::max(size, std::abs(fa[i]));
			const double fd = (fb[i] - fa[i]) / e;
			if (isDiff[i] != 0.0)
			{
				h[i] = fd;
				continue;
			}
			gap = std::max(gap, std::abs(fd - h[i]));
			scale = std::max({scale, std::abs(fd), std::abs(h[i])});
		}
		// What the difference alone could get wrong: round-off in F over e, and
		// its O(e) truncation, which the 1e-3 relative bound covers.
		const double noise = 1e3 * std::numeric_limits<double>::epsilon() * std::max(size, 1.0) / e;
		if (gap > 1e-3 * scale && gap > noise)
		{
			logmsg<LOG_LEVEL::WARNING>(
				"The case's explicit time derivatives (dLowerBoundary_dt, dSigmaFn_dt, dAuxG_dt, ...) "
				"disagree with its residual by {:g} against {:g}: its data move with t and a hook says "
				"they do not. The consistent y' at t = {:g} uses a difference of the residual instead; "
				"declaring the hooks makes it exact.",
				gap, scale, tc);
			for (sunindextype i = 0; i < n; ++i)
				if (isDiff[i] == 0.0)
					h[i] = (fb[i] - fa[i]) / e;
		}
		N_VDestroy(Fb);
	}
	N_VScale(-1.0, minusFt, minusFt);

	// F_y v by a forward difference of the residual in y, at fixed y' = u'. v is
	// x_d - u' = u'' / alpha, a small vector, and its product is an O(s / alpha)
	// correction, so the difference's own ~1e-8 relative error is negligible.
	const double yScale = std::max(1.0, N_VMaxNorm(Yc));
	auto stateDirectional = [&](N_Vector v, N_Vector out)
	{
		const double vmax = N_VMaxNorm(v);
		if (vmax == 0.0)
		{
			N_VConst(0.0, out);
			return;
		}
		const double e = std::sqrt(std::numeric_limits<double>::epsilon()) * yScale / vmax;
		N_VLinearSum(1.0, Yc, e, v, Yp);
		residual(tc, Yp, ud, Fv);
		N_VLinearSum(1.0 / e, Fv, -1.0 / e, Fa, out);
	};

	// alpha has to beat the problem's stiffness, which nothing here knows in
	// advance and which the output cadence says nothing about: tied to
	// 1 / delta_t, alpha was 1e5 on Shestakov's problem (delta_t = 1e3), and IDA
	// started from an algebraic y' 4-18% out. So alpha starts at
	// consistentDerivativeAlpha and is raised by what each pass shows: a
	// contraction slower than 1e-3 asks for alpha times the shortfall against
	// 1e-6, and a Jacobian rebuild. The iterate survives the rebuild, since the
	// system it converges to does not involve alpha.
	//
	// It is capped, lower where an algebraic row reads du/dt: that row's
	// alpha F_y' x_d and alpha F_y' u' cancel to its answer, which so carries an
	// error of about alpha eps |F_y' u'| -- 1e-6 relative at 1e10, O(1) by 1e16.
	constexpr double converged = 1e-14, roundOff = 1e-10, slow = 1e-3, wanted = 1e-6;
	const double alphaMax = readsYdot > 0.0 ? 1e10 : 1e14;
	constexpr int maxPasses = 32, maxBuilds = 6;

	const double alphaOnEntry = alpha;
	double a = consistentDerivativeAlpha;
	setJacTime(tc);
	updateBoundaryConditions(tc);
	auto build = [&]()
	{
		setAlpha(a);
		setJacEvalY(Yc, ud);
		updateMatricesForJacSolve();
	};
	build();

	// One pass: rhs = alpha F_y' u' - F_t + F_y P_d (x_d - u'), solved for x.
	auto pass = [&]()
	{
		N_VLinearSum(1.0, x, -1.0, ud, d);
		N_VProd(d, id, d);
		stateDirectional(d, rhs);
		N_VLinearSum(a, base, 1.0, rhs, rhs);
		N_VLinearSum(1.0, minusFt, 1.0, rhs, rhs);
		N_VScale(1.0, x, xPrevious);
		solveJacEq(rhs, x);
	};
	// How far a pass moved the algebraic components, relative to them or to u'.
	auto change = [&]()
	{
		double moved = 0.0, size = N_VMaxNorm(ud);
		double const *xv = N_VGetArrayPointer(x), *xp = N_VGetArrayPointer(xPrevious);
		for (sunindextype i = 0; i < n; ++i)
			if (isDiff[i] == 0.0)
			{
				moved = std::max(moved, std::abs(xv[i] - xp[i]));
				size = std::max(size, std::abs(xv[i]));
			}
		return moved / std::max(size, 1e-300);
	};

	// The first pass starts from x_d = u', where the correction term vanishes.
	// What the second moves the answer by, against a start of 1, is the first
	// rate.
	N_VScale(1.0, ud, x);
	pass();
	int passes = 1, builds = 1;
	double moved = 1.0, rate = 1.0, rateBeforeRaise = -1.0;
	bool raising = true;
	for (;;)
	{
		pass();
		++passes;
		const double previous = moved;
		moved = change();
		rate = moved / previous;
		if (moved <= converged || passes == maxPasses)
			break;
		if (rate > 0.5 && moved <= roundOff)
			break;
		// A raise that did not buy a tenth of what it should have says the slow
		// mode is not alpha's, and further raises would only cost round-off.
		if (rateBeforeRaise > 0.0 && rate > 0.1 * rateBeforeRaise)
			raising = false;
		rateBeforeRaise = -1.0;
		if (rate > slow && raising && builds < maxBuilds && a < alphaMax)
		{
			rateBeforeRaise = rate;
			a = std::min(a * std::clamp(rate / wanted, 10.0, 1e8), alphaMax);
			build();
			++builds;
		}
	}
	setAlpha(alphaOnEntry);
	consistentDerivativePasses = passes;
	consistentDerivativeBuilds = builds;
	consistentDerivativeAlphaUsed = a;
	consistentDerivativeChange = moved;
	if (moved > roundOff)
		logmsg<LOG_LEVEL::WARNING>(
			"The algebraic derivatives at t = {:g} took {} passes over {} Jacobian builds, up to "
			"alpha = {:g}, and the last still moved them by {:g} relative; the algebraic y' handed "
			"to IDA carries that error.",
			tc, passes, builds, a, moved);

	double *dy = N_VGetArrayPointer(dYdtc);
	double const *xv = N_VGetArrayPointer(x);
	for (sunindextype i = 0; i < n; ++i)
		if (isDiff[i] == 0.0)
			dy[i] = xv[i];

	for (N_Vector v : {ud, zero, F0, Fa, Fv, base, rhs, x, xPrevious, d, Yp, minusFt})
		N_VDestroy(v);
}

void SystemSolver::integrate(double tFinal)
{
	int retval;

	if (IDA_mem == nullptr)
		throw std::logic_error("integrate() called before initialize()");

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
			ApplyDirichletBCs(y, t0);
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

		// Put the boundary datum back into the Dirichlet trace entries before
		// anything reads Y.
		//
		// They are frozen at whatever setInitialConditions seeded: their row and
		// column in K_global are identically zero and imposeDirichletTraceRows
		// pins the correction to zero, so no step of the integration can move
		// them, and with time-dependent Dirichlet data they are stale by the
		// first output. Measured on MatTest, whose g_D decays like
		// exp(-t pi^2/4): at t = 0.5 the stored lower-face trace read 0.99999993
		// against a datum of 0.29121 and a first interior trace node of 0.28962 --
		// discontinuous from its own neighbour by a factor of three.
		//
		// Writing here rather than into the residual keeps this out of the
		// equations. IDASolve treats Y as an output and continues from its own
		// internal state, so nothing written between calls reaches the
		// integration; the cell rows take the datum from RF_cellwise, which
		// updateBoundaryConditions has always refreshed at the residual's own
		// time. What changes is only what a reader gets: the netCDF slice, the
		// .dat files, the restart file's DOF vector and yJac all alias this
		// memory.
		ApplyDirichletBCs(y, tret);

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

		// The steady solvers' own SteadyStateTest, at the state IDA returned;
		// see timeMarchIsSteady.
		if (TerminateOnSteadyState && timeMarchIsSteady(tret))
		{
			std::println("Steady State achieved at time t = {:g}", tret);
			// The objective estimate, as a steady solve makes it on finishing:
			// a march that passed its steady test is as near the fixed point the
			// correction extrapolates to (docs/adjoints.rst has the measurement).
			// Made after the loop's last use of IDA's factorisation.
			if (estimateObjectiveOnFinish && adjointProblem != nullptr)
				objectiveEstimate = estimateObjective(tret);
			break;
		}

		// Diagnostics go here
	}

	long int nsteps, nresevals, njacevals;
	IDAGetNumSteps(IDA_mem, &nsteps);
	IDAGetNumResEvals(IDA_mem, &nresevals);
	IDAGetNumLinSolvSetups(IDA_mem, &njacevals);
	nresevals += idaResidualEvalsBeforeReinit;
	njacevals += idaJacobianSetupsBeforeReinit;

	std::println("Total Number of Timesteps             :{}", nsteps);
	std::println("Total Number of Residual Evaluations  :{}", nresevals);
	std::println("Total Number of Jacobian Computations :{}", njacevals);

	if (reducedCoupling())
	{
		auto fs = getFieldSweepStats();
		std::println("Reduced coupled solves                :{} sweeps over {} solves, {} GMRES "
					 "iterations over {} solves, {} responses",
					 fs.iterations, fs.solves, fs.krylovIterations, fs.krylovSolves, fs.responses);
		if (fs.fallbacks > 0)
			logmsg<LOG_LEVEL::INFO>(
				"{} of {} reduced sweeps exhausted FieldSolveMaxSweeps = {} and went on to GMRES.",
				fs.fallbacks, fs.solves, fieldSolveMaxSweeps);
	}
	if (nField > 0 && getFieldSolveMode() == SystemSolver::FieldSolveMode::Krylov)
	{
		auto fs = getFieldSweepStats();
		std::println("Coupled field GMRES                   :{} iterations over {} solves ({} exact fallbacks)",
					 fs.krylovIterations, fs.krylovSolves, fs.fallbacks);
		if (fs.fallbacks > 0)
			logmsg<LOG_LEVEL::WARNING>(
				"{} of {} coupled GMRES solves did not converge and fell back to the exact Schur "
				"solve. The answers are correct; the run is paying for both.",
				fs.fallbacks, fs.krylovSolves);
	}
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
	// See the call in the time loop. A steady solve evaluates everything at t0,
	// so that is the time its boundary datum is taken at -- the same one
	// setInitialConditions used, which is why this only ever differs from the
	// seed on the cold path, where EvaluateLambda overwrites the datum with
	// {{u}} a few lines after it is applied.
	ApplyDirichletBCs(y, t0);

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
			if (writeAdjoints && writeOutput)
				WriteAdjoints();
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

	if (steadyNLS)
	{
		SUNNonlinSolFree(steadyNLS);
		steadyNLS = nullptr;
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
	                      &uPrev, &ptcDYdt, &kinScale, &resScale, &fScaleScratch,
	                      &nlsYcor, &nlsU, &nlsEwt, &nlsTmp, &nlsDprev})
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
	// here: the adjoint fills geometry before it evaluates anything
	// (physicsNodesAt), assembleAdjointOperator stores A1^T and A2^T beside M^T,
	// and solveAdjointState eliminates them exactly (FieldSolve = exact) or
	// sweeps to a checked backward error that *throws* if it is not reached. An
	// objective reading the geometry reaches dG/dpsi through dg's geometry rows,
	// and a field row reading a parameter reaches dG/dp through the adjoint
	// problem's dFieldResidual_dp.

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
