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

        void setSteadyMode(SteadyMode mode) { steadyMode = mode; };
        SteadyMode getSteadyMode() const { return steadyMode; };
        void setPseudoTransientInitialStep(double dt) { ptcInitialStep = dt; };
        void setPseudoTransientMaxStep(double dt) { ptcMaxStep = dt; };

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

        // Solves the Jy = g equation. Dispatches on whether a field model is
        // attached: without one this *is* solveTransportJac, bit for bit.
        void solveJacEq(N_Vector g, N_Vector delY);

        // The uncoupled transport operator: HDG static condensation plus the
        // scalar bordering, and nothing about the field.
        //
        // Kept separate from solveJacEq because solveCoupledJacExact applies it
        // nField + 1 times as its *inner* solve. Anything that wrote the field
        // block in here -- as Task 6's block-Jacobi psi solve did, before this
        // split -- would corrupt every one of those.
        void solveTransportJac(N_Vector g, N_Vector delY);

        // Exact Schur complement onto psi. See the definition; costs one
        // transport solve per field degree of freedom, so it is a verification
        // tool rather than a production path.
        void solveCoupledJacExact(N_Vector g, N_Vector delY);

        // Block Gauss-Seidel between the transport and field blocks: one
        // transport solve and one field solve per sweep, against the exact
        // path's nField + 1 transport solves. Stops once the relative change in
        // psi between sweeps is below FieldSolveTolerance, up to
        // FieldSolveMaxSweeps -- neither is a guarantee the sweep converges in
        // that many steps. The production path -- safe because the Jacobian is
        // never assembled, so an under-converged sweep costs Newton iterations
        // rather than correctness. Returns its last iterate when the cap is
        // reached first, rather than throwing; see the definition.
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
        //                      condition, open the output files, run IDACalcIC
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
        // live SUNDIALS vectors, and it needs the *post*-IDACalcIC derivative.
        // Before initialize() there is nothing mapped; after destroySundials()
        // they dangle.
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
        //
        //   Iterative -- block Gauss-Seidel between the transport block and the
        //                field block. The production path, and the default.
        //   Exact     -- the Schur complement onto psi, formed by applying the
        //                transport inverse to every column of A1. That is
        //                nField + 1 transport solves per Jacobian solve, so it
        //                is a verification tool: it is what makes the coupled
        //                system checkable by SolveJacTests' method (finite
        //                difference the residual, require J dy = g), and it is
        //                the oracle the iterative path is compared against.
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

        // Adjoints
        void setSolveAdjoint(bool a) { solveAdjoint = a; }

        void initializeMatricesForAdjointSolve();

        // Solve J^T z = dG/dy at the state the matrices above were built from.
        // Dispatches on whether a field model is attached, exactly as solveJacEq
        // does forwards: without one this *is* solveTransportAdjoint.
        void solveAdjointState(Index i);

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

        // The transposed block Gauss-Seidel sweep. Unlike
        // solveCoupledJacIterative this one THROWS on non-convergence rather
        // than returning its last iterate: see the definition for why the
        // asymmetry is the point rather than an inconsistency.
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
        double ptcMaxStep = std::numeric_limits<double>::infinity();
        double ptcStep = 0.0;        // the current dt; infinite in Newton mode
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

        // Allocate (or reallocate) the three buffers yJac, dydtJac and
        // dydtComplete map. Called from the constructor, and again from
        // setFieldModel, which changes how long they have to be.
        void allocateJacobianStorage();

        // Free the a2 vectors, using the *current* nField as the count -- so
        // call it before changing nField, not after.
        void freeFieldWorkVectors();

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

        // Still on that Gauss rule, and so still the derivative of the wrong
        // functional -- see the TODO entry. It is live, unlike the four above,
        // because dgFn_dphi is the only pointwise dg hook a case can still reach.
        void dGdaux_Vec(Index, Vector &, DGSoln const &, Index);

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
