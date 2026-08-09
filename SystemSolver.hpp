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

        void dSources_dScalars_Mat(Matrix &, DGSoln const &, Index );

        // Superconvergent counterpart. The scalars do not enter the
        // postprocessing, so there is no chain matrix -- only the star nodes and
        // A9 in place of the mass matrix. Takes the states and positions the
        // caller already has rather than re-deriving them from a DGSoln, which it
        // could not do for the star nodes anyway.
        void dSources_dScalars_StarMat(Matrix &, GlobalState const &,
                                       std::vector<Position> const &, Index);

        void dSourcedPhi_Mat(Matrix &, DGSoln const &, Index );
        void dPhi_Mat(Matrix &, std::vector<Eigen::Ref<Matrix>> const dX_dZ, DGSoln const &, Index );

        void dAux_Mat(Eigen::Ref<Matrix>, DGSoln const &, Index );
        void dAux_Mat(Eigen::Ref<Matrix>, GlobalStateMatrix&, DGSoln const &, Index );

        void DerivativeSubVector(Index, Vector &, void (AdjointProblem::*dX_dZ)(Index, VectorRef, const State &, Position), DGSoln const &Y, Index I);
        void DerivativeSubVector(Index, Vector &, Eigen::Ref<Matrix> const dX_dZ, DGSoln const &, Index intervalIndex);

        void dGdu_Vec(Index, Vector &, DGSoln const &, Index);
        void dGdq_Vec(Index, Vector &, DGSoln const &, Index);
        void dGdsigma_Vec(Index, Vector &, DGSoln const &, Index);
        void dGdaux_Vec(Index, Vector &, DGSoln const &, Index);

        double resNorm = 0.0; // Exclusively for unit testing purposes

        double dt;
        double t0, t, jt;

        // Really we should do init in the constructor and not need this flag. TODO
        bool initialised = false;

        bool zeroFlux = false; // used to switch between zero-flux and zero-gradient BCs

        // Text output is opt-in; netCDF is what a run produces by default.
        bool writeDatFile = false;
        bool writeDebugDatFiles = false;

        // IDASetEtaMax(10.0) rather than IDA's default 2.0. See
        // setAggressiveTimesteps.
        bool aggressiveTimesteps = false;

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
