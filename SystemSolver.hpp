#ifndef SYSTEMSOLVER_HPP
#define SYSTEMSOLVER_HPP

#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of sunrealtype, sunindextype  */
#include <nvector/nvector_serial.h>
#include <filesystem>

#include "Types.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <memory>
#include <optional>

#include "gridStructures.hpp"
#include "TransportSystem.hpp"
#include "DGSoln.hpp"
#include "NetCDFIO.hpp"

#ifdef TEST
namespace system_solver_test_suite
{
	struct systemsolver_init_tests;
	struct systemsolver_multichannel_init_tests;
	struct systemsolver_matrix_tests;
};
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

	double getdt() const { return dt; }

	void setTesting(bool t) { testing = t; }
	bool isTesting() const { return testing; }

	void updateBoundaryConditions(double t);

	Vector resEval(std::vector<Vector> resTerms);

	void mapDGtoSundials(std::vector<VectorWrapper> &SQU_cell, VectorWrapper &lam, sunrealtype *const &Y) const;

	static SystemSolver *ConstructFromConfig(std::string fname);

	// Initialise
	void runSolver(double);

	void setJacTime(double tt) { jt = tt; };
	void setTime(double tt) { t = tt; };
	void setTau(double tau) { tauc = tau; };

	void setInputFile(std::string const &fn) { inputFilePath = fn; };

	void setJacEvalY(N_Vector &);
	int residual(sunrealtype, N_Vector, N_Vector, N_Vector);

private:
	Grid grid;
	unsigned int k;		   // polynomial degree per cell
	unsigned int nCells;   // Total cell count
	unsigned int nVars;	   // Total number of variables
	unsigned int nScalars; // Any global scalars
	unsigned int nAux;	   // Any auxiliary constraints

	using EigenCellwiseSolver = Eigen::PartialPivLU<Matrix>;
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
	std::vector<Matrix> A_cellwise, B_cellwise, D_cellwise, E_cellwise, C_cellwise, G_cellwise, H_cellwise;

	SUNContext ctx;
	N_Vector *v, *w;

	std::vector<Matrix> W_cellwise;
	Matrix N_global; // Scalar-scalar coupling matrix

	//?Point the duplicated matrices to the same place?
	std::vector<EigenCellwiseSolver> MXSolvers;

	DGSoln y, dydt; // memory owned by SUNDIALS

	double *yJacMem = nullptr;
	DGSoln yJac; // memory owned by us

	void NLqMat(Matrix &, DGSoln const &, Interval);
	void NLuMat(Matrix &, DGSoln const &, Interval);

	void dSourcedu_Mat(Matrix &, DGSoln const &, Interval);
	void dSourcedq_Mat(Matrix &, DGSoln const &, Interval);
	void dSourcedsigma_Mat(Matrix &, DGSoln const &, Interval);

	void DerivativeSubMatrix(Matrix &mat, void (TransportSystem::*dX_dZ)(Index, Values &, const State &, Position, Time), DGSoln const &Y, Interval I);

	void dSources_dScalars_Mat(Matrix &, DGSoln const &, Interval);

	void dSourcedPhi_Mat(Matrix &, DGSoln const &, Interval);

	void dAux_Mat(Eigen::Ref<Matrix>, DGSoln const &, Interval);

    void dSourcedPhi_Mat( Matrix &, DGSoln const&, Interval );

    void dAux_Mat( Eigen::Ref<Matrix>, DGSoln const&, Interval );

	double resNorm = 0.0; // Exclusively for unit testing purposes

	double dt;
	double t0, t, jt;

	// Really we should do init in the constructor and not need this flag. TODO
	bool initialised = false;

	double alpha = 1.0;
	bool testing = false;

	// Why do we need to know? Surely everything is encoded in the construction of the Grid, which is done elsewhere?
	bool highGridBoundary = true;

	// Hide all physics-specific info in here
	TransportSystem *problem = nullptr;

	// Tau
	double tauc;
	double tau(double x) const { return tauc; };

	double rtol;
	std::vector<double> atol;

	NetCDFIO nc_output;
	void initialiseNetCDF(std::string const &fname, size_t nOut);
	void WriteTimeslice(double tNew);

	size_t S_DOF, U_DOF, Q_DOF, AUX_DOF, SQU_DOF;
	size_t localDOF;

	bool TerminateOnSteadyState = false;
	double steady_state_tol = 1e-3;
#ifdef PHYSICS_DEBUG
	constexpr static bool physics_debug = true;
#else
	constexpr static bool physics_debug = false;
#endif

#ifdef TEST
	friend struct system_solver_test_suite::systemsolver_init_tests;
	friend struct system_solver_test_suite::systemsolver_multichannel_init_tests;
	friend struct system_solver_test_suite::systemsolver_matrix_tests;
#endif

	std::filesystem::path inputFilePath;
	double dt0; // initial dt for CalcIC
	int nOut;
	double min_step_size;

    int getErrorWeights( N_Vector y, N_Vector ewt );
    static int getErrorWeights_static( N_Vector, N_Vector, void * );

};

#endif // SYSTEMSOLVER_HPP
