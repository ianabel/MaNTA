#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>		/* defs of realtype, sunindextype  */

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
	SystemSolver(Grid const &Grid, unsigned int polyNum, double Dt, double tau, TransportSystem *pProblem);
	SystemSolver(Grid const &Grid, unsigned int polyNum, double Dt, double tau, double SteadyStateTol, TransportSystem *pProblem);

	// This has been moved elsewhere, SystemSolver should be constructed after the parsing is done.
	// SystemSolver(std::string const& inputFile);
	~SystemSolver() {
		delete yJacMem;
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
	void print(std::ostream &out, double t, int nOut);
	void print(std::ostream &out, double t, int nOut, N_Vector const &tempY);

	double getdt() const { return dt; }

	void setTesting(bool t) { testing = t; }
	bool isTesting() const { return testing; }

	void updateBoundaryConditions(double t);

	Vector resEval(std::vector<Vector> resTerms);

	void mapDGtoSundials(std::vector<VectorWrapper> &SQU_cell, VectorWrapper &lam, realtype *const &Y) const;

	static SystemSolver *ConstructFromConfig(std::string fname);

	// Initialise
	void runSolver(std::string);

	void setJacTime(double tt) { jt = tt; };
	void setTime(double tt) { t = tt; };
	void setTau(double tau) { tauc = tau; };

	void setJacEvalY( N_Vector & );
	int residual(realtype, N_Vector, N_Vector, N_Vector );

	void setPhysicsDebug( bool x ) { physics_debug = x; };

private:
	Grid grid;
	unsigned int k;		 // polynomial degree per cell
	unsigned int nCells; // Total cell count
	unsigned int nVars;	 // Total number of variables
	unsigned int nScalars;	 // Total number of variables

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

	std::vector<Matrix> V_cellwise;
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

	double resNorm = 0.0; // Exclusively for unit testing purposes

	double dt;
	double t,jt;

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


	NetCDFIO nc_output;
	void initialiseNetCDF(std::string const &fname, size_t nOut);
	void WriteTimeslice(double tNew);

	size_t S_DOF, U_DOF, Q_DOF, SQU_DOF;

	bool TerminateOnSteadyState = false;
	double steady_state_tol = 1e-3;
	bool physics_debug = false;

#ifdef TEST
	friend struct system_solver_test_suite::systemsolver_init_tests;
	friend struct system_solver_test_suite::systemsolver_multichannel_init_tests;
	friend struct system_solver_test_suite::systemsolver_matrix_tests;
#endif
};
