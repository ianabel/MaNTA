#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <Eigen/Core>
#include <Eigen/Dense>

#include "gridStructures.hpp"

typedef std::function<double( double )> Fn;
using Matrix = Eigen::Matrix<realtype,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixWrapper = Eigen::Map<Matrix>;
using Vector = Eigen::Matrix<realtype,Eigen::Dynamic,1>;
using VectorWrapper = Eigen::Map<Vector>;

class SystemSolver
{
public:
	SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c, Fn const& kappa, BoundaryConditions const& boundary );
	~SystemSolver() = default;

	//Initialises u, q and lambda to satisfy residual equation at t=0
	void setInitialConditions(std::function< double (double)> u_0);

	//Builds initial matrices
	void initialiseMatrices();

	// Takes n vector and parses it into the RHS of a Jacobian equation
	void sundialsToDGVecConversion(N_Vector const& g, std::vector< Eigen::VectorXd >& g1g2_cellwise, Eigen::VectorXd& g3_global);

	void sundialsToDGVecConversion(N_Vector& delY, std::vector< Eigen::VectorXd >& UQLamCellwise);

	// Returnable n_vector for sundials linear solver
	void DGtoSundialsVecConversion(DGApprox delU, DGApprox delQ, Eigen::VectorXd delLambda, N_Vector& delY);

	//Creates the ABBDX cellwise matrices used at each Jacobian iteration
	void updateABBDForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& tempABBDBlocks, double const alpha);

	//Solves the Jy = -G equation
	void solveJacEq(double const alpha, N_Vector const& g, N_Vector& delY);

	//Residual Function
	int residual(realtype tres, N_Vector uu, N_Vector up, N_Vector resval,
            void *user_data);

private:

	Grid grid;
	std::vector< Eigen::MatrixXd > ABCBDECGHMats;
	std::vector< Eigen::MatrixXd > XMats;
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > ABBDSolvers;
	Eigen::MatrixXd K_global{};
	Eigen::VectorXd L_global{};
	std::vector< Eigen::MatrixXd > CG_cellwise;
	std::vector< Eigen::VectorXd > RF_cellwise;
	std::vector< Eigen::MatrixXd > QU_0_cellwise;
	std::vector< Eigen::MatrixXd > E_cellwise;

	double dt;
	double t;
	unsigned int const k; 		//polynomial degree per cell
	unsigned int const nCells;	//Total cell count
	bool initialised = false;

	Fn RHS; //Forcing function
	Fn c_fn,kappa_fn, tau; // convection velocity and diffusivity
	DGApprox u,q, dudt;
	BoundaryConditions BCs;
};
