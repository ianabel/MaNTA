#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>

#include "gridStructures.hpp"

typedef std::function<double( double )> Fn;
using Matrix = Eigen::Matrix<realtype,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixWrapper = Eigen::Map<Matrix>;
using Vector = Eigen::Matrix<realtype,Eigen::Dynamic,1>;
using VectorWrapper = Eigen::Map<Vector>;
typedef std::vector<std::vector< std::pair< Interval, Eigen::Map<Eigen::VectorXd >>>> Coeff_t;

class SystemSolver
{
public:
	SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Variables, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c, Eigen::MatrixXd kappaMat);
	~SystemSolver() = default;

	//Initialises u, q and lambda to satisfy residual equation at t=0
	void setInitialConditions(std::function< double (double)> u_0, N_Vector& Y , N_Vector& dYdt);

	//Builds initial matrices
	void initialiseMatrices();
	
	void clearCellwiseVecs();

	//Points coeefficients to sundials vector so no copying needs to occur
	void mapDGtoSundials(DGApprox& q, DGApprox& u, realtype* Y);
	void mapDGtoSundials(DGApprox& u, realtype* Y);
	void mapDGtoSundials(std::vector< Eigen::VectorXd >& QU_cell, realtype* const& Y);

	// Set the q and u coefficients at each time step
	void updateCoeffs(N_Vector const& Y, N_Vector const& dYdt);

	//Creates the ABBDX cellwise matrices used at each Jacobian iteration
	void updateABBDForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& tempABBDBlocks, double const alpha);

	//Solves the Jy = -G equation
	void solveJacEq(N_Vector const& g, N_Vector& delY);

	//Testing function, solving without IDA but using Sundials interface, Backward Euler used
	//No longer works, and no point maintaining consistently
	//void solveNonIDA(N_Vector& Y, N_Vector& dYdt, double dt);

	void setAlpha(double const a) {alpha = a;}

	//print current output for u and q to output file
	void print( std::ostream& out, double t, int nOut, int var );

	//Find the profiles from the coefficients
	double EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var );

	Fn getcfn() const {return c_fn;}
	double getdt() const {return dt;}

	void setBoundaryConditions(BoundaryConditions* BC) {BCs.reset(BC);}
	void setKappaInv(Eigen::MatrixXd kappa);

	Grid grid;
	unsigned int const k; 		//polynomial degree per cell
	unsigned int const nCells;	//Total cell count
	int nVar;					//Total number of variables
	
	std::vector< Eigen::MatrixXd > XMats;
	std::vector< Eigen::MatrixXd > ABBDBlocks;
	std::vector< Eigen::MatrixXd > CEBlocks;
	Eigen::MatrixXd K_global{};
	Eigen::VectorXd L_global{};
	Eigen::FullPivLU< Eigen::MatrixXd > H_global{};
	std::vector< Eigen::MatrixXd > CG_cellwise, RF_cellwise;
	std::vector< Eigen::MatrixXd > A_cellwise, B_cellwise, D_cellwise, E_cellwise, C_cellwise, G_cellwise, H_cellwise; //?Point the dublicated matrices to the same place?

	DGApprox u,q, dudt;
	std::shared_ptr<BoundaryConditions> BCs;
private:

	double dt;
	double t;
	bool initialised = false;
	double alpha = 1.0;

	Fn RHS; //Forcing function
	Fn c_fn,kappa_fn, tau; // convection velocity and diffusivity

	Eigen::MatrixXd kappaInv;
};

struct UserData {
	SystemSolver* system;
};