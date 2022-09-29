#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>

#include "gridStructures.hpp"
#include "DiffusionObj.hpp"

class SystemSolver
{
public:
	SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Variables, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c);
	~SystemSolver() = default;

	//Initialises u, q and lambda to satisfy residual equation at t=0
	void setInitialConditions(std::function< double (double)> u_0, std::function< double ( double )> gradu_0, std::function< double ( double )> sigma_0, N_Vector& Y, N_Vector& dYdt );

	//Builds initial matrices
	void initialiseMatrices();
	
	void clearCellwiseVecs();

	//Points coeefficients to sundials vector so no copying needs to occur
	void mapDGtoSundials(DGApprox& sigma, DGApprox& q, DGApprox& u, Eigen::Map<Eigen::VectorXd>& lam, realtype* Y);
	void mapDGtoSundials(DGApprox& u, realtype* Y);
	void mapDGtoSundials(std::vector< Eigen::Map<Eigen::VectorXd > >& SQU_cell, Eigen::Map<Eigen::VectorXd>& lam, realtype* const& Y);

	void resetCoeffs();

	//Creates the MX cellwise matrices used at each Jacobian iteration
	//Factorization of these matrices is done here
	//?TO DO: the values of the non-linear matrices will be added in this function?
	void updateMForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& tempABBDBlocks, double const alpha, DGApprox& delQ, DGApprox& delU);

	//Solves the Jy = -G equation
	void solveJacEq(N_Vector& g, N_Vector& delY);

	void setAlpha(double const a) {alpha = a;}

	//print current output for u and q to output file
	void print( std::ostream& out, double t, int nOut, int var );
	void print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY  );

	//Find the profiles from the coefficients
	double EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var );

	Fn getcfn() const {return c_fn;}
	double getdt() const {return dt;}
	void setDiffobj(std::shared_ptr<DiffusionObj> diffObj_) {diffObj = diffObj_;}
	std::shared_ptr<DiffusionObj> getDiffObj();


	void setBoundaryConditions(BoundaryConditions* BC) {BCs.reset(BC);}
	void updateBoundaryConditions(double t);

	Grid grid;
	unsigned int const k; 		//polynomial degree per cell
	unsigned int const nCells;	//Total cell count
	int nVar;					//Total number of variables
	
	std::vector< Eigen::MatrixXd > XMats;
	std::vector< Eigen::MatrixXd > MBlocks;
	std::vector< Eigen::MatrixXd > CEBlocks;
	Eigen::MatrixXd K_global{};
	Eigen::VectorXd L_global{};
	Eigen::MatrixXd H_global_mat{};
	Eigen::FullPivLU< Eigen::MatrixXd > H_global{};
	std::vector< Eigen::MatrixXd > CG_cellwise, RF_cellwise;
	std::vector< Eigen::MatrixXd > A_cellwise, B_cellwise, D_cellwise, E_cellwise, C_cellwise, G_cellwise, H_cellwise; //?Point the dublicated matrices to the same place?

	DGApprox u, q, sig, dudt, dqdt, dsigdt;
	std::optional<Eigen::Map<Eigen::VectorXd>> lambda, dlamdt;
	std::shared_ptr<BoundaryConditions> BCs;
private:

	double dt;
	double t;
	bool initialised = false;
	double alpha = 1.0;

	Fn RHS; //Forcing function
	Fn c_fn,kappa_fn, tau; // convection velocity and diffusivity

	std::shared_ptr<DiffusionObj> diffObj;
};

struct UserData {
	SystemSolver* system;
};