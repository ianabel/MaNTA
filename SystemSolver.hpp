#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <memory>
#include <optional>

#include "gridStructures.hpp"
#include "TransportSystem.hpp"

class SystemSolver
{
public:
	using Matrix = Eigen::MatrixXd;

	SystemSolver(Grid const& Grid, unsigned int polyNum, double Dt, TransportSystem *pProblem );

	// This has been moved elsewhere, SystemSolver should be constructed after the parsing is done.
	// SystemSolver(std::string const& inputFile);
	~SystemSolver() = default;

	//Initialises u, q and lambda to satisfy residual equation at t=0
	void setInitialConditions(N_Vector& Y, N_Vector& dYdt );

	//Initialize the a_fns
	void seta_fns();

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
	void updateMForJacSolve(std::vector< Eigen::FullPivLU< Matrix > >& tempABBDBlocks, double const alpha, DGApprox& delSig, DGApprox& delQ, DGApprox& delU);

	//Solves the Jy = -G equation
	void solveJacEq(N_Vector& g, N_Vector& delY);

	void setAlpha(double const a) {alpha = a;}

	//print current output for u and q to output file
	void print( std::ostream& out, double t, int nOut, int var );
	void print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY  );
	void print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY, N_Vector& tempRes );

	//Find the profiles from the coefficients
	double EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var );

	double getdt() const {return dt;}

	void setTesting(bool t) {testing = t;}
	bool isTesting() const {return testing;}

	void updateBoundaryConditions(double t);

	Vector resEval(std::vector<Vector> resTerms);

private:
	Grid grid;
	unsigned int k; 		//polynomial degree per cell
	unsigned int nCells;	//Total cell count
	int nVar;					//Total number of variables
	
	std::vector< Matrix > XMats;
	std::vector< Matrix > MBlocks;
	std::vector< Matrix > CEBlocks;
	Matrix K_global{};
	Eigen::VectorXd L_global{};
	Matrix H_global_mat{};
	Eigen::FullPivLU< Matrix > H_global{};
	std::vector< Matrix > CG_cellwise, RF_cellwise;
	std::vector< Matrix > A_cellwise, B_cellwise, D_cellwise, E_cellwise, C_cellwise, G_cellwise, H_cellwise; //?Point the dublicated matrices to the same place?

	using DGVector = std::vector<DGApprox>;
	DGVector u, q, sig, dudt, dqdt, dsigdt;
	std::optional<Eigen::Map<Eigen::VectorXd>> lambda, dlamdt;

	DGSoln y, dydt;

	void NLqMat( Matrix &, DGVector const&, DGVector const& );
	void NLuMat( Matrix &, DGVector const&, DGVector const& );

	std::shared_ptr<BoundaryConditions> BCs;
	int total_steps = 0;
	double resNorm = 0.0; //Exclusively for unit testing purposes

	double dt;
	double t;

	// Really we should do init in the constructor and not need this flag. TODO
	bool initialised = false;

	double alpha = 1.0;
	bool testing = false;
	bool highGridBoundary = true;

	// Hide all physics-specific info in here
	TransportSystem *problem = nullptr;
};

struct UserData {
	std::shared_ptr<SystemSolver> system;
};
