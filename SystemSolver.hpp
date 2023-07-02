#pragma once
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <memory>
#include <optional>

#include "gridStructures.hpp"
#include "DiffusionObj.hpp"
#include "SourceObj.hpp"
#include "InitialConditionLibrary.hpp"
#include "Plasma_cases/Plasma.hpp"

class SystemSolver
{
public:
	SystemSolver(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, unsigned int N_Variables, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c);
	SystemSolver(std::string const& inputFile);
	~SystemSolver() = default;

	//Initialises u, q and lambda to satisfy residual equation at t=0
	void setInitialConditions(N_Vector& Y, N_Vector& dYdt );
	void setInitialConditions(std::function< double ( double, int )> u_0, std::function< double ( double, int )> gradu_0, std::function< double ( double, int )> sigma_0, N_Vector& Y, N_Vector& dYdt );

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
	void updateMForJacSolve(std::vector< Eigen::FullPivLU< Eigen::MatrixXd > >& tempABBDBlocks, double const alpha, DGApprox& delSig, DGApprox& delQ, DGApprox& delU);

	//Solves the Jy = -G equation
	void solveJacEq(N_Vector& g, N_Vector& delY);

	void setAlpha(double const a) {alpha = a;}

	//print current output for u and q to output file
	void print( std::ostream& out, double t, int nOut, int var );
	void print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY  );
	void print( std::ostream& out, double t, int nOut, int var, N_Vector& tempY, N_Vector& tempRes );

	//Find the profiles from the coefficients
	double EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var );

	Fn getcfn() const {return c_fn;}
	double getdt() const {return dt;}

	void setDiffobj(std::shared_ptr<DiffusionObj> diffObj_) {diffObj = diffObj_; diffObj->k = k;}
	std::shared_ptr<DiffusionObj> getDiffObj();

	void setSourceobj(std::shared_ptr<SourceObj> sourceObj_) {sourceObj = sourceObj_; sourceObj->k = k;}
	std::shared_ptr<SourceObj> getSourceObj();

	void setTesting(bool t) {testing = t;}
	bool isTesting() const {return testing;}

	void setBoundaryConditions(std::shared_ptr<BoundaryConditions> BC) {BCs = BC;}
	void updateBoundaryConditions(double t);

	Vector resEval(std::vector<Vector> resTerms);

	Grid grid;
	unsigned int k; 		//polynomial degree per cell
	unsigned int nCells;	//Total cell count
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
	std::shared_ptr<Plasma> plasma;
	int total_steps = 0.0;
	double resNorm = 0.0; //Exclusively for unit testing purposes
private:

	double dt;
	double t;
	bool initialised = false;
	double alpha = 1.0;
	bool testing = false;
	bool highGridBoundary = true;

	//??To Be fixed - c and RHS can go. Tau needs to be user input, eventually will be come x dependent 
	Fn RHS = [ = ]( double x ){ return 0.0;};
	Fn c_fn = [ = ]( double x ){ return 0.0;};
	Fn tau = [ = ]( double x ){ return 0.5;};

	//a_fn in the channel dependent term that multiplies the time derivateive. i.e. a_fn*du/dt
	std::vector<std::function<double( double )>> a_fn;

	std::shared_ptr<DiffusionObj> diffObj;
	std::shared_ptr<SourceObj> sourceObj;

	InitialConditionLibrary initConditionLibrary;
};

struct UserData {
	std::shared_ptr<SystemSolver> system;
};
