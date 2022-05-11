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

class systemLS
{
public:
	systemLS(Grid const& Grid, unsigned int polyNum, unsigned int N_cells, double Dt, Fn const& rhs, Fn const& Tau, Fn const& c, Fn const& kappa, BoundaryConditions const& boundary );
	~systemLS();

	void initialiseMatrices();
	void setCoefficients(N_Vector U, N_Vector Q);
	void solve(double c);
	void returnSunVectors(N_Vector Jv);
private:

	Grid grid;
	std::vector< Eigen::FullPivLU< Eigen::MatrixXd > > ABBDSolvers;
	Eigen::MatrixXd K_global;
	Eigen::VectorXd L_global;
	std::vector< Eigen::MatrixXd > CG_cellwise;
	std::vector< Eigen::VectorXd > RF_cellwise;
	std::vector< Eigen::MatrixXd > QU_0_cellwise;
	std::vector< Eigen::MatrixXd > E_cellwise;

	double dt;
	double t;
	unsigned int const k; 		//polynomial degree per cell
	unsigned int const nCells;	//Total cell count

	Fn RHS; //Forcing function
	Fn c_fn,kappa_fn, tau; // convection velocity and diffusivity
	DGApprox u,q, dudt;
	BoundaryConditions BCs;
};
