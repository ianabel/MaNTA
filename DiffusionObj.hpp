#pragma once

#include <Eigen/Core>
#include <string>

#include "gridStructures.hpp"

class DiffusionObj
{
public:
	DiffusionObj() {};
	DiffusionObj(int k_, int nVar_);
	DiffusionObj(int k_, int nVar_, std::string diffCase);
	~DiffusionObj() = default;

	std::function<double (double, DGApprox, DGApprox)> getKappaFunc(int var) {return kappaFuncs[var];}

	//Big nVar*(k+1) matrices used in the Jac equation
	void NLqMat(Eigen::MatrixXd& NLq, DGApprox q, DGApprox u, Interval I);
	void NLuMat(Eigen::MatrixXd& NLu, DGApprox q, DGApprox u, Interval I);
	
	//Sub-matrices that make up the larger ones above
	void delqKappaMat(Eigen::MatrixXd& dqkappaMat, int kappa_var, int q_var, DGApprox q, DGApprox u, Interval I);
	void deluKappaMat(Eigen::MatrixXd& dukappaMat, int kappa_var, int u_var, DGApprox q, DGApprox u, Interval I);

	void clear();

	//kept public for ease of assignment
	std::vector<std::function<double (double, DGApprox, DGApprox)>> kappaFuncs;
	std::vector<std::vector<std::function<double (double, DGApprox, DGApprox)>>> delqKappaFuncs, deluKappaFuncs; //[kappa_variable][q/u_varible]
	int k, nVar;

private:
	/*
	Each function here duilds a different diffusion function
	To build new functions first build out the function you want with the correct number of variables
	then inculde the case in the if statements in the constructor of this class so that input files can call your case
	*/
	void buildSingleVariableLinearTest();
	void build2DLinear();
	void build3VariableLinearTest();
	void build1DCritDiff();
	void buildCylinderPlasmaConstDensity();
};
