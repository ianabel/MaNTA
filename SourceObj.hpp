#pragma once

#include <Eigen/Core>

#include "gridStructures.hpp"

class SourceObj
{
public:
	SourceObj(int k_, int nVar_);
	SourceObj(int k_, int nVar_, std::string diffCase);
	~SourceObj() = default;

	std::function<double (double, DGApprox, DGApprox)> getSourceFunc(int var) {return sourceFuncs[var];}

	//Big nVar*(k+1) matrices used in the Jac equation
	void setdFdqMat(Eigen::MatrixXd& dFdqMatrix, DGApprox q, DGApprox u, Interval I);
	void setdFduMat(Eigen::MatrixXd& dFduMatrix, DGApprox q, DGApprox u, Interval I);
	
	//Sub-matrices that make up the larger ones above
	void dFdqSubMat(Eigen::MatrixXd& dFdqSubMatrix, int F_var, int q_var, DGApprox q, DGApprox u, Interval I);
	void dFduSubMat(Eigen::MatrixXd& dFduSubMatrix, int F_var, int u_var, DGApprox q, DGApprox u, Interval I);

	void clear();

	//kept public for ease of assignment
	std::vector<std::function<double (double, DGApprox, DGApprox)>> sourceFuncs;
	std::vector<std::vector<std::function<double (double, DGApprox, DGApprox)>>> dFdqFuncs, dFduFuncs; //[F_variable][q/u_varible]
	int k, nVar; 

private:
	/*
	Each function here duilds a different diffusion function
	To build new functions first build out the function you want with the correct number of variables
	then inculde the case in the if statements in the constructor of this class so that input files can call your case
	*/
	void buildSingleVariableLinearTest();
	void build3VariableLinearTest();
	void build1DFisherSource();
	void build1DConstSource();
};