#pragma once

#include <Eigen/Core>

#include "gridStructures.hpp"

class DiffusionObj
{
public:
	DiffusionObj(int k_, int nVar_);
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
};