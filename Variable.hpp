#pragma once

#include <string>
#include <memory>
#include <map>
#include <algorithm>
#include <vector>
#include <functional>

class DGApprox;

class Variable
{
public:
	Variable(std::string Name, int index) : name(Name), index(index) {};
	~Variable() = default;

	void setVarMap(std::map<std::string, Variable> vars ){variables.reset(&vars);}
	void setKappaFunc(std::function<double (double)> a_fn_){a_fn = a_fn_;}
	void setKappaFunc(std::function<double (double, DGApprox, DGApprox)> kappa){kappaFunc = kappa;}
	void setSourceFunc(std::function<double (double, DGApprox, DGApprox, DGApprox)> source){sourceFunc = source;}

	void addDelqKappaFunc(int vIndex, std::function<double (double, DGApprox, DGApprox)> func){delqKappaFuncs[vIndex] = func;}
	void addDeluKappaFunc(int vIndex, std::function<double (double, DGApprox, DGApprox)> func){deluKappaFuncs[vIndex] = func;}

	void addDelsigSourceFunc(int vIndex,  std::function<double (double, DGApprox, DGApprox, DGApprox)> func){delsigSourceFuncs[vIndex] = func;}
	void addDelqSourceFunc(int vIndex, std::function<double (double, DGApprox, DGApprox, DGApprox)> func){delqSourceFuncs[vIndex] = func;}
	void addDeluSourceFunc(int vIndex,  std::function<double (double, DGApprox, DGApprox, DGApprox)> func){deluSourceFuncs[vIndex] = func;}

	void resize(int size);

	const std::string name;
	const int index;

	std::function<double (double)> a_fn;
	std::function<double (double, DGApprox, DGApprox)> kappaFunc;
	std::function<double (double, DGApprox, DGApprox, DGApprox)> sourceFunc;
	std::shared_ptr<std::map<std::string, Variable>> variables = nullptr;

	//both maps should be the same size as variables map
	//Note: Vectors should be filled by the index of the differentiating variable
	std::vector<std::function<double (double, DGApprox, DGApprox)>> delqKappaFuncs, deluKappaFuncs;
	std::vector<std::function<double (double, DGApprox, DGApprox, DGApprox)>> delsigSourceFuncs, delqSourceFuncs, deluSourceFuncs;
};
