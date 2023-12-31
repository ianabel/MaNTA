#include "Plasma.hpp"
#include "../Variable.hpp"
#include "CylindricalPlasmaConstDensity.hpp"
#include "Cylinder3Var.hpp"
#include "pouseille.hpp"
#include "ConstVoltage.hpp"
#include "CMFXparallellosses.hpp"

// Update with new variables
const std::list<std::string> Plasma::possibleVars { "n_ion", "n_e", "n_s", "P_ion", "P_e", "P_s", "omega", "velocity" };

void makePlasmaCase(std::string const& plasmaCase, std::shared_ptr<Plasma>& plasma)
{
	if(plasmaCase == "CylindricalPlasmaConstDensity") plasma = std::make_shared<CylindricalPlasmaConstDensity>();
	else if(plasmaCase == "Cylinder3Var") plasma = std::make_shared<Cylinder3Var>();
	else if(plasmaCase == "pouseille") plasma = std::make_shared<Pouseille>();
	else if(plasmaCase == "ConstVoltage") plasma = std::make_shared<ConstVoltage>();
	else if(plasmaCase == "CMFXparallellosses") plasma = std::make_shared<CMFXparallellosses>();
	else throw std::runtime_error("Plasma Case does not exist");
}

void Plasma::constructPlasma()
{
	pickVariables();
	for(auto& var : variables) { var.second.resize(variables.size());}

	seta_fns();
	setKappas();
	setSources();
	setdqdKappas();
	setdudKappas();
	setdsigdSources();
	setdqdSources();
	setdudSources();

	makeDiffObj();
	makeSourceObj();

	checkReady();
}

void Plasma::addVariable(std::string name)
{
	if(std::find(possibleVars.begin(), possibleVars.end(), name) == possibleVars.end()) throw std::runtime_error(name + " is not a viable variable name. Please use a variable located in the possibleVars list found in Plasma.hpp or add a new variable to said list.");
	int index = variables.size();
	variables.insert({name, Variable{name, index}});
	variables.at(name).setVarMap(variables);
	//Note: this means that the order you add your variables is the order they are indexed
	//The core solver will only ever use the index values
}

Variable& Plasma::getVariable(int index)
{
	//inefficient search O(n), but n will never be very big so its grand
	for(auto& var : variables)
	{
		if(var.second.index == index) return var.second;
	}
	throw std::runtime_error("Attempt to call variable which does not exist.");
}

Variable& Plasma::getVariable(std::string name)
{
	return variables.at(name);
}

void Plasma::makeDiffObj()
{
	diffObj = std::make_shared<DiffusionObj>();
	int numVar = variables.size();

	diffObj->nVar = numVar;
	diffObj->delqKappaFuncs.resize(numVar);
	diffObj->deluKappaFuncs.resize(numVar);
	diffObj->clear();

	for(int i=0; i<numVar; i++)
	{
		auto rowVariable = getVariable(i);
		diffObj->kappaFuncs.push_back(rowVariable.kappaFunc);

		for(int j=0; j<numVar; j++)
		{
			diffObj->deluKappaFuncs[i].push_back(rowVariable.deluKappaFuncs[j]);
			diffObj->delqKappaFuncs[i].push_back(rowVariable.delqKappaFuncs[j]);
		}
	}
}

void Plasma::makeSourceObj()
{
	sourceObj = std::make_shared<SourceObj>();
	int numVar = variables.size();

	sourceObj->nVar = numVar;
	sourceObj->delsigSourceFuncs.resize(numVar);
	sourceObj->delqSourceFuncs.resize(numVar);
	sourceObj->deluSourceFuncs.resize(numVar);
	sourceObj->clear();

	for(int i=0; i<numVar; i++)
	{
		auto rowVariable = getVariable(i);
		sourceObj->sourceFuncs.push_back(rowVariable.sourceFunc);

		for(int j=0; j<numVar; j++)
		{
			sourceObj->deluSourceFuncs[i].push_back(rowVariable.deluSourceFuncs[j]);
			sourceObj->delqSourceFuncs[i].push_back(rowVariable.delqSourceFuncs[j]);
			sourceObj->delsigSourceFuncs[i].push_back(rowVariable.delsigSourceFuncs[j]);
		}
	}
}

void Plasma::checkReady()
{
	//??To Do??
}
