#pragma once

#include "../Constants.hpp"
#include "../DiffusionObj.hpp"
#include "../SourceObj.hpp"
#include "../Variable.hpp"
#include "map"
#include "string"
#include "list"

/*
!!!!!!!!!!!!!---NB: Please Read---!!!!!!!!!!!!!
Base class for all plasma properties
When building a Plasma you must build all of the derived classes
All functions of  base plasma do not need to be touched
DO NO BUILD CASE SPECFIC TOOLS HERE!!!
*/

class Plasma
{
public:
	Plasma() {};
	virtual ~Plasma() {};

	std::shared_ptr<DiffusionObj> diffObj;
	std::shared_ptr<SourceObj> sourceObj;

	std::map<std::string, Variable> variables;

	//??TO DO: Function called after derived class constructor finishes to make sure you have built a viable plasma class
	void checkReady();

	//Function to call all virtual functions once overriding from derived class has kicked in
	//Called by system solver after plasma constructors have been called
	void constructPlasma();

	//Searches variable by index. Index is set in the order variables are added in Plasma constructor
	Variable& getVariable(int index);
	Variable& getVariable(std::string name);
protected:
	void addVariable(std::string name);


	void makeDiffObj();
	void makeSourceObj();

	//!!!!!!!---Virtual Functions---!!!!!!!!!!!!
	//The following functions must be made in each plasma case
	//Once correctly made the whole thing should work fine

	virtual void pickVariables() {};	//adds the required variables from the list of possible variables

	//a_fn sits in front of the time derivative term of the diffusion equation.
	//It cannot contain any dependence on any u or qs nor can it contain time dependence.
	virtual void seta_fns() {};

	virtual void setKappas() {};
	virtual void setdudKappas() {};
	virtual void setdqdKappas() {};

	virtual void setSources() {};
	virtual void setdudSources() {};
	virtual void setdqdSources() {};
	virtual void setdsigdSources() {};
	//-------------------------------------------

	//Any new variables must be added here to work for a derived plasma
	//Derived plasmas do not need to use every variable
	static const std::list<std::string> possibleVars;
};
