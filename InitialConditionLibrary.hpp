#pragma once

#include <string>
#include <functional>
#include "Plasma_cases/Plasma.hpp"
#include "gridStructures.hpp"

class InitialConditionLibrary
{
public:
	InitialConditionLibrary() = default;
	~InitialConditionLibrary() = default;
	void set(std::string initCond, std::string diffCase)
	{
		initialCondition = initCond;
		diffusionCase = diffCase;
	}
		void set(std::string initCond)
	{
		initialCondition = initCond;
	}

	std::function<double( double, int )> getqInitial();
	std::function<double( double, int )> getuInitial();
	std::function<double( double, int )> getSigInitial(std::shared_ptr<Plasma> plasma, DGApprox& q, DGApprox& u);

private:
	std::string initialCondition = "", diffusionCase = "";
};