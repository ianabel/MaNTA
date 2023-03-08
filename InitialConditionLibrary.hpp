#pragma once

#include <string>
#include <functional>

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

	std::function<double( double, int )> getqInitial();
	std::function<double( double, int )> getuInitial();
	std::function<double( double, int )> getSigInitial();

private:
	std::string initialCondition = "", diffusionCase = "";
};