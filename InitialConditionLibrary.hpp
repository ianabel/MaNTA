#pragma once

#include <string>
#include <functional>

typedef std::function<double( double )> Fn;

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

	Fn getqInitial();
	Fn getuInitial();
	Fn getSigInitial();

private:
	std::string initialCondition = "", diffusionCase = "";
};