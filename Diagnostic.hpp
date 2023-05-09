#pragma once
#include <memory>
#include "Plasma_cases/Plasma.hpp"
#include "SystemSolver.hpp"

/*
Diagnostic is essentially a suite of methods that can output measurable quantities of the plasma. 
Each diagnostic function should check that the necessary quantites are being modeled.
Then if possible the quantity can be calsulated
*/

class Diagnostic
{
public:
	Diagnostic(std::shared_ptr<SystemSolver> system_, std::shared_ptr<Plasma> plasma_) : system(system_), plasma(plasma_) {}
	~Diagnostic() = default;
	double Voltage() const;
private:
	std::shared_ptr<SystemSolver> system;
	std::shared_ptr<Plasma> plasma;
};
