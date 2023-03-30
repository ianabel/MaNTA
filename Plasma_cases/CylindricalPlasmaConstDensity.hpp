#pragma once

#include "Plasma.hpp"
#include "../Variable.hpp"

class CylindricalPlasmaConstDensity : public Plasma
{
public:
	CylindricalPlasmaConstDensity() {};
	~CylindricalPlasmaConstDensity() = default;
private:
	//-----Over ride functions-----
	//These functions must be built for every derived plasma class
	void pickVariables() override;

	void setKappas() override;
	void setdudKappas() override;
	void setdqdKappas() override;

	void setSources() override;
	void setdudSources() override;
	void setdqdSources() override;
	//-----------------------------

	double tauI(double Pi);
	double dtauIdP(double Pi);
	double lambda();

	//Constants
	double n = 1.0;
	double Te = 30;
	double beta = 1.0; //??TO DO: replace with actual constants
	double gamma = 1e-6; //??TO DO
	double I_r = 1e-2; 
};
