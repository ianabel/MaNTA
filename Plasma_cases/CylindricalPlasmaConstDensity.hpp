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

	void seta_fns() override;

	void setKappas() override;
	void setdudKappas() override;
	void setdqdKappas() override;

	void setSources() override;
	void setdudSources() override;
	void setdqdSources() override;
	//-----------------------------

	double tauI(double Pi, double R);
	double dtauIdP(double Pi, double R);
	double lambda(double R);
	double n(double R) {return 3.0e17;}
	double J(double R){return mi*n(R)*R*R;}
	double I_r(double R){ return 1.0e-1*(-1.0*R*R+10.0*R-21.0)/4.0;} //Amperes
	double beta(double R){return 4.0/(3.0*Om*Om*mi*n(R));}

	//Constants
	const double Te = eV_J(40.0);
	const double B_mid = 0.3; //Tesla
	const double Om = e_charge*B_mid/mi;
	const double gamma = 3.0e5/(10.0*Om*Om); //??TO DO: this only really affects things when its 10^6 times larger
};