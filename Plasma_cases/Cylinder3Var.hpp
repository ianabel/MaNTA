#pragma once

#include "Plasma.hpp"
#include "../Variable.hpp"

class Cylinder3Var : public Plasma
{
public:
	Cylinder3Var() {};
	~Cylinder3Var() = default;
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
	void setdsigdSources() override;
	//-----------------------------

	double tauI(double Pi, double R);
	double dtauIdP_i(double Pi, double R);
	double tauE(double Pi, double R);
	double dtauEdP_e(double Pe, double R);
	double lambda(double R);
	double nu(double Pi, double Pe,  double R); //Collision operator
	double dnudPi(double Pi, double Pe,  double R);
	double dnudPe(double Pi, double Pe,  double R);

	double Ce(double Pi, double Pe, double R); //Collisions ion-electron heat flux
	double dCedPe(double Pi, double Pe, double R);
	double dCedPi(double Pi, double Pe, double R);

	double Ci(double Pi, double Pe, double R); //Collisions electron-ion heat flux
	double dCidPe(double Pi, double Pe, double R);
	double dCidPi(double Pi, double Pe, double R);

	double n(double R) {return 3.0e18;}
	double J(double R){return ionMass*n(R)*R*R;}
	double I_r(double R); //Amperes

	//Constants
	const double B_mid = 0.3; //Tesla
	const double Om_i = e_charge*B_mid/ionMass;
	const double Om_e = e_charge*B_mid/electronMass;
};