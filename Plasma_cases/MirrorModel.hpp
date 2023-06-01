#pragma once

#include "Plasma.hpp"
#include "../Variable.hpp"

class MirrorModel : public Plasma
{
public:
	MirrorModel() {};
	~MirrorModel() = default;
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

	double J(double R){return mi*n(R)*R*R;}
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

	double n(double R) const {return 3.0e18;}
	double Jr(DGApprox sigma, double R) const;
	double I_r(double R) const { return 5.0e-3;} //Amperes
	double L_i(double R) const;

	double EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x, int var ) const;

	//Constants
	static const double B_mid = 0.3; //Tesla
	static const double Om_i = e_charge*B_mid/mi;
	static const double Om_e = e_charge*B_mid/me;

	using DGFunction = std::function<double( double, DGApprox, DGApprox, DGApprox )>;

	static double IonParallelLossRate();
	static double IonViscousHeating( double dOmegadPsi, double Ti, double ni );

};
