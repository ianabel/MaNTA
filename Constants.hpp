#pragma once

const double me = 9.1094e-31;	//Electron Mass, kg
const double mi = 1.6726e-27;	//Ion Mass ( = proton mass) kg
const double mp = 1.6726e-27;	//Proton mass, kg
const double e_charge = 1.60217663e-19; //Coulombs
const double eps_0 = 8.8541878128e-12; //vecuum permativity, farads per meter

double eV_J(double T_ev);
double J_eV(double T_J);